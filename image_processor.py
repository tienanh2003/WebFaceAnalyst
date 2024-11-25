import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import torch
import numpy as np
from retinaface import RetinaFace
from transformers import ViTForImageClassification, ViTFeatureExtractor
from insightface.app import FaceAnalysis
import faiss
from torchvision import transforms
import traceback
import base64

class ImageProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load RetinaFace model
        self.retinaface_model = RetinaFace.build_model()

        # Load ViT model for emotion recognition
        model_path = "./model/vit_best_model.pt"
        self.vit_model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=7,
            ignore_mismatched_sizes=True
        )
        self.vit_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.vit_model.to(self.device)
        self.vit_model.eval()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

        # Load ArcFace model for face recognition
        try:
            self.arcface_model = FaceAnalysis(name="buffalo_l")
            self.arcface_model.prepare(ctx_id=-1)
            self.arcface_model.models['recognition']
            print("ArcFace model loaded successfully")
        except Exception as e:
            print(f"Error loading ArcFace model: {e}")

        # Initialize FAISS index for face embeddings
        self.data = []
        self.dimension = 512  # Embedding dimension
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        self.names = []

    def resize_with_padding(self, image, target_size=(224, 224), pad_color=(0, 0, 0)):
        if image is None:
            raise ValueError("Input image is None. Please check the input.")

        original_height, original_width = image.shape[:2]
        target_width, target_height = target_size

        scale = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))

        pad_width = target_width - new_width
        pad_height = target_height - new_height
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left

        padded_image = cv2.copyMakeBorder(
            resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
        )
        return padded_image

    def run_vit_model(self, image):
        try:
            if image is None:
                raise ValueError("Frame is None. Please check the input image.")

            image = self.resize_with_padding(image, target_size=(224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            with torch.no_grad():
                outputs = self.vit_model(**inputs)
                logits = outputs.logits
                predicted_class = logits.argmax(-1).item()

            classes = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Angry", "Neutral"]
            return classes[predicted_class]
        except Exception as e:
            print(f"Error in run_vit_model: {e}")
            return None

    def get_face_embedding(self, image):
        try:
            if image is None or not isinstance(image, np.ndarray):
                raise ValueError("Input image is invalid or None. Please check the input.")

            image_normalized = (image - 127.5) / 127.5  # Normalize to [-1, 1]
            image_normalized = image_normalized.astype(np.float32)
            input_data = np.expand_dims(image_normalized, axis=0)  # (1, height, width, channels)
            input_data = np.transpose(input_data, (0, 3, 1, 2))  # (1, channels, height, width)

            embedding = self.arcface_model.models['recognition'].forward(input_data)
            if embedding is None:
                raise ValueError("Failed to generate embedding for the face.")

            return embedding[0]  # Return the first embedding
        except Exception as e:
            print(f"Error in get_face_embedding: {e}")
            return None

    def detect_faces_with_model(self, frame_rgb):
        try:
            faces = RetinaFace.detect_faces(frame_rgb, model=self.retinaface_model)
            if not isinstance(faces, dict):
                faces = {}
            return faces
        except Exception as e:
            print(f"Error during face detection: {e}")
            return {}

    def find_closest_face(self, embedding, threshold=0.6):
        if self.faiss_index.ntotal == 0:
            return "Guest", None

        try:
            embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
            distances, indices = self.faiss_index.search(embedding, 1)

            if distances[0][0] > threshold:
                return "Guest", float(distances[0][0])

            nearest_index = indices[0][0]
            return self.names[nearest_index], float(distances[0][0])
        except Exception as e:
            print(f"Error in find_closest_face: {e}")
            return "Guest", None

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            faces = self.detect_faces_with_model(frame_rgb)

            for _, face in faces.items():
                x1, y1, x2, y2 = face["facial_area"]

                cropped_face = frame_rgb[y1:y2, x1:x2]
                if cropped_face.size == 0:
                    continue

                emotion = self.run_vit_model(cropped_face)

                cropped_face_resized = cv2.resize(cropped_face, (112, 112))
                embedding = self.get_face_embedding(cropped_face_resized)
                if embedding is None:
                    continue

                name, _ = self.find_closest_face(embedding)
                if not name:
                    name = "Guest"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{name}: {emotion}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error in process_frame: {e}")

        return frame

    def crop_face_by_coordinates(self, image, x1, y1, x2, y2):
        try:
            height, width, _ = image.shape
            x1 = max(0, min(width, x1))
            y1 = max(0, min(height, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))
            cropped_face = image[y1:y2, x1:x2]

            if cropped_face.size == 0:
                print(f"Invalid crop region with coordinates: {(x1, y1, x2, y2)}")
                return None
            return cropped_face
        except Exception as e:
            print(f"Error in crop_face_by_coordinates: {e}")
            return None

    def similarity_face(self, embedding1, embedding2):
        cosine_similarity = np.dot(embedding1, embedding2.T) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return cosine_similarity
