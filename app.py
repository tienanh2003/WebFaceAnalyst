from flask import Flask, request, jsonify, render_template,  Response
from utils.database import Database
from utils.accounts import AccountService
from utils.users import UserService
from utils.face_embeddings import FaceEmbeddingService
from utils.sessions import SessionService
from utils.emotions import EmotionService
from utils.emotion_analysis_results import EmotionAnalysisResultService
from utils.videos import VideoService
from utils.person_appearances import PersonAppearanceService
from utils.strangers import StrangerService

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import tensorflow as tf
import torch
import numpy as np
from retinaface import RetinaFace
import multiprocessing
from transformers import ViTForImageClassification, ViTFeatureExtractor
#from insightface.model_zoo import get_model
from insightface.app import FaceAnalysis
from PIL import Image
from torchvision import transforms
import traceback


app = Flask(__name__)

db = Database()

account_service = AccountService(db)
user_service = UserService(db)
face_embedding_service = FaceEmbeddingService(db)
session_service = SessionService(db)
emotion_service = EmotionService(db)
emotion_analysis_result_service = EmotionAnalysisResultService(db)
video_service = VideoService(db)
person_appearance_service = PersonAppearanceService(db)
stranger_service = StrangerService(db)

device = "cuda" if torch.cuda.is_available() else "cpu"
retinaface_model = RetinaFace.build_model()
model_path = "./model/vit_best_model.pt"
frame = "./image/face.jpg"

vit_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", 
    num_labels=7,  
    ignore_mismatched_sizes=True
)
vit_model.load_state_dict(torch.load(model_path, map_location=device))
vit_model.to(device)
vit_model.eval()

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

try:
    arcface_model = FaceAnalysis(name="buffalo_l") 
    arcface_model.prepare(ctx_id=-1)  
    arcface_model.models['recognition']
    print("ArcFace model loaded successfully")
except Exception as e:
    print(f"Error loading ArcFace model: {e}")

def resize_with_padding(image, target_size=(224, 224), pad_color=(0, 0, 0)):

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

def face_analyst_frame(frame):
    try:
        if frame is None:
            raise ValueError("Frame is None. Please check the input image.")

        frame = resize_with_padding(frame, (224, 224))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inputs = feature_extractor(images=frame_rgb, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = vit_model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()

        classes = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Class 7"]
        return classes[predicted_class]
    except Exception as e:
        print(f"Error in face_analyst_frame: {e}")
        return None

def preprocess_face(image, target_size=(112, 112)):
    if image is None:
        raise ValueError("Input image is None. Please check the input.")

    image = resize_with_padding(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = Image.fromarray(image)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(image).unsqueeze(0)

def get_face_embedding(image):
    try:
        if image is None:
            raise ValueError(f"Cannot read image from {image}. Please check the file path.")

        faces = arcface_model.get(image)  
        if len(faces) == 0:
            raise ValueError("No faces detected in the image.")

        embedding = faces[0].embedding
        return embedding
    except Exception as e:
        print(f"Error in get_face_embedding: {e}")
        return None

def similarity_face(embedding1, embedding2):
    cosine_similarity = np.dot(embedding1, embedding2.T) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return cosine_similarity

@app.route("/detect", methods=["GET"])
def detect_face():
    frame_path = "./image/face.jpg"

    try:
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Cannot read image from {frame_path}. Please check the file path.")
        
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame is not a valid numpy array.")

        print(f"Frame shape: {frame.shape if frame is not None else 'None'}")
        result = face_analyst_frame(frame)
        if result is None:
            return jsonify({"error": "Failed to analyze face in the frame."}), 400

        return jsonify({"result": result}), 200
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error occurred:\n{error_details}")
        return jsonify({"error": str(e), "details": error_details}), 500

@app.route("/embedding", methods=["GET"])
def embedding_arcface():
    frame_path = "./image/face.jpg"
    if not frame_path:
        return jsonify({"error": "Frame path is required"}), 400

    try:
        frame = cv2.imread(frame_path)
        if frame is not None:
            print(f"Image shape: {frame.shape}")
        else:
            print("Error: Unable to read image.")

        face_image = preprocess_face(frame)
        embedding = get_face_embedding(frame)

        if embedding is None:
            raise ValueError("Failed to extract embedding from the image.")

        return jsonify({"embedding": embedding.tolist()}), 200
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error occurred:\n{error_details}")
        return jsonify({"error": str(e), "details": error_details}), 500

@app.route("/compare", methods=["GET"])
def compare_faces():
    try:
        frame1_path = "./image/face.jpg"
        frame2_path = "./image/face1.jpg"

        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)

        if frame1 is None or frame2 is None:
            return jsonify({"error": "One or both images could not be read. Please check the file paths."}), 400

        embedding1 = get_face_embedding(frame1)
        embedding2 = get_face_embedding(frame2)

        if embedding1 is None or embedding2 is None:
            return jsonify({"error": "Failed to extract embeddings from one or both images."}), 400

        similarity = similarity_face(embedding1, embedding2)
        return jsonify({"similarity": float(similarity)}), 200

    except Exception as e:
        error_details = traceback.format_exc()
        print("Error occurred:\n", error_details)
        return jsonify({"error": str(e), "details": error_details}), 500

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/test_connection")
def test_connection():
    try:
        test_collection = db.get_collection("testdb")
        test_collection.insert_one({"message": "Test connection successful"})
        return jsonify({"status": "success", "message": "Connection to MongoDB successful!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/login")
def user_login():
    return render_template("login.html")

@app.route("/register")
def admin_login():
    return render_template("register.html")

@app.route("/home", methods=["POST"])
def home():
    username = request.form.get("username")
    password = request.form.get("password")

    if username == "tienanh" and password == "tienanh":
        user_id = 1  
        return render_template("user.html", username=username, user_id=user_id)

    if username == "admin" and password == "admin":
        admin_id = 1  
        return render_template("admin.html", username=username, admin_id=admin_id)
    return render_template("login.html", error="Invalid username or password")

def detect_faces_with_model(frame_rgb):

    try:
        faces = RetinaFace.detect_faces(frame_rgb, model=retinaface_model)
        return faces
    except Exception as e:
        print(f"Error during face detection: {e}")
        return {}

def process_frame_worker(input_queue, output_queue, process_id):

    while True:
        frame = input_queue.get()
        if frame is None:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            faces = detect_faces_with_model(frame_rgb)

            for _, face in faces.items():
                x1, y1, x2, y2 = face["facial_area"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            output_queue.put((process_id, frame))
        except Exception as e:
            print(f"Process {process_id} error: {e}")
            output_queue.put((process_id, frame))

def generate_camera():

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)

    input_queues = [multiprocessing.Queue() for _ in range(4)]
    output_queue = multiprocessing.Queue()

    processes = [
        multiprocessing.Process(
            target=process_frame_worker, args=(input_queues[i], output_queue, i)
        )
        for i in range(4)
    ]

    for p in processes:
        p.start()

    frame_count = 0

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            process_id = frame_count % 4
            input_queues[process_id].put(frame)
            frame_count += 1

            processed_frames = {}
            while len(processed_frames) < 4:
                process_id, processed_frame = output_queue.get()
                processed_frames[process_id] = processed_frame

            ordered_frames = [processed_frames[i] for i in range(4)]

            combined_frame = np.hstack(ordered_frames)
            cv2.imshow("Processed Frames", combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:

        camera.release()
        cv2.destroyAllWindows()

        for q in input_queues:
            q.put(None)
        for p in processes:
            p.join()

@app.route("/camera_user")
def camera_user():
    return Response(generate_camera(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
    
