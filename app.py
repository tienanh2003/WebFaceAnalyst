from flask import Flask, request, jsonify, render_template,  Response, send_file, url_for, send_from_directory
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
from torchvision import transforms
import traceback
import base64
from io import BytesIO
from flask import send_from_directory
import faiss
from werkzeug.utils import secure_filename

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

data = []

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

def run_vit_model(image):
    try:
        if image is None:
            raise ValueError("Frame is None. Please check the input image.")

        image = resize_with_padding(image, target_size=(224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = vit_model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()

        classes = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Angry", "Neutral"]
        return classes[predicted_class]
    except Exception as e:
        print(f"Error in face_analyst_frame: {e}")
        return None

def preprocess_face(image, target_size=(112, 112)):

    try:
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Input image is invalid or None. Please check the input.")

        image = resize_with_padding(image, target_size)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ])

        image_tensor = transform(image).unsqueeze(0) 

        return image_tensor
    except Exception as e:
        print(f"Error in preprocess_face: {e}")
        return None

def get_face_embedding(image):
    try:
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Input image is invalid or None. Please check the input.")
        
        # Đảm bảo ảnh đã chuẩn hóa về [-1, 1]
        image_normalized = (image - 127.5) / 127.5  # Chuẩn hóa về [-1, 1]

        # Chuyển kiểu dữ liệu sang float32
        image_normalized = image_normalized.astype(np.float32)

        # Thêm chiều batch size và hoán đổi thứ tự các chiều
        input_data = np.expand_dims(image_normalized, axis=0)  # (1, height, width, channels)
        input_data = np.transpose(input_data, (0, 3, 1, 2))  # (1, channels, height, width)

        # Tính embedding
        embedding = arcface_model.models['recognition'].forward(input_data)
        if embedding is None:
            raise ValueError("Failed to generate embedding for the face.")

        return embedding[0]  # Return the first embedding
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

        result = run_vit_model(frame)
        if result is None:
            return jsonify({"error": "Failed to analyze face in the frame."}), 400

        return jsonify({"result": result}), 200
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error occurred:\n{error_details}")
        return jsonify({"error": str(e), "details": error_details}), 500

def crop_face_by_coordinates(image, x1, y1, x2, y2):

    try:
        height, width, _ = image.shape

        # Ensure the coordinates are within the image size
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))

        # Crop the region
        cropped_face = image[y1:y2, x1:x2]

        # Check if the cropped region is valid
        if cropped_face.size == 0:
            print(f"Invalid crop region with coordinates: {(x1, y1, x2, y2)}")
            return None

        return cropped_face
    except Exception as e:
        print(f"Error in crop_face_by_coordinates: {e}")
        return None

@app.route("/embedding", methods=["GET"])
def embedding_arcface():
    frame_path = "./image/face1.jpg"
    save_path = "./imagehandle"
    if not os.path.exists(save_path):
        os.makedirs(save_path)  # Create folder if it doesn't exist

    try:
        # Load the image
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Cannot read image from {frame_path}. Please check the file path.")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using the model
        faces = detect_faces_with_model(frame_rgb)
        if not faces:
            return jsonify({"error": "No faces detected in the image."}), 400

        embeddings = []
        face_index = 0  # Index for saving cropped faces
        for _, face in faces.items():
            x1, y1, x2, y2 = face["facial_area"]

            # Step 1: Crop the face region
            cropped_face = crop_face_by_coordinates(frame_rgb, x1, y1, x2, y2)
            if cropped_face is None:
                print(f"Failed to crop face region for coordinates: {(x1, y1, x2, y2)}")
                continue

            # Save the cropped face
            cropped_face_path = os.path.join(save_path, f"cropped_face_{face_index}.jpg")
            cv2.imwrite(cropped_face_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
            print(f"Saved cropped face: {cropped_face_path}")

            # Step 2: Resize the cropped face
            cropped_face_resized = resize_with_padding(cropped_face, target_size=(112, 112))
            if cropped_face_resized is None:
                print(f"Failed to resize the cropped face for coordinates: {(x1, y1, x2, y2)}")
                continue

            # Save the resized face
            resized_face_path = os.path.join(save_path, f"resized_face_{face_index}.jpg")
            cv2.imwrite(resized_face_path, cv2.cvtColor(cropped_face_resized, cv2.COLOR_RGB2BGR))
            print(f"Saved resized face: {resized_face_path}")

            print(f"Type before embedding: {type(cropped_face_resized)}")
            print(f"Shape before embedding: {cropped_face_resized.shape}")
            print(f"Data type before embedding: {cropped_face_resized.dtype}")

            # Step 3: Generate the face embedding
            embedding = get_face_embedding(cropped_face_resized)
            if embedding is None:
                print(f"Failed to generate embedding for the cropped face.")
                continue

            embeddings.append(embedding.tolist())  # Convert to list for JSON serialization
            face_index += 1  # Increment face index

        if not embeddings:
            return jsonify({"error": "Failed to generate embeddings for any faces in the image."}), 400

        return jsonify({"embeddings": embeddings}), 200

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"An error occurred:\n{error_details}")
        return jsonify({"error": str(e), "details": error_details}), 500

# @app.route("/detect_embedding", methods=["POST"])
# def detect_embedding_arcface():
#     try:
#         # Check if an image was uploaded
#         if 'image' not in request.files:
#             return jsonify({"error": "No image uploaded"}), 400

#         file = request.files['image']
#         if file.filename == '':
#             return jsonify({"error": "No selected file"}), 400

#         # Read the image in memory
#         image_stream = file.stream
#         file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
#         frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#         if frame is None:
#             return jsonify({"error": "Invalid image"}), 400

#         # Convert image to RGB for face detection
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Detect faces using the model
#         faces = detect_faces_with_model(frame_rgb)
#         if not faces:
#             return jsonify({"error": "No faces detected in the image."}), 400

#         face_data = []
#         face_index = 1

#         # Process each face
#         for _, face in faces.items():
#             x1, y1, x2, y2 = face["facial_area"]

#             # Ensure coordinates are within image bounds
#             height, width, _ = frame.shape  # Use original frame dimensions
#             x1 = max(0, min(width, x1))
#             y1 = max(0, min(height, y1))
#             x2 = max(0, min(width, x2))
#             y2 = max(0, min(height, y2))

#             # Crop the face from the BGR image
#             cropped_face_bgr = frame[y1:y2, x1:x2]
#             if cropped_face_bgr is None or cropped_face_bgr.size == 0:
#                 continue

#             # For embedding extraction, we need the face in RGB
#             cropped_face_rgb = cv2.cvtColor(cropped_face_bgr, cv2.COLOR_BGR2RGB)

#             # Resize face for embedding extraction if necessary
#             if cropped_face_rgb.shape[:2] != (112, 112):
#                 cropped_face_resized = cv2.resize(cropped_face_rgb, (112, 112))
#             else:
#                 cropped_face_resized = cropped_face_rgb

#             # Extract embedding
#             embedding = get_face_embedding(cropped_face_resized)
#             if embedding is None:
#                 continue

#             # For ViT model, we can use the cropped face (resize if necessary)
#             emotion_label = run_vit_model(cropped_face_rgb)
#             if emotion_label is None:
#                 emotion_label = "Unknown"

#             # Draw bounding box and label on the original BGR image
#             label = f"Face {face_index}: {emotion_label}"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5, (0, 255, 0), 2)

#             # Encode cropped face image to base64
#             _, face_buffer = cv2.imencode('.jpg', cropped_face_bgr)
#             face_base64 = base64.b64encode(face_buffer).decode('utf-8')

#             # Collect face data
#             face_info = {
#                 "face_index": face_index,
#                 "embedding": embedding.tolist(),  # Convert numpy array to list
#                 "emotion": emotion_label,
#                 "face_image": face_base64
#             }
#             face_data.append(face_info)
#             face_index += 1

#         # Encode the result image to base64
#         _, buffer = cv2.imencode('.jpg', frame)
#         result_image_base64 = base64.b64encode(buffer).decode('utf-8')

#         response_data = {
#             "result_image": result_image_base64,
#             "faces": face_data
#         }

#         return jsonify(response_data), 200

#     except Exception as e:
#         error_details = traceback.format_exc()
#         print(f"An error occurred:\n{error_details}")
#         return jsonify({"error": str(e), "details": error_details}), 500

def find_nearest_face(embedding, threshold=0.6):
    try:
        # Lấy embedding từ data
        if not data:
            return None, None

        embeddings = np.array([np.array(user['embedding']) for user in data]).astype('float32')
        names = [user['name'] for user in data]

        # Tạo FAISS index
        d = embeddings.shape[1]  # Số chiều của embedding
        index = faiss.IndexFlatL2(d)  # Index L2
        index.add(embeddings)  # Thêm embedding vào index

        # Tìm embedding gần nhất
        distances, indices = index.search(np.array([embedding]).astype('float32'), 1)

        # Kiểm tra khoảng cách có nằm trong ngưỡng
        nearest_distance = distances[0][0]
        nearest_index = indices[0][0]

        if nearest_distance <= threshold:
            return names[nearest_index], nearest_distance  # Trả về tên và khoảng cách
        else:
            return "Người lạ", nearest_distance

    except Exception as e:
        print(f"Error in find_nearest_face: {e}")
        return None, None

def find_closest_face(embedding, threshold=0.6):
    if not data:
        return None, None

    min_distance = float("inf")
    closest_name = None

    for user in data:
        stored_embedding = np.array(user["embedding"])
        distance = np.linalg.norm(embedding - stored_embedding)

        if distance < min_distance and distance < threshold:
            min_distance = distance
            closest_name = user["name"]

    return closest_name, min_distance if closest_name else None

@app.route("/detect_embedding", methods=["POST"])
def detect_embedding_arcface():
    try:
        # Kiểm tra xem ảnh có được gửi không
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Đọc ảnh từ file upload
        image_stream = file.stream
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image"}), 400

        # Chuyển đổi ảnh sang RGB để phát hiện khuôn mặt
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detect_faces_with_model(frame_rgb)

        if not faces:
            return jsonify({"error": "No faces detected in the image."}), 400

        face_data = []  # Danh sách lưu thông tin khuôn mặt
        for face_id, face in faces.items():
            x1, y1, x2, y2 = face["facial_area"]

            # Crop và phân tích khuôn mặt
            cropped_face = frame_rgb[y1:y2, x1:x2]
            if cropped_face.size == 0:
                continue

            cropped_face_resized = cv2.resize(cropped_face, (112, 112))
            embedding = get_face_embedding(cropped_face_resized)
            emotion = run_vit_model(cropped_face)

            # Tìm tên người gần nhất hoặc đánh dấu là Guest
            name, distance = find_closest_face(embedding)

            if not name:
                name = "Guest"

            # Vẽ bounding box và tên trên ảnh gốc
            label = f"{name}: {emotion}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

            # Encode khuôn mặt đã cắt sang Base64
            _, face_buffer = cv2.imencode('.jpg', cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
            face_base64 = base64.b64encode(face_buffer).decode('utf-8')

            # Lưu thông tin khuôn mặt
            face_data.append({
                "face_index": face_id,
                "name": name,
                "distance": distance,
                "emotion": emotion,
                "face_image": face_base64,
            })

        # Encode ảnh kết quả (ảnh gốc có bounding box) sang Base64
        _, buffer = cv2.imencode('.jpg', frame)
        result_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "result_image": result_image_base64,
            "faces": face_data
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/compare", methods=["GET"])
def compare_faces():
    try:
        frame1_path = "./image/face.jpg"
        frame2_path = "./image/face1.jpg"

        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)

        if frame1 is None or frame2 is None:
            return jsonify({"error": "One or both images could not be read. Please check the file paths."}), 400
        
        frame1 = resize_with_padding(frame1, target_size=(112, 112))
        frame2 = resize_with_padding(frame2, target_size=(112, 112))

        embedding1 = get_face_embedding(frame1)
        embedding2 = get_face_embedding(frame2)
        print("After Compare")
        print(embedding1)
        print(embedding2)

        if embedding1 is None or embedding2 is None:
            return jsonify({"error": "Failed to extract embeddings from one or both images."}), 400

        similarity = similarity_face(embedding1, embedding2)
        return jsonify({"similarity": float(similarity)}), 200

    except Exception as e:
        error_details = traceback.format_exc()
        print("Error occurred:\n", error_details)
        return jsonify({"error": str(e), "details": error_details}), 500

def process_video(input_path, output_path, time_s):
    try:
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError("Cannot open video file.")

        # Get video information
        fps = min(30, int(cap.get(cv2.CAP_PROP_FPS)))  # Limit FPS to a maximum of 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate the maximum number of frames to process
        max_frames = min(total_frames, int(time_s * fps))

        # Define codec and writer for the output video
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            raise ValueError("VideoWriter failed to open. Check codec or file path.")

        frame_count = 0

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                print("No more frames to read.")
                break

            # Process each frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detect_faces_with_model(frame_rgb)

            for _, face in faces.items():
                x1, y1, x2, y2 = face["facial_area"]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)

                # Crop and analyze face
                cropped_face = frame_rgb[y1:y2, x1:x2]
                if cropped_face.size == 0:
                    continue  # Skip invalid face

                # Resize and compute embedding
                cropped_face_resized = cv2.resize(cropped_face, (112, 112))
                embedding = get_face_embedding(cropped_face_resized)
                emotion = run_vit_model(cropped_face)

                # Find closest face in data
                name, _ = find_closest_face(embedding)
                if name is None:
                    name = "Guest"

                # Draw bounding box and label on the frame
                label = f"{name}: {emotion}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

            # Write frame to output video
            writer.write(frame)
            frame_count += 1
            print(f"Processing frame {frame_count}/{max_frames}...")

        # Release resources
        cap.release()
        writer.release()
        print(f"Finished processing {frame_count} frames. Video saved at: {output_path}")

    except Exception as e:
        print(f"An error occurred in process_video:\n{traceback.format_exc()}")
# def process_video(input_path, output_path, time_s):
#     try:
#         cap = cv2.VideoCapture(input_path)
#         if not cap.isOpened():
#             raise ValueError("Cannot open video file.")

#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         print(f"Video info: FPS={fps}, Total Frames={total_frames}, Resolution={width}x{height}")

#         # Mở VideoWriter
#         fourcc = cv2.VideoWriter_fourcc(*'H264')
#         writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#         if not writer.isOpened():
#             raise ValueError("Failed to open VideoWriter.")

#         frame_count = 0
#         while frame_count < total_frames:
#             ret, frame = cap.read()
#             if not ret:
#                 print("No more frames to read.")
#                 break

#             # Ghi từng frame (không chỉnh sửa để kiểm tra)
#             writer.write(frame)
#             frame_count += 1
#             print(f"Processing frame {frame_count}/{total_frames}...")

#         cap.release()
#         writer.release()
#         print(f"Video processing complete. Saved to {output_path}")

#     except Exception as e:
#         print(f"Error in process_video: {e}")

@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory('./videos', filename, mimetype='video/mp4')

@app.route("/detect_video", methods=["POST"])
def detect_video():
    try:
        # Kiểm tra file video
        if 'video' not in request.files:
            return jsonify({"error": "Chưa upload video."}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "Không có video nào được chọn."}), 400

        # Lưu file video upload
        input_dir = "./videos"
        os.makedirs(input_dir, exist_ok=True)
        input_filename = secure_filename(file.filename)
        video_path = os.path.join(input_dir, input_filename)
        file.save(video_path)

        # Xử lý video (ví dụ thêm các bước xử lý tại đây)
        output_filename = "processed_" + input_filename
        output_path = os.path.join(input_dir, output_filename)
        process_video(video_path, output_path, time_s=1)  # Hàm xử lý video

        # Trả về đường dẫn của video đã xử lý
        video_url = url_for('serve_video', filename=output_filename)
        return jsonify({"video_url": video_url}), 200

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Lỗi:\n{error_details}")
        return jsonify({"error": str(e), "details": error_details}), 500
    
@app.route("/add_user", methods=["POST"])
def add_user():
    global data
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        image_stream = file.stream
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image"}), 400

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detect_faces_with_model(frame_rgb)

        if not faces:
            return jsonify({"error": "No face detected in the image."}), 400
        if len(faces) > 1:
            return jsonify({"error": "More than one face detected."}), 400

        _, face = next(iter(faces.items()))
        x1, y1, x2, y2 = face["facial_area"]
        cropped_face = frame[y1:y2, x1:x2]
        cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        cropped_face_resized = cv2.resize(cropped_face_rgb, (112, 112))

        embedding = get_face_embedding(cropped_face_resized)
        if embedding is None:
            return jsonify({"error": "Failed to generate embedding."}), 400

        username = request.form.get("name")
        if not username:
            return jsonify({"error": "Username is required."}), 400

        os.makedirs('profile', exist_ok=True)
        filename = f"{username}.png"
        filepath = os.path.join("profile", filename)
        # cv2.imwrite(filepath, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
        cv2.imwrite(filepath, cropped_face)

        user_info = {"name": username, "image": filepath, "embedding": embedding.tolist()}
        data.append(user_info)

        return jsonify({"message": f"User {username} added successfully.", "data": data}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

        # Kiểm tra nếu kết quả không phải là dictionary
        if not isinstance(faces, dict):
            raise ValueError("Unexpected output from RetinaFace.detect_faces. Expected a dictionary.")

        return faces
    except Exception as e:
        print(f"Error during face detection: {e}")
        return {}  # Trả về dictionary rỗng nếu gặp lỗi


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
    
