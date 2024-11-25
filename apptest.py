from flask import Flask, request, jsonify, render_template, Response, send_from_directory, url_for
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import traceback
import base64

# Import the classes from the separate files
from image_processor import ImageProcessor
from video_processor import VideoProcessor

app = Flask(__name__)

# Initialize processors
image_processor = ImageProcessor()
video_processor = VideoProcessor(image_processor)

@app.route("/camera_user")
def camera_user():
    return Response(video_processor.stream_camera_feed(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory('./videos', filename, mimetype='video/mp4')

# @app.route("/detect_video", methods=["POST"])
# def detect_video():
#     try:
#         if 'video' not in request.files:
#             return jsonify({"error": "No video uploaded."}), 400

#         file = request.files['video']
#         if file.filename == '':
#             return jsonify({"error": "No selected video."}), 400

#         input_dir = "./videos"
#         os.makedirs(input_dir, exist_ok=True)
#         input_filename = secure_filename(file.filename)
#         video_path = os.path.join(input_dir, input_filename)
#         file.save(video_path)

#         output_filename = "processed_" + input_filename
#         output_path = os.path.join(input_dir, output_filename)
#         video_processor.process_video(video_path, output_path, time_s=2)

#         video_url = url_for('serve_video', filename=output_filename)
#         return jsonify({"video_url": video_url}), 200

#     except Exception as e:
#         error_details = traceback.format_exc()
#         print(f"Error:\n{error_details}")
#         return jsonify({"error": str(e), "details": error_details}), 500
@app.route("/detect_video", methods=["POST"])
def detect_video():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video uploaded."}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No selected video."}), 400

        input_dir = "./videos"
        os.makedirs(input_dir, exist_ok=True)
        input_filename = secure_filename(file.filename)
        video_path = os.path.join(input_dir, input_filename)
        file.save(video_path)

        output_filename = "processed_" + input_filename
        output_path = os.path.join(input_dir, output_filename)

        # Call process_video and get emotions_per_second_per_id
        emotions_per_second_per_id = video_processor.process_video(video_path, output_path, time_s=1)

        video_url = url_for('serve_video', filename=output_filename)
        return jsonify({
            "video_url": video_url,
            "emotions_per_second_per_id": emotions_per_second_per_id
        }), 200

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error:\n{error_details}")
        return jsonify({"error": str(e), "details": error_details}), 500

@app.route("/add_user", methods=["POST"])
def add_user():
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
        faces = image_processor.detect_faces_with_model(frame_rgb)
        if not faces:
            return jsonify({"error": "No face detected in the image."}), 400
        if len(faces) > 1:
            return jsonify({"error": "More than one face detected."}), 400

        _, face = next(iter(faces.items()))
        x1, y1, x2, y2 = face["facial_area"]
        cropped_face = frame[y1:y2, x1:x2]
        cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        cropped_face_resized = cv2.resize(cropped_face_rgb, (112, 112))

        embedding = image_processor.get_face_embedding(cropped_face_resized)
        if embedding is None:
            return jsonify({"error": "Failed to generate embedding."}), 400

        username = request.form.get("name")
        if not username:
            return jsonify({"error": "Username is required."}), 400

        os.makedirs('profile', exist_ok=True)
        filename = f"{username}.png"
        filepath = os.path.join("profile", filename)
        cv2.imwrite(filepath, cropped_face)

        user_info = {"name": username, "image": filepath, "embedding": embedding.tolist()}
        image_processor.faiss_index.add(np.array([embedding], dtype=np.float32))
        image_processor.names.append(username)
        image_processor.data.append(user_info)

        return jsonify({"message": f"User {username} added successfully.", "data": image_processor.data}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/detect_embedding", methods=["POST"])
def detect_embedding_arcface():
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
        faces = image_processor.detect_faces_with_model(frame_rgb)

        if not faces:
            return jsonify({"error": "No faces detected in the image."}), 400

        face_data = []
        for face_id, face in faces.items():
            x1, y1, x2, y2 = face["facial_area"]

            cropped_face = frame_rgb[y1:y2, x1:x2]
            if cropped_face.size == 0:
                continue

            cropped_face_resized = cv2.resize(cropped_face, (112, 112))
            embedding = image_processor.get_face_embedding(cropped_face_resized)
            emotion = image_processor.run_vit_model(cropped_face)

            name, distance = image_processor.find_closest_face(embedding)
            if not name:
                name = "Guest"

            label = f"{name}: {emotion}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

            _, face_buffer = cv2.imencode('.jpg', cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
            face_base64 = base64.b64encode(face_buffer).decode('utf-8')

            face_data.append({
                "face_index": face_id,
                "name": name,
                "distance": distance,
                "emotion": emotion,
                "face_image": face_base64,
            })

        _, buffer = cv2.imencode('.jpg', frame)
        result_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "result_image": result_image_base64,
            "faces": face_data
        }), 200

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"An error occurred:\n{error_details}")
        return jsonify({"error": str(e), "details": error_details}), 500

@app.route("/")
def index():
    return render_template("user.html")

if __name__ == "__main__":
    app.run(debug=True)
