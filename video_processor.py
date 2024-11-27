import cv2
import numpy as np
from tracker.byte_tracker import BYTETracker
import traceback
from multiprocessing import Pool, cpu_count, Manager

class VideoProcessor:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    # def process_frame(self, frame_data):
    #     frame, frame_index, tracker, height, width = frame_data
    #     try:
    #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         faces = self.image_processor.detect_faces_with_model(frame_rgb)

    #         detections = []
    #         for _, face in faces.items():
    #             x1, y1, x2, y2 = face["facial_area"]
    #             score = face.get("score", 1.0)
    #             detections.append([x1, y1, x2, y2, score])

    #         detections = np.array(detections) if detections else np.empty((0, 5))
    #         tracked_faces = tracker.update(detections, (height, width), (height, width))

    #         result = []
    #         for track in tracked_faces:
    #             if track.is_activated:
    #                 x1, y1, x2, y2 = map(int, track.tlbr)
    #                 track_id = track.track_id

    #                 cropped_face = frame_rgb[y1:y2, x1:x2]
    #                 emotion = "Unknown"
    #                 if cropped_face.size > 0:
    #                     emotion = self.image_processor.run_vit_model(cropped_face)

    #                 result.append({
    #                     "track_id": track_id,
    #                     "emotion": emotion,
    #                     "bbox": (x1, y1, x2, y2)
    #                 })

    #         return frame_index, frame, result
    #     except Exception as e:
    #         print(f"Error processing frame {frame_index}: {e}")
    #         return frame_index, frame, []

    # def process_video(self, input_path, output_path, time_s=None):
    #     try:
    #         cap = cv2.VideoCapture(input_path)
    #         if not cap.isOpened():
    #             raise ValueError("Cannot open video file.")

    #         fps = int(cap.get(cv2.CAP_PROP_FPS))
    #         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #         max_frames = min(total_frames, int(time_s * fps)) if time_s else total_frames

    #         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #         writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    #         if not writer.isOpened():
    #             raise ValueError("VideoWriter failed to open. Check codec or file path.")

    #         tracker_args = type('Args', (), {
    #             "track_thresh": 0.5,
    #             "track_buffer": 30,
    #             "match_thresh": 0.7,
    #             "mot20": False
    #         })

    #         # Initialize ByteTracker
    #         tracker = BYTETracker(tracker_args, frame_rate=fps)

    #         # Prepare frames for multiprocessing
    #         frames = []
    #         for frame_index in range(max_frames):
    #             ret, frame = cap.read()
    #             if not ret:
    #                 break
    #             frames.append((frame, frame_index, tracker, height, width))

    #         cap.release()

    #         # Process frames using multiprocessing
    #         with Pool(processes=cpu_count()) as pool:
    #             results = pool.map(self.process_frame, frames)

    #         # Collect results and write frames
    #         results = sorted(results, key=lambda x: x[0])  # Sort by frame index
    #         for _, frame, _ in results:
    #             writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    #         writer.release()
    #         print(f"Finished processing {len(results)} frames. Video saved at: {output_path}")

    #     except Exception as e:
    #         print(f"An error occurred in process_video:\n{traceback.format_exc()}")

    # def process_video(self, input_path, output_path, time_s):
    #     try:
    #         cap = cv2.VideoCapture(input_path)
    #         if not cap.isOpened():
    #             raise ValueError("Cannot open video file.")

    #         fps = min(30, int(cap.get(cv2.CAP_PROP_FPS)))
    #         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #         max_frames = min(total_frames, int(time_s * fps))

    #         fourcc = cv2.VideoWriter_fourcc(*'H264')
    #         writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    #         if not writer.isOpened():
    #             raise ValueError("VideoWriter failed to open. Check codec or file path.")

    #         tracker_args = type('Args', (), {
    #             "track_thresh": 0.5,
    #             "track_buffer": 30,
    #             "match_thresh": 0.7,
    #             "mot20": False
    #         })
    #         byte_tracker = BYTETracker(tracker_args, frame_rate=fps)
    #         frame_count = 0

    #         while frame_count < max_frames:
    #             ret, frame = cap.read()
    #             if not ret:
    #                 print("No more frames to read.")
    #                 break

    #             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             faces = self.image_processor.detect_faces_with_model(frame_rgb)

    #             detections = []
    #             for _, face in faces.items():
    #                 x1, y1, x2, y2 = face["facial_area"]
    #                 score = face.get("score", 1.0)
    #                 detections.append([x1, y1, x2, y2, score])

    #             detections = np.array(detections) if detections else np.empty((0, 5))
    #             tracked_faces = byte_tracker.update(detections, (height, width), (height, width))

    #             for track in tracked_faces:
    #                 if track.is_activated:
    #                     x1, y1, x2, y2 = map(int, track.tlbr)
    #                     track_id = track.track_id

    #                     cropped_face = frame_rgb[y1:y2, x1:x2]
    #                     if cropped_face.size > 0:
    #                         emotion = self.image_processor.run_vit_model(cropped_face)
    #                         cropped_face_resized = cv2.resize(cropped_face, (112, 112))
    #                         embedding = self.image_processor.get_face_embedding(cropped_face_resized)
    #                         name, _ = self.image_processor.find_closest_face(embedding)
    #                     else:
    #                         emotion = "Unknown"
    #                         name = "Guest"

    #                     label = f"ID {track_id}: {name} ({emotion})"
    #                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #                     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
    #                                 0.5, (0, 255, 0), 2)

    #             writer.write(frame)
    #             frame_count += 1
    #             print(f"Processing frame {frame_count}/{max_frames}...")

    #         cap.release()
    #         writer.release()
    #         print(f"Finished processing {frame_count} frames. Video saved at: {output_path}")

    #     except Exception as e:
    #         print(f"An error occurred in process_video:\n{traceback.format_exc()}")
    def process_video(self, input_path, output_path, time_s): 
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError("Cannot open video file.")

            fps = min(30, int(cap.get(cv2.CAP_PROP_FPS)))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            max_frames = min(total_frames, int(time_s * fps)) if time_s else total_frames

            fourcc = cv2.VideoWriter_fourcc(*'H264')  
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise ValueError("VideoWriter failed to open. Check codec or file path.")

            tracker_args = type('Args', (), {
                "track_thresh": 0.5,
                "track_buffer": 30,
                "match_thresh": 0.7,
                "mot20": False
            })
            byte_tracker = BYTETracker(tracker_args, frame_rate=fps)
            frame_count = 0

            # Initialize variables to store emotion counts per track_id
            emotion_counts_per_id = {}  # {track_id: {emotion: count}}
            total_emotion_counts_per_id = {}  # {track_id: total_emotion_counts}

            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    print("No more frames to read.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.image_processor.detect_faces_with_model(frame_rgb)

                detections = []
                for _, face in faces.items():
                    x1, y1, x2, y2 = face["facial_area"]
                    score = face.get("score", 1.0)
                    detections.append([x1, y1, x2, y2, score])

                detections = np.array(detections) if detections else np.empty((0, 5))
                tracked_faces = byte_tracker.update(detections, (height, width), (height, width))

                for track in tracked_faces:
                    if track.is_activated:
                        x1, y1, x2, y2 = map(int, track.tlbr)
                        track_id = track.track_id

                        cropped_face = frame_rgb[y1:y2, x1:x2]
                        if cropped_face.size > 0:
                            emotion = self.image_processor.run_vit_model(cropped_face)
                            cropped_face_resized = cv2.resize(cropped_face, (112, 112))
                            embedding = self.image_processor.get_face_embedding(cropped_face_resized)
                            name, _ = self.image_processor.find_closest_face(embedding)
                        else:
                            emotion = "Unknown"
                            name = "Guest"

                        # Update emotion counts per track_id
                        if track_id not in emotion_counts_per_id:
                            emotion_counts_per_id[track_id] = {}
                            total_emotion_counts_per_id[track_id] = 0
                        if emotion not in emotion_counts_per_id[track_id]:
                            emotion_counts_per_id[track_id][emotion] = 0
                        emotion_counts_per_id[track_id][emotion] += 1
                        total_emotion_counts_per_id[track_id] += 1

                        label = f"ID {track_id}: {name} ({emotion})"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 2)

                writer.write(frame)
                frame_count += 1
                print(f"Processing frame {frame_count}/{max_frames}...")

            cap.release()
            writer.release()
            print(f"Finished processing {frame_count} frames. Video saved at: {output_path}")

            # Calculate emotion percentages per track_id
            emotion_percentages_per_id = {}
            for track_id, emotion_counts in emotion_counts_per_id.items():
                total_counts = total_emotion_counts_per_id[track_id]
                emotion_percentages_per_id[track_id] = {}
                for emotion, count in emotion_counts.items():
                    percentage = (count / total_counts) * 100
                    emotion_percentages_per_id[track_id][emotion] = percentage

            return emotion_percentages_per_id

        except Exception as e:
            print(f"An error occurred in process_video:\n{traceback.format_exc()}")
            return {}

    def stream_camera_feed(self):
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Failed to capture frame from camera.")
                    break

                processed_frame = self.image_processor.process_frame(frame)

                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            camera.release()
