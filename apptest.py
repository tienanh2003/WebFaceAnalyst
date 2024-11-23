import cv2

def process_and_save_video(input_path, output_path):
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Cannot open input video file.")

        # Lấy thông tin video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video Info - FPS: {fps}, Width: {width}, Height: {height}, Total Frames: {total_frames}")

        # Thử codec H.264 (avc1)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec H.264
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Nếu codec không hoạt động, thử codec khác (mp4v)
        if not writer.isOpened():
            print("Failed with H.264 codec. Trying mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            raise ValueError("VideoWriter failed to open. Check codec or file path.")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            writer.write(frame)
            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames}...")

        cap.release()
        writer.release()
        print(f"Finished processing video. Output saved at: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Đường dẫn video đầu vào và đầu ra
input_video_path = "./videos/1721303-hd_1920_1080_25fps.mp4"
output_video_path = "./videos/output_video.mp4"

process_and_save_video(input_video_path, output_video_path)
