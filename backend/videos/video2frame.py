# 이 코드는 OpenCV를 사용하여 동영상 파일을 읽고 지정된 간격으로 프레임을 추출하여 이미지로 저장하는 역할을 합니다.
# 주어진 경로에 있는 동영상 파일을 사용하며, 추출된 프레임은 frames 폴더에 저장됩니다.

import cv2
import time
import os


def video_to_frames():
    # 동영상 파일 경로들
    video_paths = ['/workspace/wpqkf/videos/video1.mp4', '/workspace/wpqkf/videos/video2.mp4', '/workspace/wpqkf/videos/video3.mp4']

    # 프레임 처리 간격 설정
    frame_interval = 0.15
    
    # 동영상 번호 초기화
    video_num = 0
    
    
    for video_path in video_paths:
        # 동영상 열기
        cap = cv2.VideoCapture(video_path)
        video_num += 1
        # 프레임을 저장할 폴더 생성
        output_folder = '/workspace/wpqkf/videos/frames'
        os.makedirs(output_folder, exist_ok=True)

        # 프레임 추출을 위한 반복문
        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()  # 프레임 읽기
            if not ret:
                break

            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time >= frame_interval:
                # 프레임 처리 (예: 저장)
                frame_filename = os.path.join(output_folder, f'video{video_num}_frame_{frame_count:04d}.jpg')  # 프레임 파일명
                cv2.imwrite(frame_filename, frame)  # 프레임 이미지 저장

                frame_count += 1
                start_time = current_time  # 다음 프레임의 시작 시간 업데이트

        # 동영상 처리 종료
        cap.release()

    cv2.destroyAllWindows()

    
# video_to_frames()