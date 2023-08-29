# 이 코드에서는 얼굴 감정 분석을 위한 VGGNet 모델을 사용합니다.
# 이미지에서 얼굴을 감지하고 크롭한 후 모델을 활용하여 감정을 예측합니다.
# 여러 이미지 중에서 가장 빈도가 높은 감정을 추출합니다.
# 해당 감정의 색상 및 텍스트를 이미지에 추가한 후, 별도 폴더에 저장합니다.


import numpy as np
import cv2
import mediapipe as mp
import os
import glob
import shutil

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam


# 이미지 크롭 사이즈
CROP_SIZE = (48,48)

# GPU 사용하지 않도록 환경 변수 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 감정 카테고리 및 색상 설정
emotions = {
    '0': ['angry', (0,0,255), (255,255,255)],
    '1': ['fear', (0,255,0), (255,255,255)],   # 초록색, 흰색 텍스트
    '2': ['happy', (255,0,0), (255,255,255)],  # 파란색, 흰색 텍스트
    '3': ['neutral', (0,255,255), (0,0,0)],    # 노란색, 검은색 텍스트
    '4': ['sad', (255,0,255), (0,0,0)]         # 보라색, 검은색 텍스트
}
num_classes = len(emotions)
input_shape = CROP_SIZE+(1,)
weights = '/workspace/wpqkf/facial_emotion_recognition/vggnet_weight.h5'


class VGGNet(Sequential):
    def __init__(self, input_shape, num_classes, checkpoint_path, lr=1e-3):
        super().__init__()
        self.add(Rescaling(1./255, input_shape=input_shape))
        self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.5))

        self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.4))

        self.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.5))

        self.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.4))

        self.add(Flatten())

        self.add(Dense(1024, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(256, activation='relu'))

        self.add(Dense(num_classes, activation='softmax'))

        self.compile(optimizer=Adam(learning_rate=lr),
                    loss=categorical_crossentropy,
                    metrics=['accuracy'])

        self.checkpoint_path = checkpoint_path


# VGGNet 모델 초기화 및 가중치 로드
model = VGGNet(input_shape = input_shape,num_classes = num_classes, checkpoint_path = weights)
model.load_weights(model.checkpoint_path)


# Mediapipe 모듈 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)


# 이미지 전처리

def detection_preprocessing(image, h_max=360):
    h, w, _ = image.shape
    if h > h_max:
        ratio = h_max / h
        w_ = int(w * ratio)
        image = cv2.resize(image, (w_,h_max))
    return image


def resize_face(face):
    x = tf.expand_dims(tf.convert_to_tensor(face), axis=2)
    return tf.image.resize(x, CROP_SIZE)


def recognition_preprocessing(faces):
    x = tf.convert_to_tensor([resize_face(f) for f in faces])
    return x


def inference(image):
    H, W, _ = image.shape

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)

    if results.detections:
        faces = []
        pos = []
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box

            x = int(box.xmin * W)
            y = int(box.ymin * H)
            w = int(box.width * W)
            h = int(box.height * H)

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(x + w, W)
            y2 = min(y + h, H)

            face = image[y1:y2,x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            faces.append(face)
            pos.append((x1, y1, x2, y2))

        x = recognition_preprocessing(faces)

        y = model.predict(x)
        #가장 높은 감정 값의 index
        l = np.argmax(y, axis=1)
        res_emotion = emotions[str(l[0])][0]  

        #박스 치고 감정 표시하는 코드 필요없을시 삭제
        for i in range(1):
            cv2.rectangle(image, (pos[i][0],pos[i][1]),
                            (pos[i][2],pos[i][3]), emotions[str(l[i])][1], 2, lineType=cv2.LINE_AA)

            cv2.rectangle(image, (pos[i][0],pos[i][1]-20),
                            (pos[i][2]+20,pos[i][1]), emotions[str(l[i])][1], -1, lineType=cv2.LINE_AA)

            cv2.putText(image, f'{emotions[str(l[i])][0]}', (pos[i][0],pos[i][1]-5),
                            0, 0.6, emotions[str(l[i])][2], 2, lineType=cv2.LINE_AA)

        return res_emotion


def infer_multi_images(files):
    lst = []
    for path in files:
        image = cv2.imread(path)
        image = detection_preprocessing(image)
        emotion = inference(image)
        if emotion is None:continue
        cv2.imwrite(path,image)
        lst.append((path,emotion))
    return lst


# 특정 폴더의 모든 jpg 파일 삭제 함수
def delete_all_jpg_files(folder_path):
    jpg_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    for file_path in jpg_files:
        os.remove(file_path)


# temp 이미지들의 감정 인식을 위한 함수
# 가장 빈도수 높은 감정 및 이미지 추출
def find_max_emotion(stage_name, arr):
    source_folder_name = "/workspace/wpqkf/facial_emotion_recognition/temp/"
    des_folder_name = "/workspace/wpqkf/facial_emotion_recognition/inference/"
    path_arr, emotion_arr = zip(*arr)
    path_arr = np.array(path_arr)
    emotion_arr = np.array(emotion_arr)
    #가장 많이 나온 감정과 해당감정의 사진 찾기
    # 최빈값 찾기
    unique_emotions, counts = np.unique(emotion_arr, return_counts=True)
    idx_max_emotion = np.argmax(counts)
    most_common_emotion = unique_emotions[idx_max_emotion]

    # 해당 최빈값의 첫번째 인덱스 찾기
    idx_in_emotion_arr = np.where(emotion_arr == most_common_emotion)[0][0]
    max_path = path_arr[idx_in_emotion_arr]


    file_name = os.path.basename(max_path)
    #destination_filename = f'{most_common_emotion}_{file_name}'
    destination_filename = 'picture.jpg'
    stage_folder_path = os.path.join(des_folder_name, stage_name)
    if not os.path.exists(stage_folder_path):   # 폴더가 없을 경우 생성
        os.makedirs(stage_folder_path)

    destination_path = os.path.join(stage_folder_path, destination_filename)

    if os.path.exists(destination_path):
        print(f"Warning: Destination file '{destination_path}' already exists. Overwriting the copy.")
        shutil.copy2(max_path, destination_path)
    else:
        shutil.move(max_path, destination_path)



    # temp폴더의 모든 사진 삭제
    delete_all_jpg_files(source_folder_name)

    return most_common_emotion


# videos의 frame의 감정 인식을 위한 함수
# 가장 빈도수 높은 감정 및 이미지 추출
def find_max_emotion2(stage_name, arr):
    source_folder_name2 = "/workspace/wpqkf/videos/frames/"
    des_folder_name2 = "/workspace/wpqkf/videos"
    path_arr, emotion_arr = zip(*arr)
    path_arr = np.array(path_arr)
    emotion_arr = np.array(emotion_arr)
    #가장 많이 나온 감정과 해당감정의 사진 찾기
    # 최빈값 찾기
    unique_emotions, counts = np.unique(emotion_arr, return_counts=True)
    idx_max_emotion = np.argmax(counts)
    most_common_emotion = unique_emotions[idx_max_emotion]

    # 해당 최빈값의 첫번째 인덱스 찾기
    idx_in_emotion_arr = np.where(emotion_arr == most_common_emotion)[0][0]
    max_path = path_arr[idx_in_emotion_arr]


    file_name = os.path.basename(max_path)
    #destination_filename = f'{most_common_emotion}_{file_name}'
    destination_filename = 'picture.jpg'
    stage_folder_path = os.path.join(des_folder_name2, stage_name)
    if not os.path.exists(stage_folder_path):   # 폴더가 없을 경우 생성
        os.makedirs(stage_folder_path)

    destination_path = os.path.join(stage_folder_path, destination_filename)
    
    if os.path.exists(destination_path):
        print(f"Warning: Destination file '{destination_path}' already exists. Overwriting the copy.")
        shutil.copy2(max_path, destination_path)
    else:
        shutil.move(max_path, destination_path)


    # frames 폴더의 모든 사진 삭제
    delete_all_jpg_files(source_folder_name2)

    return most_common_emotion


# 사용 예시
# paths = np.sort(np.array(glob.glob('temp/*.jpg')))
# infers = infer_multi_images(paths)
# 단계명은 단계 마다 다른이름,inference 폴더 안에 단계 이름의 폴더를 만들어야함
# final_emotion = find_max_emotion('',infers)
# print(final_emotion)