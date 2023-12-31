# 이 코드에서는 얼굴 감정 분석을 위한 Inception-ResNet-V2모델을 사용합니다.
# 이미지에서 얼굴을 감지하고 크롭한 후 모델을 활용하여 감정을 예측합니다.
# 여러 이미지 중에서 가장 빈도가 높은 감정을 추출합니다.
# 해당 감정의 색상 및 텍스트를 이미지에 추가한 후, 별도 폴더에 저장합니다.

#GPU관련 tensorflow 경고메시지 뜨지 않게 설정
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
 

import numpy as np
import cv2
import mediapipe as mp
import os
import glob
import shutil
# GPU 사용하지 않도록 환경 변수 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import InceptionResNetV2
# 모든 GPU를 숨깁니다.


# 이미지 크롭 사이즈
CROP_SIZE = (96,96)


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
weights = '/workspace/wpqkf/facial_emotion_recognition/best_model.h5'


# Inceptionn-ResNet-v2 모델 
class EmotionClassifier:
    def __init__(self, num_classes=num_classes, checkpoint_path=weights,input_shape = input_shape):
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        self.input_shape = input_shape
        self.model = self.build_model()

    def build_model(self):
        base_model = InceptionResNetV2(weights=None, include_top=False, input_shape=self.input_shape)
        x = Flatten()(base_model.output)
        x = Dropout(0.2)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=x)
        return model

    def save(self):
        self.model.save_weights(self.checkpoint_path)

    def load(self):
        self.model.load_weights(self.checkpoint_path)

    def predict(self, input_data):
        return self.model.predict(input_data)


# Inceptionn-ResNet-v2 모델 초기화 및 가중치 로드
emotion_classifier = EmotionClassifier(num_classes=num_classes, checkpoint_path=weights, input_shape = input_shape)
emotion_classifier.load()

# Mediapipe 모듈 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)


# 이미지 전처리

def detection_preprocessing(image, h_max=360):
    h, w, _ = image.shape
    if h > h_max:
        ratio = h_max / h
        w_ = int(w * ratio)
        image = cv2.resize(image, (w_, h_max))
    return image


def resize_face(face):
    x = tf.expand_dims(tf.convert_to_tensor(face), axis=2)
    return tf.image.resize(x, CROP_SIZE)


def recognition_preprocessing(faces):
    x = tf.convert_to_tensor([resize_face(f) for f in faces])
    return x


def get_face_coordinates(detection, H, W):
    """
    감지된 얼굴 객체를 기반으로 얼굴 경계 상자의 좌표를 반환합니다.
    
    인자:
    - detection: 감지된 얼굴 객체.
    - H: 원본 이미지의 높이.
    - W: 원본 이미지의 너비.
    
    반환값:
    - x1, y1, x2, y2: 얼굴 경계 상자의 좌표.
    """
    box = detection.location_data.relative_bounding_box
    x = int(box.xmin * W)
    y = int(box.ymin * H)
    w = int(box.width * W)
    h = int(box.height * H)

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(x + w, W)
    y2 = min(y + h, H)

    return x1, y1, x2, y2

def draw_emotion_on_image(image, pos, emotion_label, color, text_color):
    """
    주어진 이미지에 감지된 감정을 그립니다.

    인자:
    - image: 감정을 그릴 이미지.
    - pos: 얼굴 경계 상자의 좌표.
    - emotion_label: 감지된 감정의 라벨.
    - color: 경계 상자의 색상.
    - text_color: 텍스트 라벨의 색상.
    """
    cv2.rectangle(image, (pos[0], pos[1]), (pos[2], pos[3]), color, 2, lineType=cv2.LINE_AA)
    cv2.rectangle(image, (pos[0], pos[1]-20), (pos[2]+20, pos[1]), color, -1, lineType=cv2.LINE_AA)
    cv2.putText(image, emotion_label, (pos[0], pos[1]-5), 0, 0.6, text_color, 2, lineType=cv2.LINE_AA)

def inference(image):
    """
    주어진 이미지를 처리하여 얼굴을 감지하고 해당 얼굴에 대한 감정을 예측한 후 이미지에 감정을 그립니다.

    인자:
    - image: 입력 이미지.

    반환값:
    - res_emotion: 감지된 얼굴의 감정.
    """
    H, W, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)

    if not results.detections:
        return None

    # 첫 번째 검출된 얼굴에 대한 좌표를 가져옵니다.
    x1, y1, x2, y2 = get_face_coordinates(results.detections[0], H, W)
    face = image[y1:y2, x1:x2]
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    x = recognition_preprocessing([face_gray])
    y_pred = emotion_classifier.predict(x)
    emotion_index = np.argmax(y_pred, axis=1)
    res_emotion = emotions[str(emotion_index[0])][0]

    # 시각화
    draw_emotion_on_image(image, (x1, y1, x2, y2), res_emotion, emotions[str(emotion_index[0])][1], emotions[str(emotion_index[0])][2])

    return res_emotion



def infer_multi_images(files):
    lst = []
    for path in files:
        image = cv2.imread(path)
        image = detection_preprocessing(image)
        emotion = inference(image)
        print(emotion)
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