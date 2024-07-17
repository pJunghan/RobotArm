import os
import shutil
import cv2
from tqdm import tqdm

# Haar Cascade 분류기 파일 경로 설정
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # OpenCV의 Haar Cascade 분류기 파일 경로를 설정
face_cascade = cv2.CascadeClassifier(cascade_path)  # Haar Cascade 분류기를 초기화

def is_face_image(image_path):
    """이미지에서 얼굴을 검출하여 얼굴 이미지인지 확인하는 함수"""
    image = cv2.imread(image_path)  # 이미지를 읽어옴
    if image is None:  # 이미지가 제대로 읽히지 않은 경우
        return False  # 얼굴 이미지가 아니라고 판단

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 이미지를 회색조로 변환
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,    # 이미지 피라미드를 생성할 때 사용할 스케일링 비율
        minNeighbors=10,     # 얼굴 영역으로 인정받기 위해 필요한 최소 이웃 사각형 수
        minSize=(10, 10)     # 검출할 얼굴의 최소 크기
    )

    return len(faces) > 0  # 얼굴이 하나 이상 검출되면 True, 아니면 False 반환

def process_image(image_name):
    """이미지를 처리하여 얼굴 이미지인지 여부를 확인하고 이동하는 함수"""
    image_path = os.path.join(image_dir, image_name)  # 이미지의 전체 경로 생성
    if is_face_image(image_path):  # 얼굴 이미지인지 확인
        shutil.move(image_path, os.path.join(face_image_dir, image_name))  # 얼굴 이미지인 경우 얼굴 이미지 디렉토리로 이동
        return f"{image_name} is a face image."  # 얼굴 이미지임을 나타내는 메시지 반환
    else:  # 얼굴 이미지가 아닌 경우
        shutil.move(image_path, os.path.join(non_face_image_dir, image_name))  # 비얼굴 이미지 디렉토리로 이동
        return f"{image_name} is NOT a face image."  # 얼굴 이미지가 아님을 나타내는 메시지 반환

# 테스트할 이미지 경로 설정
image_dir = "/home/hui/Downloads/annoyed1"  # 이미지들이 저장된 디렉토리 경로

# 결과를 저장할 디렉토리 설정
face_image_dir = os.path.join(image_dir, "faces")  # 얼굴 이미지를 저장할 디렉토리 경로 생성
non_face_image_dir = os.path.join(image_dir, "non_faces")  # 비얼굴 이미지를 저장할 디렉토리 경로 생성
os.makedirs(face_image_dir, exist_ok=True)  # 얼굴 이미지 디렉토리가 없으면 생성
os.makedirs(non_face_image_dir, exist_ok=True)  # 비얼굴 이미지 디렉토리가 없으면 생성

# 이미지 파일 리스트 생성
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  # 지정된 확장자의 이미지 파일 리스트 생성

# 이미지 구별 및 이동
for image_name in tqdm(image_files, desc="Processing images", unit="image"):  # 이미지 파일들을 순회하면서 처리
    image_path = os.path.join(image_dir, image_name)  # 이미지의 전체 경로 생성
    if os.path.isfile(image_path):  # 파일인지 확인
        result = process_image(image_name)  # 이미지를 처리하여 결과를 얻음
        print(result)  # 결과 출력

