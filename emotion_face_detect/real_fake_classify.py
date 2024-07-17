import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from deepface import DeepFace

# Haar Cascade 분류기 파일 경로
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

def analyze_image(img_path, edge_density_threshold=0.1, hist_mean_threshold=0.1, min_faces=1):
    """이미지가 실제 사람 얼굴인지 또는 만화인지 여부를 판별"""
    image = cv2.imread(img_path)  # 이미지 파일을 읽어옴
    if image is None:  # 이미지가 없으면
        return "unknown"  # 'unknown'을 반환

    # 이미지 크기를 조정하여 처리 속도 향상
    resized_image = cv2.resize(image, (256, 256))  # 이미지를 256x256 크기로 조정합니다.

    # 이미지 색상 공간을 HSV로 변환
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)  # 이미지를 HSV 색상 공간으로 변환합니다.

    # 히스토그램을 계산하여 색상 분포 분석
    hist = cv2.calcHist([hsv_image], [0, 1], None, [180, 256], [0, 180, 0, 256])  # 히스토그램을 계산합니다.
    hist = cv2.normalize(hist, hist).flatten()  # 히스토그램을 정규화하고 평탄화합니다.

    # 이미지의 에지 검출
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)  # 이미지를 회색조로 변환합니다.
    edges = cv2.Canny(gray_image, 100, 200)  # Canny 에지 검출기를 사용하여 에지를 검출합니다.

    # 에지 밀도를 계산하여 텍스처 분석
    edge_density = np.sum(edges) / (256 * 256)  # 에지 밀도를 계산합니다.

    # Haar Cascade를 사용하여 얼굴 검출
    faces_haar = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,  # 스케일링 비율
        minNeighbors=12,  # 최소 이웃 수
        minSize=(15, 15)  # 최소 크기
    )

    # 얼굴이 검출되면 감정 인식을 시도
    if len(faces_haar) >= min_faces:  # 검출된 얼굴의 수가 최소 얼굴 수 이상일 경우
        try:
            analysis = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)  # DeepFace를 사용하여 감정 분석을 시도합니다.
            if 'emotion' in analysis:  # 감정 분석 결과가 있으면
                return "real"  # 실제 사람 얼굴로 간주하고 'real'을 반환합니다.
        except Exception as e:
            print(f"Error in emotion analysis for {img_path}: {e}")  # 감정 분석 중 에러가 발생하면 에러 메시지를 출력합니다.

    # 만화 이미지와 실제 사람 이미지 구분
    if edge_density > edge_density_threshold and np.mean(hist) < hist_mean_threshold:  # 에지 밀도와 히스토그램 평균값을 기준으로
        return "cartoon"  # 만화 이미지로 간주하고 'cartoon'을 반환합니다.
    else:
        return "real"  # 그렇지 않으면 실제 사람 얼굴로 간주하고 'real'을 반환합니다.

# 테스트할 이미지 경로
source_dir = "/home/hui/Downloads/annoyed1/faces"  # 처리할 이미지들이 저장된 디렉토리 경로
real_faces_dir = os.path.join(source_dir, "real_faces")  # 실제 얼굴 이미지를 저장할 디렉토리 경로
cartoon_faces_dir = os.path.join(source_dir, "cartoon_faces")  # 만화 얼굴 이미지를 저장할 디렉토리 경로

# 결과 저장할 디렉토리 생성
os.makedirs(real_faces_dir, exist_ok=True)  # 실제 얼굴 이미지 저장 디렉토리를 생성합니다.
os.makedirs(cartoon_faces_dir, exist_ok=True)  # 만화 얼굴 이미지 저장 디렉토리를 생성합니다.

# 이미지 파일 리스트 생성
image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  # 디렉토리 내 이미지 파일 리스트를 생성합니다.

# 이미지 구별 및 이동
for image_name in tqdm(image_files, desc="Processing images", unit="image"):  # 진행 상황을 표시하며 이미지 파일을 처리합니다.
    image_path = os.path.join(source_dir, image_name)  # 이미지 파일 경로를 설정합니다.
    if os.path.isfile(image_path):  # 파일이 존재하는지 확인합니다.
        result = analyze_image(image_path)  # 이미지를 분석하여 결과를 반환받습니다.
        if result == "real":  # 결과가 'real'이면
            print(f"{image_name} is a real face image.")  # 해당 이미지는 실제 얼굴 이미지입니다.
            shutil.move(image_path, os.path.join(real_faces_dir, image_name))  # 이미지를 실제 얼굴 이미지 디렉토리로 이동합니다.
        elif result == "cartoon":  # 결과가 'cartoon'이면
            print(f"{image_name} is a cartoon face image.")  # 해당 이미지는 만화 얼굴 이미지입니다.
            shutil.move(image_path, os.path.join(cartoon_faces_dir, image_name))  # 이미지를 만화 얼굴 이미지 디렉토리로 이동합니다.
        else:
            print(f"{image_name} could not be classified.")  # 이미지가 분류되지 않았음을 출력합니다.

