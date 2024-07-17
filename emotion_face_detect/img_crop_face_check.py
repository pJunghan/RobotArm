import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm

# InsightFace 모델 초기화
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])  # GPU를 사용할 경우 'CUDAExecutionProvider'
app.prepare(ctx_id=0, det_size=(640, 640))

# 이미지 디렉토리 경로 설정
image_dir = '/home/hui/emotion_detect/src/asian_train (copy)/dataset/sad (1)_cropped'  # 원본 이미지 디렉토리 경로
output_dir = '/home/hui/emotion_detect/src/asian_train (copy)/dataset/sad (1)_cropped/output'  # 출력 디렉토리 경로

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)  # 출력 디렉토리가 없으면 생성

# 이미지 확장자
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp')  # 유효한 이미지 파일 확장자 목록

# 이미지 파일 리스트 가져오기
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]  # 디렉토리 내 유효한 확장자의 이미지 파일 목록 생성

# 이미지 순회
for image_name in tqdm(image_files, desc="Processing images"):  # 이미지 파일 목록을 순회하며 진행 상황을 표시
    image_path = os.path.join(image_dir, image_name)  # 각 이미지 파일의 전체 경로 생성
    img = cv2.imread(image_path)  # 이미지를 읽어옴

    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 이미지를 RGB 형식으로 변환
        faces = app.get(img_rgb)  # 얼굴 검출 수행
        
        if len(faces) > 0:
            print(f"Face detected in {image_name}")  # 얼굴이 검출된 경우
            output_path = os.path.join(output_dir, image_name)  # 출력 파일 경로 생성
            cv2.imwrite(output_path, img)  # 검출된 얼굴 이미지를 출력 디렉토리에 저장
        else:
            print(f"No face detected in {image_name}")  # 얼굴이 검출되지 않은 경우
    else:
        print(f"Error: Unable to load image at {image_path}")  # 이미지를 읽어올 수 없는 경우

print("Processing complete.")  # 모든 이미지 처리 완료 메시지 출력

