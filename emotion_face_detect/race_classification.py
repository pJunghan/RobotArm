import os  # 운영 체제 관련 기능을 제공하는 모듈
import cv2  # OpenCV 라이브러리, 이미지 처리 기능 제공
import shutil  # 파일 및 디렉토리 작업을 위한 모듈
from tqdm import tqdm  # 진행 상태 표시바 제공
import numpy as np  # 수학 및 배열 연산을 위한 라이브러리
from insightface.app import FaceAnalysis  # InsightFace의 얼굴 분석 모듈
from sklearn.metrics.pairwise import cosine_similarity  # 코사인 유사도 계산을 위한 모듈

# 데이터셋 경로 및 출력 경로 설정
base_path = '/home/hui/Downloads/image'  # 이미지 데이터셋의 기본 경로
output_path = os.path.join(base_path, 'UTKresult')  # 동양인 이미지 저장 경로
os.makedirs(output_path, exist_ok=True)  # 출력 경로가 없으면 생성

# 데이터셋 폴더 목록 (asian 폴더는 제외)
folders = ['UTKFace']  # 처리할 데이터셋 폴더 목록

# InsightFace 모델 초기화
app = FaceAnalysis(providers=['CPUExecutionProvider'])  # CPU 사용 설정
app.prepare(ctx_id=0, det_size=(640, 640))  # 모델 준비, det_size는 감지 크기 설정

# 사전 학습된 동양인 얼굴 특징 벡터 로드
asian_embedding = np.load('asian_face_embedding.npy')  # 사전 학습된 동양인 특징 벡터 로드

# 얼굴 특징 벡터 추출 및 정렬 함수
def get_face_embeddings(face_image):
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)  # 이미지를 RGB로 변환
    faces = app.get(np.asarray(face_image))  # 얼굴 감지 및 특징 추출
    
    embeddings = []
    for face in faces:
        aligned_face = face.image  # 이미 정렬된 얼굴 이미지
        embeddings.append(face.embedding)  # 얼굴 특징 벡터 추출
    return embeddings

# 유사도 측정 함수
def is_asian(face_embedding, threshold=0.5):  # 임계값을 0.5로 설정
    similarity = cosine_similarity([face_embedding], [asian_embedding])[0][0]  # 코사인 유사도 계산
    return similarity > threshold  # 유사도가 임계값보다 크면 동양인으로 판단

# 이미지 분류 및 저장
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    if not os.path.isdir(folder_path):
        continue

    print(f"Processing folder: {folder}")

    # 이미지 파일 리스트 가져오기
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for image_file in tqdm(image_files, desc=f"Processing {folder}"):
        image_path = os.path.join(folder_path, image_file)

        try:
            # 이미지 읽기
            image = cv2.imread(image_path)
            if image is None:
                continue

            # 얼굴 특징 벡터 추출 및 정렬
            face_embeddings = get_face_embeddings(image)
            if face_embeddings:
                for face_embedding in face_embeddings:
                    # 유사도 측정을 통해 동양인 여부 판단
                    if is_asian(face_embedding):
                        shutil.copy(image_path, output_path)  # 동양인 이미지로 판단되면 복사
                        break  # 한 명이라도 동양인이면 이미지를 복사하고 다음 이미지로 넘어감
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

print("Processing complete. Asian images are saved in the 'UTKresult' folder.")

