import os
import cv2
import numpy as np
import dlib
from insightface.app import FaceAnalysis
from tqdm import tqdm

# InsightFace 모델 초기화
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# dlib의 HOG 기반 얼굴 감지기 초기화
detector = dlib.get_frontal_face_detector()

# 동양인 얼굴 이미지 경로
asian_face_image_folder = '/home/hui/emotion_detect/src/asian_train (copy)/dataset/asian_dataset'

# 동양인 얼굴 특징 벡터 리스트
embeddings = []

# 동양인 얼굴 이미지에서 특징 벡터 추출
for image_name in tqdm(os.listdir(asian_face_image_folder), desc="Processing images"):
    image_path = os.path.join(asian_face_image_folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        continue

    # 이미지 크기 조정 (너무 큰 이미지는 인식이 어려울 수 있으므로 적절한 크기로 조정)
    image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)

    # 얼굴 검출 (dlib 사용)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)  # 업샘플링 횟수 1

    if len(rects) > 0:
        for rect in rects:
            x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
            face_img = image[y1:y2, x1:x2]

            # InsightFace 모델을 사용하여 얼굴 특징 벡터 추출
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            faces = app.get(np.asarray(face_img_rgb))

            if len(faces) > 0:
                embedding = faces[0].embedding
                embeddings.append(embedding)
            else:
                print(f"No face detected in the cropped face: {image_path}")
    else:
        print(f"No face detected in the image: {image_path}")

# 평균 특징 벡터 계산 및 저장
if len(embeddings) > 0:
    asian_embedding = np.mean(embeddings, axis=0)
    np.save('asian_face_embedding.npy', asian_embedding)
    print("Feature vector saved successfully.")
else:
    print("No faces detected in any images.")
