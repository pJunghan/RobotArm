import os
import dlib
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

def rect_to_bb(rect, scale=2.0):
    """dlib의 'rect' 객체를 바운딩 박스 [x, y, w, h]로 변환하고 스케일링 진행."""
    x = rect.left()  # 바운딩 박스의 왼쪽 좌표
    y = rect.top()  # 바운딩 박스의 위쪽 좌표
    w = rect.right() - x  # 바운딩 박스의 너비
    h = rect.bottom() - y  # 바운딩 박스의 높이
    
    # 중심 좌표 계산
    center_x = x + w // 2
    center_y = y + h // 2
    
    # 새 크기 계산
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 새로운 좌표 계산
    new_x = max(center_x - new_w // 2, 0)  # 새로운 x 좌표 (0보다 작으면 0으로 설정)
    new_y = max(center_y - new_h // 2, 0)  # 새로운 y 좌표 (0보다 작으면 0으로 설정)
    new_w = min(new_w, rect.right() + w // 2 - new_x)  # 새로운 너비 (이미지 경계를 벗어나지 않도록 설정)
    new_h = min(new_h, rect.bottom() + h // 2 - new_y)  # 새로운 높이 (이미지 경계를 벗어나지 않도록 설정)
    
    return (new_x, new_y, new_w, new_h)  # 새로운 바운딩 박스 좌표와 크기 반환

def crop_and_save_faces(directory, output_directory, upsample=1):
    # dlib의 HOG 기반 얼굴 감지기 초기화
    detector = dlib.get_frontal_face_detector()

    # InsightFace 모델 초기화
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])  # GPU를 사용할 경우 'CUDAExecutionProvider'
    app.prepare(ctx_id=0, det_size=(640, 640))  # 모델 준비

    if not os.path.exists(output_directory):  # 출력 디렉토리가 없으면 생성
        os.makedirs(output_directory)

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp')  # 유효한 이미지 파일 확장자

    image_files = [f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]  # 디렉토리에서 이미지 파일 리스트 가져오기

    for image_name in tqdm(image_files, desc="Processing images", unit="image"):  # 이미지 파일 리스트 순회
        image_path = os.path.join(directory, image_name)  # 이미지 파일 경로 생성
        if os.path.isfile(image_path):  # 파일이 존재하는지 확인
            try:
                img = cv2.imread(image_path)  # 이미지를 읽어옴
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 이미지를 회색조로 변환
                rects = detector(gray, upsample)  # 얼굴 감지 (업샘플링 횟수 설정)

                if len(rects) == 0:  # 얼굴이 감지되지 않으면
                    print(f"No face detected in {image_name}, deleting...")
                    os.remove(image_path)  # 이미지를 삭제
                    continue

                for (i, rect) in enumerate(rects):  # 감지된 얼굴들 순회
                    (x, y, w, h) = rect_to_bb(rect, scale=2.0)  # 바운딩 박스를 계산
                    
                    # 바운딩 박스가 이미지 경계를 벗어나지 않도록 설정
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, img.shape[1] - x)
                    h = min(h, img.shape[0] - y)
                    
                    face_img = img[y:y+h, x:x+w]  # 얼굴 이미지를 잘라냄
                    
                    # InsightFace를 사용하여 얼굴 검증
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # 얼굴 이미지를 RGB로 변환
                    faces = app.get(np.asarray(face_rgb))  # 얼굴 검증
                    
                    if len(faces) > 0:  # 얼굴이 검증되면
                        output_path = os.path.join(output_directory, f"face_{i}_{image_name}")  # 출력 경로 생성
                        cv2.imwrite(output_path, face_img)  # 잘라낸 얼굴 이미지를 저장
                        print(f"Saved cropped and validated face to {output_path}")
                    else:
                        print(f"No face detected in the cropped face from {image_name}")

            except Exception as e:  # 예외 발생 시
                print(f"Error processing {image_name}: {e}")
                os.remove(image_path)  # 이미지를 삭제
                print(f"Deleted {image_name} due to processing error.")

# 테스트할 이미지 경로
source_directory = "/home/hui/emotion_detect/src/asian_train (copy)/dataset/angry (1)"
output_directory = "/home/hui/emotion_detect/src/asian_train (copy)/dataset/angry (1)_cropped"

# 얼굴 크롭 및 저장
crop_and_save_faces(source_directory, output_directory)  # 함수 호출

