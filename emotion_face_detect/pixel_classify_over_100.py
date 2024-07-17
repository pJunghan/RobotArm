import os
import shutil
from PIL import Image  # PIL의 Image 모듈을 가져옴

# 원본 디렉토리와 대상 디렉토리 경로 설정
source_base_dir = '/home/hui/emotion_detect/src/asian_train_copy/dataset'  # 원본 이미지들이 저장된 디렉토리 경로
target_base_dir = '/home/hui/emotion_detect/src/asian_train_copy/dataset_filtered'  # 필터링된 이미지를 저장할 디렉토리 경로
categories = ['angry', 'happy', 'sad']  # 처리할 이미지 카테고리 목록

# 대상 디렉토리가 존재하지 않으면 생성
if not os.path.exists(target_base_dir):  # 대상 디렉토리가 존재하는지 확인
    os.makedirs(target_base_dir)  # 대상 디렉토리가 없으면 생성

# 각 카테고리별로 이미지 필터링 및 복사
for category in categories:  # 각 카테고리에 대해 반복
    source_dir = os.path.join(source_base_dir, category)  # 원본 디렉토리 경로를 설정
    target_dir = os.path.join(target_base_dir, f"{category}_filtered")  # 대상 디렉토리 경로를 설정
    
    # 각 카테고리별 대상 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(target_dir):  # 대상 디렉토리가 존재하는지 확인
        os.makedirs(target_dir)  # 대상 디렉토리가 없으면 생성

    # 원본 디렉토리의 모든 파일을 검사
    for file_name in os.listdir(source_dir):  # 원본 디렉토리의 모든 파일에 대해 반복
        if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):  # 이미지 파일 확장자를 확인
            file_path = os.path.join(source_dir, file_name)  # 파일 경로를 설정
            try:
                with Image.open(file_path) as img:  # 이미지를 엶
                    width, height = img.size  # 이미지의 크기를 가져옴
                    # 이미지 크기가 100x100 픽셀보다 큰 경우
                    if width > 100 and height > 100:  # 이미지 크기를 확인합
                        shutil.copy(file_path, target_dir)  # 이미지를 대상 디렉토리로 복사
                        print(f"Copied {file_name} to {target_dir}")  # 복사 완료 메시지를 출력
                    else:
                        print(f"Skipped {file_name} (size: {width}x{height})")  # 크기가 작은 이미지는 건너뜀
            except Exception as e:  # 예외 발생 시 처리
                print(f"Failed to process {file_name}: {e}")  # 예외 메시지를 출력

print("All qualifying images have been copied.")  # 모든 작업 완료 메시지를 출력

