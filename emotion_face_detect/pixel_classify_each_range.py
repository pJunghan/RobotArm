import os
import shutil
from PIL import Image  # PIL에서 Image 모듈 import

# 원본 데이터셋 디렉토리와 대상 디렉토리 경로 설정
source_base_dir = '/home/hui/emotion_detect/src/asian_train_copy/dataset'  # 원본 이미지들이 저장된 디렉토리
target_base_dir = '/home/hui/emotion_detect/src/asian_train_copy/dataset_grouped'  # 그룹화된 이미지를 저장할 디렉토리
categories = ['angry', 'happy', 'sad']  # 카테고리 목록

# 그룹별 픽셀 범위 설정
ranges = [(100, 300), (300, 500), (500, 1000), (1000, 2000)]  # 각 그룹의 최소 및 최대 픽셀 범위 설정

# 대상 디렉토리가 존재하지 않으면 생성
if not os.path.exists(target_base_dir):  # 대상 디렉토리가 존재하는지 확인
    os.makedirs(target_base_dir)  # 대상 디렉토리가 없으면 생성

# 각 카테고리와 범위에 맞게 데이터 복사
for category in categories:  # 각 카테고리에 대해 반복
    source_dir = os.path.join(source_base_dir, category)  # 원본 디렉토리 경로 설정
    
    for range_min, range_max in ranges:  # 각 픽셀 범위에 대해 반복
        target_dir = os.path.join(target_base_dir, f"{range_min}_{range_max}", category)  # 대상 디렉토리 경로 설정
        
        # 대상 디렉토리가 존재하지 않으면 생성
        if not os.path.exists(target_dir):  # 대상 디렉토리가 존재하는지 확인
            os.makedirs(target_dir)  # 대상 디렉토리가 없으면 생성
        
        # 원본 디렉토리의 파일을 대상 디렉토리로 복사
        for file_name in os.listdir(source_dir):  # 원본 디렉토리의 모든 파일에 대해 반복
            file_path = os.path.join(source_dir, file_name)  # 파일 경로 설정
            if os.path.isfile(file_path):  # 파일인지 확인
                try:
                    img_width, img_height = Image.open(file_path).size  # 이미지 크기 읽기
                    min_size = min(img_width, img_height)  # 이미지의 최소 크기 계산
                    if range_min <= min_size < range_max:  # 최소 크기가 범위 내에 있는지 확인
                        shutil.copy(file_path, target_dir)  # 파일을 대상 디렉토리로 복사
                        print(f"Copied {file_name} to {target_dir}")  # 파일 복사 완료 메시지 출력
                except OSError:  # 파일을 열 수 없는 경우 예외 처리
                    print(f"Skipping {file_name} due to unsupported format.")  # 지원되지 않는 형식의 파일 건너뛰기 메시지 출력

print("Data has been grouped and copied according to specified pixel ranges.")  # 모든 작업 완료 메시지 출력

