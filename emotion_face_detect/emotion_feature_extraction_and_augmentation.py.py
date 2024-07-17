import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from insightface.app import FaceAnalysis
from tqdm import tqdm
from multiprocessing import Pool

# ArcFace 모델 초기화
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# 각 픽셀 범위에 대해 최적의 det_size 설정
det_size_dict = {
    '100_300': (320, 320),
    '300_500': (480, 480),
    '500_1000': (640, 640),
    '1000_2000': (800, 800)
}

def prepare_model_for_range(range_key):
    """픽셀 범위에 맞는 모델을 준비"""
    det_size = det_size_dict[range_key]  # 범위에 해당하는 det_size를 가져옴
    app.prepare(ctx_id=0, det_size=det_size)  # 모델을 해당 det_size로 준비
    return app  # 준비된 모델을 반환

# 멀티프로세싱을 사용하여 얼굴 특징 추출
def process_image(args):
    """이미지에서 얼굴 특징을 추출"""
    img_path, label, range_key = args  # 이미지 경로, 레이블, 범위 키를 인수로 받음
    img = cv2.imread(img_path)  # 이미지를 읽어옴
    if img is not None:  # 이미지가 제대로 읽혔는지 확인
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 이미지를 RGB로 변환
        faces = app.get(img_rgb)  # ArcFace 모델을 사용하여 얼굴을 검출
        if len(faces) > 0:  # 얼굴이 검출된 경우
            embedding = faces[0].embedding  # 얼굴 임베딩을 추출
            return embedding, label  # 임베딩과 레이블을 반환
    return None  # 얼굴이 검출되지 않거나 오류가 발생한 경우 None을 반환

def load_data_parallel(data_dir):
    """데이터를 병렬로 로드하여 얼굴 특징을 추출"""
    data = []  # 데이터를 저장할 리스트를 초기화
    labels = []  # 레이블을 저장할 리스트를 초기화
    args = []  # 멀티프로세싱을 위한 인수 리스트를 초기화
    for range_key in ['100_300', '300_500', '500_1000', '1000_2000']:  # 각 픽셀 범위를 순회
        prepare_model_for_range(range_key)  # 각 범위에 맞는 모델을 준비
        range_path = os.path.join(data_dir, range_key)  # 해당 범위의 디렉토리 경로를 설정
        for category in ['happy', 'sad', 'angry']:  # 각 감정 카테고리를 순회
            category_path = os.path.join(range_path, category)  # 감정 카테고리 디렉토리 경로를 설정
            label = category  # 레이블을 감정 카테고리로 설정
            for img_name in os.listdir(category_path):  # 카테고리 디렉토리 내의 각 이미지를 순회
                img_path = os.path.join(category_path, img_name)  # 이미지 경로를 설정
                args.append((img_path, label, range_key))  # 이미지 경로, 레이블, 범위 키를 인수 리스트에 추가
    
    with Pool() as pool:  # 멀티프로세싱 풀을 초기화
        for result in tqdm(pool.imap(process_image, args), total=len(args), desc="Processing images"):  # 병렬로 이미지를 처리
            if result is not None:  # 결과가 있는 경우
                embedding, label = result  # 임베딩과 레이블을 추출
                data.append(embedding)  # 데이터를 리스트에 추가
                labels.append(label)  # 레이블을 리스트에 추가
    
    return np.array(data), np.array(labels)  # 데이터를 numpy 배열로 변환하여 반환

# 데이터 로드
data_dir = '/home/hui/emotion_detect/src/asian_train_copy/dataset/dataset_filtered/dataset_grouped'  # 데이터 디렉토리 경로를 설정
data, labels = load_data_parallel(data_dir)  # 병렬로 데이터를 로드하여 얼굴 특징을 추출

# 데이터 증강: 다양한 방법 적용
def augment_data(data, labels, num_augmentations=5):
    """데이터를 증강"""
    data_augmented = []  # 증강된 데이터를 저장할 리스트를 초기화
    labels_augmented = []  # 증강된 레이블을 저장할 리스트를 초기화
    
    for img, label in zip(data, labels):  # 각 데이터와 레이블을 순회
        for _ in range(num_augmentations):  # 증강 횟수만큼 반복
            # Gaussian Noise 추가
            noisy_img = img + 0.01 * np.random.randn(*img.shape)  # 이미지에 가우시안 노이즈를 추가
            data_augmented.append(noisy_img)  # 증강된 이미지를 리스트에 추가
            labels_augmented.append(label)  # 레이블을 리스트에 추가
            
            # Scaling
            scaled_img = img * (1 + 0.01 * np.random.randn())  # 이미지를 스케일링
            data_augmented.append(scaled_img)  # 증강된 이미지를 리스트에 추가
            labels_augmented.append(label)  # 레이블을 리스트에 추가
            
            # Random Erasing
            erased_img = img.copy()  # 이미지를 복사
            num_erase = np.random.randint(1, 10)  # 랜덤으로 지울 픽셀 수를 설정
            for _ in range(num_erase):  # 지울 픽셀 수만큼 반복
                idx = np.random.randint(0, len(erased_img))  # 랜덤 인덱스를 설정
                erased_img[idx] = 0  # 해당 인덱스의 픽셀을 0으로 설정
            data_augmented.append(erased_img)  # 증강된 이미지를 리스트에 추가
            labels_augmented.append(label)  # 레이블을 리스트에 추가
            
            # Mixup
            if len(data) > 1:  # 데이터가 2개 이상인 경우
                mixup_idx = np.random.randint(0, len(data))  # 랜덤 인덱스를 설정
                lam = np.random.beta(0.2, 0.2)  # 베타 분포에서 랜덤 값을 가져옴
                mixup_img = lam * img + (1 - lam) * data[mixup_idx]  # 이미지 믹스업을 수행
                data_augmented.append(mixup_img)  # 증강된 이미지를 리스트에 추가
                labels_augmented.append(label)  # 레이블을 리스트에 추가
    
    return np.array(data_augmented), np.array(labels_augmented)  # 증강된 데이터를 numpy 배열로 변환하여 반환

# 데이터 증강 수행
data_augmented, labels_augmented = augment_data(data, labels)  # 데이터를 증강

# 모든 범위의 추출된 벡터와 레이블을 하나의 파일로 저장
np.save('data_augmented_combined.npy', data_augmented)  # 증강된 데이터를 파일로 저장
np.save('labels_augmented_combined.npy', labels_augmented)  # 증강된 레이블을 파일로 저장

# 레이블 이진화
lb = LabelBinarizer()  # LabelBinarizer 객체를 생성
labels_augmented = lb.fit_transform(labels_augmented)  # 레이블을 이진화

# 데이터를 훈련, 검증, 테스트 세트로 분할
train_data, test_data, train_labels, test_labels = train_test_split(data_augmented, labels_augmented, test_size=0.15, random_state=42)  # 데이터를 훈련, 검증, 테스트 세트로 분할
train_data, validation_data, train_labels, validation_labels = train_test_split(train_data, train_labels, test_size=0.1765, random_state=42)  # 0.1765 x 0.85 ≈ 0.15

# 데이터셋 분할 후 크기 출력
print(f"Train data shape: {train_data.shape}")  # 훈련 데이터의 크기를 출력
print(f"Validation data shape: {validation_data.shape}")  # 검증 데이터의 크기를 출력
print(f"Test data shape: {test_data.shape}")  # 테스트 데이터의 크기를 출력

