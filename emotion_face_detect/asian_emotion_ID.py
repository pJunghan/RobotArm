import os  # 운영체제의 기능을 사용하기 위한 모듈
import cv2  # OpenCV 라이브러리, 이미지 및 비디오 처리에 사용
import time  # 시간 관련 함수 제공
import uuid  # 고유 식별자(UUID)를 생성하는 모듈
import numpy as np  # 고성능 수치 계산을 위한 라이브러리
from threading import Thread  # 멀티스레딩을 위한 모듈
from queue import Queue  # 스레드 간 데이터 전달을 위한 큐(queue) 모듈
import tensorflow as tf  # 딥러닝 모델을 구축하고 훈련하는 라이브러리
from insightface.app import FaceAnalysis  # 얼굴 인식 및 분석을 위한 라이브러리

class FaceRecognition:
    def __init__(self, db_path, model_path):
        self.db_path = db_path  # 데이터베이스 경로
        self.cap = cv2.VideoCapture(0)  # 웹캠을 사용하여 비디오 캡처 객체 생성
        self.font = cv2.FONT_HERSHEY_SIMPLEX  # 폰트 설정
        self.detected = False  # 얼굴이 인식되었는지 여부를 나타내는 변수
        self.last_check = time.time()  # 마지막으로 확인한 시간을 저장
        self.check_interval = 0.5  # 0.5초마다 확인
        self.frame_queue = Queue(maxsize=1)  # 프레임을 저장할 큐
        self.result_queue = Queue(maxsize=1)  # 결과를 저장할 큐
        self.member_id = None  # 인식된 멤버의 ID
        self.phone_number = None  # 인식된 멤버의 전화번호
        self.labels = ['angry', 'happy', 'sad']  # 감정 레이블
        self.model = tf.keras.models.load_model(model_path)  # 학습된 모델 로드
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])  # InsightFace 모델 초기화
        self.app.prepare(ctx_id=0, det_size=(640, 640))  # 모델 준비, 최적의 검출 크기 설정

    def run_cam(self):
        """웹캠을 통해 얼굴을 인식하고 결과를 화면에 표시"""
        while True:
            ret, frame = self.cap.read()  # 웹캠에서 프레임 읽기
            if not ret:
                continue

            current_time = time.time()  # 현재 시간 저장
            if current_time - self.last_check > self.check_interval:
                self.last_check = current_time  # 마지막 확인 시간 업데이트
                if self.frame_queue.full():
                    self.frame_queue.get()  # 큐가 가득 차면 이전 프레임 제거
                self.frame_queue.put(frame.copy())  # 큐에 현재 프레임 추가

            if not self.result_queue.empty():
                self.detected, self.member_id, self.phone_number, emotion = self.result_queue.get()  # 결과 큐에서 값 가져오기

            display_frame = frame.copy()  # 화면에 표시할 프레임 복사
            if self.detected:
                cv2.putText(display_frame, f"ID: {self.member_id}", (10, 30), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(display_frame, f"Phone: {self.phone_number}", (10, 60), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(display_frame, f"Emotion: {emotion}", (10, 90), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(display_frame, "Not recognized", (10, 30), self.font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Face Recognition", display_frame)  # 화면에 프레임 표시
            key = cv2.waitKey(1) & 0xFF  # 키 입력 대기
            if key == ord('c'):
                self.capture_and_save(frame)  # 'c' 키를 누르면 캡처 및 저장
            elif key == ord('q'):
                break  # 'q' 키를 누르면 종료

        self.cap.release()  # 웹캠 해제
        cv2.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기

    def capture_and_save(self, frame):
        """현재 프레임을 캡처하여 저장"""
        phone_number = input("Enter phone number: ")
        member_id = str(uuid.uuid4())  # 고유 ID 생성
        member_folder = os.path.join(self.db_path, member_id)  # 멤버 폴더 경로 생성
        os.makedirs(member_folder, exist_ok=True)  # 폴더 생성
        captured_image_path = os.path.join(member_folder, "photo.jpg")
        cv2.imwrite(captured_image_path, frame)  # 프레임 저장

        with open(os.path.join(member_folder, "info.txt"), 'w') as f:
            f.write(phone_number)  # 전화번호 저장

        print(f"Member registered with ID: {member_id}")
        self.member_id = member_id
        self.phone_number = phone_number

    def extract_face_embedding(self, img):
        """이미지에서 얼굴 임베딩 추출"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 이미지를 RGB로 변환
        faces = self.app.get(img_rgb)  # 얼굴 인식 및 임베딩 추출
        if len(faces) > 0:
            return faces[0].embedding
        return None

    def predict_emotion(self, embedding):
        """임베딩을 기반으로 감정을 예측"""
        prediction = self.model.predict(np.expand_dims(embedding, axis=0))  # 예측 수행
        return self.labels[np.argmax(prediction)]  # 가장 높은 확률의 레이블 반환

    def compare_faces(self):
        """프레임 큐에서 프레임을 가져와 데이터베이스의 얼굴과 비교"""
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()  # 큐에서 프레임 가져오기
                members = os.listdir(self.db_path)  # 데이터베이스의 멤버 리스트 가져오기
                detected = False
                emotion = None

                frame_embedding = self.extract_face_embedding(frame)  # 프레임에서 임베딩 추출
                if frame_embedding is None:
                    self.result_queue.put((False, None, None, None))
                    continue

                for member_id in members:
                    member_folder = os.path.join(self.db_path, member_id)
                    photo_path = os.path.join(member_folder, "photo.jpg")
                    info_path = os.path.join(member_folder, "info.txt")

                    if not os.path.isfile(photo_path):
                        continue

                    try:
                        member_image = cv2.imread(photo_path)
                        member_embedding = self.extract_face_embedding(member_image)
                        if member_embedding is None:
                            continue

                        # 임베딩 비교
                        if frame_embedding.shape != member_embedding.shape:
                            print(f"Skipping member {member_id} due to embedding size mismatch.")
                            continue

                        similarity = np.dot(frame_embedding, member_embedding) / (np.linalg.norm(frame_embedding) * np.linalg.norm(member_embedding))

                        if similarity > 0.5:  # 유사도 임계값 설정
                            with open(info_path, 'r') as f:
                                phone_number = f.read().strip()
                            emotion = self.predict_emotion(frame_embedding)  # 감정 예측
                            self.result_queue.put((True, member_id, phone_number, emotion))
                            detected = True
                            break
                    except Exception as e:
                        print(f"Error verifying member {member_id}: {str(e)}")

                if not detected:
                    self.result_queue.put((False, None, None, None))

if __name__ == "__main__":
    db_path = "/home/hui/emotion_detect/src/asian_train_final/성공(얼굴인식,감정인식)/ID"  # 데이터베이스 경로
    model_path = "/home/hui/emotion_detect/src/asian_train_final/성공(얼굴인식,감정인식)/Asian_emotion_model.h5"  # 모델 경로
    face_recognition = FaceRecognition(db_path=db_path, model_path=model_path)

    cam_thread = Thread(target=face_recognition.run_cam)  # 웹캠 스레드 시작
    compare_thread = Thread(target=face_recognition.compare_faces)  # 얼굴 비교 스레드 시작

    cam_thread.start()
    compare_thread.start()

    cam_thread.join()
    compare_thread.join()

