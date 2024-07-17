import os
import cv2
import time
import uuid
import numpy as np
from threading import Thread
from queue import Queue
import tensorflow as tf
from insightface.app import FaceAnalysis
from config import db_path, model_path, age_prototxt, age_model, gender_prototxt, gender_model

class FaceRecognition:
    def __init__(self, db_path = db_path, model_path = model_path, age_prototxt = age_prototxt, age_model = age_model, gender_prototxt = gender_prototxt, gender_model = gender_model):
        self.db_path = db_path
        self.cap = cv2.VideoCapture(0)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.detected = False
        self.visualization = True
        self.last_check = time.time()
        self.check_interval = 0.5
        self.failed_attempts = 0
        self.frame_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        self.member_id = None
        self.phone_number = None
        self.frame = None
        self.known_person = None
        self.labels = ['angry', 'happy', 'sad']
        self.model = tf.keras.models.load_model(model_path)
        self.age_net = cv2.dnn.readNetFromCaffe(age_prototxt, age_model)
        self.gender_net = cv2.dnn.readNetFromCaffe(gender_prototxt, gender_model)
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_list = ['(0-10)', '(10-15)' '(15-20)', '(20-25)', '(25-30)', '(30-35)' '(35-40)', '(40-45)','(45-50)', '(50-55)','(55-60)', '(60-70)', '(70-80)', '(80-90)', '(90-100)']
        self.gender_list = ['Male', 'Female']
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.result_dict = {"detected" : False, "member_id" : None, "emotion" : None, "age" : None, "gender" : None}
        self.cam_to_info_deamon = True
        self.cam_deamon = True

    def run_cam(self):
        """웹캠을 통해 얼굴을 인식하고 결과를 화면에 표시"""
        while self.cam_deamon:
            ret, self.frame = self.cap.read()
            if not ret:
                continue

            current_time = time.time()
            if current_time - self.last_check > self.check_interval:
                self.last_check = current_time
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(self.frame.copy())

            if not self.result_queue.empty():
                self.detected, self.member_id, emotion, age, gender = self.result_queue.get()
            
            display_frame = self.frame.copy()
            if self.detected:
                cv2.putText(display_frame, f"ID: {self.member_id}", (10, 30), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # cv2.putText(display_frame, f"Phone: {self.phone_number}", (10, 60), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(display_frame, f"Emotion: {emotion}", (10, 90), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(display_frame, f"Age: {age}", (10, 120), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(display_frame, f"Gender: {gender}", (10, 150), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(display_frame, "Not recognized", (10, 30), self.font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        #     cv2.imshow("Face Recognition", display_frame)
        #     key = cv2.waitKey(1) & 0xFF
        #     if key == ord('c'):
        #         self.capture_and_save(frame)
        #     elif key == ord('q'):
        #         break

        # self.cap.release()
        # cv2.destroyAllWindows()

    def capture_and_save(self, frame):
        """현재 프레임을 캡처하여 저장"""
        phone_number = input("Enter phone number: ")
        member_id = str(uuid.uuid4())
        member_folder = os.path.join(self.db_path, member_id)
        os.makedirs(member_folder, exist_ok=True)
        captured_image_path = os.path.join(member_folder, "photo.jpg")
        cv2.imwrite(captured_image_path, frame)

        with open(os.path.join(member_folder, "info.txt"), 'w') as f:
            f.write(phone_number)

        print(f"Member registered with ID: {member_id}")
        self.member_id = member_id
        self.phone_number = phone_number

    def extract_face_embedding(self, img):
        """이미지에서 얼굴 임베딩 추출"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.app.get(img_rgb)
        if len(faces) > 0:
            return faces[0].embedding, faces[0].bbox
        return None, None

    def predict_emotion(self, embedding):
        """임베딩을 기반으로 감정을 예측"""
        prediction = self.model.predict(np.expand_dims(embedding, axis=0))
        return self.labels[np.argmax(prediction)]

    def predict_age_and_gender(self, face_img):
        """얼굴 이미지에서 나이와 성별 예측"""
        face_img_resized = cv2.resize(face_img, (227, 227))  # 크기 조정
        blob = cv2.dnn.blobFromImage(face_img_resized, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)

        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = self.gender_list[gender_preds[0].argmax()]

        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = self.age_list[age_preds[0].argmax()]

        return age, gender

    def compare_faces(self):
        """프레임 큐에서 프레임을 가져와 데이터베이스의 얼굴과 비교"""
        while self.cam_to_info_deamon:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                members = os.listdir(self.db_path)
                detected = False
                emotion = None
                age = None
                gender = None

                frame_embedding, bbox = self.extract_face_embedding(frame)
                if frame_embedding is None:
                    self.result_queue.put((False, None, None, None, None))
                    self.result_to_dict()
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                face_img = frame[y1:y2, x1:x2]

                for member_id in members:
                    # member_folder = os.path.join(self.db_path, member_id)
                    photo_path = os.path.join(self.db_path, member_id)
                    # info_path = os.path.join(self.db_path, "info.txt")

                    if not os.path.isfile(photo_path):
                        continue

                    try:
                        member_image = cv2.imread(photo_path)
                        member_embedding, _ = self.extract_face_embedding(member_image)
                        if member_embedding is None:
                            continue

                        # 임베딩 비교
                        if frame_embedding.shape != member_embedding.shape:
                            print(f"Skipping member {member_id} due to embedding size mismatch.")
                            continue

                        similarity = np.dot(frame_embedding, member_embedding) / (np.linalg.norm(frame_embedding) * np.linalg.norm(member_embedding))

                        if similarity > 0.5:  # 유사도 임계값 설정
                            # with open(info_path, 'r') as f:
                            #     phone_number = f.read().strip()
                            emotion = self.predict_emotion(frame_embedding)  # 감정 예측
                            age, gender = self.predict_age_and_gender(face_img)  # 나이 및 성별 예측
                            self.result_queue.put((True, member_id, emotion, age, gender))
                            detected = True
                            self.result_to_dict(detected, member_id, emotion, age, gender)
                            break
                    except Exception as e:
                        print(f"Error verifying member {member_id}: {str(e)}")

                if not detected:
                    self.result_queue.put((False, None, None, None, None))
                    self.result_to_dict()

    def get_frame(self):  # 현재 프레임을 반환
        if self.frame is None:
            return False, self.frame
        else:
            return True, self.frame
        
    def result_to_dict(self, detected = False, member_id = None, emotion = None, age = None, gender = None):
        self.result_dict["detected"] = detected
        self.result_dict["member_id"] = member_id
        self.result_dict["emotion"] = emotion
        self.result_dict["age"] = age
        self.result_dict["gender"] = gender
        if detected :
            self.known_person = member_id.split(".")[0]
            self.failed_attempts = 0
        else:
            self.known_person = None
            self.failed_attempts +=1

    def __del__(self):
        self.cap.release()

        
if __name__ == "__main__":
    face_recognition = FaceRecognition(
        db_path=db_path, 
        model_path=model_path, 
        age_prototxt=age_prototxt, 
        age_model=age_model, 
        gender_prototxt=gender_prototxt, 
        gender_model=gender_model
    )

    cam_thread = Thread(target=face_recognition.run_cam)  # 웹캠 스레드 시작
    compare_thread = Thread(target=face_recognition.compare_faces)  # 얼굴 비교 스레드 시작

    cam_thread.start()
    compare_thread.start()

    cam_thread.join()
    compare_thread.join()
