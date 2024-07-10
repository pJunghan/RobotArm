import os
import cv2
import numpy as np
import sqlite3
from threading import Thread
from deepface import DeepFace
from queue import Queue

class FaceToInfo:
    def __init__(self, db_path="RobotArm/GUI/DB/user_data.db"):
        self.db_path = db_path  # 데이터베이스 경로
        self.cap = cv2.VideoCapture(0)  # 웹캠 캡처 초기화
        self.db_images = []  # 이미지 목록 초기화
        self.color = (255, 255, 255)  # 기본 색상 설정
        self.font = cv2.FONT_HERSHEY_PLAIN  # 폰트 설정
        self.img_queue = Queue(maxsize=1)  # 이미지 큐 초기화

        self.width = 640  # 프레임의 폭 설정
        self.height = 480  # 프레임의 높이 설정

        self.analyze_result = None  # 분석 결과 초기화
        self.frame = None  # 프레임 초기화

        self.visualization = False  # 시각화 여부 초기화
        self.known_person = False  # 인식된 사람 여부 초기화
        self.pprint_ = False  # 디버그 출력 여부 초기화
        self.ret = False  # 웹캠 캡처 결과 초기화

        self.cam_to_info_deamon = True  # 카메라 데몬 플래그
        self.cam_deamon = True  # 정보 데몬 플래그

        self.load_images_from_db()  # 데이터베이스에서 이미지 로드

    def load_images_from_db(self):
        # 데이터베이스에서 이미지 로드
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, image FROM users WHERE image IS NOT NULL")
        rows = cursor.fetchall()
        for row in rows:
            image_data = row[1]
            np_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            self.db_images.append((row[0], image))
        conn.close()

    def run_cam(self):
        # 카메라 실행 및 프레임 큐에 추가
        while self.cam_deamon:
            self.ret, frame = self.cap.read()
            self.frame = frame
            if self.ret:
                if self.img_queue.full():
                    self.img_queue.get()
                self.img_queue.put(frame)
                if self.analyze_result is not None and self.visualization:
                    self.frame = self.result_visualization(frame=frame)
                elif self.analyze_result is None and self.visualization:
                    cv2.putText(frame, "Waiting Service...", (0, 50), self.font, 2, (0, 0, 255), 1, cv2.LINE_AA)
        self.cap.release()

    def result_visualization(self, frame):
        # 분석 결과를 프레임에 시각화
        cv2.putText(frame, self.analyze_result[0]["dominant_gender"], (0, 30), self.font, 2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, self.analyze_result[0]["dominant_emotion"], (0, 60), self.font, 2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.analyze_result[0]["age"]), (0, 90), self.font, 2, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.rectangle(frame,
                      (self.face_data["facial_area"]["x"], self.face_data["facial_area"]["y"]),
                      (self.face_data["facial_area"]["x"] + self.face_data["facial_area"]["w"],
                       self.face_data["facial_area"]["y"] + self.face_data["facial_area"]["h"]),
                      color=self.color,
                      thickness=3)

        if self.find_result and self.known_person:
            cv2.putText(frame, f"ID: {self.find_result['id']}", (0, 120), self.font, 2, (0, 255, 0), 1, cv2.LINE_AA)

        return frame

    def cam_to_info(self):
        # 프레임에서 얼굴 인식 및 분석 수행
        while self.cam_to_info_deamon:
            if self.img_queue.empty():
                continue

            frame = self.img_queue.get()
            faces = DeepFace.extract_faces(img_path=frame, enforce_detection=False)

            x = 999

            for face_data in faces:
                face_img = (face_data["face"] * 255).astype(np.uint8)
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                best_match = None
                best_distance = float("inf")

                for user_id, db_img in self.db_images:
                    try:
                        result = DeepFace.verify(face_img, db_img, enforce_detection=False, silent=True)
                        distance = result["distance"]

                        if distance < best_distance:
                            best_distance = distance
                            best_match = {"id": user_id, "distance": distance}
                    except Exception as e:
                        print(f"Error during verification: {e}")
                        continue

                analyze_result = DeepFace.analyze(img_path=face_img, silent=True, enforce_detection=False)

                if x > abs(face_data["facial_area"]["x"] + face_data["facial_area"]["w"] / 2 - self.width / 2) and face_data[
                    "facial_area"]["x"] != 0:
                    x = abs(face_data["facial_area"]["x"] + face_data["facial_area"]["w"] / 2 - self.width / 2)
                    mid_analyze_result = analyze_result
                    mid_find_result = best_match if best_match else {}
                    mid_face_data = face_data
                    self.are_you_member(best_match)

                self.pprint(best_distance)
                self.pprint(x)

            if 'mid_analyze_result' in locals():
                self.analyze_result = mid_analyze_result
                self.find_result = mid_find_result
                self.face_data = mid_face_data

    def are_you_member(self, find_result):
        # 인식된 사람이 회원인지 확인
        if find_result and find_result["distance"] > 0.7:
            self.color = (0, 255, 0)
            self.known_person = True
        else:
            self.color = (0, 0, 255)
            self.known_person = False

    def pprint(self, string):
        # 디버그 출력
        if self.pprint_ is True:
            print(string)

    def get_frame(self):
        # 현재 프레임 반환
        if self.frame is None:
            return False, self.frame
        else:
            return True, self.frame

    def __del__(self):
        # 객체 삭제 시 데몬 플래그 설정 및 캡처 릴리스
        self.cam_to_info_deamon = False
        self.cam_deamon = False
        self.cap.release()

if __name__ == "__main__":
    face = FaceToInfo()
    cam_thread = Thread(target=face.run_cam)
    cam_thread.start()
    deep_face_thread = Thread(target=face.cam_to_info)
    deep_face_thread.start()
