import os
import cv2
import numpy as np
from threading import Thread
from deepface import DeepFace
from queue import Queue
from config import user_img_path

class FaceToInfo():
    def __init__(self, db_path=user_img_path):
        self.db_path = db_path
        self.cap = cv2.VideoCapture(0)
        self.db_img = cv2.imread(self.db_path)
        self.db = os.listdir(self.db_path)
        self.color = (255, 255, 255)
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.img_queue = Queue(maxsize=1)
        
        self.width = 640

        self.analyze_result = None
        self.frame = None
        self.name = None

        self.visualization = False  # 이미지 시각화와 관련된 파라미터 결과값을 보고싶다면 True로
        self.known_person = None  # 사용자 ID를 저장하기 위한 변수
        self.log_print_ = False  # 디버깅 메시지를 확인하기 위한 파라미터
        self.ret = False

        self.cam_to_info_deamon = True  # 각 스레드를 정지시키기 위한 파라미터 False로 할 시 해당 스레드 정지
        self.cam_deamon = True
    
    def run_cam(self):  # 캠에서 이미지를 받아와 이미지 큐에 넣어주는 함수
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

    def result_visualization(self, frame):  # 결과값 시각화 함수
        cv2.putText(frame, self.analyze_result[0]["dominant_gender"], (0, 30), self.font, 2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, self.analyze_result[0]["dominant_emotion"], (0, 60), self.font, 2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.analyze_result[0]["age"]), (0, 90), self.font, 2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (self.face_data["facial_area"]["x"], self.face_data["facial_area"]["y"]),
                      (self.face_data["facial_area"]["x"] + self.face_data["facial_area"]["w"], self.face_data["facial_area"]["y"] + self.face_data["facial_area"]["h"]),
                      color=self.color, thickness=3)
        if not self.find_result[0].empty and self.known_person:
            self.name = (self.find_result[0]["identity"][0]).split("/")[2].split(".")[0]
            cv2.putText(frame, self.name, (0, 120), self.font, 2, (0, 255, 0), 1, cv2.LINE_AA)

        return frame

    def cam_to_info(self):  # 이미지 큐에서 사람의 얼굴을 가져와서 정보들을 넣어주는 함수
        while self.cam_to_info_deamon:
            if self.img_queue.empty():
                continue

            frame = self.img_queue.get()
            faces = DeepFace.extract_faces(img_path=frame, enforce_detection=False)
            
            x = 999
            
            for face_data in faces:
                face_img = (face_data["face"] * 255).astype(np.uint8)
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                find_result = DeepFace.find(img_path=face_img, db_path=self.db_path, threshold=0.8, silent=True, enforce_detection=False)
                analyze_result = DeepFace.analyze(img_path=face_img, silent=True, enforce_detection=False)
                
                if x > abs(face_data["facial_area"]["x"] + face_data["facial_area"]["w"] / 2 - self.width / 2) and face_data["facial_area"]["x"] != 0:
                    x = abs(face_data["facial_area"]["x"] + face_data["facial_area"]["w"] / 2 - self.width / 2)
                    mid_analyze_result = analyze_result
                    mid_find_result = find_result
                    mid_face_data = face_data
                    self.are_you_member(find_result)
                else:
                    continue 
                
                self.log_print(find_result[0].distance)
                self.log_print(x)

            if 'mid_analyze_result' in locals():
                self.analyze_result = mid_analyze_result
                self.find_result = mid_find_result
                self.face_data = mid_face_data

    def are_you_member(self, find_result):  # 멤버쉽 멤버인지 확인
        if not find_result[0].empty:
            if find_result[0].distance.iloc[0] < 0.15: # 85프로 이상 일치할때 찾음
                self.color = (0, 255, 0)
                self.known_person = find_result[0]["identity"][0].split("/")[-1].split(".")[0]  # 사용자 ID로 설정
            else:
                self.color = (0, 0, 255)
                self.known_person = None
        else:
            self.color = (0, 0, 255)
            self.known_person = None

    def log_print(self, string):  # 디버깅 로그 출력
        if self.log_print_:
            print(string)

    def get_frame(self):  # 현재 프레임을 반환
        if self.frame is None:
            return False, self.frame
        else:
            return True, self.frame

    def __del__(self):
        self.cam_to_info_deamon = False
        self.cam_deamon = False
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face = FaceToInfo()
    cam_thread = Thread(target=face.run_cam)
    cam_thread.start()
    deep_face_thread = Thread(target=face.cam_to_info)
    deep_face_thread.start()
