import os
import cv2
import pandas as pd
import numpy as np
from threading import Thread
from deepface import DeepFace
from queue import Queue

class FaceToInfo():
    def __init__(self, db_path = "test/img_db/") :
        self.db_path = db_path
        self.cap = cv2.VideoCapture(0)
        self.db_img = cv2.imread(self.db_path)
        self.db = os.listdir(self.db_path)
        self.color = (255, 255, 255) 
        self.font =  cv2.FONT_HERSHEY_PLAIN
        self.img_queue = Queue(maxsize=1)
        
        self.width = 640

        self.analyze_result = None
        self.frame = None
        
        self.visualization = False  # 이미지 시각화와 관련된 파라미터 결과값을 보고싶다면 True로
        self.known_person = False
        self.log_print_ = False        # 디버깅 메세지들을 확인하기 위한 파라미터 한번에 관리하기 용이
        self.ret = False

        self.cam_to_info_deamon = True  # 각 스래드를 정지시키기위한 파라미터 False로 할시 해당 스레드 정지
        self.cam_deamon = True          #
    
    def run_cam(self) : # 캠에서 이미지를 받아와 이미지 큐에 넣어주는 함수 결과 시각화가 켜져있다면 해당 결과도 이미지에 추가해줌
        while self.cam_deamon:
            self.ret, frame = self.cap.read()
            self.frame = frame
            if self.ret:
                # self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if self.img_queue.full():
                    self.img_queue.get()
                
                self.img_queue.put(frame)
                if self.analyze_result != None and self.visualization:
                    self.frame = self.result_visualization(frame=frame)
                elif self.analyze_result == None and self.visualization:
                    cv2.putText(frame, "Waiting Service...",  (0, 50),     self.font, 2, (0, 0, 255), 1, cv2.LINE_AA)

    
            # cv2.imshow("test", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        self.cap.release()
        # cv2.destroyAllWindows()
        
    def result_visualization(self, frame) : # 결과값 시각화 함수
        cv2.putText(frame, self.analyze_result[0]["dominant_gender"],                        (0, 30),     self.font, 2, (0, 255, 0), 1, cv2.LINE_AA)# 성별
        cv2.putText(frame, self.analyze_result[0]["dominant_emotion"],                       (0, 60),   self.font, 2, (0, 255, 0), 1, cv2.LINE_AA)  # 감정
        cv2.putText(frame, str(self.analyze_result[0]["age"]),                               (0, 90),   self.font, 2, (0, 255, 0), 1, cv2.LINE_AA)  # 나이
        
        cv2.rectangle(frame, (self.face_data["facial_area"]["x"], self.face_data["facial_area"]["y"]), 
                        (self.face_data["facial_area"]["x"] + self.face_data["facial_area"]["w"], self.face_data["facial_area"]["y"] + self.face_data["facial_area"]["h"]),
                        color=self.color,
                        thickness=3) # 얼굴 범위
        
        if not self.find_result[0].empty and self.known_person: # db에 존재하는 인물이라면 해당 이미지의 이름
            cv2.putText(frame, (self.find_result[0]["identity"][0]).split("/")[2].split(".")[0], (0, 120),   self.font, 2, (0, 255, 0), 1, cv2.LINE_AA)

        return frame

    def cam_to_info(self) : # 이미지 큐 에서 사람의 얼굴을 가져와서 정보들을 넣어주는 함수
        while self.cam_to_info_deamon:
            if self.img_queue.empty(): # 이미지큐가 비어있으면 이번 반복 스킵
                continue

            frame = self.img_queue.get() # 이미지 큐에서 이미지 가져오기
            faces = DeepFace.extract_faces(img_path=frame, enforce_detection=False) # 이미지 큐에서 얼굴부분 추출
            
            x = 999
            
            for face_data in faces: # 다수의 얼굴이 검출될 시 얼굴마다 반복
                face_img = (face_data["face"] * 255).astype(np.uint8)   # 각 얼굴의 이미지
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)    # 색상 타입 변경

                find_result = DeepFace.find(img_path = face_img, db_path = self.db_path, threshold = 0.8, silent = True, enforce_detection = False) # db에서 0.8의 일치도를 보이는 얼굴이 있는지 확인
                analyze_result = DeepFace.analyze(img_path = face_img, silent = True, enforce_detection = False)    # 사람의 정보들을 유추
                
                if x > abs(face_data["facial_area"]["x"] + face_data["facial_area"]["w"] / 2 - self.width / 2) and face_data["facial_area"]["x"] != 0: # 사람의 얼굴이 가장 중앙에 가까울 시 데이터 저장
                    x = abs(face_data["facial_area"]["x"] + face_data["facial_area"]["w"] / 2 - self.width / 2) # 다음 비교를 위해 값 저장 
                    mid_analyze_result = analyze_result
                    mid_find_result = find_result
                    mid_face_data = face_data
                    self.are_you_member(find_result)
                else:
                    continue 
                
                self.log_print(find_result[0].distance)
                self.log_print(x)

            if 'mid_analyze_result' in locals(): # 검출 결과 있는지 확인
                self.analyze_result = mid_analyze_result
                self.find_result = mid_find_result
                self.face_data = mid_face_data

    def are_you_member(self, find_result) : # 멤버쉽 멤버이면(db에 인식률이 0.85 이상의 사진이 있다면) 초록색, 아니면 빨간색으로 표시해주는 함수
        if not find_result[0].empty :
            if find_result[0].distance.iloc[0] < 0.15:
                self.color = (0, 255, 0)
                self.known_person = True
            else:
                self.color = (0, 0, 255)
                self.known_person = False
        else:
            self.color = (0, 0, 255)
            self.known_person = False

    def log_print(self, string) : # self.log_print_ 를 통해 한번에 비활성화, 활성화 가능한 print
        if self.log_print_ is True:
            print(string)

    def get_frame(self) : # 이미지 가져올 수 있는 함수
        if self.frame is None:
            return False, self.frame
        else:
            return True, self.frame

    def __del__(self) : 
        self.cam_to_info_deamon = False  # 각 스래드를 정지시키기위한 파라미터 False로 할시 해당 스레드 정지
        self.cam_deamon = False          #
        self.cap.release()
        # cv2.destroyAllWindows()


        

if __name__ == "__main__":
    face = FaceToInfo()
    cam_thread = Thread(target=face.run_cam)
    cam_thread.start()
    deep_face_thread = Thread(target=face.cam_to_info)
    deep_face_thread.start()