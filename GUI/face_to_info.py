import os
import cv2
import pandas as pd
import numpy as np
from threading import Thread
from deepface import DeepFace
from queue import Queue

class FaceToInfo():
    def __init__(self, db_path = "test/img_db/"):
        self.db_path = db_path
        self.cap = cv2.VideoCapture(0)
        self.db_img = cv2.imread(self.db_path)
        self.db = os.listdir(self.db_path)
        self.color = (255, 255, 255) 
        self.font =  cv2.FONT_HERSHEY_PLAIN
        self.img_queue = Queue(maxsize=1)

        self.analyze_result = None
        self.frame = None
        
        self.known_person = False
        self.pprint_ = False

        self.cam_to_info_deamon = True
        self.cam_deamon = True
    
    def run_cam(self):
        while self.cam_deamon:
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if self.img_queue.full():
                    self.img_queue.get()
                
                self.img_queue.put(frame)
                if self.analyze_result != None:
                    self.frame = self.result_visualization(frame=frame)
                else:
                    cv2.putText(frame, "Waiting Service...",  (0, 50),     self.font, 2, (0, 0, 255), 1, cv2.LINE_AA)
    
    
            # cv2.imshow("test", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        self.cap.release()
        # cv2.destroyAllWindows()
        
    def result_visualization(self, frame):
        cv2.putText(frame, self.analyze_result[0]["dominant_gender"],                        (0, 30),     self.font, 2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, self.analyze_result[0]["dominant_emotion"],                       (0, 60),   self.font, 2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.analyze_result[0]["age"]),                               (0, 90),   self.font, 2, (0, 255, 0), 1, cv2.LINE_AA)
        
        cv2.rectangle(frame, (self.face_data["facial_area"]["x"], self.face_data["facial_area"]["y"]), 
                        (self.face_data["facial_area"]["x"] + self.face_data["facial_area"]["w"], self.face_data["facial_area"]["y"] + self.face_data["facial_area"]["h"]),
                        color=self.color)
        
        if not self.find_result[0].empty and self.known_person:
            cv2.putText(frame, (self.find_result[0]["identity"][0]).split("/")[2].split(".")[0], (0, 120),   self.font, 2, (0, 255, 0), 1, cv2.LINE_AA)

        return frame

    def cam_to_info(self):
        while self.cam_to_info_deamon:
            if self.img_queue.empty():
                continue

            frame = self.img_queue.get()
            faces = DeepFace.extract_faces(img_path=frame, enforce_detection=False)
            
            x = 999
            
            for face_data in faces:
                face_img = (face_data["face"] * 255).astype(np.uint8) 
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                find_result = DeepFace.find(img_path = face_img, db_path = self.db_path, threshold = 0.8, silent = True, enforce_detection = False)
                analyze_result = DeepFace.analyze(img_path = face_img, silent = True, enforce_detection = False)
                
                if x > abs(face_data["facial_area"]["x"] + face_data["facial_area"]["w"] / 2 - 320) and face_data["facial_area"]["x"] != 0:
                    x = abs(face_data["facial_area"]["x"] + face_data["facial_area"]["w"] / 2 - 320)
                    mid_analyze_result = analyze_result
                    mid_find_result = find_result
                    mid_face_data = face_data
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
                else:
                    continue
                
                self.pprint(find_result[0].distance)
                self.pprint(x)

            if 'mid_analyze_result' in locals():
                self.analyze_result = mid_analyze_result
                self.find_result = mid_find_result
                self.face_data = mid_face_data

    def pprint(self, string):
        if self.pprint_ is True:
            print(string)

    def __del__(self):
        self.cap.release()
        # cv2.destroyAllWindows()


        

if __name__ == "__main__":
    face = FaceToInfo()
    cam_thread = Thread(target=face.run_cam)
    cam_thread.start()
    deep_face_thread = Thread(target=face.cam_to_info)
    deep_face_thread.start()