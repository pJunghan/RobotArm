import os
import cv2
import pandas as pd
import numpy as np
from threading import Thread
from deepface import DeepFace
from queue import Queue

class FaceToInfo():
    def __init__(self):
        self.db_path = "test/img_db/"
        self.cap = cv2.VideoCapture(0)
        self.db_img = cv2.imread(self.db_path)
        self.db = os.listdir(self.db_path)
        self.blue = (255, 0, 0)
        self.green = (0, 255, 0)
        self.red = (0, 0, 255)
        self.white = (255, 255, 255) 
        self.font =  cv2.FONT_HERSHEY_PLAIN
        self.analyze_result = None
        self.img_queue = Queue(maxsize=1)

        self.cam_to_info_deamon = True
    
    def run_cam(self):
        while 1:
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if self.img_queue.full():
                    self.img_queue.get()
                
                self.img_queue.put(frame)
                if not self.analyze_result == None:

                    # print("test1")
                    # print(analyze_result)
                    # print("test2")
                    # print(find_result)
                    # print("test3")
                    # print(pose)
                    
                    
                    cv2.putText(frame, self.analyze_result[0]["dominant_gender"],                        (0, 30),     self.font, 2, self.green, 1, cv2.LINE_AA)
                    cv2.putText(frame, self.analyze_result[0]["dominant_emotion"],                       (0, 60),   self.font, 2, self.green, 1, cv2.LINE_AA)
                    cv2.putText(frame, str(self.analyze_result[0]["age"]),                               (0, 90),   self.font, 2, self.green, 1, cv2.LINE_AA)
                    cv2.putText(frame, (self.find_result[0]["identity"][0]).split("/")[2].split(".")[0], (0, 120),   self.font, 2, self.green, 1, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "Waiting Service...",  (0, 50),     self.font, 2, self.red, 1, cv2.LINE_AA)
    
    
            cv2.imshow("test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def cam_to_info(self):
        while self.cam_to_info_deamon:
            if self.img_queue.empty():
                continue

            frame = self.img_queue.get()
            faces = DeepFace.extract_faces(img_path=frame, enforce_detection=False)
            
            print(faces)
            w = 0

            for face_data in faces:
                face_img = (face_data["face"] * 255).astype(np.uint8) 
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                find_result = (DeepFace.find(img_path = face_img, db_path = self.db_path, threshold = 0.9, silent = True, enforce_detection = False))
                analyze_result = DeepFace.analyze(img_path = face_img, silent = True, enforce_detection = False)
                
                # cv2.imshow("cut_img", face_img)
                if w < analyze_result[0]["region"]["w"]:
                    self.find_result = find_result
                    self.analyze_result = analyze_result
                    w = analyze_result[0]["region"]["w"]
                else:
                    continue




        

if __name__ == "__main__":
    face = FaceToInfo()
    cam_thread = Thread(target=face.run_cam)
    cam_thread.start()
    deep_face_thread = Thread(target=face.cam_to_info)
    deep_face_thread.start()