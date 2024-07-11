import cv2
import numpy as np
from deepface import DeepFace


class SaveImage():
    def __init__(self, path):
        self.path = path

    def save_image(self, frame, name):
        save_img = DeepFace.extract_faces(frame)
        face_img = (save_img[0]["face"] * 255).astype(np.uint8)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(self.path + name, face_img)
        

