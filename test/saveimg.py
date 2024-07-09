import cv2
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 웹캠에서 프레임 읽기

ret, frame = cap.read()
# frame = cv2.imread("/test/img_db/test.jpeg")

if ret:
    # 이미지를 파일로 저장
    save_img = DeepFace.extract_faces(frame)
    face_img = (save_img[0]["face"] * 255).astype(np.uint8)
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
else:
    print("프레임을 읽을 수 없습니다.")

# 웹캠 해제
cap.release()

# 모든 창 닫기
cv2.destroyAllWindows()