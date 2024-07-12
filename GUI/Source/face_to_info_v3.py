import cv2
import numpy as np
from deepface import DeepFace
import os

def get_images_from_directory(directory_path):
    images = {}
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            user_id = os.path.splitext(filename)[0]
            img_path = os.path.join(directory_path, filename)
            images[user_id] = cv2.imread(img_path)
    return images

image_directory = '/home/lsm/git_ws/RobotArm/GUI/Image'
images = get_images_from_directory(image_directory)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet"]
metrics = ["cosine", "euclidean", "euclidean_l2"]
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet', 'centerface']

model_name = models[0]
distance_metric = metrics[0]
detector_backend = backends[5]

threshold = 0.6

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    try:
        faces = DeepFace.extract_faces(frame, detector_backend=detector_backend, enforce_detection=False)
        
        for face in faces:
            if 'facial_area' in face:
                facial_area = face['facial_area']
                if isinstance(facial_area, dict):
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                elif isinstance(facial_area, (list, tuple)) and len(facial_area) == 4:
                    x, y, w, h = facial_area
                else:
                    print("Unexpected facial_area format")
                    continue
            else:
                print("No facial_area found in face data")
                continue

            match = False
            for user_id, db_image in images.items():
                try:
                    face_image = frame[y:y+h, x:x+w]
                    result = DeepFace.verify(face_image, db_image, 
                                             model_name=model_name, 
                                             distance_metric=distance_metric, 
                                             detector_backend=detector_backend)

                    if result['verified']:
                        match = True
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Matched: ID {user_id}", (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        break
                except Exception as e:
                    print(f"ID {user_id}와 비교 중 오류 발생:", str(e))

            if not match:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "No Match", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('Webcam', frame)

    except Exception as e:
        print("얼굴 분석 중 오류 발생:", str(e))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()