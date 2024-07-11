import cv2
import sqlite3
import numpy as np
from deepface import DeepFace
import threading

# SQLite 데이터베이스에서 이미지 데이터를 가져오는 함수
def get_images_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT ID, Image FROM users WHERE Image IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()

    images = {}
    for row in rows:
        user_id, image_data = row
        np_img = np.frombuffer(image_data, dtype=np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        images[user_id] = img

    return images

# 데이터베이스에서 이미지를 가져옵니다
db_path = '/home/lsm/git_ws/RobotArm/GUI/DB/user_data.db'
images = get_images_from_db(db_path)

# 웹캠에서 실시간 얼굴 감지 및 분석
cap = cv2.VideoCapture(0)

# 파라미터 설정
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet"]
metrics = ["cosine", "euclidean", "euclidean_l2"]
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet', 'centerface']

frame = None
ret = False

def capture_frames():
    global ret, frame
    while True:
        if cap.isOpened():
            ret, frame = cap.read()
        else:
            break

capture_thread = threading.Thread(target=capture_frames)
capture_thread.daemon = True
capture_thread.start()

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    if frame is not None:
        try:
            # 얼굴 검출
            detected_faces = DeepFace.extract_faces(frame, detector_backend=backends[5], enforce_detection=False)
            
            if detected_faces:
                # 가장 큰 얼굴을 선택
                largest_face = max(detected_faces, key=lambda face: face['facial_area']['w'] * face['facial_area']['h'])
                x, y, w, h = largest_face['facial_area']['x'], largest_face['facial_area']['y'], largest_face['facial_area']['w'], largest_face['facial_area']['h']
                
                # 데이터베이스의 각 이미지와 비교
                match = False
                for user_id, db_image in images.items():
                    try:
                        face_image = frame[y:y+h, x:x+w]
                        verify_result = DeepFace.verify(face_image, db_image, model_name=models[0], distance_metric=metrics[1], detector_backend=backends[0], threshold=0.8)

                        if verify_result['verified']:
                            match = True
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            match_text = f"Matched: ID {user_id}"
                            cv2.putText(frame, match_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            break
                    except Exception as e:
                        print("얼굴 비교 중 오류 발생:", e)

                if not match:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "No Match", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # 원본 프레임 표시
            cv2.imshow('Webcam', frame)

        except Exception as e:
            print("얼굴 분석 중 오류 발생:", e)

    # 'q' 키를 누르면 루프를 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
