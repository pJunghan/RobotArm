import numpy as np
import cv2 as cv

# 저장된 캘리브레이션 데이터 로드
calibration_data = np.load('camera_calibration/calibration_data.npz')
mtx = calibration_data['mtx']
dist = calibration_data['dist']

def get_object_coordinates(image_points, mtx, dist):
    # 물체의 실세계 좌표를 계산하기 위해 Z=0으로 설정
    object_points = cv.undistortPoints(np.expand_dims(image_points, axis=1), mtx, dist)

    # 실세계 좌표를 계산
    object_points_3D = cv.convertPointsToHomogeneous(object_points)
    object_points_3D[:, :, 2] = -180  # Z 값을 0으로 설정

    return object_points_3D

# 웹캠에서 실시간으로 물체 좌표 변환
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    # 이미지 좌표를 예시로 설정 (실제로는 물체 검출 알고리즘 필요)
    # 예시로 이미지를 클릭하여 좌표를 얻는다고 가정
    # 이 예제에서는 (500, 300) 좌표를 사용합니다.
    image_points = np.array([[500, 300]], dtype=np.float32)

    # 실세계 좌표 계산
    object_points_3D = get_object_coordinates(image_points, mtx, dist)

    # 결과 출력
    print("실세계 좌표:", object_points_3D)

    # 웹캠 프레임에 좌표 표시
    cv.circle(frame, (int(image_points[0][0]), int(image_points[0][1])), 5, (0, 255, 0), -1)
    cv.putText(frame, f"World Coord: {object_points_3D[0][0]}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv.imshow('Webcam', frame)

    # 'q' 키를 누르면 종료
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
