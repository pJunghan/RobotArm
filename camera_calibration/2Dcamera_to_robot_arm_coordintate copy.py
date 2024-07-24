import numpy as np
import cv2 as cv

"""
코드 구성 및 설명
1. 카메라 보정 데이터 로드
2. 기준점 좌표 측정
3. 변환 행렬 계산
4. 실시간 좌표 변환 및 표시

요약
기준점 좌표 수집:

카메라 좌표계와 로봇 좌표계에서 각각의 기준점 좌표를 측정합니다.
카메라 좌표는 OpenCV를 사용하여 체커보드 패턴을 통해 측정합니다.
로봇 좌표는 로봇 제어 소프트웨어를 사용하여 직접 측정합니다.
변환 행렬 계산:

수집한 좌표를 사용하여 cv.findHomography를 통해 변환 행렬을 계산합니다.
"""

"""
1. 카메라 보정 데이터 로드
카메라 보정 데이터를 로드하여 카메라 매트릭스와 왜곡 계수를 가져옵니다. 이 데이터는 이전에 cv2.calibrateCamera를 사용하여 저장된 데이터입니다.
"""

# 저장된 캘리브레이션 데이터 로드
calibration_data = np.load('camera_calibration/calibration_data.npz')
mtx = calibration_data['mtx']
dist = calibration_data['dist']

"""
2. 기준점 좌표 측정
기준점의 카메라 좌표와 로봇 좌표를 측정합니다. 이 예제에서는 가상의 좌표를 사용합니다.
"""

# 예제 기준점 좌표 (카메라 좌표계와 로봇 좌표계에서 각각 측정된 좌표)
camera_points = np.array([
    [100, 200],  # 기준점 1의 카메라 좌표
    [200, 300],  # 기준점 2의 카메라 좌표
    [300, 400],  # 기준점 3의 카메라 좌표
    [400, 500],  # 기준점 4의 카메라 좌표
], dtype=np.float32)

robot_points = np.array([
    [10, 20],  # 기준점 1의 로봇 좌표
    [20, 30],  # 기준점 2의 로봇 좌표
    [30, 40],  # 기준점 3의 로봇 좌표
    [40, 50],  # 기준점 4의 로봇 좌표
], dtype=np.float32)

"""
3. 변환 행렬 계산
카메라 좌표를 로봇 좌표로 변환하기 위해 변환 행렬을 계산합니다. cv2.findHomography를 사용하여 변환 행렬을 계산합니다.
"""

# 호모그래피 변환 행렬 계산
H, _ = cv.findHomography(camera_points, robot_points)

print("호모그래피 변환 행렬 H:\n", H)

"""
4. 실시간 좌표 변환 및 표시
웹캠을 사용하여 실시간으로 프레임을 가져오고, 특정 이미지 좌표를 로봇 좌표로 변환하여 표시합니다.
"""

def transform_to_robot_coordinates(image_points, H):
    # 이미지 좌표를 로봇 좌표로 변환
    camera_coords = np.array([image_points], dtype=np.float32)
    camera_coords = np.array([camera_coords])
    robot_coords = cv.perspectiveTransform(camera_coords, H)
    return robot_coords[0][0]

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
    image_points = [500, 300]

    # 로봇 좌표계로 변환
    robot_coords = transform_to_robot_coordinates(image_points, H)

    # 결과 출력
    print("카메라 좌표계 좌표:", image_points)
    print("로봇 좌표계 좌표:", robot_coords)

    # 웹캠 프레임에 좌표 표시
    cv.circle(frame, (int(image_points[0]), int(image_points[1])), 5, (0, 255, 0), -1)
    cv.putText(frame, f"Robot Coord: {robot_coords}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv.imshow('Webcam', frame)

    # 'q' 키를 누르면 종료
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
