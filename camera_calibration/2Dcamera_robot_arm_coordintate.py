import numpy as np
import cv2 as cv

"""
코드 구성 및 설명
1. 기준점 좌표 측정
2. 변환 행렬 계산
3. 실시간 좌표 변환 및 표시

요약
기준점 좌표 수집:

카메라 캘리브레이션 없이 카메라의 픽셀 좌료를 로봇 좌표로 변환

수집한 좌표를 사용하여 cv.findHomography를 통해 변환 행렬을 계산합니다.
"""

class CameraRobotTransformer:
    def __init__(self):
        # (카메라 좌표계와 로봇 좌표계에서 각각 측정된 좌표)
        self.camera_points =  np.array([
            [118, 210],  # 기준점 1의 카메라 좌표
            [114, 271],  # 기준점 2의 카메라 좌표
            [110, 333],  # 기준점 3의 카메라 좌표
            [480, 210],  # 기준점 4의 카메라 좌표
            [486, 268],  # 기준점 5의 카메라 좌표
            [490, 330],  # 기준점 6의 카메라 좌표
            [424, 267],  # 기준점 7의 카메라 좌표
            ], dtype=np.float32)
        
        self.robot_points = np.array([
            [300, -101],  # 기준점 1의 로봇 좌표
            [296, 0.1],  # 기준점 2의 로봇 좌표
            [298.6, 99.6],  # 기준점 3의 로봇 좌표
            [-295.4, -96.6],  # 기준점 4의 로봇 좌표
            [-296, 0.8],  # 기준점 5의 로봇 좌표
            [-301.5, 96.8],  # 기준점 6의 로봇 좌표
            [-198, -2.5],  # 기준점 7의 로봇 좌표
            ], dtype=np.float32)
        
        self.H = self.compute_homography_matrix()

    def compute_homography_matrix(self):
        H, _ = cv.findHomography(self.camera_points, self.robot_points)
        print("호모그래피 변환 행렬 H:\n", H)
        return H

    def transform_to_robot_coordinates(self, image_points):
        camera_coords = np.array([image_points], dtype=np.float32)
        camera_coords = np.array([camera_coords])
        robot_coords = cv.perspectiveTransform(camera_coords, self.H)
        # 좌표를 소수점 한 자리로 반올림
        robot_coords = [round(float(coord), 1) for coord in robot_coords[0][0]]
        return robot_coords

def main():

    transformer = CameraRobotTransformer()

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)  # 프레임 너비 설정
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)  # 프레임 높이 설정

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break

        # 이미지 좌표를 예시로 설정 (실제로는 물체 검출 알고리즘 필요)
        # 여기에 카메라로 감지된 바운딩 박스 중심 좌표 설정
        image_points = [486, 268]

        # 로봇 좌표계로 변환
        robot_coords = transformer.transform_to_robot_coordinates(image_points)

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

def start_webcam():
    # 웹캠 객체 생성
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)  # 프레임 너비 설정
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)  # 프레임 높이 설정

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break

        cv.imshow('Webcam', frame)

        # ESC 키를 누르면 종료
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
    # start_webcam()
