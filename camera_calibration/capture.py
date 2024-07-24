import cv2 as cv
import os

# 저장할 디렉토리 설정
save_dir = '/home/pjh/RobotArm/camera_calibration/calibration_imgs'
os.makedirs(save_dir, exist_ok=True)

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 이미지 캡처 및 저장
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    # 이미지 표시
    cv.imshow('Webcam - Press SPACE to capture, Q to quit', frame)

    # 키 입력 대기
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # 스페이스바를 누르면 이미지 저장
        img_path = os.path.join(save_dir, f'calib_image_{count:03d}.jpg')
        cv.imwrite(img_path, frame)
        print(f"이미지 저장: {img_path}")
        count += 1

cap.release()
cv.destroyAllWindows()
