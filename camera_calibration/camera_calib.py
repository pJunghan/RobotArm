import numpy as np
import cv2 as cv
import glob

# 종료 기준 설정
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 물체 포인트 준비, (0,0,0), (1,0,0), (2,0,0) ....,(8,13,0) 같은 점들
objp = np.zeros((9 * 14, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:14].T.reshape(-1, 2)
objp *= 15  # 각 사각형의 크기를 15mm로 설정

# 3D 물체 포인트와 2D 이미지 포인트를 저장할 배열
objpoints = []  # 실제 공간의 3D 점
imgpoints = []  # 이미지 평면의 2D 점

# 이미지 파일들을 가져오기
images = glob.glob('camera_calibration/calibration_imgs/*.jpg')

if not images:
    print("이미지 파일을 찾을 수 없습니다. 경로를 확인하세요.")
    exit()

for fname in images:
    print(f"처리 중인 이미지: {fname}")
    img = cv.imread(fname)
    if img is None:
        print(f"이미지를 로드할 수 없습니다: {fname}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # 체스보드 코너 찾기
    ret, corners = cv.findChessboardCorners(gray, (9, 14), None)
    print(f"체스보드 코너 찾기 결과: {ret}")

    # 코너를 찾았으면, 물체 포인트와 이미지 포인트를 추가
    if ret:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 코너 그리기 및 표시
        cv.drawChessboardCorners(img, (9, 14), corners2, ret)
        # 주 스레드에서 이미지를 보여주기 위해 저장
        cv.imwrite('camera_calibration/calibration_output.jpg', img)

cv.destroyAllWindows()

# 카메라 캘리브레이션 수행
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 결과 출력
print("카메라 매트릭스:\n", mtx)
print("왜곡 계수:\n", dist)
print("회전 벡터:\n", rvecs)
print("이동 벡터:\n", tvecs)

# 재프로젝션 에러 계산 및 시각적 확인
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

    # 실제 이미지와 재프로젝션된 이미지를 비교하여 시각적으로 표시
    img = cv.imread(images[i])
    for j in range(len(imgpoints[i])):
        cv.circle(img, (int(imgpoints[i][j][0][0]), int(imgpoints[i][j][0][1])), 5, (0, 0, 255), -1)  # 실제 이미지 포인트
        cv.circle(img, (int(imgpoints2[j][0][0]), int(imgpoints2[j][0][1])), 3, (0, 255, 0), -1)  # 재프로젝션된 이미지 포인트

    cv.imshow(f'Reprojection Error - Image {i+1}', img)
    cv.waitKey(0)

mean_error /= len(objpoints)
print(f"재프로젝션 에러: {mean_error}")

# 결과 저장
np.savez('camera_calibration/calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# 예제 이미지를 사용하여 보정된 이미지 보여주기
def show_calibrated_image():
    img = cv.imread('camera_calibration/calibration_output.jpg')
    if img is None:
        print("보정된 이미지를 로드할 수 없습니다.")
        return

    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # 보정된 이미지
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # 보정된 이미지 자르기
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imshow('calibrated', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

# 주 스레드에서 보정된 이미지 보여주기
if __name__ == "__main__":
    show_calibrated_image()
