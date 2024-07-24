import numpy as np
import cv2 as cv
import glob

def camera_calibration():
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
    images = glob.glob('camera_calibration/raw_imgs/*.jpg')

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
            cv.imwrite(f'camera_calibration/calibration_output_imgs/calibrate_{fname.split("/")[-1]}', img)

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

    mean_error /= len(objpoints)
    print(f"재프로젝션 에러: {mean_error}")

    # 결과 저장
    np.savez('camera_calibration/calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


def show_calibrated_image():
    # 보정된 이미지를 사용하기 위해 보정 데이터를 로드
    data = np.load('camera_calibration/calibration_data.npz')
    mtx = data['mtx']
    dist = data['dist']
    
    img = cv.imread('camera_calibration/calibration_output_imgs/calibrate_image_0009.jpg') # 보정된 예제 이미지를 불러오기
    if img is None:
        print("보정된 이미지를 로드할 수 없습니다.")
        return

    # 원본 이미지 크기 확인
    h, w = img.shape[:2]
    print(f"원본 이미지 크기: {w} x {h}")
    
    # 최적의 새로운 카메라 매트릭스 계산
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0.5, (w, h))  # 자유 크기 매개변수 값을 0으로 설정
    print(f"newcameramtx: \n{newcameramtx}")
    print(f"roi: {roi}")

    # 보정된 이미지
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    
    # 보정된 이미지 저장 및 크기 확인
    cv.imwrite('camera_calibration/calibrated_undistorted.png', dst)
    print(f"보정된 이미지 크기: {dst.shape[1]} x {dst.shape[0]}")

    # 보정된 이미지 자르기
    x, y, w, h = roi
    print(f"자르기 영역: x={x}, y={y}, w={w}, h={h}")
    
    # 자르기 영역이 유효한지 확인
    if w > 0 and h > 0:
        dst = dst[y:y+h, x:x+w]
        cv.imwrite('camera_calibration/calibrated_cropped.png', dst)
        print(f"잘린 보정된 이미지 크기: {dst.shape[1]} x {dst.shape[0]}")
    else:
        print("유효한 자르기 영역이 없습니다.")

    # 보정된 이미지 표시
    cv.namedWindow('calibrated', cv.WINDOW_NORMAL)
    cv.imshow('calibrated', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    # camera_calibration()
    show_calibrated_image()
