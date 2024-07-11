import sys
import os
import cv2
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QTimer, Qt

# UI 파일을 로드합니다.
signup_form_class = uic.loadUiType("RobotArm/GUI/signup.ui")[0]

class SignVideoThread:
    def __init__(self, graphics_view):
        self.graphics_view = graphics_view
        self.webcam = cv2.VideoCapture(0)
        if not self.webcam.isOpened():
            print("웹캠을 열 수 없습니다.")
            exit()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(60)  # 60 ms마다 프레임을 갱신합니다.
        self.current_frame = None

    def update_frame(self):
        status, frame = self.webcam.read()
        if status:
            self.current_frame = frame.copy()
            # OpenCV의 BGR 이미지를 Qt의 RGB 이미지로 변환합니다.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # graphicsView의 크기에 맞게 프레임을 리사이즈합니다.
            gv_width = self.graphics_view.width()
            gv_height = self.graphics_view.height()
            frame = cv2.resize(frame, (gv_width, gv_height))

            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.display_image(pixmap)

    def display_image(self, pixmap):
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(pixmap))
        self.graphics_view.setScene(scene)
        self.graphics_view.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def stop(self):
        self.timer.stop()
        self.webcam.release()

    def save_current_frame(self, file_path):
        if self.current_frame is not None:
            cv2.imwrite(file_path, self.current_frame)
            print(f"사진이 {file_path}에 저장되었습니다.")
        else:
            print("저장할 프레임이 없습니다.")

class SignUpDialog(QDialog, signup_form_class):
    def __init__(self):
        super(SignUpDialog, self).__init__()
        self.setupUi(self)
        self.video_thread = SignVideoThread(self.graphicsView)
        self.takePhotoBtn.clicked.connect(self.save_photo)

    def save_photo(self):
        # 현재 위치에서 Save_Image 폴더를 생성합니다.
        save_dir = os.path.join("RobotArm/GUI/Image")
        os.makedirs(save_dir, exist_ok=True)

        # 저장할 파일 이름을 결정합니다.
        file_number = 1
        while True:
            file_name = f"photo{file_number}.jpg"
            file_path = os.path.join(save_dir, file_name)
            if not os.path.exists(file_path):
                break
            file_number += 1

        # 사진을 저장합니다.
        self.video_thread.save_current_frame(file_path)

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = SignUpDialog()
    main_window.show()
    sys.exit(app.exec_())
