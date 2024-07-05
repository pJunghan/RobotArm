import sys
import cv2
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

# ui 파일을 로드합니다.
login_form_class = uic.loadUiType("/home/lsm/Downloads/RobotArm/GUI/login.ui")[0]

class VideoThread:
    def __init__(self, graphics_view):
        self.graphics_view = graphics_view
        self.webcam = cv2.VideoCapture(0)
        if not self.webcam.isOpened():
            print("Could not open webcam")
            exit()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(60)  # 30 ms마다 프레임을 갱신합니다.

    def update_frame(self):
        status, frame = self.webcam.read()
        if status:
            # OpenCV의 BGR 이미지를 Qt의 RGB 이미지로 변환합니다.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.display_image(pixmap)

    def display_image(self, pixmap):
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(pixmap))
        self.graphics_view.setScene(scene)
        self.graphics_view.fitInView(scene.itemsBoundingRect(), 1)  # 1은 AspectRatioMode의 KeepAspectRatio 값입니다.

    def stop(self):
        self.timer.stop()
        self.webcam.release()

class LoginWindow(QMainWindow, login_form_class):
    def __init__(self):
        super(LoginWindow, self).__init__()
        self.setupUi(self)
        self.video_thread = VideoThread(self.graphicsView)

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = LoginWindow()
    main_window.show()
    sys.exit(app.exec_())
