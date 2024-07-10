import cv2
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

class VideoThread(QThread):
    frame_updated = pyqtSignal(QPixmap)

    def __init__(self, graphics_view):
        super().__init__()
        self.graphics_view = graphics_view
        self.webcam = None  # 웹캠 객체 초기화
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_running = False

    def start_camera(self):
        self.webcam = cv2.VideoCapture(-1)
        if not self.webcam.isOpened():
            print("웹캠을 열 수 없습니다.")
            return False
        self.is_running = True
        self.timer.start(60)
        return True

    def stop(self):
        self.stop_camera()

    def stop_camera(self):
        self.is_running = False
        if self.webcam and self.webcam.isOpened():
            self.webcam.release()
        self.timer.stop()

    def update_frame(self):
        status, frame = self.webcam.read()
        if status:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.frame_updated.emit(pixmap)  # 프레임 업데이트 시그널 발생
        else:
            print("프레임을 읽을 수 없습니다.")

    def run(self):
        while self.is_running:
            status, frame = self.webcam.read()
            if status:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.frame_updated.emit(pixmap)  # 프레임 업데이트 시그널 발생
            else:
                print("프레임을 읽을 수 없습니다.")
                break
