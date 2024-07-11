import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem

class SignVideoThread:
    def __init__(self, graphics_view):
        self.graphics_view = graphics_view
        self.webcam = cv2.VideoCapture(0)
        if not self.webcam.isOpened():
            print("웹캠을 열 수 없습니다.")
            exit()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(60)
        self.current_frame = None

    def update_frame(self):
        status, frame = self.webcam.read()
        if status:
            self.current_frame = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        if self.timer.isActive():
            self.timer.stop()
        if self.webcam.isOpened():
            self.webcam.release()

    def save_current_frame(self, file_path):
        if self.current_frame is not None:
            cv2.imwrite(file_path, self.current_frame)
            print(f"사진이 {file_path}에 저장되었습니다.")
            return self.current_frame
        else:
            print("저장할 프레임이 없습니다.")
            return None
