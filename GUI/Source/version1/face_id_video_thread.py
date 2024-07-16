import cv2
from threading import Thread
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
import face_to_info as face

class FaceIDVideoThread:
    def __init__(self, graphics_view):
        self.face = face.FaceToInfo("/home/lsm/git_ws/RobotArm/GUI/Image")
        self.cam_thread = Thread(target=self.face.run_cam)
        self.cam_thread.start()
        self.deep_face_thread = Thread(target=self.face.cam_to_info)
        self.deep_face_thread.start()
        self.face.visualization = True
        self.graphics_view = graphics_view
        self.webcam = self.face.cap
        if not self.webcam.isOpened():
            print("Unable to open webcam.")
            exit()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(60)

    def update_frame(self):
        ret, frame = self.face.get_frame()
        if ret:
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
        self.graphics_view.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def stop(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.webcam.isOpened():
            self.webcam.release()
        self.face.cam_to_info_deamon = False
        self.face.cam_deamon = False

    def __del__(self):
        self.stop()
