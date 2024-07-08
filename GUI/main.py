import sys
import cv2
import threading
import face_to_info as face
from threading import Thread
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QEvent, QTimer

# UI 파일 연결
main_form_class = uic.loadUiType("GUI/main2.ui")[0]
login_form_class = uic.loadUiType("GUI/login.ui")[0]
menu_form_class = uic.loadUiType("GUI/order_ice_cream2.ui")[0]

class VideoThread:
    def __init__(self, graphics_view):
        self.face = face.FaceToInfo("test/img_db/")
        cam_thread = Thread(target=self.face.run_cam)
        cam_thread.start()
        deep_face_thread = Thread(target=self.face.cam_to_info)
        deep_face_thread.start()
        self.graphics_view = graphics_view
        self.webcam = self.face.cap
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
        self.face.cam_to_info_deamon = False
        self.face.cam_deamon = False

    def __del__(self):
        self.stop()


# 메인 창 클래스
class MainWindow(QMainWindow, main_form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.orderButton.clicked.connect(self.go_to_next_window)

    def go_to_next_window(self):
        self.next_window = LoginWindow()
        self.next_window.show()
        self.close()


class LoginWindow(QMainWindow, login_form_class):
    def __init__(self):
        super(LoginWindow, self).__init__()
        self.setupUi(self)
        self.video_thread = VideoThread(self.graphicsView)
        self.orderbtn.clicked.connect(self.go_to_menu_window)
        self.memberBtn.clicked.connect(self.go_to_menu_window)


    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()
    
    def go_to_menu_window(self):
        self.next_window = MenuWindow()
        self.next_window.show()
        self.close()


# 메뉴 창 클래스
class MenuWindow(QMainWindow, menu_form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # graphicsView들에 마우스 이벤트 필터 등록
        self.Vanila.installEventFilter(self)
        self.Choco.installEventFilter(self)
        self.Strawberry.installEventFilter(self)
        self.topping1.installEventFilter(self)
        self.topping_2.installEventFilter(self)
        self.topping_3.installEventFilter(self)
        
        # 홈 버튼과 다음 버튼에 클릭 이벤트 연결
        self.Home_Button.clicked.connect(self.go_to_main_window)
        self.Next_Button.clicked.connect(self.go_to_next_tab)
        
    # 이벤트 필터 메서드
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Enter:
            obj.setStyleSheet("border: 2px solid blue;")  # 마우스가 들어왔을 때 테두리 색상 변경
        elif event.type() == QEvent.Leave:
            obj.setStyleSheet("")  # 마우스가 나갔을 때 테두리 색상 초기화
        return super().eventFilter(obj, event)
    
    # 홈 버튼 클릭 시 메인 창으로 이동하는 메서드
    def go_to_main_window(self):
        self.main_window = MainWindow()
        self.main_window.show()
        self.close()
    
    # 다음 버튼 클릭 시 다음 탭으로 이동하는 메서드
    def go_to_next_tab(self):
        current_index = self.tabWidget.currentIndex()
        next_index = (current_index + 1) % self.tabWidget.count()
        self.tabWidget.setCurrentIndex(next_index)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
