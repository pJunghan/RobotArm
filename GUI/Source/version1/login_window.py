from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow
from face_id_video_thread import FaceIDVideoThread

login_form_class = uic.loadUiType("/home/lsm/git_ws/RobotArm/GUI/UI/login.ui")[0]

class LoginWindow(QMainWindow, login_form_class):
    def __init__(self):
        super(LoginWindow, self).__init__()
        self.setupUi(self)
        self.video_thread = FaceIDVideoThread(self.graphicsView)
        self.orderbtn.clicked.connect(self.go_to_menu_window)
        self.memberBtn.clicked.connect(self.go_to_menu_window)

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

    def go_to_menu_window(self):
        from menu_window import MenuWindow  # 지연 임포트
        self.next_window = MenuWindow()
        self.next_window.show()
        self.close()
