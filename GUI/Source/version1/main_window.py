import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QDialog
from signup_dialog import SignUpDialog

main_form_class = uic.loadUiType("/home/lsm/git_ws/RobotArm/GUI/UI/main_v1.ui")[0]

class MainWindow(QMainWindow, main_form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.orderButton.clicked.connect(self.go_to_next_window)
        self.signupBtn.clicked.connect(self.open_signup_dialog)

    def go_to_next_window(self):
        from login_window import LoginWindow  # 지연 임포트
        self.next_window = LoginWindow()
        self.next_window.show()
        self.close()

    def open_signup_dialog(self):
        self.signup_dialog = SignUpDialog()
        self.signup_dialog.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
