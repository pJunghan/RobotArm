import os
import cv2
import tts
from PyQt5 import uic
from PyQt5.QtCore import Qt, QRectF, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import QDialog, QGraphicsScene
from config import check_ui_path, user_img_path
from check_account_window import CheckAccountWindow
from menu_window import MenuWindow
from PyQt5.QtWidgets import QApplication

class CheckLoginWindow(QDialog):
    def __init__(self, user_image_path, user_info, parent=None, main = None):
        super(CheckLoginWindow, self).__init__(parent)
        uic.loadUi(check_ui_path, self)

        self.main = main
        self.user_info = user_info
        self.parent = parent

        # 하나의 메서드로 두 가지 작업 수행
        self.yesBtn.clicked.connect(self.open_menu_window)
        self.noBtn.clicked.connect(self.open_check_account_window)

        self.scene = QGraphicsScene(self)
        self.user_photo.setScene(self.scene)

        self.display_user_info(user_image_path, user_info)
        
        self.setFixedSize(self.size())  # 현재 창 크기로 고정

        # 화면 크기를 가져와 창의 중앙 위치를 계산
        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)
        
        self.customize_ui()


    def customize_ui(self):
        # QFrame에 배경 이미지 설정
        ui_image_path = "ui/pic"
        image_path = os.path.join(ui_image_path, "login_background.png")
        if os.path.exists(image_path):
            self.frame.setStyleSheet(f"QFrame {{background-image: url('{image_path}'); background-repeat: no-repeat; background-position: center;}}")
        else:
            print(f"Error: Image file {image_path} does not exist.")

        # QPushButton 스타일 설정
        button_style = """
            QPushButton {
                background-color: #62A0EA;
                border: 2px solid #62A0EA;
                border-radius: 15px;
                color: white;
                font-size: 16pt;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #3B82F6;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """
        self.yesBtn.setStyleSheet(button_style)
        self.noBtn.setStyleSheet(button_style)

        # QTextBrowser 스타일 설정
        textBrowser_style = ("""
            QTextBrowser {
                border: 2px solid #62A0EA;
                border-radius: 10px;
                padding: 10px;
                font-weight: bold;
                color: rgb(0,255,0);
                background-color: white;
            }
        """)
        self.Name.setStyleSheet(textBrowser_style)
        self.Birth.setStyleSheet(textBrowser_style)


    def display_user_info(self, user_image_path, user_info):
        # 사용자 사진 표시
        frame = cv2.imread(user_image_path)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(convertToQtFormat)

            if not pixmap.isNull():
                self.scene.clear()
                pixmap = pixmap.scaled(self.user_photo.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.scene.addPixmap(pixmap)
                self.user_photo.setSceneRect(QRectF(pixmap.rect()))
            else:
                print("Error: Pixmap is null after conversion.")
        else:
            print(f"Error: Cannot read image from {user_image_path}")

        # 사용자 정보 표시 예외 처리 추가
        if user_info is not None and 'name' in user_info and 'birthday' in user_info:
            self.Name.setHtml(f"""
                <div style="text-align: center;">
                    <span style="font-size: 12pt; font-weight: bold;">{user_info['name']}</span>
                </div>
            """)
            self.Birth.setHtml(f"""
                <div style="text-align: center;">
                    <span style="font-size: 12pt; font-weight: bold;">{user_info['birthday'].strftime('%Y-%m-%d')}</span>
                </div>
            """)
        else:
            print("Error: user_info is None or missing required keys")

    def open_menu_window(self):
        self.menu_window = MenuWindow(self.parent.db_config, self.main)  # parent.db_config을 사용
        self.menu_window.show()
        self.accept()  # 현재 창을 닫고
        self.parent.close()  # LoginWindow를 닫음

    def open_check_account_window(self):
        self.check_account_window = CheckAccountWindow(self.parent, self.main)  # parent를 전달
        self.check_account_window.exec_()  # CheckAccountWindow를 모달 창으로 열림
        self.accept()  # 현재 창을 닫고

        
    def closeEvent(self, event):
        event.accept()
        gui_windows = QApplication.allWidgets()
        main_windows = [win for win in gui_windows if isinstance(win, (MenuWindow, CheckAccountWindow)) and win.isVisible()]
        if not main_windows:
            self.main.home()


if __name__ == "__main__":
    import sys
    from datetime import datetime

    app = QApplication(sys.argv)

    # 테스트용 사용자 정보 설정
    user_image_path = "path_to_test_image.jpg"  # 테스트 이미지 경로 설정
    user_info = {
        "name": "Test User",
        "birthday": datetime.strptime("1990-01-01", "%Y-%m-%d")
    }

    main_window = CheckLoginWindow(user_image_path, user_info)
    main_window.show()
    sys.exit(app.exec_())