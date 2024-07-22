import os
import cv2
import tts
from PyQt5 import uic
from PyQt5.QtCore import Qt, QRectF, QTimer
from PyQt5.QtGui import QPixmap, QImage
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
                    <span style="font-size: 14pt; font-weight: bold;">{user_info['name']}</span>
                </div>
            """)
            self.Birth.setHtml(f"""
                <div style="text-align: center;">
                    <span style="font-size: 14pt; font-weight: bold;">{user_info['birthday'].strftime('%Y-%m-%d')}</span>
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
