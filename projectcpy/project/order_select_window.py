import sys
import os
import cv2
import pymysql
from login_window import LoginWindow
from seat_select_window import Seat_Selection_Window
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt, QThread, QTimer, QStringListModel, pyqtSlot, QSize, QRectF
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsPixmapItem, QMessageBox, QDialog, QTextBrowser, QStyledItemDelegate, QStyleOptionViewItem
from config import db_config, order_select_ui_path

class Order_Select_Window(QDialog):

    def __init__(self, main):
        super().__init__()  
        self.main = main
        # UI 파일 로드
        uic.loadUi(order_select_ui_path, self) 

        # UI 설정
        self.customize_ui()

        # 버튼 클릭시 다음 창으로 이동
        self.take_out_btn.clicked.connect(self.go_to_login_window)
        self.dine_btn.clicked.connect(self.go_to_seat_select_window)


    def customize_ui(self):
        # 현재 창 크기로 고정
        self.setFixedSize(self.size())  
        
        # 화면 크기를 가져와 창의 중앙 위치를 계산 및 화면 중앙 표시
        screen_geometry = QApplication.primaryScreen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)


        # QPushButton 스타일 설정
        button_style = """
            QPushButton {
                background-color: white;
                border: 2px solid gray;
                border-radius: 15px;
                color: black;
                font-size: 20pt;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #87CEFA;
            }
            QPushButton:pressed {
                background-color: #00BFFF;
            }
        """
        
        self.take_out_btn.setStyleSheet(button_style)
        self.dine_btn.setStyleSheet(button_style)


    def go_to_seat_select_window(self):
        if not hasattr(self, 'seat_select_window'):
            self.seat_select_window = Seat_Selection_Window(self.main)
            self.seat_select_window.show()
            self.close() 


    def go_to_login_window(self):
        if not hasattr(self, 'login_window'):
            self.login_window = LoginWindow(self.main)
            self.login_window.show()
            self.close()  

    
    def closeEvent(self, event):
        event.accept()
        gui_windows = QApplication.allWidgets()
        main_windows = [win for win in gui_windows if isinstance(win, (Seat_Selection_Window, LoginWindow)) and win.isVisible()]
        if not main_windows:
            self.main.home()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Main window 설정 및 보여주기
    main_window = Order_Select_Window(main= None)
    main_window.show()
    sys.exit(app.exec_())
