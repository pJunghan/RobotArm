import os
import cv2
import pymysql
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from login_window import LoginWindow
from menu_window import MenuWindow
from new_account_window import NewAccountWindow
from config import manager_ui_path, db_config

class ManagerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(manager_ui_path, self)
        
        self.mainBtn.clicked.connect(self.go_to_main_window)
        self.orderBtn.clicked.connect(self.show_order_message)  # Connect orderBtn to show_order_message

    def go_to_main_window(self):
        from main_window import MainWindow
        self.main_window = MainWindow()
        self.main_window.show()
        self.close()

    def show_order_message(self):
        QMessageBox.information(self, "알림", "발주 되었습니다.")
        self.go_to_main_window()
    
    