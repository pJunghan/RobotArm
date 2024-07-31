import sys
import os
import pymysql
import socket
import json
import time
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt, QSize
from threading import Thread
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QTextBrowser
from login_window import LoginWindow
from menu_window import MenuWindow
from PyQt5.QtGui import QMovie, QIcon
from new_account_window import NewAccountWindow
from config import main_ui_path, db_config

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(main_ui_path, self)

        self.data = {"topping1" : 0, "topping2" : 0, "topping3" : 0, "gender" : "", "age" : 0}
        self.none_data = {"topping1" : 0, "topping2" : 0, "topping3" : 0, "gender" : "", "age" : 0}
        self.movie = QMovie("ui/pic/aris_main.gif")
        self.label.setMovie(self.movie)
        self.movie.setScaledSize(self.label.size())
        self.movie.start()

        self.setFixedSize(self.size())  # 현재 창 크기로 고정

        # 화면 크기를 가져와 창의 중앙 위치를 계산
        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

        # self.orderButton.setGeometry(350, 500, 200, 50)

        self.orderButton.setStyleSheet("""
            QPushButton {
                background-color: rgb(251, 191, 196);
                border: 2px solid rgb(251, 191, 196);
                border-radius: 20px;
                color: white;
                font-size: 28px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: rgb(255, 200, 200);
            }
            QPushButton:pressed {
                background-color: rgb(255, 150, 150);
            }
        """)
        self.autoButton.setStyleSheet("""
            QPushButton {
                background-color: transparent; 
                border: 0px solid black; 
                color: black; 
                font-size: 16px; 
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 30); 
            }
        """)

        self.autoButton.setIcon(QIcon("ui/pic/aris.png"))
        self.autoButton.setIconSize(QSize(80,80))
        self.orderButton.clicked.connect(self.go_to_login_window)
        self.autoButton.clicked.connect(self.go_to_auto_order_window)
        self.update_purchase_count(db_config)


 

    def go_to_login_window(self):
        if not hasattr(self, 'login_window'):
            self.login_window = LoginWindow(self)
            self.login_window.show()
            self.close()  # 메인 윈도우를 숨깁니다.

    def go_to_auto_order_window(self):
        from auto_order import OrderManager
        self.win = OrderManager(self)
        self.win.show()
        self.close()
        
    def update_purchase_count(self, db_config):
        try:
            # 데이터베이스 연결
            conn = pymysql.connect(**db_config)
            cursor = conn.cursor()


            # 실행할 SQL 쿼리문
            sql_query = """
                UPDATE purchase_record_table
                SET choco_count = 0, vanila_count = 0, strawberry_count = 0,
                    topping1_count = 0, topping2_count = 0, topping3_count = 0
                """
            # 쿼리 실행
            cursor.execute(sql_query)
            conn.commit()

            # 연결 종료
            cursor.close()
            conn.close()

        except pymysql.MySQLError as e:
            print(f"Error updating purchase counts: {e}")

    def home(self):
        self.update_purchase_count(db_config)
        self.show()
        # Check if login_window exists before attempting to delete
        if hasattr(self, 'login_window'):
            del self.login_window

    def socket_run(self):
        # self.HOST = '192.168.1.167'
        self.HOST = '127.0.0.1'
        self.PORT = 10002
        self.BUFSIZE = 1024
        self.ADDR = (self.HOST, self.PORT)
        
        
        clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                clientSocket.connect(self.ADDR)
                break 

            except:
                continue

        while True:
            if self.data == self.none_data:
                continue
            else:
                print(f"send data{str(self.data)}")
                msg = json.dumps(self.data)
                clientSocket.send(msg.encode())
                self.data = self.none_data.copy()


    def set_data(self, gender = "", age = 20):
        self.data["gender"] = gender
        self.data["age"] = age

    def closeEvent(self, event):
        pass # 



if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    socket_thread = Thread(target=main_window.socket_run)
    socket_thread.start()
    main_window.show()
    sys.exit(app.exec_())