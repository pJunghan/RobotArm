import sys
import os
import cv2
import pymysql
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt, QThread, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsPixmapItem, QMessageBox
from video_thread import VideoThread
from menu_window import MenuWindow
from new_account_window import NewAccountWindow
from config import login_ui_path, db_config,new_account_ui_path

class LoginWindow(QMainWindow):
    def __init__(self):
        super(LoginWindow, self).__init__()
        uic.loadUi(login_ui_path, self)
        self.video_thread = VideoThread(self.graphicsView)
        self.orderbtn.clicked.connect(self.handle_guest_login)
        self.memberBtn.clicked.connect(self.go_to_new_account_window)

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

    def handle_guest_login(self):
        # 데이터베이스에 새로운 비회원 사용자 추가
        guest_name = self.create_guest_user()
        if guest_name:
            self.go_to_menu_window()

    def create_guest_user(self):
        try:
            conn = pymysql.connect(**db_config)
            with conn.cursor() as cursor:
                # undefined로 시작하는 사용자 수 확인
                query = "SELECT COUNT(*) AS undefined_count FROM user_info_table WHERE name LIKE 'undefined%'"
                cursor.execute(query)
                result = cursor.fetchone()
                
                undefined_count = 0
                if result and 'undefined_count' in result:
                    undefined_count = result['undefined_count']

                # 새로운 undefined 사용자 이름 생성
                new_guest_name = f"undefined_{undefined_count + 1}"

                # 새로운 사용자 레코드 삽입
                insert_query = """
                INSERT INTO user_info_table (name, point)
                VALUES (%s, %s)
                """
                cursor.execute(insert_query, (new_guest_name, 0))
                conn.commit()
                return new_guest_name
        except pymysql.MySQLError as err:
            print(f"데이터베이스 오류 발생: {err}")
            QMessageBox.warning(self, "오류", f"데이터베이스 오류 발생: {err}")
            return None
        finally:
            if 'conn' in locals():
                conn.close()

    def go_to_menu_window(self):
        self.next_window = MenuWindow(db_config)
        self.next_window.show()
        self.close()

    def go_to_new_account_window(self):
        self.next_window = NewAccountWindow(new_account_ui_path, db_config)
        self.next_window.show()
        self.close()

