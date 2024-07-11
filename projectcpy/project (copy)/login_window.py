import sys
import cv2
import pymysql
import time
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QGraphicsScene
from face_to_info import FaceToInfo
from menu_window import MenuWindow
from threading import Thread
from new_account_window import NewAccountWindow
from config import login_ui_path, db_config, new_account_ui_path, user_img_path

class LoginWindow(QMainWindow):
    def __init__(self):
        super(LoginWindow, self).__init__()
        uic.loadUi(login_ui_path, self)

        self.face = FaceToInfo(user_img_path)
        cam_thread = Thread(target=self.face.run_cam)
        cam_thread.start()
        deep_face_thread = Thread(target=self.face.cam_to_info)
        deep_face_thread.start()
        self.face.visualization = True

        # self.camera = cv2.VideoCapture(0)  # Open the default camera (0)
        # if not self.camera.isOpened():
        #     QMessageBox.critical(self, "카메라 연결 오류", "카메라를 열 수 없습니다.")
        #     return

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // 30)  # Update frame every 33 milliseconds (30 fps)

        self.orderbtn.clicked.connect(self.handle_guest_login)
        self.memberBtn.clicked.connect(self.go_to_new_account_window)

        self.scene = QGraphicsScene(self)
        self.graphicsView.setScene(self.scene)

    def update_frame(self):
        ret, frame = self.face.get_frame()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(convertToQtFormat)
            
            # Clear the previous items in the scene
            self.scene.clear()
            
            # Add the new pixmap item to the scene
            pixmap_item = self.scene.addPixmap(pixmap)
            
            # Scale the pixmap item to fit the graphics view
            pixmap_item.setScale(1)
            
            # Ensure the pixmap item is centered in the graphics view
            pixmap_item.setPos((self.graphicsView.width() - pixmap.width()) / 2,
                               (self.graphicsView.height() - pixmap.height()) / 2)
        self.check_user()
        
    def check_user(self):
        if self.face.known_person and not self.isHidden():
            self.go_to_menu_window()


    def close_event(self):
        self.timer.stop()
        self.face.cam_to_info_deamon = False
        self.face.cam_deamon = False
        del(self.face)
        self.close()
        

    def handle_guest_login(self):
        # 데이터베이스에 새로운 비회원 사용자 추가
        guest_name = self.create_guest_user()
        if guest_name:
            self.go_to_menu_window()

    def create_guest_user(self):
        try:
            conn = pymysql.connect(**db_config)
            with conn.cursor() as cursor:
                # Find the smallest unused user_ID
                cursor.execute("SELECT user_ID FROM user_info_table ORDER BY user_ID")
                existing_ids = {row['user_ID'] for row in cursor.fetchall()}

                new_user_id = 1
                while new_user_id in existing_ids:
                    new_user_id += 1

                # Generate new undefined guest name
                new_guest_name = f"undefined_{new_user_id}"

                # Insert new user record
                insert_query = """
                INSERT INTO user_info_table (user_ID, name, point)
                VALUES (%s, %s, %s)
                """
                cursor.execute(insert_query, (new_user_id, new_guest_name, 0))
                conn.commit()

                return new_guest_name

        except pymysql.MySQLError as err:
            print(f"데이터베이스 오류 발생: {err}")
            QMessageBox.warning(self, "오류", f"데이터베이스 오류 발생: {err}")
            return None

        finally:
            if 'conn' in locals():
                conn.close()
                print("데이터베이스 연결을 닫았습니다.")


    def go_to_menu_window(self):
        self.close_event()
        # self.hide()  # 메인 윈도우를 숨깁니다.
        self.next_window = MenuWindow(db_config)
        self.next_window.show()



    def go_to_new_account_window(self):
        self.close_event()
        # self.hide()  # 메인 윈도우를 숨깁니다.
        self.next_window = NewAccountWindow(new_account_ui_path, db_config)
        self.next_window.show()

