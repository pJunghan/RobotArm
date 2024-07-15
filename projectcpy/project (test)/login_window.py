import sys
import os
import cv2
import pymysql
import time
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QGraphicsScene, QDialog
from face_to_info import FaceToInfo
from menu_window import MenuWindow
from threading import Thread
from new_account_window import NewAccountWindow
from check_login_window import CheckLoginWindow
from check_account_window import CheckAccountWindow
from config import login_ui_path, db_config, new_account_ui_path, user_img_path, check_ui_path

class LoginWindow(QMainWindow):
    def __init__(self, main=None):
        super(LoginWindow, self).__init__()
        uic.loadUi(login_ui_path, self)
        self.main = main
        self.db_config = db_config  # db_config을 속성으로 추가

        self.face = FaceToInfo(user_img_path)
        self.cam_thread = Thread(target=self.face.run_cam)
        self.cam_thread.start()
        self.deep_face_thread = Thread(target=self.face.cam_to_info)
        self.deep_face_thread.start()
        self.face.visualization = True

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // 30)  # 매 33 밀리초마다 프레임 업데이트 (30 fps)

        self.orderbtn.clicked.connect(self.handle_guest_login)
        self.memberBtn.clicked.connect(self.go_to_new_account_window)

        self.scene = QGraphicsScene(self)
        self.graphicsView.setScene(self.scene)

    def update_frame(self):
        ret, frame = self.face.get_frame()
        if ret and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(convertToQtFormat)
            
            if not pixmap.isNull():
                self.scene.clear()
                pixmap = pixmap.scaled(self.graphicsView.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                pixmap_item = self.scene.addPixmap(pixmap)
                self.graphicsView.setSceneRect(QRectF(pixmap.rect()))
            else:
                print("Error: Pixmap is null after conversion.")
        else:
            print("Error: Failed to get frame from camera or frame is None.")
        
        self.check_user()

    def check_user(self):
        if self.face.known_person and not self.isHidden():
            self.stop_camera()  # 카메라 동작 중지
            user_id = self.face.known_person
            user_info = self.get_user_info(user_id)
            user_image_path = os.path.join(user_img_path, f"{user_id}.jpeg")
            if 'photo_path' in user_info and user_info['photo_path']:
                user_image_path = os.path.join(user_img_path, user_info['photo_path'])
            
            if os.path.exists(user_image_path):
                check_window = CheckLoginWindow(user_image_path, user_info, self)
                if check_window.exec_() == QDialog.Rejected:
                    self.start_camera()  # 카메라 재시작
            else:
                print(f"Error: Image file {user_image_path} does not exist.")

    def get_user_info(self, user_id):
        try:
            conn = pymysql.connect(**self.db_config)  # self.db_config을 사용
            with conn.cursor() as cursor:
                query = ("SELECT name, birthday, photo_path FROM user_info_table "
                         "WHERE user_ID = %s")
                cursor.execute(query, (user_id,))
                result = cursor.fetchone()

                if result:
                    update_query = ("UPDATE user_info_table "
                                    "SET last_modified = CURRENT_TIMESTAMP "
                                    "WHERE user_ID = %s")
                    cursor.execute(update_query, (user_id,))
                    conn.commit()
                    return {'name': result['name'], 'birthday': result['birthday'], 'photo_path': result['photo_path']}
        except pymysql.MySQLError as err:
            print(f"데이터베이스 오류 발생: {err}")
            QMessageBox.warning(self, "오류", f"데이터베이스 오류 발생: {err}")
            return None
        finally:
            if 'conn' in locals():
                conn.close()
                print("데이터베이스 연결을 닫았습니다.")
    
    def stop_camera(self):
        self.timer.stop()
        self.face.cam_to_info_deamon = False
        self.face.cam_deamon = False

    def start_camera(self):
        self.face.cam_to_info_deamon = True
        self.face.cam_deamon = True
        self.cam_thread = Thread(target=self.face.run_cam)
        self.cam_thread.start()
        self.deep_face_thread = Thread(target=self.face.cam_to_info)
        self.deep_face_thread.start()
        self.timer.start(1000 // 30)  # 타이머 재시작

    def handle_guest_login(self):
        self.stop_camera()  # 카메라 동작 중지
        guest_name = self.create_guest_user()
        if guest_name:
            self.go_to_menu_window()

    def create_guest_user(self):
        try:
            conn = pymysql.connect(**self.db_config)  # self.db_config을 사용
            with conn.cursor() as cursor:
                cursor.execute("SELECT user_ID FROM user_info_table ORDER BY user_ID")
                existing_ids = {row['user_ID'] for row in cursor.fetchall()}

                new_user_id = 1
                while new_user_id in existing_ids:
                    new_user_id += 1

                new_guest_name = f"undefined_{new_user_id}"

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
        self.stop_camera()
        self.next_window = MenuWindow(self.db_config, self.main)  # self.db_config을 전달
        self.next_window.show()

    def go_to_new_account_window(self):
        self.stop_camera()
        self.next_window = NewAccountWindow(new_account_ui_path, self.db_config)  # self.db_config을 전달
        self.next_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoginWindow()
    window.show()
    sys.exit(app.exec_())
