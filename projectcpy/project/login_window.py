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
    def __init__(self, main):
        super(LoginWindow, self).__init__()
        uic.loadUi(login_ui_path, self)  # UI 파일 로드
        self.db_config = db_config  # db_config 속성 추가
        self.main = main
        self.face = FaceToInfo(user_img_path)  # 얼굴 인식 객체 초기화
        self.cam_thread = Thread(target=self.face.run_cam)  # 카메라 스레드 시작
        self.cam_thread.start()
        self.deep_face_thread = Thread(target=self.face.cam_to_info)  # 얼굴 인식 처리 스레드 시작
        self.deep_face_thread.start()
        self.face.visualization = True

        self.timer = QTimer(self)  # 타이머 설정
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // 30)  # 매 33 밀리초마다 프레임 업데이트 (30 fps)

        self.orderbtn.clicked.connect(self.handle_guest_login)  # 비회원 로그인 버튼 클릭 시 처리
        self.memberBtn.clicked.connect(self.go_to_new_account_window)  # 회원 가입 버튼 클릭 시 처리

        self.scene = QGraphicsScene(self)  # 그래픽 씬 설정
        self.graphicsView.setScene(self.scene)

    def update_frame(self):
        # 카메라에서 프레임을 가져와서 그래픽 뷰에 업데이트
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
                self.check_user()

        else:
            print("Error: Failed to get frame from camera or frame is None.")
        

    def check_user(self):
        # 얼굴 인식된 사용자가 있으면 로그인 처리
        if self.face.known_person:
            self.stop_camera()  # 카메라 동작 중지
            user_id = self.face.known_person
            user_info = self.get_user_info(user_id)  # 사용자 정보 가져오기
            user_image_path = os.path.join(user_img_path, f"{user_id}.jpeg")

            
            if os.path.exists(user_image_path):
                check_window = CheckLoginWindow(user_image_path, user_info, self)
                if check_window.exec_() == QDialog.Rejected:
                    self.start_camera()  # 로그인 실패 시 카메라 재시작
            else:
                print(f"Error: Image file {user_image_path} does not exist.")

    def get_user_info(self, user_id):
        # 데이터베이스에서 사용자 정보 가져오기
        try:
            conn = pymysql.connect(**db_config)
            with conn.cursor() as cursor:
                query = ("SELECT name, birthday, photo_path FROM user_info_table WHERE user_ID = %s")
                cursor.execute(query, (user_id,))
                result = cursor.fetchone()

                if result:
                    update_query = ("UPDATE user_info_table SET last_modified = CURRENT_TIMESTAMP WHERE user_ID = %s")
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
        # 카메라와 관련된 스레드 및 타이머 중지
        self.timer.stop()
        self.face.cam_to_info_deamon = False
        self.face.cam_deamon = False

    def start_camera(self):
        # 카메라와 관련된 스레드 및 타이머 시작
        self.face.cam_to_info_deamon = True
        self.face.cam_deamon = True
        self.cam_thread = Thread(target=self.face.run_cam)
        self.cam_thread.start()
        self.deep_face_thread = Thread(target=self.face.cam_to_info)
        self.deep_face_thread.start()
        self.timer.start(1000 // 30)

    def handle_guest_login(self):
        # 비회원 로그인 처리
        self.stop_camera()  # 카메라 동작 중지
        guest_name = self.create_guest_user()  # 새로운 비회원 사용자 생성
        if guest_name:
            self.close()
            self.go_to_menu_window()

    def create_guest_user(self):
        # 데이터베이스에 새로운 비회원 사용자 추가
        try:
            conn = pymysql.connect(**db_config)
            with conn.cursor() as cursor:
                cursor.execute("SELECT user_ID FROM user_info_table ORDER BY user_ID")
                existing_ids = {row['user_ID'] for row in cursor.fetchall()}

                new_user_id = 1
                while new_user_id in existing_ids:
                    new_user_id += 1

                new_guest_name = f"guest_{new_user_id}"

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
        # 메뉴 창으로 이동
        self.stop_camera()
        self.next_window = MenuWindow(db_config, self.main)
        self.next_window.show()

    def go_to_new_account_window(self):
        # 새로운 계정 생성 창으로 이동
        self.stop_camera()
        self.close()
        self.next_window = NewAccountWindow(new_account_ui_path, db_config, self.main)
        self.next_window.show()


