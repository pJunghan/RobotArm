import sys
import cv2
import sqlite3
import gtts
import os
import re
import threading
import mysql
import face_to_info as face
from save_image import SaveImage
from threading import Thread
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QEvent, QTimer
from datetime import datetime
import mysql.connector
from mysql.connector import errorcode


# UI 파일 연결
main_form_class = uic.loadUiType("GUI/main_v1.ui")[0]
login_form_class = uic.loadUiType("GUI/login.ui")[0]
menu_form_class = uic.loadUiType("GUI/order_ice_cream2.ui")[0]
signup_form_class = uic.loadUiType("GUI/signup.ui")[0]
signPhoto_form_class = uic.loadUiType("GUI/signPhoto.ui")[0]


class FaceIDVideoThread:
    def __init__(self, graphics_view):
        self.connect_sql()
        self.face = face.FaceToInfo("GUI/Image")
        cam_thread = Thread(target=self.face.run_cam)
        cam_thread.start()
        deep_face_thread = Thread(target=self.face.cam_to_info)
        deep_face_thread.start()
        self.face.visualization = True  # 이미지에 데이터 시각화 할 것인지
        self.graphics_view = graphics_view
        self.webcam = self.face.cap
        if not self.webcam.isOpened():
            print("웹캠을 열 수 없습니다.")
            exit()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(60)  # 60 ms마다 프레임을 갱신합니다.

    def connect_sql(self):
        config = {
            'user': 'rds',
            'password': '8470',
            'host': '127.0.0.1',
            'database': 'ARIS',  # 사용할 데이터베이스 이름
            'raise_on_warnings': False
        }
        self.cnx = mysql.connector.connect(**config)
        self.cursor = self.cnx.cursor()

    def close_sql(self):
        self.cursor.close()
        self.cnx.close()

    def update_frame(self):
        ret, frame = self.face.get_frame()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.display_image(pixmap)

    def display_image(self, pixmap):
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(pixmap))
        self.graphics_view.setScene(scene)
        self.graphics_view.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def stop(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.webcam.isOpened():
            self.webcam.release()
        self.face.cam_to_info_deamon = False
        self.face.cam_deamon = False

    def __del__(self):
        self.stop()
        self.close_sql()

# 메인 창 클래스
class MainWindow(QMainWindow, main_form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.orderButton.clicked.connect(self.go_to_next_window)
        self.signupBtn.clicked.connect(self.open_signup_dialog)

    def go_to_next_window(self):
        self.next_window = LoginWindow()
        self.next_window.show()
        self.close()

    def open_signup_dialog(self):
        self.signup_dialog = SignUpDialog()
        self.signup_dialog.exec_()

class LoginWindow(QMainWindow, login_form_class):
    def __init__(self):
        super(LoginWindow, self).__init__()
        self.setupUi(self)
        self.video_thread = FaceIDVideoThread(self.graphicsView)
        self.orderbtn.clicked.connect(self.go_to_menu_window)
        self.memberBtn.clicked.connect(self.go_to_menu_window)

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

    def go_to_menu_window(self):
        self.next_window = MenuWindow()
        self.next_window.show()
        self.close()

    def tts(self, name):
        text = name + f"반갑습니다 {name}님"
        ko_tts = gtts.gTTS(lang="ko", text=text)
        gtts.gTTS.save(ko_tts, "say.mp3")

# 메뉴 창 클래스
class MenuWindow(QMainWindow, menu_form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # graphicsView들에 마우스 이벤트 필터 등록
        self.Vanila.installEventFilter(self)
        self.Choco.installEventFilter(self)
        self.Strawberry.installEventFilter(self)
        self.topping1.installEventFilter(self)
        self.topping_2.installEventFilter(self)
        self.topping_3.installEventFilter(self)
        
        # 홈 버튼과 다음 버튼에 클릭 이벤트 연결
        self.Home_Button.clicked.connect(self.go_to_main_window)
        self.Next_Button.clicked.connect(self.go_to_next_tab)
        
    # 이벤트 필터 메서드
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Enter:
            obj.setStyleSheet("border: 2px solid blue;")  # 마우스가 들어왔을 때 테두리 색상 변경
        elif event.type() == QEvent.Leave:
            obj.setStyleSheet("")  # 마우스가 나갔을 때 테두리 색상 초기화
        return super().eventFilter(obj, event)
    
    # 홈 버튼 클릭 시 메인 창으로 이동하는 메서드
    def go_to_main_window(self):
        self.main_window = MainWindow()
        self.main_window.show()
        self.close()
    
    # 다음 버튼 클릭 시 다음 탭으로 이동하는 메서드
    def go_to_next_tab(self):
        current_index = self.tabWidget.currentIndex()
        next_index = (current_index + 1) % self.tabWidget.count()
        self.tabWidget.setCurrentIndex(next_index)

class SignVideoThread:
    def __init__(self, graphics_view):
        self.graphics_view = graphics_view
        self.webcam = cv2.VideoCapture(0)
        if not self.webcam.isOpened():
            print("웹캠을 열 수 없습니다.")
            exit()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(60)  # 60 ms마다 프레임을 갱신합니다.
        self.current_frame = None

    def update_frame(self):
        status, frame = self.webcam.read()
        if status:
            self.current_frame = frame.copy()
            # OpenCV의 BGR 이미지를 Qt의 RGB 이미지로 변환합니다.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # graphicsView의 크기에 맞게 프레임을 리사이즈합니다.
            gv_width = self.graphics_view.width()
            gv_height = self.graphics_view.height()
            frame = cv2.resize(frame, (gv_width, gv_height))

            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.display_image(pixmap)

    def display_image(self, pixmap):
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(pixmap))
        self.graphics_view.setScene(scene)
        self.graphics_view.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def stop(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.webcam.isOpened():
            self.webcam.release()

    def save_current_frame(self, file_path):
        if self.current_frame is not None:
            cv2.imwrite(file_path, self.current_frame)
            print(f"사진이 {file_path}에 저장되었습니다.")
            return self.current_frame
        else:
            print("저장할 프레임이 없습니다.")
            return None

class SignUpDialog(QDialog, signup_form_class):
    def __init__(self):
        super(SignUpDialog, self).__init__()
        self.setupUi(self)
        self.signupBtn.clicked.connect(self.save_to_db)
        self.checkMale.stateChanged.connect(self.on_check_male)
        self.checkFemale.stateChanged.connect(self.on_check_female)

        self.connect_sql()
        
    def connect_sql(self):
        config = {
            'user': 'rds',
            'password': '8470',
            'host': '127.0.0.1',
            'database': 'ARIS',  # 사용할 데이터베이스 이름
            'raise_on_warnings': False
        }
        self.cnx = mysql.connector.connect(**config)
        self._cursor = self.cnx.cursor()

    def close_sql(self):
        self._cursor.close()
        self.cnx.close()


    def on_check_male(self, state):
        if state == Qt.Checked:
            self.checkFemale.setChecked(False)

    def on_check_female(self, state):
        if state == Qt.Checked:
            self.checkMale.setChecked(False)

    def save_to_db(self):
        name = self.Name.text()
        phone_number = self.PhoneNumber.text()
        birth = self.Birth.text()
        gender = "Male" if self.checkMale.isChecked() else "Female" if self.checkFemale.isChecked() else ""
        last_modified = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        phone_pattern = re.compile(r'^\d{10,11}$')
        birth_pattern = re.compile(r'^\d{6}$')
        
        
        if not name:
            self.show_warning("입력 오류", "이름을 입력해주세요.")
            return
        if not phone_pattern.match(phone_number):
            self.show_warning("입력 오류", "전화번호를 올바르게 입력해주세요.\n" + "(ex: 01012345678)")
            return
        if not birth_pattern.match(birth):
            self.show_warning("입력 오류", "생년월일을 올바르게 입력해주세요.\n" + "(ex: 901020)")
            return
        if not gender:
            self.show_warning("입력 오류", "성별을 선택해주세요.")
            return
        if int(birth[0:2]) <= int(datetime.now().strftime('%y')):
            birth = '20' + birth
        else:
            birth = '19' + birth
        birth_date = datetime.strptime(birth, '%Y%m%d')
        # birth_date = birth_date.strftime('%Y-%m-%d')

        self._cursor.execute(f"SELECT EXISTS (SELECT 1 FROM user_info_table WHERE phone_num = '{phone_number}') AS value_exists;")
        if self._cursor.fetchone()[0]:
            self.show_warning("입력 오류", "이미 가입된 사용자입니다.")
            return
        
        user_id = self._cursor.lastrowid
        print(user_id)
        add_user = ("INSERT INTO user_info_table "
                    "(point, gender, name, phone_num, birthday, photo_path, last_modified) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)")
        data_user = (0, gender, name, phone_number, birth_date, user_id, last_modified)
        self._cursor.execute(add_user, data_user)
        self.cnx.commit()


        QMessageBox.information(self, "성공", "회원가입 성공")
        self.open_photo_dialog(user_id)
        

    def open_photo_dialog(self, user_id):
        self.photo_dialog = SignUpPhotoDialog(user_id)
        self.photo_dialog.show()
        self.close()
        

    def show_warning(self, title, message):
        msg_box = QMessageBox(QMessageBox.Warning, title, message, QMessageBox.Ok, self)
        msg_box.exec_()
        msg_box.move(self.frame.geometry().center().x() - msg_box.width() // 2, self.frame.geometry().center().y() - msg_box.height() // 2)

    def closeEvent(self, event):
        self.close_sql()
        event.accept()

class SignUpPhotoDialog(QDialog, signPhoto_form_class):
    def __init__(self, user_id):
        super(SignUpPhotoDialog, self).__init__()
        self.setupUi(self)
        self.user_id = user_id
        self.video_thread = SignVideoThread(self.graphicsView)
        self.takePhotoBtn.clicked.connect(self.save_photo)

    def save_photo(self):
        save_dir = os.path.join("GUI/Image")
        os.makedirs(save_dir, exist_ok=True)

        file_name = f"{self.user_id}.jpg"
        file_path = os.path.join(save_dir, file_name)

        frame = self.video_thread.save_current_frame(file_path)

        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            image_data = buffer.tobytes()

            conn = sqlite3.connect('GUI/DB/user_data.db')
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET image = ? WHERE id = ?", (image_data, self.user_id))
            conn.commit()
            conn.close()

            QMessageBox.information(self, "성공", "사진 촬영 성공")
        
        self.close()

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())