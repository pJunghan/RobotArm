import sys
import os
import cv2
import sqlite3
from datetime import datetime
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QTimer, Qt
import re  # 정규 표현식 사용을 위해 추가

# UI 파일을 로드합니다.
signup_form_class = uic.loadUiType("RobotArm/GUI/signup.ui")[0]
signPhoto_form_class = uic.loadUiType("RobotArm/GUI/signPhoto.ui")[0]


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
        self.timer.stop()
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

        # QCheckBox의 상태 변경 시그널을 연결합니다.
        self.checkMale.stateChanged.connect(self.on_check_male)
        self.checkFemale.stateChanged.connect(self.on_check_female)

        # 데이터베이스 초기화
        self.init_db()

    def init_db(self):
        # 데이터베이스 연결 및 테이블 생성
        self.conn = sqlite3.connect('RobotArm/GUI/DB/user_data.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS users
                               (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                name TEXT NOT NULL,
                                phone_number INTEGER NOT NULL,
                                birth INTEGER NOT NULL,
                                gender TEXT NOT NULL,
                                created_at TEXT NOT NULL)''')
        self.cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in self.cursor.fetchall()]
        if 'image' not in columns:
            self.cursor.execute("ALTER TABLE users ADD COLUMN image BLOB")
        self.conn.commit()

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
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 입력 데이터 검증
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

        # 중복 사용자 검증
        self.cursor.execute("SELECT * FROM users WHERE phone_number = ?", (phone_number,))
        if self.cursor.fetchone():
            self.show_warning("입력 오류", "이미 가입된 사용자입니다.")
            return

        # 데이터베이스에 데이터 저장
        self.cursor.execute("INSERT INTO users (name, phone_number, birth, gender, created_at) VALUES (?, ?, ?, ?, ?)",
                            (name, phone_number, int(birth), gender, created_at))
        self.conn.commit()

        # 생성된 사용자의 PRIMARY KEY 가져오기
        user_id = self.cursor.lastrowid

        QMessageBox.information(self, "성공", "회원가입 성공")

        # SignUpPhotoDialog로 창 전환
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
        self.conn.close()  # 데이터베이스 연결 종료
        event.accept()

class SignUpPhotoDialog(QDialog, signPhoto_form_class):
    def __init__(self, user_id):
        super(SignUpPhotoDialog, self).__init__()
        self.setupUi(self)
        self.user_id = user_id
        self.video_thread = SignVideoThread(self.graphicsView)
        self.takePhotoBtn.clicked.connect(self.save_photo)

    def save_photo(self):
        # 현재 위치에서 Save_Image 폴더를 생성합니다.
        save_dir = os.path.join("RobotArm/GUI/Image")
        os.makedirs(save_dir, exist_ok=True)

        # PRIMARY KEY 번호를 사용하여 파일 이름을 설정합니다.
        file_name = f"ID{self.user_id}.jpg"
        file_path = os.path.join(save_dir, file_name)

        # 사진을 저장합니다.
        frame = self.video_thread.save_current_frame(file_path)

        # 이미지 데이터를 데이터베이스에 저장합니다.
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            image_data = buffer.tobytes()

            # 데이터베이스에 이미지 데이터 저장
            conn = sqlite3.connect('RobotArm/GUI/DB/user_data.db')
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET image = ? WHERE id = ?", (image_data, self.user_id))
            conn.commit()
            conn.close()

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = SignUpDialog()
    main_window.show()
    sys.exit(app.exec_())
