from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5.QtCore import Qt
import mysql.connector
import re
from datetime import datetime
from sign_photo_dialog import SignUpPhotoDialog

signup_form_class = uic.loadUiType("/home/lsm/git_ws/RobotArm/GUI/UI/signup.ui")[0]

class SignUpDialog(QDialog, signup_form_class):
    def __init__(self):
        super(SignUpDialog, self).__init__()
        self.setupUi(self)
        self.signupBtn.clicked.connect(self.save_to_db)
        self.checkMale.stateChanged.connect(self.on_check_male)
        self.checkFemale.stateChanged.connect(self.on_check_female)
        self.init_db()

    def init_db(self):
        self.conn = mysql.connector.connect(
            host='localhost',
            user='user',  # MySQL 사용자 이름으로 변경
            password='1111',  # MySQL 비밀번호로 변경
            database='Aris'
        )
        self.cursor = self.conn.cursor()
        table_name = 'User_Data'  # 원하는 테이블 이름으로 변경
        self.cursor.execute(f'''CREATE TABLE IF NOT EXISTS {table_name} (
                                ID INT AUTO_INCREMENT PRIMARY KEY,
                                Name VARCHAR(100) NOT NULL,
                                Phone_Number VARCHAR(11) NOT NULL,
                                Birth VARCHAR(6) NOT NULL,
                                Gender VARCHAR(6) NOT NULL,
                                Point INT DEFAULT 0,
                                Image_Path VARCHAR(255) DEFAULT NULL,
                                Created_at DATETIME NOT NULL
                                )''')
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
        phone_pattern = re.compile(r'^\d{10,11}$')
        birth_pattern = re.compile(r'^\d{6}$')

        if not name:
            self.show_warning("입력 오류", "이름을 입력해주세요.")
            return
        if not phone_pattern.match(phone_number):
            self.show_warning("입력 오류", "전화번호를 올바르게 입력해주세요.\n(ex: 01012345678)")
            return
        if not birth_pattern.match(birth):
            self.show_warning("입력 오류", "생년월일을 올바르게 입력해주세요.\n(ex: 901020)")
            return
        if not gender:
            self.show_warning("입력 오류", "성별을 선택해주세요.")
            return

        table_name = 'User_Data'  # 테이블 이름을 변수로 지정
        self.cursor.execute(f"SELECT * FROM {table_name} WHERE Phone_Number = %s", (phone_number,))
        if self.cursor.fetchone():
            self.show_warning("입력 오류", "이미 가입된 사용자입니다.")
            return

        self.cursor.execute(f"INSERT INTO {table_name} (Name, Phone_Number, Birth, Gender, Created_at) VALUES (%s, %s, %s, %s, %s)",
                            (name, phone_number, birth, gender, created_at))
        self.conn.commit()

        user_id = self.cursor.lastrowid
        QMessageBox.information(self, "성공", "회원가입 성공")
        self.open_photo_dialog(user_id)

    def open_photo_dialog(self, user_id):
        self.photo_dialog = SignUpPhotoDialog(user_id)
        self.photo_dialog.show()
        self.close()

    def show_warning(self, title, message):
        msg_box = QMessageBox(QMessageBox.Warning, title, message, QMessageBox.Ok, self)
        msg_box.exec_()

    def closeEvent(self, event):
        self.conn.close()
        event.accept()
