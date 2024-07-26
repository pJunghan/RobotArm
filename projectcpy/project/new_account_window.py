from PyQt5.QtWidgets import QDialog, QMessageBox, QFrame
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt
import pymysql
from kiosk_window import KioskWindow
import re
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys, os
from config import db_config, new_account_ui_path, user_img_path, check_ui_path

class NewAccountWindow(QDialog):  # Inherit from QDialog
    def __init__(self, ui_path, db_config, main):
        super().__init__()
        uic.loadUi(ui_path, self)
        self.main = main
        self.db_config = db_config
        self.gender = None
        self.signupBtn.clicked.connect(self.save_user_info)  # Correct button name
        self.checkMale.stateChanged.connect(self.on_check_gender)
        self.checkFemale.stateChanged.connect(self.on_check_gender)
        self.customize_ui()
        self.setFixedSize(self.size())  # 현재 창 크기로 고정
        
        # 화면 크기를 가져와 창의 중앙 위치를 계산
        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

    def customize_ui(self):

        # 배경 이미지 설정
        ui_image_path = "ui/pic"
        image_path = os.path.join(ui_image_path, "signup_background.png")
        if os.path.exists(image_path):
            self.setStyleSheet(f"QDialog {{background-image: url('{image_path}'); background-repeat: no-repeat; background-position: center;}}")
        else:
            print(f"Error: Image file {image_path} does not exist.")

        # QPushButton 스타일 설정
        self.signupBtn.setStyleSheet("""
            QPushButton {
                background-color: rgb(251, 191, 196);
                border: 2px solid rgb(251, 191, 196);
                border-radius: 20px;
                color: white;
                font-size: 24px;
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

        # QLineEdit 스타일 설정
        lineedit_style = """
            QLineEdit {
                font-size: 12pt;
                font-weight: bold;
                padding: 5px;
                border: 2px solid rgb(160, 207, 198);
                border-radius: 20px;
                background-color: white;
                color: rgb(0, 255, 0);
            }
            QLineEdit:focus {
                border: 2px solid rgb(251, 191, 196);
            }
        """
        self.Name.setStyleSheet(lineedit_style)
        self.PhoneNumber.setStyleSheet(lineedit_style)
        self.Birth.setStyleSheet(lineedit_style)

        # QLineEdit 중앙 정렬
        self.Name.setAlignment(Qt.AlignCenter)
        self.PhoneNumber.setAlignment(Qt.AlignCenter)
        self.Birth.setAlignment(Qt.AlignCenter)

        # QCheckBox 스타일 설정
        checkbox_style = """
            QCheckBox {
                font-size: 12pt;
                color: rgb(255, 255, 0);
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                }
            }
        """
        self.checkMale.setStyleSheet(checkbox_style)
        self.checkFemale.setStyleSheet(checkbox_style)



    


    def on_check_gender(self, state):
        # Update gender based on checkbox state
        if self.sender() == self.checkMale and state == Qt.Checked:
            self.gender = 'Male'
            self.checkFemale.setChecked(False)  # Uncheck Female
        elif self.sender() == self.checkFemale and state == Qt.Checked:
            self.gender = 'Female'
            self.checkMale.setChecked(False)  # Uncheck Male

    def show_warning(self, title, message):
        QMessageBox.warning(self, title, message)

    def save_user_info(self):
        name = self.Name.text().strip()
        phone = self.PhoneNumber.text().strip()
        birthday = self.Birth.text().strip()
        gender = "Male" if self.checkMale.isChecked() else "Female" if self.checkFemale.isChecked() else ""

        # # Check for empty fields
        # if not name or not phone or not birthday:
        #     QMessageBox.warning(self, "경고", "모든 정보를 입력해주세요.")
        #     return

        # # Check for selected gender
        # if not self.gender:
        #     QMessageBox.warning(self, "경고", "성별을 선택해주세요.")
        #     return
        

        # 형식에 맞춰 제대로 입력 안하면 오류 메세지 출력
        phone_pattern = re.compile(r'^\d{10,11}$')

        if not name:
            self.show_warning("입력 오류", "이름을 입력해주세요.")
            return
        if not phone_pattern.match(phone):
            self.show_warning("입력 오류", "전화번호를 올바르게 입력해주세요.\n(ex: 01012345678)")
            return
        if not gender:
            self.show_warning("입력 오류", "성별을 선택해주세요.")
            return
        
        # 생년월일 형식과 유효성 검사
        try:
            datetime.strptime(birthday, '%y%m%d')
        except ValueError:
            self.show_warning("입력 오류", "생년월일을 올바르게 입력해주세요.\n(ex: 901020)")
            return

        try:
            with pymysql.connect(**self.db_config) as conn:
                with conn.cursor() as cursor:
                    # Check if the user already exists
                    query = ("SELECT user_ID FROM user_info_table "
                             "WHERE name = %s AND phone_num = %s AND birthday = %s AND gender = %s")
                    cursor.execute(query, (name, phone, birthday, self.gender))
                    existing_user = cursor.fetchone()

                    if existing_user:
                        user_id = existing_user['user_ID']
                        update_query = ("UPDATE user_info_table "
                                         "SET last_modified = CURRENT_TIMESTAMP "
                                         "WHERE user_ID = %s")
                        cursor.execute(update_query, (user_id,))
                        conn.commit()
                        QMessageBox.information(self, "알림", "기존 사용자입니다.")
                    else:
                        cursor.execute("SELECT user_ID FROM user_info_table ORDER BY user_ID")
                        existing_ids = {row['user_ID'] for row in cursor.fetchall()}

                        new_user_id = 1
                        while new_user_id in existing_ids:
                            new_user_id += 1

                        insert_query = ("INSERT INTO user_info_table "
                                         "(user_ID, name, phone_num, birthday, gender, point, last_modified) "
                                         "VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)")
                        cursor.execute(insert_query, (new_user_id, name, phone, birthday, self.gender, 0))
                        conn.commit()
                        QMessageBox.information(self, "알림", "새로운 사용자가 등록되었습니다.")

        except pymysql.MySQLError as err:
            QMessageBox.critical(self, "오류", f"데이터베이스 오류 발생: {err}")

        finally:
            self.go_to_next_window()

    def go_to_next_window(self):
        self.next_window = KioskWindow(self.db_config, self.main)
        self.next_window.show()
        self.close()


    def closeEvent(self, event):
        event.accept()
        gui_windows = QApplication.allWidgets()
        main_windows = [win for win in gui_windows if isinstance(win, (KioskWindow)) and win.isVisible()]
        if not main_windows:
            self.main.home()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    ui_path = new_account_ui_path
    new_account_window = NewAccountWindow(ui_path, db_config, main_window)
    new_account_window.show()
    sys.exit(app.exec_())