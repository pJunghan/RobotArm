from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt
import pymysql
from kiosk_window import KioskWindow

class NewAccountWindow(QDialog):  # Inherit from QDialog
    def __init__(self, ui_path, db_config):
        super().__init__()
        uic.loadUi(ui_path, self)
        self.db_config = db_config
        self.gender = None
        self.signupBtn.clicked.connect(self.save_user_info)  # Correct button name
        self.checkMale.stateChanged.connect(self.on_check_gender)
        self.checkFemale.stateChanged.connect(self.on_check_gender)

    def on_check_gender(self, state):
        # Update gender based on checkbox state
        if self.sender() == self.checkMale and state == Qt.Checked:
            self.gender = 'Male'
            self.checkFemale.setChecked(False)  # Uncheck Female
        elif self.sender() == self.checkFemale and state == Qt.Checked:
            self.gender = 'Female'
            self.checkMale.setChecked(False)  # Uncheck Male

    def save_user_info(self):
        name = self.Name.text().strip()
        phone = self.PhoneNumber.text().strip()
        birthday = self.Birth.text().strip()

        # Check for empty fields
        if not name or not phone or not birthday:
            QMessageBox.warning(self, "경고", "모든 정보를 입력해주세요.")
            return

        # Check for selected gender
        if not self.gender:
            QMessageBox.warning(self, "경고", "성별을 선택해주세요.")
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
        self.next_window = KioskWindow(self.db_config)
        self.next_window.show()
        self.close()
