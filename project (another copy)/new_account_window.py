from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5 import uic
import pymysql
from kiosk_window import KioskWindow

class NewAccountWindow(QMainWindow):
    def __init__(self, ui_path, db_config):
        super().__init__()
        uic.loadUi(ui_path, self)
        self.db_config = db_config
        self.nextButton.clicked.connect(self.save_user_info)

    def save_user_info(self):
        name = self.nameLineEdit.text().strip()
        phone = self.phoneLineEdit.text().strip()
        birthday = self.birthdayLineEdit.text().strip()

        if not name or not phone or not birthday:
            QMessageBox.warning(self, "경고", "모든 정보를 입력해주세요.")
            return

        try:
            conn = pymysql.connect(**self.db_config)
            with conn.cursor() as cursor:
                # Check if the user already exists
                query = ("SELECT user_ID FROM user_info_table "
                        "WHERE name = %s AND phone_num = %s AND birthday = %s")
                cursor.execute(query, (name, phone, birthday))
                existing_user = cursor.fetchone()

                if existing_user:
                    user_id = existing_user['user_ID']

                    # If the user exists, update last_modified
                    update_query = ("UPDATE user_info_table "
                                    "SET last_modified = CURRENT_TIMESTAMP "
                                    "WHERE user_ID = %s")
                    cursor.execute(update_query, (user_id,))
                    conn.commit()

                    QMessageBox.information(self, "알림", "기존 사용자입니다.")
                else:
                    # Find the first available user_id that is not in use
                    cursor.execute("SELECT user_ID FROM user_info_table ORDER BY user_ID")
                    existing_ids = {row['user_ID'] for row in cursor.fetchall()}

                    new_user_id = 1
                    while new_user_id in existing_ids:
                        new_user_id += 1

                    # Insert new user with the determined new_user_id
                    insert_query = ("INSERT INTO user_info_table "
                                    "(user_ID, name, phone_num, birthday, point, last_modified) "
                                    "VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)")
                    cursor.execute(insert_query, (new_user_id, name, phone, birthday, 0))
                    conn.commit()

                    QMessageBox.information(self, "알림", f"새로운 사용자가 등록되었습니다.")

        except pymysql.MySQLError as err:
            QMessageBox.critical(self, "오류", f"데이터베이스 오류 발생: {err}")

        finally:
            if 'conn' in locals() and conn.open:
                conn.close()
                print("데이터베이스 연결을 닫았습니다.")

            self.go_to_next_window()

    def go_to_next_window(self):
        self.next_window = KioskWindow(self.db_config)
        self.next_window.show()
        self.close()

