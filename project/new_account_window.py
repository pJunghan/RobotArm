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
                query = ("SELECT user_id, point FROM user_info_table "
                         "WHERE name = %s AND phone_num = %s AND birthday = %s")
                cursor.execute(query, (name, phone, birthday))
                existing_user = cursor.fetchone()

                if existing_user:
                    user_id, current_point = existing_user['user_id'], existing_user['point']
                    new_point = current_point + 1

                    update_query = ("UPDATE user_info_table "
                                    "SET point = %s, last_modified = CURRENT_TIMESTAMP "
                                    "WHERE user_id = %s")
                    cursor.execute(update_query, (new_point, user_id))
                    conn.commit()

                    QMessageBox.information(self, "알림", f"기존 사용자입니다. 포인트가 {new_point}점으로 증가되었습니다.")
                else:
                    insert_query = ("INSERT INTO user_info_table "
                                    "(name, phone_num, birthday, point, last_modified) "
                                    "VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)")
                    cursor.execute(insert_query, (name, phone, birthday, 1))
                    conn.commit()

                    QMessageBox.information(self, "알림", "새로운 사용자가 등록되었습니다. 포인트는 1점으로 초기화됩니다.")

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

