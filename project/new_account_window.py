import os
from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUi
import pymysql
from kiosk_window import KioskWindow

new_account_form_class = uic.loadUiType("/home/pjh/dev_ws/EDA/ui/new_account.ui")[0]

class NewAccountWindow(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        ui_path = os.path.expanduser("~/dev_ws/EDA/ui/new_account.ui")
        loadUi(ui_path, self)
        
        self.main_window = main_window
        self.nextButton.clicked.connect(self.save_user_info)

    def save_user_info(self):
        name = self.nameLineEdit.text().strip()
        phone = self.phoneLineEdit.text().strip()
        birthday = self.birthdayLineEdit.text().strip()

        if not name or not phone or not birthday:
            print("모든 정보를 입력해주세요.")
            return

        mysql_config = {
            'host': 'localhost',
            'user': 'junghan',
            'password': '6488',
            'database': 'order_db',
            'charset': 'utf8',
            'cursorclass': pymysql.cursors.DictCursor
        }

        try:
            conn = pymysql.connect(**mysql_config)
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

                    print(f"기존 사용자입니다. 포인트가 {new_point}점으로 증가되었습니다.")
                else:
                    insert_query = ("INSERT INTO user_info_table "
                                    "(name, phone_num, birthday, point, last_modified) "
                                    "VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)")
                    cursor.execute(insert_query, (name, phone, birthday, 1))
                    conn.commit()

                    print("새로운 사용자가 등록되었습니다. 포인트는 1점으로 초기화됩니다.")

        except pymysql.MySQLError as err:
            print(f"오류: {err}")

        finally:
            if 'conn' in locals():
                conn.close()
                print("데이터베이스 연결을 닫았습니다.")

            self.go_to_next_window()

    def go_to_next_window(self):
        if self.main_window:
            self.next_window = KioskWindow(self.main_window)
            self.next_window.show()
        self.close()
