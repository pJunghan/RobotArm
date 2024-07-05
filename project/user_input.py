import sys
import pymysql
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QMessageBox

class UserInfoForm(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.create_db_connection()
        self.delete_duplicate_users()

    def initUI(self):
        self.setWindowTitle('사용자 정보 입력 폼')
        self.setGeometry(100, 100, 300, 200)

        self.lbl_name = QLabel('이름:', self)
        self.lbl_phone_num = QLabel('전화번호:', self)
        self.lbl_birthday = QLabel('생년월일(yyyymmdd):', self)

        self.txt_name = QLineEdit(self)
        self.txt_phone_num = QLineEdit(self)
        self.txt_birthday = QLineEdit(self)

        self.btn_submit = QPushButton('확인', self)
        self.btn_submit.clicked.connect(self.check_and_save_user_info)

        vbox = QVBoxLayout()
        vbox.addWidget(self.lbl_name)
        vbox.addWidget(self.txt_name)
        vbox.addWidget(self.lbl_phone_num)
        vbox.addWidget(self.txt_phone_num)
        vbox.addWidget(self.lbl_birthday)
        vbox.addWidget(self.txt_birthday)
        vbox.addWidget(self.btn_submit)

        self.setLayout(vbox)

    def create_db_connection(self):
        try:
            self.conn = pymysql.connect(host='127.0.0.1', user='junghan', password='6488', db='order_db', charset='utf8')
            self.cur = self.conn.cursor()
        except pymysql.MySQLError as e:
            QMessageBox.critical(self, '오류', f"데이터베이스 연결 오류: {e}", QMessageBox.Ok)
            sys.exit(1)

    def delete_duplicate_users(self):
        try:
            # 중복된 사용자 정보 삭제
            delete_query = """
                DELETE t1 FROM user_info_table t1
                JOIN user_info_table t2 ON t1.name = t2.name AND t1.phone_num = t2.phone_num AND t1.birthday = t2.birthday
                WHERE t1.user_ID > t2.user_ID
            """
            self.cur.execute(delete_query)
            self.conn.commit()

            # 사용자 정보 초기화
            reset_query = "ALTER TABLE user_info_table AUTO_INCREMENT = 1"
            self.cur.execute(reset_query)
            self.conn.commit()

        except pymysql.MySQLError as e:
            QMessageBox.critical(self, '오류', f"데이터베이스 오류 발생: {e}", QMessageBox.Ok)

    def check_and_save_user_info(self):
        name = self.txt_name.text().strip()
        phone_num = self.txt_phone_num.text().strip()
        birthday = self.txt_birthday.text().strip()

        # yyyymmdd 형식으로 변환
        if len(birthday) == 8 and birthday.isdigit():
            birthday_formatted = f"{birthday[:4]}-{birthday[4:6]}-{birthday[6:8]}"
        else:
            QMessageBox.warning(self, '경고', '생년월일을 올바른 형식으로 입력하세요 (yyyymmdd).', QMessageBox.Ok)
            return

        if not name or not phone_num or not birthday_formatted:
            QMessageBox.warning(self, '경고', '모든 필드를 입력하세요.', QMessageBox.Ok)
            return

        try:
            # 사용자 정보 존재 여부 확인
            select_query = "SELECT user_ID, point FROM user_info_table WHERE name = %s AND phone_num = %s AND birthday = %s"
            self.cur.execute(select_query, (name, phone_num, birthday_formatted))
            result = self.cur.fetchone()

            if result:
                user_id, point = result
                # 이미 존재하는 사용자 정보의 point 증가
                update_query = "UPDATE user_info_table SET point = %s WHERE user_ID = %s"
                new_point = point + 1 if point else 1
                self.cur.execute(update_query, (new_point, user_id))
                self.conn.commit()
                QMessageBox.information(self, '알림', '이미 존재하는 사용자입니다. Point가 증가되었습니다.', QMessageBox.Ok)
            else:
                # 사용자 정보가 존재하지 않을 경우 새로운 정보 저장
                insert_query = "INSERT INTO user_info_table (name, phone_num, birthday) VALUES (%s, %s, %s)"
                self.cur.execute(insert_query, (name, phone_num, birthday_formatted))
                self.conn.commit()
                QMessageBox.information(self, '알림', '사용자 정보가 성공적으로 저장되었습니다.', QMessageBox.Ok)

        except pymysql.MySQLError as e:
            QMessageBox.critical(self, '오류', f"데이터베이스 오류 발생: {e}", QMessageBox.Ok)

    def closeEvent(self, event):
        if hasattr(self, 'conn') and self.conn.open:
            self.conn.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = UserInfoForm()
    ex.show()
    sys.exit(app.exec_())