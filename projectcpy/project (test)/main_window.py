import sys
import os
import pymysql
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from login_window import LoginWindow
from menu_window import MenuWindow
from new_account_window import NewAccountWindow
from config import main_ui_path, db_config

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(main_ui_path, self)
        self.orderButton.clicked.connect(self.go_to_login_window)

        # main.ui가 불려올 때마다 데이터 초기화 함수 실행
        self.update_purchase_count(db_config)

    def go_to_login_window(self):
        if not hasattr(self, 'login_window'):
            self.login_window = LoginWindow(self)
            self.login_window.show()
            self.hide()  # 메인 윈도우를 숨깁니다.

    def update_purchase_count(self, db_config):
        try:
            # 데이터베이스 연결
            conn = pymysql.connect(**db_config)
            cursor = conn.cursor()

            # 실행할 SQL 쿼리문
            sql_query = """
                UPDATE purchase_record_table
                SET choco_count = 0, vanila_count = 0, strawberry_count = 0,
                    topping1_count = 0, topping2_count = 0, topping3_count = 0
                """
            # 쿼리 실행
            cursor.execute(sql_query)
            conn.commit()

            # 연결 종료
            cursor.close()
            conn.close()

        except pymysql.MySQLError as e:
            print(f"Error updating purchase counts: {e}")

    def home(self):
        self.update_purchase_count(db_config)
        self.show()
        del(self.login_window)

    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())