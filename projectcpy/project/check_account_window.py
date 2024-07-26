import os
import pymysql
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QMessageBox
from config import check_account_ui_path, db_config
from menu_window import MenuWindow
from PyQt5.QtWidgets import QApplication

class CheckAccountWindow(QDialog):
    def __init__(self, parent=None, main=None):
        super(CheckAccountWindow, self).__init__(parent)
        uic.loadUi(check_account_ui_path, self)
        
        self.db_config = db_config
        self.parent = parent
        self.main = main  # main 속성 추가
        self.checkBtn.clicked.connect(self.check_account)

        self.setFixedSize(self.size())  # 현재 창 크기로 고정

        # 화면 크기를 가져와 창의 중앙 위치를 계산
        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

        self.customize_ui()



    def customize_ui(self):
        # QDialog에 배경 이미지 설정
        ui_image_path = "ui/pic"
        image_path = os.path.join(ui_image_path, "check_login_background.png")
        if os.path.exists(image_path):
            self.setStyleSheet(f"QDialog {{background-image: url('{image_path}'); background-repeat: no-repeat; background-position: center;}}")
        else:
            print(f"Error: Image file {image_path} does not exist.")

        # QPushButton 스타일 설정
        button_style = """
            QPushButton {
                background-color: #97d5cd;
                border: 2px solid #97d5cd;
                border-radius: 15px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7bc0b9;
            }
            QPushButton:pressed {
                background-color: #68a49f;
            }
        """
        self.checkBtn.setStyleSheet(button_style)

        # QLineEdit 스타일 설정
        lineedit_style = """
            QLineEdit {
                font-size: 11pt;
                font-weight: bold;
                padding: 5px;
                border: 2px solid rgb(160, 207, 198);
                border-radius: 20px;
                background-color: white;
                color: rgb(248,141,164);
            }
            QLineEdit:focus {
                border: 2px solid rgb(251, 191, 196);
            }
        """
        self.phoneLineEdit.setStyleSheet(lineedit_style)



    def check_account(self):
        phone_num = self.phoneLineEdit.text()
        
        if not phone_num:
            QMessageBox.warning(self, "경고", "휴대폰 번호를 입력하세요.")
            return

        user_id = self.get_user_id_by_phone(phone_num)
        
        if user_id:
            self.go_to_menu_window(user_id)
        else:
            QMessageBox.warning(self, "오류", "일치하는 회원정보가 없습니다.")

    def get_user_id_by_phone(self, phone_num):
        try:
            conn = pymysql.connect(**self.db_config)
            with conn.cursor() as cursor:
                query = ("SELECT user_ID FROM user_info_table WHERE phone_num = %s")
                cursor.execute(query, (phone_num,))
                existing_user = cursor.fetchone()

                if existing_user:
                    user_id = existing_user['user_ID']
                    update_query = ("UPDATE user_info_table SET last_modified = CURRENT_TIMESTAMP WHERE user_ID = %s")
                    cursor.execute(update_query, (user_id,))
                    conn.commit()
                    return user_id  # user_id를 반환
                else:
                    return None  # 일치하는 회원정보가 없으면 None 반환

        except pymysql.MySQLError as err:
            print(f"데이터베이스 오류 발생: {err}")
            QMessageBox.warning(self, "오류", f"데이터베이스 오류 발생: {err}")
            return None
        finally:
            if 'conn' in locals():
                conn.close()
                print("데이터베이스 연결을 닫았습니다.")

    def go_to_menu_window(self, user_id):
        self.accept()  # 현재 창을 닫음
        self.parent.close()  # LoginWindow를 닫음
        self.next_window = MenuWindow(self.db_config, self.main)  # user_id를 전달
        self.next_window.show()


    def closeEvent(self, event):
        event.accept()
        gui_windows = QApplication.allWidgets()
        main_windows = [win for win in gui_windows if isinstance(win, (MenuWindow)) and win.isVisible()]
        if not main_windows:
            self.main.home()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    main = None  # main 객체를 None으로 설정
    window = CheckAccountWindow(main=main)
    window.show()
    sys.exit(app.exec_())