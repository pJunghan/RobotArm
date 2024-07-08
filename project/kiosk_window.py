import sys
import os
import cv2
import pymysql
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsPixmapItem, QMessageBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

# UI 파일 경로
ui_base_path = "/home/pjh/dev_ws/EDA/ui"
main_ui_path = os.path.join(ui_base_path, "main.ui")
login_ui_path = os.path.join(ui_base_path, "login.ui")
menu_ui_path = os.path.join(ui_base_path, "order_ice_cream_window.ui")
new_account_ui_path = os.path.join(ui_base_path, "new_account.ui")
kiosk_ui_path = os.path.join(ui_base_path, "kiosk_cam.ui")

# 메인 창 클래스
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(main_ui_path, self)
        self.orderButton.clicked.connect(self.go_to_login_window)

    def go_to_login_window(self):
        self.next_window = LoginWindow()
        self.next_window.show()
        self.close()

class LoginWindow(QMainWindow):
    def __init__(self):
        super(LoginWindow, self).__init__()
        uic.loadUi(login_ui_path, self)
        self.video_thread = VideoThread(self.graphicsView)
        self.orderbtn.clicked.connect(self.go_to_menu_window)
        self.memberBtn.clicked.connect(self.go_to_new_account_window)

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

    def go_to_menu_window(self):
        self.next_window = MenuWindow()
        self.next_window.show()
        self.close()

    def go_to_new_account_window(self):
        self.next_window = NewAccountWindow(self)
        self.next_window.show()
        self.close()

class MenuWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(menu_ui_path, self)
        self.Home_Button.clicked.connect(self.go_to_main_window)
        self.Next_Button.clicked.connect(self.go_to_next_tab)

    def go_to_main_window(self):
        self.main_window = MainWindow()
        self.main_window.show()
        self.close()

    def go_to_next_tab(self):
        current_index = self.tabWidget.currentIndex()
        next_index = (current_index + 1) % self.tabWidget.count()
        self.tabWidget.setCurrentIndex(next_index)

class KioskWindow(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        uic.loadUi(kiosk_ui_path, self)
        self.main_window = main_window
        self.captureButton.clicked.connect(self.capture_image)
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // 30)
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(convertToQtFormat)
            self.cameraLabel.setPixmap(pixmap.scaled(self.cameraLabel.size(), Qt.KeepAspectRatio))
        
    def capture_image(self):
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
                query = "SELECT user_id FROM user_info_table ORDER BY last_modified DESC LIMIT 1"
                cursor.execute(query)
                result = cursor.fetchone()
                if result:
                    user_id = result['user_id']
                    image_path = os.path.expanduser(f"~/dev_ws/EDA/user_pic/{user_id}.jpeg")
                    
                    ret, frame = self.cap.read()
                    if ret:
                        cv2.imwrite(image_path, frame)
                    else:
                        QMessageBox.warning(self, "촬영 실패", "카메라에서 이미지를 가져오지 못했습니다.")
                else:
                    QMessageBox.warning(self, "사용자 없음", "등록된 사용자가 없습니다.")

        except pymysql.MySQLError as err:
            print(f"오류: {err}")

        finally:
            if 'conn' in locals():
                conn.close()
                print("데이터베이스 연결을 닫았습니다.")

            self.go_to_menu_window()

    def go_to_menu_window(self):
        self.menu_window = MenuWindow()
        self.menu_window.show()
        self.close()

class NewAccountWindow(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        uic.loadUi(new_account_ui_path, self)
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

class VideoThread:
    def __init__(self, graphics_view):
        self.graphics_view = graphics_view
        self.webcam = cv2.VideoCapture(0)
        if not self.webcam.isOpened():
            print("Could not open webcam")
            exit()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(60)

    def update_frame(self):
        status, frame = self.webcam.read()
        if status:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.display_image(pixmap)

    def display_image(self, pixmap):
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(pixmap))
        self.graphics_view.setScene(scene)
        self.graphics_view.fitInView(scene.itemsBoundingRect(), 1)

    def stop(self):
        self.timer.stop()
        self.webcam.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
