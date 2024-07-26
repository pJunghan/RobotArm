import sys
import os
import cv2
import pymysql
import time
from PyQt5 import uic, QtCore, QtGui
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox, QGraphicsScene,  QMainWindow
from menu_window import MenuWindow
from config import kiosk_ui_path, user_img_path, db_config


class KioskWindow(QDialog):
    def __init__(self, db_config, main):
        super().__init__()
        uic.loadUi(kiosk_ui_path, self)
        self.main = main
        self.db_config = db_config
        self.cap = cv2.VideoCapture(0)  # 기본 카메라 사용
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
        self.captureButton.clicked.connect(self.capture_image)
        
        if not self.cap.isOpened():
            QMessageBox.warning(self, "카메라 연결 오류", "카메라를 열 수 없습니다.")
        
        self.timer.start(1000 // 30)  # 30 fps로 설정

        # QGraphicsScene 초기화
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)

        self.customize_ui()
        self.setFixedSize(self.size())  # 현재 창 크기로 고정

        # 화면 크기를 가져와 창의 중앙 위치를 계산
        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

    def customize_ui(self):
        # QFrame에 배경 이미지 설정
        ui_image_path = "ui/pic"
        image_path = os.path.join(ui_image_path, "login_background.png")
        if os.path.exists(image_path):
            self.frame_2.setStyleSheet(f"QFrame {{background-image: url('{image_path}'); background-repeat: no-repeat; background-position: center;}}")
        else:
            print(f"Error: Image file {image_path} does not exist.")


        # QPushButton 스타일 설정
        self.captureButton.setStyleSheet("""
            QPushButton {
                background-color: #62A0EA;
                border: 2px solid #62A0EA;
                border-radius: 15px;
                color: white;
                font-size: 16pt;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #3B82F6;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(convertToQtFormat)
            
            # QGraphicsScene에 pixmap 추가
            self.scene.clear()
            self.scene.addPixmap(pixmap.scaled(self.graphicsView.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


    def capture_image(self):
        try:
            conn = pymysql.connect(**self.db_config)
            with conn.cursor() as cursor:
                query = "SELECT user_id FROM user_info_table ORDER BY last_modified DESC LIMIT 1"
                cursor.execute(query)
                result = cursor.fetchone()
                if result:
                    user_id = result['user_id']
                    image_path = os.path.expanduser(f"{user_img_path}/{user_id}.jpeg")

                    ret, frame = self.cap.read()
                    if ret:
                        cv2.imwrite(image_path, frame)
                        QMessageBox.information(self, "촬영 완료", "사진이 성공적으로 저장되었습니다.")
                        self.go_to_menu_window()
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
            self.close()

    def closeEvent(self, event):
        self.cap.release()
        self.timer.stop()
        event.accept()
        time.sleep(1)

    def go_to_menu_window(self):
        try:
            if not hasattr(self, 'menu_window') or not self.menu_window.isVisible():
                self.menu_window = MenuWindow(self.db_config, self.main)
            self.menu_window.show()
        except Exception as e:
            print(f"메뉴 창을 열던 중 에러 발생: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    new_account_window = KioskWindow(db_config, main_window)
    new_account_window.show()
    sys.exit(app.exec_())