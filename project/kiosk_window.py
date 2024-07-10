import sys
import os
import cv2
import pymysql
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from menu_window import MenuWindow
from config import kiosk_ui_path

class KioskWindow(QMainWindow):
    def __init__(self, db_config):
        super().__init__()
        uic.loadUi(kiosk_ui_path, self)
        self.db_config = db_config
        
        self.captureButton.clicked.connect(self.capture_image)
        
        self.cap = cv2.VideoCapture(-1)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "카메라 연결 오류", "카메라를 열 수 없습니다.")
        
        self.timer = QtCore.QTimer(self)
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
        try:
            conn = pymysql.connect(**self.db_config)
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
                        QMessageBox.information(self, "촬영 완료", "사진이 성공적으로 저장되었습니다.")
                        # 메뉴 창으로 이동
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

    def go_to_menu_window(self):
        try:
            if not hasattr(self, 'menu_window') or not self.menu_window.isVisible():
                self.menu_window = MenuWindow(self.db_config)
            self.menu_window.show()
        except Exception as e:
            print(f"메뉴 창을 열던 중 에러 발생: {e}")