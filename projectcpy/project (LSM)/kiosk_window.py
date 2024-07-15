import sys
import os
import cv2
import pymysql
import time
from PyQt5 import uic, QtCore, QtGui
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QGraphicsScene
from menu_window import MenuWindow
from config import kiosk_ui_path

class KioskWindow(QMainWindow):
    def __init__(self, db_config):
        super().__init__()
        uic.loadUi(kiosk_ui_path, self)
        self.db_config = db_config
        self.cap = cv2.VideoCapture(0)  # 기본 카메라 사용
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
        self.takePhotoBtn.clicked.connect(self.capture_image)
        
        if not self.cap.isOpened():
            QMessageBox.warning(self, "카메라 연결 오류", "카메라를 열 수 없습니다.")
        
        self.timer.start(1000 // 30)  # 30 fps로 설정

        self.scene = QGraphicsScene(self)
        self.graphicsView.setScene(self.scene)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(convertToQtFormat)

            # Clear the previous items in the scene
            self.scene.clear()

            # Scale the pixmap to fit the graphics view geometry while maintaining aspect ratio
            pixmap = pixmap.scaled(self.graphicsView.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Add the new pixmap item to the scene
            pixmap_item = self.scene.addPixmap(pixmap)

            # Center the pixmap item in the graphics view
            self.graphicsView.setSceneRect(QRectF(pixmap.rect()))
            self.graphicsView.fitInView(QRectF(pixmap.rect()), Qt.KeepAspectRatio)

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
                self.menu_window = MenuWindow(self.db_config)
            self.menu_window.show()
        except Exception as e:
            print(f"메뉴 창을 열던 중 에러 발생: {e}")


if __name__ == "__main__":
    from config import db_config  # db_config를 import합니다.
    app = QApplication(sys.argv)
    window = KioskWindow(db_config)
    window.show()
    sys.exit(app.exec_())
