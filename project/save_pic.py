import sys
import os
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import pymysql

class CameraWidget(QWidget):
    def __init__(self, user_id):
        super().__init__()
        self.user_id = user_id
        self.initUI()
        self.create_db_connection()

    def initUI(self):
        self.setWindowTitle('사진 촬영 및 저장')
        self.setGeometry(100, 100, 640, 480)

        self.lbl_camera = QLabel(self)
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        self.lbl_camera.setFixedSize(640, 480)

        self.btn_capture = QPushButton('사진 촬영', self)
        self.btn_capture.clicked.connect(self.capture_photo)

        vbox = QVBoxLayout()
        vbox.addWidget(self.lbl_camera)
        vbox.addWidget(self.btn_capture, alignment=Qt.AlignCenter)

        self.setLayout(vbox)

        # 카메라 초기화
        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000/30)  # 30 FPS로 설정

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            # OpenCV BGR 이미지를 PyQt QPixmap으로 변환하여 QLabel에 표시
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.lbl_camera.setPixmap(pixmap.scaled(self.lbl_camera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def capture_photo(self):
        ret, frame = self.capture.read()
        if ret:
            photo_path = f"~/dev_ws/EDA/user_pic/{self.user_id}.jpeg"  # 저장할 경로 및 파일명
            cv2.imwrite(os.path.expanduser(photo_path), frame)
            self.save_to_db(photo_path)
            QMessageBox.information(self, '알림', '사진이 성공적으로 저장되었습니다.', QMessageBox.Ok)

    def save_to_db(self, photo_path):
        try:
            # 데이터베이스 연결
            conn = pymysql.connect(host='127.0.0.1', user='junghan', password='6488', db='order_db', charset='utf8')
            cur = conn.cursor()

            # 사용자 정보 업데이트
            update_query = "UPDATE user_info_table SET photo_path = %s WHERE user_ID = %s"
            cur.execute(update_query, (photo_path, self.user_id))
            conn.commit()

            cur.close()
            conn.close()
        except pymysql.MySQLError as e:
            QMessageBox.critical(self, '오류', f"데이터베이스 오류 발생: {e}", QMessageBox.Ok)

    def create_db_connection(self):
        try:
            self.conn = pymysql.connect(host='127.0.0.1', user='junghan', password='6488', db='order_db', charset='utf8')
        except pymysql.MySQLError as e:
            QMessageBox.critical(self, '오류', f"데이터베이스 연결 오류: {e}", QMessageBox.Ok)
            sys.exit(1)

    def closeEvent(self, event):
        self.capture.release()
        self.timer.stop()
        if hasattr(self, 'conn'):
            self.conn.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("사용법: python save_pic.py [user_id]")
        sys.exit(1)
    
    user_id = int(sys.argv[1])
    
    app = QApplication(sys.argv)
    ex = CameraWidget(user_id)
    ex.show()
    sys.exit(app.exec_())
