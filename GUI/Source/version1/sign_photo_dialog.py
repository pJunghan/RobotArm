import os
import cv2
import mysql.connector
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QMessageBox
from sign_video_thread import SignVideoThread

signPhoto_form_class = uic.loadUiType("/home/lsm/git_ws/RobotArm/GUI/UI/signPhoto.ui")[0]

class SignUpPhotoDialog(QDialog, signPhoto_form_class):
    def __init__(self, user_id):
        super(SignUpPhotoDialog, self).__init__()
        self.setupUi(self)
        self.user_id = user_id
        self.video_thread = SignVideoThread(self.graphicsView)
        self.takePhotoBtn.clicked.connect(self.save_photo)
        self.init_db()

    def init_db(self):
        self.conn = mysql.connector.connect(
            host='localhost',
            user='user',  # MySQL 사용자 이름으로 변경
            password='1111',  # MySQL 비밀번호로 변경
            database='Aris'
        )
        self.cursor = self.conn.cursor()

    def save_photo(self):
        save_dir = os.path.join("RobotArm/GUI/Image")
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"ID{self.user_id}.jpg"
        file_path = os.path.join(save_dir, file_name)
        frame = self.video_thread.save_current_frame(file_path)

        if frame is not None:
            cv2.imwrite(file_path, frame)  # 이미지를 파일로 저장합니다
            
            # 이미지 파일 경로를 데이터베이스에 저장합니다
            self.cursor.execute("UPDATE User_Data SET Image_Path = %s WHERE ID = %s", (file_path, self.user_id))
            self.conn.commit()
            QMessageBox.information(self, "성공", "사진 촬영 성공")
        self.close()

    def closeEvent(self, event):
        self.video_thread.stop()
        self.conn.close()
        event.accept()
