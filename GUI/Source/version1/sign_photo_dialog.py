import os
import cv2
import sqlite3
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

    def save_photo(self):
        save_dir = os.path.join("RobotArm/GUI/Image")
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"ID{self.user_id}.jpg"
        file_path = os.path.join(save_dir, file_name)
        frame = self.video_thread.save_current_frame(file_path)

        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            image_data = buffer.tobytes()
            conn = sqlite3.connect('RobotArm/GUI/DB/user_data.db')
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET image = ? WHERE id = ?", (image_data, self.user_id))
            conn.commit()
            conn.close()
            QMessageBox.information(self, "성공", "사진 촬영 성공")
        self.close()

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()
