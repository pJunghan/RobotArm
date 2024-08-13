import sys
import os
import cv2
import pymysql
from login_window import LoginWindow
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt, QThread, QTimer, QStringListModel, pyqtSlot, QSize, QRectF
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsPixmapItem, QMessageBox, QDialog, QTextBrowser, QStyledItemDelegate, QStyleOptionViewItem
from config import db_config, seat_select_ui_path, seat_select_image_path
import logging

logging.basicConfig(level=logging.INFO)

class DatabaseManager:
    def __init__(self, db_config):
        self.db_config = db_config
        
    def connect(self):
        try:
            return pymysql.connect(**self.db_config)
        except pymysql.MySQLError as err:
            logging.error(f"데이터베이스 연결 오류: {err}")
            return None
        
    def get_latest_user_id(self):
        conn = self.connect()
        if not conn:
            return None
        try:
            with conn.cursor() as cursor:
                query = "SELECT user_ID FROM user_info_table ORDER BY last_modified DESC LIMIT 1"
                cursor.execute(query)
                result = cursor.fetchone()
                return result['user_ID'] if result else None
        except pymysql.MySQLError as err:
            logging.error(f"데이터베이스 쿼리 오류: {err}")
            return None
        except Exception as e:
            logging.error(f"알 수 없는 오류 발생: {e}")
            return None
        finally:
            if conn:
                conn.close()
            
    def get_user_info(self, user_id):
        conn = self.connect()
        if not conn:
            return None, None, None
        try:
            with conn.cursor() as cursor:
                query = "SELECT gender, birthday, name FROM user_info_table WHERE user_ID = %s"
                cursor.execute(query, (user_id,))
                result = cursor.fetchone()
                if result:
                    gender = result['gender']
                    birthday = result['birthday']
                    name = result['name']
                    return birthday, gender, name
                else:
                    logging.warning("등록된 사용자 정보가 없습니다.")
                    return None, None, None
        except pymysql.MySQLError as err:
            logging.error(f"데이터베이스 쿼리 오류: {err}")
            return None, None, None
        except Exception as e:
            logging.error(f"알 수 없는 오류 발생: {e}")
            return None, None, None
        finally:
            if conn:
                conn.close()


class Seat_Selection_Window(QMainWindow):

    def __init__(self, main):
        super().__init__()

        # MainWindow 객체 참조  
        self.main = main

        # UI 파일 로드
        uic.loadUi(seat_select_ui_path, self) 

        # 데이터베이스 로드 
        self.db_manager = DatabaseManager(db_config)  
        
        # 사용자 ID를 가져옴
        self.user_id = self.db_manager.get_latest_user_id()  

        # UI 설정
        self.customize_ui()

        # 버튼 클릭 시 메시지 박스 출력 연결
        buttons = [(self.seat1, 1), (self.seat2, 2), (self.seat3, 3), (self.seat4, 4)]
        for button, seat_number in buttons:
            button.clicked.connect(lambda _, sn=seat_number: self.show_seat_message(sn))


    def customize_ui(self):
        # 현재 창 크기로 고정
        self.setFixedSize(self.size())  
        
        # 화면 크기를 가져와 창의 중앙 위치를 계산 및 화면 중앙 표시
        screen_geometry = QApplication.primaryScreen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

        # 자리 도면 이미지 
        self.display_image_in_graphics_view(seat_select_image_path)

        # QPushButton 스타일 설정
        button_style = """
            QPushButton {
                background-color: white;
                border: 2px solid gray;
                border-radius: 15px;
                color: black;
                font-size: 12pt;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #87CEFA;
            }
            QPushButton:pressed {
                background-color: #00BFFF;
            }
        """
        
        self.seat1.setStyleSheet(button_style)
        self.seat2.setStyleSheet(button_style)
        self.seat3.setStyleSheet(button_style)
        self.seat4.setStyleSheet(button_style)

    def show_seat_message(self, seat_number):
        # 좌석 선택 시 출력할 메시지 박스
        reply = QMessageBox.question(self, "질문", f"{seat_number}번 좌석을 선택하시겠습니까?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            # 선택한 좌석 번호 저장
            self.main.data["seat"] = int(seat_number)

            # 데이터 저장 확인
            print(self.main.data)
            self.go_to_login_window()


    def display_image_in_graphics_view(self, image_path):
        try:
            # 이미지를 QImage로 로드
            image = QImage(image_path)
            
            # 이미지 로드 실패 시 예외 발생
            if image.isNull():
                raise ValueError(f"이미지 파일을 로드할 수 없습니다: {image_path}")
            
            # QImage를 QPixmap으로 변환
            pixmap = QPixmap.fromImage(image)
            
            # QGraphicsView의 크기 얻기
            view_size = self.graphicsView.size()
            
            # QPixmap 크기 조정 (QGraphicsView 크기에 맞게)
            scaled_pixmap = pixmap.scaled(view_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # QGraphicsPixmapItem 생성
            pixmap_item = QGraphicsPixmapItem(scaled_pixmap)
            
            # QGraphicsScene 생성 및 QGraphicsPixmapItem 추가
            scene = QGraphicsScene(self)
            scene.addItem(pixmap_item)
            
            # QGraphicsView에 Scene 설정
            self.graphicsView.setScene(scene)
            
        
        except ValueError as ve:
            logging.error(f"이미지 로드 오류: {ve}")
            QMessageBox.critical(self, "이미지 로드 오류", str(ve))
        except Exception as e:
            logging.exception("알 수 없는 오류가 발생했습니다.")
            QMessageBox.critical(self, "오류", f"알 수 없는 오류가 발생했습니다: {str(e)}")


    def go_to_login_window(self):
        if not hasattr(self, 'login_window'):
            self.login_window = LoginWindow(self.main)
            self.login_window.show()
            self.close()  

    
    def closeEvent(self, event):
        event.accept()
        gui_windows = QApplication.allWidgets()
        main_windows = [win for win in gui_windows if isinstance(win, (LoginWindow)) and win.isVisible()]
        if not main_windows:
            self.main.home()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Main window 설정 및 보여주기
    main_window = Seat_Selection_Window(main= None)
    main_window.show()
    sys.exit(app.exec_())
