import sys
import os
import cv2
import pymysql
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt, QThread, QTimer,QStringListModel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsPixmapItem, QMessageBox
from purchase import ConfirmWindow  # ConfirmWindow import 추가
import pymysql
from config import menu_ui_path,db_config,ice_cream_images,topping_images



class MenuWindow(QMainWindow):
    def __init__(self, db_config, main):
        super().__init__()
        self.main = main
        self.menu_items = {}
        uic.loadUi(menu_ui_path, self)  # UI 파일 로드

        self.db_config = db_config
        self.user_id = self.get_latest_user_id()  # 사용자 ID를 가져옴
        self.Home_Button.clicked.connect(self.go_to_main_window)  # Home_Button 클릭 시 메인 창으로 이동
        self.Next_Button.clicked.connect(self.go_to_purchase_window)  # Next_Button 클릭 시 결제 창으로 이동

        self.item_click_count = {
            'choco': 0,
            'vanila': 0,
            'strawberry': 0,
            'topping1': 0,
            'topping2': 0,
            'topping3': 0
        }

        # QStringListModel 초기화
        self.list_model = QStringListModel()
        self.listView.setModel(self.list_model)  # QListView에 모델 설정

        self.add_image_to_graphics_view(ice_cream_images[0], self.graphicsView_1, 'choco')
        self.add_image_to_graphics_view(ice_cream_images[1], self.graphicsView_2, 'vanila')
        self.add_image_to_graphics_view(ice_cream_images[2], self.graphicsView_3, 'strawberry')
        self.add_image_to_graphics_view(topping_images[0], self.graphicsView_4, 'topping1')
        self.add_image_to_graphics_view(topping_images[1], self.graphicsView_5, 'topping2')
        self.add_image_to_graphics_view(topping_images[2], self.graphicsView_6, 'topping3')

    def get_latest_user_id(self):
        try:
            conn = pymysql.connect(**self.db_config)
            with conn.cursor() as cursor:
                query = "SELECT user_ID FROM user_info_table ORDER BY last_modified DESC LIMIT 1"
                cursor.execute(query)
                result = cursor.fetchone()
                if result:
                    return result['user_ID']
                else:
                    QMessageBox.warning(self, "사용자 정보 없음", "등록된 사용자 정보가 없습니다.")
                    return None
        except pymysql.MySQLError as err:
            print(f"데이터베이스 오류 발생: {err}")
            return None
        finally:
            if 'conn' in locals():
                conn.close()

    def go_to_main_window(self):
        self.main.home()
        self.close()

    def go_to_purchase_window(self):
        self.update_purchase_record()  # purchase_record_table 업데이트
        # Pass the string list from the model to the ConfirmWindow
        self.confirm_window = ConfirmWindow(self.db_config, self.item_click_count, self.list_model.stringList(), self.main)
        self.confirm_window.show()
        self.close()


    def update_purchase_record(self):
        try:
            conn = pymysql.connect(**self.db_config)
            with conn.cursor() as cursor:
                # 사용자의 레코드가 있는지 확인
                query = f"SELECT * FROM purchase_record_table WHERE user_id = {self.user_id}"
                cursor.execute(query)
                result = cursor.fetchone()
                if not result:
                    # 레코드가 없으면 새로 추가
                    cursor.execute("INSERT INTO purchase_record_table (user_id) VALUES (%s)", (self.user_id,))
                
                # 아이템 수 업데이트
                for item_name, count in self.item_click_count.items():
                    update_query = f"UPDATE purchase_record_table SET {item_name}_count = {item_name}_count + %s WHERE user_id = %s"
                    cursor.execute(update_query, (count, self.user_id))
            conn.commit()
        except pymysql.MySQLError as err:
            print(f"데이터베이스 오류 발생: {err}")
        finally:
            if 'conn' in locals():
                conn.close()

    def add_image_to_graphics_view(self, image_path, graphics_view, item_name):
        pixmap = QPixmap(image_path)
        pixmap_resized = pixmap.scaled(200, 200, Qt.KeepAspectRatio)
        item = QGraphicsPixmapItem(pixmap_resized)
        scene = QGraphicsScene()
        scene.addItem(item)
        graphics_view.setScene(scene)

        item.mousePressEvent = lambda event: self.item_click_event(event, item_name)

    def item_click_event(self, event, item_name):
        self.item_click_count[item_name] += 1
        self.update_list_view()

    def update_list_view(self):
        items_to_show = [f"{item}: {count}" for item, count in self.item_click_count.items() if count > 0]
        self.list_model.setStringList(items_to_show)  # QStringList로 업데이트
        self.listView.setModel(self.list_model) 