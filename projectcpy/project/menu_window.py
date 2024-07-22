import sys
import os
import cv2
import pymysql
import tts
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt, QThread, QTimer,QStringListModel, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsPixmapItem, QMessageBox, QDialog
from purchase import ConfirmWindow  # ConfirmWindow import 추가
from config import menu_ui_path, db_config, ice_cream_images, topping_images, user_img_path # user_img_path 추가
from deepface import DeepFace
import numpy as np
from datetime import datetime
from face_emotion_age_gender_detect import FaceRecognition

class GreetingThread(QThread):
    def __init__(self, parent, gender, name):
        QThread.__init__(self)
        self.parent = parent
        self.gender = gender
        self.name = name

    def run(self):
        if self.name.startswith("guest"):
            if self.gender == "Male":
                tts.google_tts_and_play("남성 회원님 안녕하세요.")
            elif self.gender == "Female":
                tts.google_tts_and_play("여성 회원님 안녕하세요.")
            else:
                tts.google_tts_and_play("게스트 회원님 안녕하세요.")
        else:
            first_name = self.name.split(maxsplit=1)[-1]
            tts.google_tts_and_play(f"{first_name}님 안녕하세요.")


class MenuWindow(QMainWindow):
    flavors = ['choco', 'vanila', 'strawberry']
    topping_flavors = ['topping1', 'topping2', 'topping3']
    def __init__(self, db_config, main):
        super().__init__()
        self.main = main
        self.menu_items = {}
        uic.loadUi(menu_ui_path, self)  # UI 파일 로드
        self.current_emotion = None

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
        self.graphicsView_1.mousePressEvent = lambda event: self.item_click_event(event, flavor='choco', topping=None)

        self.add_image_to_graphics_view(ice_cream_images[1], self.graphicsView_2, 'vanila')
        self.graphicsView_2.mousePressEvent = lambda event: self.item_click_event(event, flavor='vanila', topping=None)

        self.add_image_to_graphics_view(ice_cream_images[2], self.graphicsView_3, 'strawberry')
        self.graphicsView_3.mousePressEvent = lambda event: self.item_click_event(event, flavor='strawberry', topping=None)

        self.add_image_to_graphics_view(topping_images[0], self.graphicsView_4, 'topping1')
        self.graphicsView_4.mousePressEvent = lambda event: self.item_click_event(event, flavor=None, topping='topping1')

        self.add_image_to_graphics_view(topping_images[1], self.graphicsView_5, 'topping2')
        self.graphicsView_5.mousePressEvent = lambda event: self.item_click_event(event, flavor=None, topping='topping2')

        self.add_image_to_graphics_view(topping_images[2], self.graphicsView_6, 'topping3')
        self.graphicsView_6.mousePressEvent = lambda event: self.item_click_event(event, flavor=None, topping='topping3')
        self.setup_recommendations()
        self.greeting_tts()
         

    # 0.05초 후에 tts.google_tts_and_play("안녕하세요.") 호출
    def greeting_tts(self):
        age, gender, name = self.get_user_info(self.user_id)
        self.greeting_thread = GreetingThread(self, gender, name)
        self.greeting_thread.start()

        self.main.set_data(gender = gender, age = age)

        # if name.startswith("guest"):
        #     if gender == "Male":
        #         QTimer.singleShot(50, lambda: tts.google_tts_and_play("남성 회원님 안녕하세요."))

        #     elif gender == "Female":
        #         QTimer.singleShot(50, lambda: tts.google_tts_and_play("여성 회원님 안녕하세요."))

        #     else:
        #         QTimer.singleShot(50, lambda: tts.google_tts_and_play("게스트 회원님 안녕하세요."))

        # else:
        #     # 이름이 공백으로 구분되어 있는 경우를 처리 및 성 제외하고 이름만 출력
        #     first_name = name.split(maxsplit=1)[-1]
        #     QTimer.singleShot(50, lambda: tts.google_tts_and_play(f"{first_name}님 안녕하세요."))

        # self.add_image_to_graphics_view(topping_images[2], self.recommendView_5, 'topping3')

    def setup_recommendations(self):
        age, gender, _ = self.get_user_info(self.user_id)
        
        if age is None or gender is None:
            recommended_flavor = 'choco'  # 기본값으로 'choco' 설정
            topping_recommendation = 'topping1'
        else:
            recommended_flavor = self.recommend_flavor(age, gender)  # 추천 아이스크림 가져오기
            topping_recommendation = self.recommend_topping(age, gender)
        
        # 추천 아이스크림을 각 recommendView에 추가
        self.add_image_to_graphics_view(
            ice_cream_images[self.flavors.index(recommended_flavor)],
            self.recommendView_1,
            recommended_flavor
        )
        self.recommendView_1.mousePressEvent = lambda event: self.recommend_item_click_event(event, recommended_flavor, topping_recommendation)
        
        # 토핑 추천을 위한 이미지를 추가합니다.
        self.add_image_to_graphics_view(
            topping_images[self.topping_flavors.index(topping_recommendation)],
            self.recommendView_5,
            topping_recommendation
        )
        self.recommendView_5.mousePressEvent = lambda event: self.recommend_item_click_event(event, recommended_flavor, topping_recommendation)

        # Note: No separate click event for recommendView_5 as it is triggered by recommendView_1's event

        # 과거 구매 기록을 바탕으로 한 추천 아이스크림 및 토핑 설정
        historical_recommendation, historical_topping_recommendation = self.recommend_based_on_history(self.user_id)
        self.add_image_to_graphics_view(
            ice_cream_images[self.flavors.index(historical_recommendation)],
            self.recommendView_3,
            historical_recommendation
        )
        self.recommendView_3.mousePressEvent = lambda event: self.recommend_item_click_event(event, historical_recommendation, historical_topping_recommendation)

        self.add_image_to_graphics_view(
            topping_images[self.topping_flavors.index(historical_topping_recommendation)],
            self.recommendView_7,
            historical_topping_recommendation
        )
        self.recommendView_7.mousePressEvent = lambda event: self.recommend_item_click_event(event, historical_recommendation, historical_topping_recommendation)
        
        self.process_emotion()
        
        # 재고 기반 맛 추천(가장 많이 팔림)
        inventory_flavor, inventory_topping = self.recommend_by_inventory()
        self.add_image_to_graphics_view(
            ice_cream_images[self.flavors.index(inventory_flavor)],
            self.recommendView_4,
            inventory_flavor
        )
        self.recommendView_4.mousePressEvent = lambda event: self.recommend_item_click_event(event, inventory_flavor, inventory_topping)
        
        self.add_image_to_graphics_view(
            topping_images[self.topping_flavors.index(inventory_topping)],
            self.recommendView_8,
            inventory_topping
        )
        self.recommendView_8.mousePressEvent = lambda event: self.recommend_item_click_event(event, inventory_flavor, inventory_topping)

    def set_emotion(self, emotion):
        """감정 정보 설정 메서드"""
        self.current_emotion = emotion

    def process_emotion(self):
        """감정에 따른 처리 메서드"""
        emotion_ice, emotion_topping = self.get_emotion_recommendations(self.current_emotion)
        self.add_image_to_graphics_view(
            ice_cream_images[self.flavors.index(emotion_ice)],
            self.recommendView_2,
            emotion_ice
        )
        self.recommendView_2.mousePressEvent = lambda event: self.recommend_item_click_event(event, emotion_ice, emotion_topping)
        
        self.add_image_to_graphics_view(
            topping_images[self.topping_flavors.index(emotion_topping)],
            self.recommendView_6,
            emotion_topping
        )
        self.recommendView_6.mousePressEvent = lambda event: self.recommend_item_click_event(event, emotion_ice, emotion_topping)

    def get_emotion_recommendations(self, emotion):
        """감정에 따라 추천 아이스크림 및 토핑 반환 메서드"""
        emotion_ice = 'choco'
        emotion_topping = 'topping1'

        if emotion == 'angry':
            emotion_ice = self.flavors[1]  # '바닐라 아이스크림'
            emotion_topping = self.topping_flavors[0]  # 'topping1'
        elif emotion == 'sad':
            emotion_ice = self.flavors[0]  # '초코 아이스크림'
            emotion_topping = self.topping_flavors[1]  # 'topping2'
        elif emotion == 'happy':
            emotion_ice = self.flavors[2]  # '딸기맛 아이스크림'
            emotion_topping = self.topping_flavors[2]  # 'topping3'

        return emotion_ice, emotion_topping
    
    def recommend_by_inventory(self):
        try:
            conn = pymysql.connect(**self.db_config)
            with conn.cursor() as cursor:
                # 가장 최근 레코드를 가져오는 쿼리
                query = """
                SELECT flavor1, flavor2, flavor3, topping1, topping2, topping3
                FROM inventory_management_table
                ORDER BY date_time DESC
                LIMIT 1
                """
                cursor.execute(query)
                result = cursor.fetchone()
                if result:
                    flavors_count = np.array([result['flavor1'], result['flavor2'], result['flavor3']])
                    toppings_count = np.array([result['topping1'], result['topping2'], result['topping3']])
                    
                    # 가장 재고가 많은 아이스크림 맛 찾기
                    max_flavor_index = np.argmax(flavors_count)
                    flavors = ['choco', 'vanila', 'strawberry']
                    inventory_ice = flavors[max_flavor_index]
                    
                    # 가장 재고가 많은 토핑 찾기
                    max_topping_index = np.argmax(toppings_count)
                    toppings = ['topping1', 'topping2', 'topping3']
                    inventory_topping = toppings[max_topping_index]

                    return inventory_ice, inventory_topping
                else:
                    QMessageBox.warning(self, "재고 정보 없음", "재고 정보를 가져올 수 없습니다.")
                    return 'choco', 'topping1'  # 기본값으로 'choco'와 'topping1' 반환
                    
        except pymysql.MySQLError as err:
            print(f"데이터베이스 오류 발생: {err}")
            return 'choco', 'topping1'  # 기본값으로 'choco'와 'topping1' 반환
        finally:
            if 'conn' in locals():
                conn.close()

    def recommend_based_on_history(self, user_id):
        try:
            conn = pymysql.connect(**self.db_config)
            with conn.cursor() as cursor:
                query = """
                SELECT choco_count, vanila_count, strawberry_count, topping1_count, topping2_count, topping3_count
                FROM purchase_record_table
                WHERE user_id = %s
                """
                cursor.execute(query, (user_id,))
                result = cursor.fetchone()
                if result:
                    ice_cream_counts = np.array([
                        result['choco_count'],
                        result['vanila_count'],
                        result['strawberry_count']
                    ])
                    topping_counts = np.array([
                        result['topping1_count'],
                        result['topping2_count'],
                        result['topping3_count']
                    ])
                    
                    # 가장 많이 주문한 아이스크림 맛 찾기
                    max_ice_cream_index = np.argmax(ice_cream_counts)
                    ice_cream_flavors = ['choco', 'vanila', 'strawberry']
                    recommended_ice_cream = ice_cream_flavors[max_ice_cream_index]
                    
                    # 가장 많이 주문한 토핑 찾기
                    max_topping_index = np.argmax(topping_counts)
                    topping_flavors = ['topping1', 'topping2', 'topping3']
                    recommended_topping = topping_flavors[max_topping_index]

                    return recommended_ice_cream, recommended_topping
                else:
                    QMessageBox.warning(self, "기록 없음", "사용자 기록이 없습니다.")
                    return 'choco', 'topping1'  # 기본값으로 'choco'와 'topping1' 반환
        except pymysql.MySQLError as err:
            print(f"데이터베이스 오류 발생: {err}")
            return 'choco', 'topping1'  # 기본값으로 'choco'와 'topping1' 반환
        finally:
            if 'conn' in locals():
                conn.close()

    def get_user_info(self, user_id):
        try:
            conn = pymysql.connect(**self.db_config)
            with conn.cursor() as cursor:
                query = "SELECT gender, birthday, name FROM user_info_table WHERE user_ID = %s"
                cursor.execute(query, (user_id,))
                result = cursor.fetchone()
                if result:
                    gender = result['gender']
                    birthday = result['birthday']
                    name = result['name']
                    if birthday is not None:  # birthday가 None인지 확인
                        age = self.calculate_age(birthday)
                    else:
                        QMessageBox.warning(self, "생년월일 없음", "사용자의 생년월일이 등록되어 있지 않습니다.")
                        return None, gender  # 생년월일이 없으면 나이를 None으로 반환
                    return age, gender, name
                else:
                    QMessageBox.warning(self, "사용자 정보 없음", "등록된 사용자 정보가 없습니다.")
                    return None, None, None
        except pymysql.MySQLError as err:
            print(f"데이터베이스 오류 발생: {err}")
            return None, None, None
        finally:
            if 'conn' in locals():
                conn.close()

    def calculate_age(self, birthdate):
        today = datetime.today()
        age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
        return age
    
    def recommend_flavor(self, age, gender):
        m = np.array([35.6, 38.8, 25.6])  # Male preference
        F = np.array([48.8, 36.8, 16.4])  # Female preference
        y10 = np.array([31, 43, 26])      # Age < 20
        y20 = np.array([51, 27, 22])      # Age < 30
        y30 = np.array([67, 20, 13])      # Age < 40
        y40 = np.array([26, 51, 23])      # Age < 50
        y50 = np.array([31, 48, 21])      # Age >= 50

        if age < 20:
            age_group_value = y10
        elif age < 30:
            age_group_value = y20
        elif age < 40:
            age_group_value = y30
        elif age < 50:
            age_group_value = y40
        else:
            age_group_value = y50

        male_female_value = m if gender == 'Male' else F
        result = male_female_value + age_group_value
        max_index = np.argmax(result)

        flavors = ['choco', 'vanila', 'strawberry']
        return flavors[max_index]
    
    def recommend_topping(self, age, gender):
        m = np.array([30.0, 40.0, 30.0])  # Male preference
        f = np.array([25.0, 50.0, 25.0])  # Female preference
        y10 = np.array([50.0, 30.0, 20.0])  # Age < 20
        y20 = np.array([40.0, 35.0, 25.0])  # Age < 30
        y30 = np.array([30.0, 40.0, 30.0])  # Age < 40
        y40 = np.array([25.0, 50.0, 25.0])  # Age < 50
        y50 = np.array([20.0, 55.0, 25.0])  # Age >= 50

        if age < 20:
            age_group_value = y10
        elif age < 30:
            age_group_value = y20
        elif age < 40:
            age_group_value = y30
        elif age < 50:
            age_group_value = y40
        else:
            age_group_value = y50

        male_female_value = m if gender == 'Male' else f
        result = male_female_value + age_group_value
        max_index = np.argmax(result)

        toppings = ['topping1', 'topping2', 'topping3']
        return toppings[max_index]
    
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


    def item_click_event(self, event, flavor, topping):
        # Set the clicked item count to 1 based on whether it's a flavor or topping
        if flavor in ['choco', 'vanila', 'strawberry']:
            for key in ['choco', 'vanila', 'strawberry']:
                self.item_click_count[key] = 0
            self.item_click_count[flavor] = 1
            selected_topping = next((key for key, value in self.item_click_count.items() if key in ['topping1', 'topping2', 'topping3'] and value == 1), None)
            self.update_list_view(selected_flavor=flavor, selected_topping=selected_topping)  # Update list view with selected flavor
        elif topping in ['topping1', 'topping2', 'topping3']:
            for key in ['topping1', 'topping2', 'topping3']:
                self.item_click_count[key] = 0
            self.item_click_count[topping] = 1
            selected_flavor = next((key for key, value in self.item_click_count.items() if key in ['choco', 'vanila', 'strawberry'] and value == 1), None)
            self.update_list_view(selected_flavor=selected_flavor, selected_topping=topping)  # Update list view with selected topping

    def recommend_item_click_event(self, event, flavor=None, topping=None):
        # 이전 선택을 초기화합니다.
        for key in ['choco', 'vanila', 'strawberry']:
            self.item_click_count[key] = 0
        for key in ['topping1', 'topping2', 'topping3']:
            self.item_click_count[key] = 0

        # 새로 클릭된 아이템을 업데이트합니다.
        if flavor in ['choco', 'vanila', 'strawberry']:
            self.item_click_count[flavor] = 1
        if topping in ['topping1', 'topping2', 'topping3']:
            self.item_click_count[topping] = 1
        
        # 선택된 아이스크림과 토핑을 리스트 뷰에 추가합니다.
        selected_flavor = next((key for key, value in self.item_click_count.items() if key in ['choco', 'vanila', 'strawberry'] and value == 1), None)
        selected_topping = next((key for key, value in self.item_click_count.items() if key in ['topping1', 'topping2', 'topping3'] and value == 1), None)
        
        self.update_list_view(selected_flavor=selected_flavor, selected_topping=selected_topping)



    def update_list_view(self, selected_flavor=None, selected_topping=None):
        item_list = self.list_model.stringList()

        # 기존의 아이템을 모두 제거
        item_list.clear()

        # 선택된 아이스크림 추가
        if selected_flavor:
            item_list.append(selected_flavor)

        # 선택된 토핑 추가
        if selected_topping:
            item_list.append(selected_topping)

        # 모델을 업데이트
        self.list_model.setStringList(item_list)
    
    def closeEvent(self, event):
        event.accept()
        gui_windows = QApplication.allWidgets()
        main_windows = [win for win in gui_windows if isinstance(win, (ConfirmWindow)) and win.isVisible()]
        if not main_windows:
            self.main.home()