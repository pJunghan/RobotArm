import requests
import json
import math
import pickle
import pymysql
from config import db_config, order_manage_ui_path, model_file_path
from main_window import MainWindow  # Assuming you have this import for the main window
from datetime import datetime
from PyQt5 import uic, QtCore
from PyQt5.QtCore import QStringListModel
from PyQt5.QtWidgets import QApplication, QMainWindow, QListView, QTextBrowser, QPushButton

class OrderManager(QMainWindow):
    def __init__(self,main):
        super().__init__()
        uic.loadUi(order_manage_ui_path, self)  # UI 파일 로드
        self.show()  # UI 창 보여주기
        self.main = main
        city = "Seoul"
        apikey = "c1f6c98d79482985f66c26c037fe666a"
        
        self.setup_weather(city, apikey)
        self.setup_buttons()

    def setup_weather(self, city, apikey):
        try:
            data = self.fetch_weather_data(city, apikey)

            if data["cod"] == 200:
                temperature = data['main']['temp']
                weather_description = data['weather'][0]['description']
                predicted_sales = self.predict_sales(temperature)
                adjusted_sales = self.adjust_sales_by_weekday(predicted_sales)

                weather_info = f"{weather_description}: {temperature}°C"
                weather_browser = self.findChild(QTextBrowser, 'textBrowser_2')  # UI에서 textBrowser_2 찾기
                weather_browser.setText(weather_info)  # 날씨 정보를 텍스트 브라우저에 설정

                conn = pymysql.connect(**db_config)
                with conn.cursor() as cursor:
                    select_query = "SELECT * FROM inventory_management_table WHERE DATE(date_time) = CURDATE()"
                    cursor.execute(select_query)
                    inventory_result = cursor.fetchone()

                    if inventory_result:
                        self.update_inventory_status(inventory_result, adjusted_sales)
        
        except requests.exceptions.RequestException as e:
            print(f"API 요청 실패: {e}")
        except Exception as e:
            print(f"오류 발생: {e}")

    def fetch_weather_data(self, city, apikey):
        api = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={apikey}&lang=kr&units=metric"
        result = requests.get(api)
        result.raise_for_status()
        return json.loads(result.text)

    def predict_sales(self, temperature):
        with open(model_file_path, 'rb') as file:
            loaded_model = pickle.load(file)
        x_test = [[temperature]]
        predictions = loaded_model.predict(x_test)
        return predictions[0][0] / 58.11

    def adjust_sales_by_weekday(self, predicted_sales):
        current_weekday = datetime.now().weekday()
        if current_weekday in [0, 1, 2, 3]:  # 월~목
            return predicted_sales
        elif current_weekday in [4, 5]:  # 금, 토
            return predicted_sales * 2
        elif current_weekday == 6:  # 일요일
            return predicted_sales * 1.5

    def update_inventory_status(self, inventory_result, adjusted_sales):
        # 재고 상태 정보를 listView에 추가
        status_items = [
            f"Flavor 1 : {inventory_result['flavor1_status']}",
            f"Flavor 2 : {inventory_result['flavor2_status']}",
            f"Flavor 3 : {inventory_result['flavor3_status']}",
            f"Topping 1 : {inventory_result['topping1_status']}",
            f"Topping 2 : {inventory_result['topping2_status']}",
            f"Topping 3 : {inventory_result['topping3_status']}"
        ]

        listView = self.findChild(QListView, 'listView')  # UI에서 listView 찾기
        status_model = QStringListModel()
        status_model.setStringList(status_items)  # 리스트 모델 설정
        listView.setModel(status_model)  # ListView에 모델 설정

        # 판매량 정보를 listView_2에 추가
        sales_items = [
            f"Flavor 1: {inventory_result['flavor1'] }",
            f"Flavor 2: {inventory_result['flavor2']}",
            f"Flavor 3: {inventory_result['flavor3'] }",
            f"Topping 1: {inventory_result['topping1'] }",
            f"Topping 2: {inventory_result['topping2'] }",
            f"Topping 3: {inventory_result['topping3'] }"
        ]

        listView_2 = self.findChild(QListView, 'listView_2')  # UI에서 listView_2 찾기
        sales_model = QStringListModel()
        sales_model.setStringList(sales_items)  # 리스트 모델 설정
        listView_2.setModel(sales_model)  # ListView에 모델 설정

        flavors = ['flavor1', 'flavor2', 'flavor3']
        toppings = ['topping1', 'topping2', 'topping3']
        flavor_names = ['Choco', 'Vanila', 'Strawberry']
        topping_names = ['Topping1', 'Topping2', 'Topping3']

        # Flavor 주문량 설정
        for i, flavor in enumerate(flavors):
            order_browser = self.findChild(QTextBrowser, f'{flavor_names[i]}Order')
            order_amount = math.ceil(inventory_result[flavor] * adjusted_sales)  # 올림 처리
            order_browser.setText(f"{flavor_names[i]} Order: {order_amount}")

        # Topping 주문량 설정
        for i, topping in enumerate(toppings):
            order_browser = self.findChild(QTextBrowser, f'{topping_names[i]}Order')
            order_amount = math.ceil(inventory_result[topping] * adjusted_sales)  # 올림 처리
            print(f"{topping_names[i]} amount: {order_amount}")  # 디버깅 로그
            order_browser.setText(f"{topping_names[i]} Order: {order_amount}")

    def setup_buttons(self):
        pushButton = self.findChild(QPushButton, 'pushButton')  # 주문 제출 버튼
        pushButton.clicked.connect(self.submit_order)  # 클릭 시 submit_order 호출

        pushButton_2 = self.findChild(QPushButton, 'pushButton_2')  # UI에서 pushButton_2 찾기
        pushButton_2.clicked.connect(self.go_to_main_window)

        pushButton_3 = self.findChild(QPushButton, 'pushButton_3')  # UI에서 pushButton_3 찾기
        pushButton_3.clicked.connect(self.go_to_self_order)  # 클릭 시 self_order로 이동


    def submit_order(self):
        try:
            # 주문량 가져오기
            choco_order = int(self.findChild(QTextBrowser, 'ChocoOrder').toPlainText().split(": ")[1])
            vanila_order = int(self.findChild(QTextBrowser, 'VanilaOrder').toPlainText().split(": ")[1])
            strawberry_order = int(self.findChild(QTextBrowser, 'StrawberryOrder').toPlainText().split(": ")[1])
            topping1_order = int(self.findChild(QTextBrowser, 'Topping1Order').toPlainText().split(": ")[1])
            topping2_order = int(self.findChild(QTextBrowser, 'Topping2Order').toPlainText().split(": ")[1])
            topping3_order = int(self.findChild(QTextBrowser, 'Topping3Order').toPlainText().split(": ")[1])

            weather_info = self.findChild(QTextBrowser, 'textBrowser_2').toPlainText()

            # 데이터베이스에 저장
            conn = pymysql.connect(**db_config)
            with conn.cursor() as cursor:
                insert_query = """
                    INSERT INTO order_management_table 
                    (date_time, choco_order, vanila_order, strawberry_order, topping1_order, topping2_order, topping3_order, weather) 
                    VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(insert_query, (choco_order, vanila_order, strawberry_order, topping1_order, topping2_order, topping3_order, weather_info))
                conn.commit()
            
            print("Order submitted successfully.")  # 디버깅용 메시지

        except Exception as e:
            print(f"Error while submitting order: {e}")  # 에러 메시지 출력


    def go_to_main_window(self,main):
        self.main.home()  # 메인 윈도우 보여주기
        self.close()

    def go_to_self_order(self):
        from self_order import SelfOrder  # Import here to avoid circular import
        self.close()
        self.self_order_window = SelfOrder(self.main)  # Pass main to SelfOrder
        self.self_order_window.show()

    
