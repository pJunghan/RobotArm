import sys
import requests
import json
import math
import pickle
import pymysql
from config import db_config, order_manage_ui_path, model_file_path
from main_window import MainWindow  
from datetime import datetime
from PyQt5 import uic, QtCore, QtGui
from PyQt5.QtCore import QStringListModel, Qt, QTimer
from PyQt5.QtGui import QPixmap, QBrush, QColor, QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QListView, QTextBrowser, QPushButton, QStyledItemDelegate, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsTextItem, QGraphicsLineItem

class CenterAlignDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        option.displayAlignment = Qt.AlignCenter
        super().paint(painter, option, index)

class OrderManager(QMainWindow):
    def __init__(self,main):
        super().__init__()
        uic.loadUi(order_manage_ui_path, self)  # UI 파일 로드
        self.show()  # UI 창 보여주기

        # 화면 크기를 가져와 창의 중앙 위치를 계산
        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

        self.main = main
        city = "Seoul"
        apikey = "c1f6c98d79482985f66c26c037fe666a"
        
        self.setup_weather(city, apikey)
        self.setup_buttons()
        self.display_current_datetime()

        self.setFixedSize(self.size())  # 현재 창 크기로 고정
        self.customize_ui()

        self.start_datetime_timer()  # 타이머 시작


    def customize_ui(self):

        # QPushButton 스타일 설정
        button_style = """
            QPushButton {
                background-color: #77DD77;  /* 파스텔 그린 */
                border: 2px solid #77DD77;
                border-radius: 15px;
                color: white;
                font-size: 16pt;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #66CC66;  /* 호버 시 더 어두운 그린 */
            }
            QPushButton:pressed {
                background-color: #55AA55;  /* 눌렀을 때 더 어두운 그린 */
            }
        """
        self.pushButton.setStyleSheet(button_style)
        self.pushButton_2.setStyleSheet(button_style)
        self.pushButton_3.setStyleSheet(button_style)

        # QTextBrowser 스타일 설정
        top_textBrowser_style = ("""
            QTextBrowser {
                background-color: transparent;
                border: none;
                border-radius: 10px;
                padding: 7px;
                font-weight: normal;
                font-size: 12pt;
                color: rgb(0,0,0);
            }
        """)

        order_textBrowser_style = ("""
            QTextBrowser {
                background-color: transparent;
                border: none;
                border-radius: 10px;
                padding: 7px;
                font-weight: bold;
                font-size: 12pt;
                color: rgb(0,0,0);
            }
        """)
        self.textBrowser.setStyleSheet(top_textBrowser_style)
        self.textBrowser_2.setStyleSheet(top_textBrowser_style)

        self.textBrowser_3.setStyleSheet(order_textBrowser_style)
        self.textBrowser_6.setStyleSheet(order_textBrowser_style)
        self.textBrowser_7.setStyleSheet(order_textBrowser_style)
        self.textBrowser_11.setStyleSheet(order_textBrowser_style)
        self.textBrowser_12.setStyleSheet(order_textBrowser_style)
        self.textBrowser_13.setStyleSheet(order_textBrowser_style)
        self.ChocoOrder.setStyleSheet(order_textBrowser_style)
        self.VanilaOrder.setStyleSheet(order_textBrowser_style)
        self.StrawberryOrder.setStyleSheet(order_textBrowser_style)
        self.Topping1Order.setStyleSheet(order_textBrowser_style)
        self.Topping2Order.setStyleSheet(order_textBrowser_style)
        self.Topping3Order.setStyleSheet(order_textBrowser_style)


        # QListView 스타일 설정
        listView_style = ("""
            QListView {
                background-color: rgb(255,255,255);
                border: 2px solid black;
                border-radius: 10px;
                padding: 7px;
                font-size: 16pt;
                font-weight: bold;
                color: black;
            }
        """)
        self.listView.setStyleSheet(listView_style)
        self.listView.setItemDelegate(CenterAlignDelegate(self.listView))

        self.listView_2.setStyleSheet(listView_style)
        self.listView_2.setItemDelegate(CenterAlignDelegate(self.listView))


        # QGraphicsView 스타일 설정
        weather_graphicsView_style = ("""
            QGraphicsView {
                border: none;
                border-radius: 10px;
                background-color: rgb(255,255,255);
            }
        """)

        graph_graphicsView_style = ("""
            QGraphicsView {
                border: 2px solid black;
                border-radius: 10px;
                background-color: rgb(255,255,255);
            }
        """)

        self.graphicsView.setStyleSheet(weather_graphicsView_style)
        self.graphicsView_sales.setStyleSheet(graph_graphicsView_style)
        self.graphicsView_status.setStyleSheet(graph_graphicsView_style)

        

    def start_datetime_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.display_current_datetime)
        self.timer.start(1000)  # 1초마다 업데이트

    def display_current_datetime(self):
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        datetime_browser = self.findChild(QTextBrowser, 'textBrowser')  # UI에서 textBrowser 찾기
        datetime_browser.setText(f"<div style='text-align: center;'>{current_datetime}")

    def setup_weather(self, city, apikey):
        try:
            data = self.fetch_weather_data(city, apikey)

            icon_id = data["weather"][0]["icon"]
            icon_url = f"http://openweathermap.org/img/wn/{icon_id}@2x.png"
            self.display_weather_icon(icon_url)  # 날씨 아이콘 표시

            if data["cod"] == 200:
                temperature = data['main']['temp']
                weather_description = data['weather'][0]['description']
                predicted_sales = self.predict_sales(temperature)
                adjusted_sales = self.adjust_sales_by_weekday(predicted_sales)

                weather_info = f"서울 현재 날씨: {int(temperature)}°C {weather_description}"
                weather_browser = self.findChild(QTextBrowser, 'textBrowser_2')  # UI에서 textBrowser_2 찾기
                weather_browser.setHtml(f"<div style='text-align: center;'>{weather_info}</div>")  # 날씨 정보를 텍스트 브라우저에 설정

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

    def display_weather_icon(self, icon_url):
        response = requests.get(icon_url)
        response.raise_for_status()
        pixmap = QPixmap()
        pixmap.loadFromData(response.content)

        graphics_view = self.findChild(QGraphicsView, 'graphicsView')  # UI에서 graphicsView 찾기
        if graphics_view:
            # graphicsView의 크기에 맞게 pixmap 리사이즈
            graphics_view_size = graphics_view.size()
            resized_pixmap = pixmap.scaled(graphics_view_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

            scene = QGraphicsScene()
            scene.addPixmap(resized_pixmap)

            graphics_view.setScene(scene)

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
            f"Choco : {inventory_result['flavor1_status']}",
            f"Vanila : {inventory_result['flavor2_status']}",
            f"Strawberry : {inventory_result['flavor3_status']}",
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
            f"Choco: {inventory_result['flavor1']}",
            f"Vanila : {inventory_result['flavor2']}",
            f"Strawberry : {inventory_result['flavor3']}",
            f"Topping 1: {inventory_result['topping1']}",
            f"Topping 2: {inventory_result['topping2']}",
            f"Topping 3: {inventory_result['topping3']}"
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
        
        # 그래픽 뷰에 막대 그래프 그리기
        self.draw_bar_graph('graphicsView_status', status_items, '재고 상태')
        self.draw_bar_graph('graphicsView_sales', sales_items, '판매량')

    def draw_bar_graph(self, view_name, items, title_text):
        scene = QGraphicsScene()

        # items에서 값을 추출
        values = [int(item.split(': ')[1].split(' ')[0]) for item in items]
        max_value = max(values)

        # 그래프의 최대 높이와 너비 설정
        max_height = max_value

        bar_width = 30
        spacing = 10

        # 각 막대의 높이 계산 및 추가
        colors = [QColor("lightcoral"), QColor("lightgreen"), QColor("lightblue"), QColor("lightcyan"), QColor("lightpink"), QColor("lightyellow"),
                QColor("red"), QColor("green"), QColor("blue"), QColor("cyan"), QColor("magenta"), QColor("yellow")]

        for i, value in enumerate(values):
            height = (value / max_value) * max_height
            bar = QGraphicsRectItem(0, i * (bar_width + spacing), height, bar_width)
            bar.setBrush(QBrush(colors[i % len(colors)]))
            scene.addItem(bar)

            # 막대 레이블 추가
            label = QGraphicsTextItem(items[i].split(': ')[0])
            label.setDefaultTextColor(QColor("black"))
            label.setFont(QFont("Arial", 10))
            label.setPos(-70, i * (bar_width + spacing))  # 막대 왼쪽에 레이블 위치 설정
            scene.addItem(label)

            # 막대 값 레이블 추가
            value_label = QGraphicsTextItem(str(value))
            value_label.setDefaultTextColor(QColor("black"))
            value_label.setFont(QFont("Arial", 10))
            value_label.setPos(height + 5, i * (bar_width + spacing))  # 막대 오른쪽에 값 위치 설정
            scene.addItem(value_label)

        # QGraphicsView 찾기 및 확인
        graphics_view = self.findChild(QGraphicsView, view_name)
        if graphics_view is not None:
            # 그래프 제목 추가
            title = QGraphicsTextItem(title_text)
            title.setDefaultTextColor(QColor("black"))
            title.setFont(QFont("Arial", 16))

            # QGraphicsView의 실제 너비를 가져와서 제목의 위치를 조정
            view_width = graphics_view.viewport().width()
            title_width = title.boundingRect().width()
            title.setPos((title_width) / 2, -50)
            scene.addItem(title)

            graphics_view.setScene(scene)
        else:
            print(f"Error: QGraphicsView with name '{view_name}' not found.")


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

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()  # Assuming this is the main window class
    order_manager = OrderManager(main_window)
    order_manager.show()
    sys.exit(app.exec_())
