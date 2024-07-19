import os
from PyQt5 import uic, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QMessageBox
import pymysql
from config import db_config, self_manage_ui_path
from auto_order import OrderManager  # Adjust the import as needed

class SelfOrder(QMainWindow):
    def __init__(self, main):
        super().__init__()
        uic.loadUi(self_manage_ui_path, self)  # Load UI file
        self.show()  # Show the UI window
        self.main = main

        # Set up buttons
        pushButton = self.findChild(QPushButton, 'pushButton')  # Order button
        if pushButton:
            pushButton.clicked.connect(self.submit_order)
        else:
            QMessageBox.critical(self, "Error", "Order button not found.")
        
        backButton = self.findChild(QPushButton, 'pushButton_2')  # Back button
        if backButton:
            backButton.clicked.connect(self.go_back)
        else:
            QMessageBox.critical(self, "Error", "Back button not found.")

    def submit_order(self):
        try:
            # QTextEdit에서 발주량 가져오기
            choco_order_widget = self.findChild(QTextEdit, 'ChocoOrder1')
            vanila_order_widget = self.findChild(QTextEdit, 'VanilaOrder')
            strawberry_order_widget = self.findChild(QTextEdit, 'StrawberryOrder')
            topping1_order_widget = self.findChild(QTextEdit, 'ToppingOrder')
            topping2_order_widget = self.findChild(QTextEdit, 'Topping2Order')
            topping3_order_widget = self.findChild(QTextEdit, 'Topping3Order')

            # 각 위젯이 제대로 로드되었는지 확인
            if not choco_order_widget:
                raise ValueError("초코 주문 위젯을 찾을 수 없습니다.")
            if not vanila_order_widget:
                raise ValueError("바닐라 주문 위젯을 찾을 수 없습니다.")
            if not strawberry_order_widget:
                raise ValueError("딸기 주문 위젯을 찾을 수 없습니다.")
            if not topping1_order_widget:
                raise ValueError("토핑1 주문 위젯을 찾을 수 없습니다.")
            if not topping2_order_widget:
                raise ValueError("토핑2 주문 위젯을 찾을 수 없습니다.")
            if not topping3_order_widget:
                raise ValueError("토핑3 주문 위젯을 찾을 수 없습니다.")

            choco_order = int(choco_order_widget.toPlainText().strip())
            vanila_order = int(vanila_order_widget.toPlainText().strip())
            strawberry_order = int(strawberry_order_widget.toPlainText().strip())
            topping1_order = int(topping1_order_widget.toPlainText().strip())
            topping2_order = int(topping2_order_widget.toPlainText().strip())
            topping3_order = int(topping3_order_widget.toPlainText().strip())

            # MySQL 데이터베이스 연결
            conn = pymysql.connect(**db_config)
            with conn.cursor() as cursor:
                # INSERT 쿼리 실행
                insert_query = """
                INSERT INTO order_management_table (choco_order, vanila_order, strawberry_order, topping1_order, topping2_order, topping3_order)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(insert_query, (choco_order, vanila_order, strawberry_order, topping1_order, topping2_order, topping3_order))
                conn.commit()

            # 성공 메시지 표시
            QMessageBox.information(self, "성공", "주문이 성공적으로 저장되었습니다.")
        except Exception as e:
            # 오류 메시지 표시
            QMessageBox.critical(self, "오류", f"주문 저장 중 오류가 발생했습니다: {e}")
        finally:
            if conn:
                conn.close()
    
    def go_back(self):
        self.close()  # Hide current window
        self.auto_order_window = OrderManager(self.main)  # Return to OrderManager
        self.auto_order_window.show() 

    def closeEvent(self, event):
        self.go_back()