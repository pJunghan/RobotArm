import sys
import os
import cv2
import pymysql
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt, QStringListModel
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QListView
from datetime import datetime, timedelta
from config import confirm_ui_path

MAX_STOCK = 200
ICECREAMPRICE = 3000
TOPPINGPRICE = 1000

class ConfirmWindow(QMainWindow):
    def __init__(self, db_config, item_click_count, purchase_list, main):
        super().__init__()
        uic.loadUi(confirm_ui_path, self)  # UI 파일 로드
        self.main = main
        self.used_points = 0  # 초기화
        self.db_config = db_config
        self.item_click_count = item_click_count
        
        # Initialize the model and set the list of purchased items
        self.list_model = QStringListModel()
        self.PurchaseList.setModel(self.list_model)

        # Update the list view with the passed purchase list
        self.list_model.setStringList(purchase_list)

        self.UsePointBtn.clicked.connect(self.use_points)
        self.PurchaseBtn.clicked.connect(self.confirm_purchase)

        # 가장 최근에 수정된 사용자의 user_ID 가져오기
        self.get_latest_user_info()
    def load_purchase_record(self):
        try:
            conn = pymysql.connect(**self.db_config)
            with conn.cursor() as cursor:
                for item_name in self.item_click_count:
                    query = f"SELECT {item_name}_count FROM purchase_record_table WHERE user_id = %s"
                    cursor.execute(query, (self.user_id,))
                    result = cursor.fetchone()
                    if result:
                        self.item_click_count[item_name] = result[f"{item_name}_count"]
        except pymysql.MySQLError as err:
            print(f"데이터베이스 오류 발생: {err}")
        finally:
            if 'conn' in locals():
                conn.close()

    def update_ui(self):
        self.total_price = self.calculate_total_price()
        # 구매 목록 업데이트
        self.list_model.setStringList(self.get_purchase_list())

        self.CurrentPoint.setPlainText(str(self.current_points))
        self.TotalPrice.setPlainText(f"{self.total_price}원")
        self.UsedPoint.setPlainText("0")
        self.DiscountPrice.setPlainText("0원")

    def get_purchase_list(self):
        purchase_list = []
        for item_name, count in self.item_click_count.items():
            if count > 0:
                if item_name in ['choco', 'vanila', 'strawberry']:
                    price = ICECREAMPRICE * count
                else:
                    price = TOPPINGPRICE * count
                purchase_list.append(f"{item_name}: {count}개 - {price}원")
        return purchase_list

    def get_latest_user_info(self):
        try:
            conn = pymysql.connect(**self.db_config)
            with conn.cursor() as cursor:
                query = "SELECT user_ID, point, name FROM user_info_table ORDER BY last_modified DESC LIMIT 1"
                cursor.execute(query)
                result = cursor.fetchone()
                if result:
                    self.user_id = result['user_ID']
                    self.current_points = result['point']
                    user_name = result['name']
                    if user_name.startswith('undefined'):
                        QMessageBox.information(self, "비회원 사용자", f"비회원 사용자입니다. 누적 포인트가 {self.current_points}점 있습니다.")
                    else:
                        QMessageBox.information(self, "기존 사용자", f"기존 사용자입니다. 누적 포인트가 {self.current_points}점 있습니다.")
                    self.load_purchase_record()
                    self.update_ui()
                else:
                    # 사용자가 존재하지 않는 경우
                    self.user_id = None
                    self.current_points = 0
                    QMessageBox.warning(self, "사용자 정보 없음", "등록된 사용자 정보가 없습니다.")
        except pymysql.MySQLError as err:
            print(f"데이터베이스 오류 발생: {err}")
        finally:
            if 'conn' in locals():
                conn.close()

    
    def calculate_total_price(self):
        self.ice_cream_count = self.item_click_count['choco'] + self.item_click_count['vanila'] + self.item_click_count['strawberry']
        self.topping_count = self.item_click_count['topping1'] + self.item_click_count['topping2'] + self.item_click_count['topping3']
        return ICECREAMPRICE * self.ice_cream_count + TOPPINGPRICE * self.topping_count

    def use_points(self):
        if self.current_points >= 10:
            self.total_price = max(0, self.total_price - 3000)  # 10 points = 3000원 discount
            self.current_points -= 10
            self.used_points += 10

            self.CurrentPoint.setPlainText(str(self.current_points))
            self.TotalPrice.setPlainText(f"{self.total_price}원")
            self.UsedPoint.setPlainText(str(self.used_points))
            self.DiscountPrice.setPlainText(f"{self.used_points // 10 * 3000}원")

            try:
                conn = pymysql.connect(**self.db_config)
                with conn.cursor() as cursor:
                    cursor.execute("UPDATE user_info_table SET point = %s WHERE user_ID = %s", (self.current_points, self.user_id))
                conn.commit()
            except pymysql.MySQLError as err:
                print(f"데이터베이스 오류 발생: {err}")
                QMessageBox.warning(self, "오류", f"데이터베이스 오류 발생: {err}")
            finally:
                if 'conn' in locals():
                    conn.close()

        else:
            QMessageBox.warning(self, "포인트 부족", "포인트가 부족합니다. 포인트는 10점부터 사용 가능합니다.")


    def confirm_purchase(self):
        try:
            conn = pymysql.connect(**self.db_config)
            with conn.cursor() as cursor:
                # 구매 기록 테이블에서 현재 사용자의 아이스크림 및 토핑 개수를 불러옴
                cursor.execute("""
                    SELECT choco, vanila, strawberry, choco_count, vanila_count, strawberry_count, topping1, topping2, topping3, topping1_count, topping2_count, topping3_count
                    FROM purchase_record_table
                    WHERE user_id = %s
                """, (self.user_id,))
                result = cursor.fetchone()

                if result:
                    # 기존 개수를 불러옴
                    choco_count = result['choco_count']
                    vanila_count = result['vanila_count']
                    strawberry_count = result['strawberry_count']
                    topping1_count = result['topping1_count']
                    topping2_count = result['topping2_count']
                    topping3_count = result['topping3_count']

                    current_date = datetime.now().date()

                    # 재고 업데이트 또는 추가
                    cursor.execute("SELECT * FROM inventory_management_table WHERE DATE(date_time) = %s", (current_date,))
                    inventory_result = cursor.fetchone()

                    if inventory_result:
                        # 데이터가 존재할 경우 업데이트
                        inventory_query = """
                            UPDATE inventory_management_table
                            SET 
                                flavor1 = flavor1 + %s,
                                flavor2 = flavor2 + %s,
                                flavor3 = flavor3 + %s,
                                topping1 = topping1 + %s,
                                topping2 = topping2 + %s,
                                topping3 = topping3 + %s,
                                flavor1_status = %s,
                                flavor2_status = %s,
                                flavor3_status = %s,
                                topping1_status = %s,
                                topping2_status = %s,
                                topping3_status = %s
                            WHERE DATE(date_time) = %s;
                        """
                        cursor.execute(inventory_query, (
                            self.item_click_count['choco'],
                            self.item_click_count['vanila'],
                            self.item_click_count['strawberry'],
                            self.item_click_count['topping1'],
                            self.item_click_count['topping2'],
                            self.item_click_count['topping3'],
                            MAX_STOCK - (inventory_result['flavor1'] + self.item_click_count['choco']),
                            MAX_STOCK - (inventory_result['flavor2'] + self.item_click_count['vanila']),
                            MAX_STOCK - (inventory_result['flavor3'] + self.item_click_count['strawberry']),
                            MAX_STOCK - (inventory_result['topping1'] + self.item_click_count['topping1']),
                            MAX_STOCK - (inventory_result['topping2'] + self.item_click_count['topping2']),
                            MAX_STOCK - (inventory_result['topping3'] + self.item_click_count['topping3']),
                            current_date
                        ))
                    else:
                        # 현재 날짜의 데이터가 없으면 어제 날짜의 데이터를 불러옴
                        yesterday_date = current_date - timedelta(days=1)
                        cursor.execute("SELECT * FROM inventory_management_table WHERE DATE(date_time) = %s", (yesterday_date,))
                        yesterday_inventory_result = cursor.fetchone()

                        if yesterday_inventory_result:
                            # 어제 날짜의 데이터가 있을 경우 새로운 행 추가
                            new_flavor1 = yesterday_inventory_result['flavor1'] + self.item_click_count['choco']
                            new_flavor2 = yesterday_inventory_result['flavor2'] + self.item_click_count['vanila']
                            new_flavor3 = yesterday_inventory_result['flavor3'] + self.item_click_count['strawberry']
                            new_topping1 = yesterday_inventory_result['topping1'] + self.item_click_count['topping1']
                            new_topping2 = yesterday_inventory_result['topping2'] + self.item_click_count['topping2']
                            new_topping3 = yesterday_inventory_result['topping3'] + self.item_click_count['topping3']

                            inventory_query = """
                                INSERT INTO inventory_management_table (date_time, flavor1, flavor2, flavor3, topping1, topping2, topping3, flavor1_status, flavor2_status, flavor3_status, topping1_status, topping2_status, topping3_status)
                                VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                            """
                            cursor.execute(inventory_query, (
                                new_flavor1,
                                new_flavor2,
                                new_flavor3,
                                new_topping1,
                                new_topping2,
                                new_topping3,
                                MAX_STOCK - new_flavor1,
                                MAX_STOCK - new_flavor2,
                                MAX_STOCK - new_flavor3,
                                MAX_STOCK - new_topping1,
                                MAX_STOCK - new_topping2,
                                MAX_STOCK - new_topping3
                            ))
                        else:
                            # 어제 날짜의 데이터가 없을 경우, 초기값으로 새로운 행 추가
                            inventory_query = """
                                INSERT INTO inventory_management_table (date_time, flavor1, flavor2, flavor3, topping1, topping2, topping3, flavor1_status, flavor2_status, flavor3_status, topping1_status, topping2_status, topping3_status)
                                VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                            """
                            cursor.execute(inventory_query, (
                                self.item_click_count['choco'],
                                self.item_click_count['vanila'],
                                self.item_click_count['strawberry'],
                                self.item_click_count['topping1'],
                                self.item_click_count['topping2'],
                                self.item_click_count['topping3'],
                                MAX_STOCK - self.item_click_count['choco'],
                                MAX_STOCK - self.item_click_count['vanila'],
                                MAX_STOCK - self.item_click_count['strawberry'],
                                MAX_STOCK - self.item_click_count['topping1'],
                                MAX_STOCK - self.item_click_count['topping2'],
                                MAX_STOCK - self.item_click_count['topping3']
                            ))

                    # 사용자의 포인트 업데이트
                    self.total_cnt = self.ice_cream_count * 2 + self.topping_count
                    self.save_points = self.current_points + self.total_cnt
                    cursor.execute("UPDATE user_info_table SET point = %s WHERE user_ID = %s", (self.save_points, self.user_id))

                    # 구매 기록 업데이트
                    update_query = """
                        UPDATE purchase_record_table
                        SET choco = choco + %s, vanila = vanila + %s, strawberry = strawberry + %s,
                            choco_count = choco_count + %s, vanila_count = vanila_count + %s, strawberry_count = strawberry_count + %s,
                            topping1 = topping1 + %s, topping2 = topping2 + %s, topping3 = topping3 + %s,
                            topping1_count = topping1_count + %s, topping2_count = topping2_count + %s, topping3_count = topping3_count + %s
                        WHERE user_id = %s
                    """
                    cursor.execute(update_query, (
                        self.item_click_count['choco'], self.item_click_count['vanila'], self.item_click_count['strawberry'],
                        self.item_click_count['choco'], self.item_click_count['vanila'], self.item_click_count['strawberry'],
                        self.item_click_count['topping1'], self.item_click_count['topping2'], self.item_click_count['topping3'],
                        self.item_click_count['topping1'], self.item_click_count['topping2'], self.item_click_count['topping3'],
                        self.user_id
                    ))

                    # 트랜잭션 커밋
                    conn.commit()
                    QMessageBox.information(self, "결제 완료", "결제가 성공적으로 완료되었습니다.")
                    self.main.data["topping1"] = self.item_click_count["topping1"]
                    self.main.data["topping2"] = self.item_click_count["topping2"]
                    self.main.data["topping3"] = self.item_click_count["topping3"]
                    self.go_to_main_window()

        except pymysql.MySQLError as err:
            print(f"데이터베이스 오류 발생: {err}")
            QMessageBox.critical(self, "오류", f"데이터베이스 오류 발생: {err}")

        finally:
            if 'conn' in locals() and conn.open:
                conn.close()
                print("데이터베이스 연결을 닫았습니다.")

    def go_to_main_window(self):
        self.main.home()
        self.close()