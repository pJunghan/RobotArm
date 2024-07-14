import mysql

# MySQL 데이터베이스 연결
connection = mysql.connector.connect(
    host='localhost',
    user='your_username',
    password='your_password',
    database='order_db'
)

cursor = connection.cursor()
cursor.execute("SELECT * FROM inventory_management_table")
rows = cursor.fetchall()

# 요일별 가중치 설정
day_weights = {
    '금요일': 1.5,
    '토요일': 2.0,
    '일요일': 2.0,
    '평일': 1.0,
}

# 평균 판매량 설정
average_sales = 55  # 예시: 55만원
max_stock = 300     # 예시: 최대 재고 300만원

# 판매 예측 및 재고 관리
for row in rows:
    date_time, flavor1, flavor2, flavor3, topping1, topping2, topping3, flavor1_status, flavor2_status, flavor3_status, topping1_status, topping2_status, topping3_status = row

    # 남은 재고 계산 (예시: flavor1의 남은 수량)
    current_stock = flavor1_status
    
    # 요일 추출 (예: date_time에서 요일 정보 가져오기)
    day_of_week = date_time.strftime('%A')  # 요일을 추출하는 방법 (예: '금요일')
    
    # 예측 판매량 계산
    if day_of_week in day_weights:
        day_weight = day_weights[day_of_week]
    else:
        day_weight = day_weights['평일']  # 기본 가중치

    days_until_order = 3  # 예시: 발주까지 남은 날짜
    forecast_sales = day_weight * days_until_order * average_sales

    # 주문 결정
    if forecast_sales < current_stock:
        order_decision = "주문 안 함"
    elif forecast_sales > max_stock:
        order_decision = "주문을 일찍 함"
    else:
        order_decision = "주문 필요"

    # 결과 출력
    print(f"{date_time}: 예측 판매량 {forecast_sales}만원, 주문 결정: {order_decision}")

# 연결 종료
cursor.close()
connection.close()
