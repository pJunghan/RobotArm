import pymysql
import os

# UI 파일 경로
ui_base_path = "/home/pjh/dev_ws/EDA/ui"
main_ui_path = os.path.join(ui_base_path, "main.ui")
login_ui_path = os.path.join(ui_base_path, "login.ui")
menu_ui_path = os.path.join(ui_base_path, "order_ice_cream_window.ui")
new_account_ui_path = os.path.join(ui_base_path, "new_account.ui")
kiosk_ui_path = os.path.join(ui_base_path, "kiosk_cam.ui")
confirm_ui_path = os.path.join(ui_base_path, "purchase.ui")

# 데이터베이스 연결 설정
db_config = {
    'host': 'localhost',      # 데이터베이스 호스트 주소
    'user': 'junghan',        # 사용자 이름
    'password': '6488',       # 비밀번호
    'database': 'order_db',   # 데이터베이스 이름
    'charset': 'utf8',        # 문자셋 설정
    'cursorclass': pymysql.cursors.DictCursor  # 결과를 딕셔너리 형태로 반환하는 커서 설정
}
