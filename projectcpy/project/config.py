import pymysql
import os

# UI 파일 경로
ui_base_path = "ui"
main_ui_path = os.path.join(ui_base_path, "main.ui")
login_ui_path = os.path.join(ui_base_path, "login.ui")
menu_ui_path = os.path.join(ui_base_path, "ice_cream_window2.ui")
new_account_ui_path = os.path.join(ui_base_path, "signup.ui")
kiosk_ui_path = os.path.join(ui_base_path, "signPhoto.ui")
confirm_ui_path = os.path.join(ui_base_path, "purchase.ui")
manager_ui_path = os.path.join(ui_base_path, "manager.ui")
order_manage_ui_path = os.path.join(ui_base_path,"manager.ui")
self_manage_ui_path = os.path.join(ui_base_path, "self_manage.ui")

check_ui_path = os.path.join(ui_base_path, "check_login.ui")
check_account_ui_path = os.path.join(ui_base_path, "check_account.ui")

model_file_path = "projectcpy/project/linear_regression_model.pkl"
user_img_path = "projectcpy/user_pic"

# 데이터베이스 연결 설정
# db_config = {
#     'host': 'localhost',      # 데이터베이스 호스트 주소
#     'user': 'junghan',        # 사용자 이름
#     'password': '6488',       # 비밀번호
#     'database': 'order_db',   # 데이터베이스 이름
#     'charset': 'utf8',        # 문자셋 설정
#     'cursorclass': pymysql.cursors.DictCursor  # 결과를 딕셔너리 형태로 반환하는 커서 설정
# }

db_config = {
    'user': 'junghan',
    'password': '6488',
    'host': '172.30.1.29',  # 변경된 IP 주소
    'database': 'order_db',
    'charset': 'utf8',
    'cursorclass': pymysql.cursors.DictCursor
}



# db_config = {
#     'host': '172.30.1.53',      # 데이터베이스 호스트 주소
#     'user': 'user3',        # 사용자 이름
#     'password': 'test1234',       # 비밀번호
#     'database': 'order_db',   # 데이터베이스 이름
#     'charset': 'utf8',        # 문자셋 설정
#     'cursorclass': pymysql.cursors.DictCursor  # 결과를 딕셔너리 형태로 반환하는 커서 설정
# }

ice_cream_images = [
            "flavor/choco.jpeg",
            "flavor/vanila.jpeg",
            "flavor/strawberry.jpeg"
        ]

topping_images = [
            "flavor/topping1.jpeg",
            "flavor/topping2.jpeg",
            "flavor/topping3.jpeg"
        ]