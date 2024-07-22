import pymysql
import os

# UI 파일 경로
ui_base_path = "ui"
gui_img_path = "ui/pic"
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
image_folder = os.path.join("projectcpy/flavor")

model_file_path = "projectcpy/project/linear_regression_model.pkl"
user_img_path = "projectcpy/user_pic"
db_path = "projectcpy/user_pic"  # 데이터베이스 경로
model_path = "projectcpy/project/Asian_emotion_model.h5"  # 감정 인식 모델 경로
age_prototxt = "projectcpy/project/deploy_age.prototxt"  # 나이 예측 모델 prototxt 경로
age_model = "projectcpy/project/age_net.caffemodel"  # 나이 예측 모델 caffemodel 경로
gender_prototxt = "projectcpy/project/deploy_gender.prototxt"  # 성별 예측 모델 prototxt 경로
gender_model = "projectcpy/project/gender_net.caffemodel"  # 성별 예측 모델 caffemodel 경로
tts_account_path = "projectcpy/project/aris-tts-db0d4caef6e0.json" # 구글 클라우드 로그인 계정

db_config = {
    'user': 'junghan',
    'password': '6488',
    'host': '172.30.1.12',  # 변경된 IP 주소
    'database': 'order_db',
    'charset': 'utf8',
    'cursorclass': pymysql.cursors.DictCursor
}

# db_config = {
#     'host': 'localhost',      # 데이터베이스 호스트 주소
#     'user': 'junghan',        # 사용자 이름
#     'password': '6488',       # 비밀번호
#     'database': 'order_db',   # 데이터베이스 이름
#     'charset': 'utf8',        # 문자셋 설정
#     'cursorclass': pymysql.cursors.DictCursor  # 결과를 딕셔너리 형태로 반환하는 커서 설정
# }


# db_config = {
#     'host': '172.30.1.53',      # 데이터베이스 호스트 주소
#     'user': 'user3',        # 사용자 이름
#     'password': 'test1234',       # 비밀번호
#     'database': 'order_db',   # 데이터베이스 이름
#     'charset': 'utf8',        # 문자셋 설정
#     'cursorclass': pymysql.cursors.DictCursor  # 결과를 딕셔너리 형태로 반환하는 커서 설정
# }

# db_config = {
#     'host': 'localhost',      # 데이터베이스 호스트 주소
#     'user': 'root',        # 사용자 이름
#     'password': '8470',       # 비밀번호
#     'database': 'ARIS',   # 데이터베이스 이름
#     'charset': 'utf8',        # 문자셋 설정
#     'cursorclass': pymysql.cursors.DictCursor  # 결과를 딕셔너리 형태로 반환하는 커서 설정
# }

# ice_cream_images = [
#     os.path.join(image_folder, "choco.jpeg"),
#     os.path.join(image_folder, "vanila.jpeg"),
#     os.path.join(image_folder, "strawberry.jpeg")
# ]

ice_cream_images = [
    os.path.join(image_folder, "choco2.png"),
    os.path.join(image_folder, "vanila2.png"),
    os.path.join(image_folder, "strawberry2.png")
]

# 토핑 이미지 경로
topping_images = [
    os.path.join(image_folder, "topping1.png"),
    os.path.join(image_folder, "topping2.png"),
    os.path.join(image_folder, "topping3.png")
]