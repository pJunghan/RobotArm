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


db_config = {
    'host': 'localhost',
    'user': 'junghan',
    'password': '6488',
    'database': 'order_db',
    'charset': 'utf8',
    'cursorclass': pymysql.cursors.DictCursor,
}
