import sys
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QEvent

# UI 파일 연결
main_form_class = uic.loadUiType("/home/lsm/git_ws/RobotArm/GUI/main.ui")[0]
next_form_class = uic.loadUiType("/home/lsm/git_ws/RobotArm/GUI/order_ice_cream.ui")[0]

# 메인 창 클래스
class MainWindow(QMainWindow, main_form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.orderButton.clicked.connect(self.go_to_next_window)

    def go_to_next_window(self):
        self.next_window = NextWindow()
        self.next_window.show()
        self.close()

# 다음 창 클래스
class NextWindow(QMainWindow, next_form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # graphicsView들에 마우스 이벤트 필터 등록
        self.Vanila.installEventFilter(self)
        self.Choco.installEventFilter(self)
        self.Strawberry.installEventFilter(self)
        self.topping1.installEventFilter(self)
        self.topping_2.installEventFilter(self)
        self.topping_3.installEventFilter(self)
        
        # 홈 버튼과 다음 버튼에 클릭 이벤트 연결
        self.Home_Button.clicked.connect(self.go_to_main_window)
        self.Next_Button.clicked.connect(self.go_to_next_tab)
        
    # 이벤트 필터 메서드
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Enter:
            obj.setStyleSheet("border: 2px solid blue;")  # 마우스가 들어왔을 때 테두리 색상 변경
        elif event.type() == QEvent.Leave:
            obj.setStyleSheet("")  # 마우스가 나갔을 때 테두리 색상 초기화
        return super().eventFilter(obj, event)
    
    # 홈 버튼 클릭 시 메인 창으로 이동하는 메서드
    def go_to_main_window(self):
        self.main_window = MainWindow()
        self.main_window.show()
        self.close()
    
    # 다음 버튼 클릭 시 다음 탭으로 이동하는 메서드
    def go_to_next_tab(self):
        current_index = self.tabWidget.currentIndex()
        next_index = (current_index + 1) % self.tabWidget.count()
        self.tabWidget.setCurrentIndex(next_index)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
