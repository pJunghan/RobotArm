from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QEvent

menu_form_class = uic.loadUiType("/home/lsm/git_ws/RobotArm/GUI/UI/order_ice_cream2.ui")[0]

class MenuWindow(QMainWindow, menu_form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.Vanila.installEventFilter(self)
        self.Choco.installEventFilter(self)
        self.Strawberry.installEventFilter(self)
        self.topping1.installEventFilter(self)
        self.topping_2.installEventFilter(self)
        self.topping_3.installEventFilter(self)
        self.Home_Button.clicked.connect(self.go_to_main_window)
        self.Next_Button.clicked.connect(self.go_to_next_tab)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Enter:
            obj.setStyleSheet("border: 2px solid blue;")
        elif event.type() == QEvent.Leave:
            obj.setStyleSheet("")
        return super().eventFilter(obj, event)

    def go_to_main_window(self):
        from main_window import MainWindow  # 지연 임포트
        self.main_window = MainWindow()
        self.main_window.show()
        self.close()

    def go_to_next_tab(self):
        current_index = self.tabWidget.currentIndex()
        next_index = (current_index + 1) % self.tabWidget.count()
        self.tabWidget.setCurrentIndex(next_index)
