from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUi
from main_window import MainWindow

class OrderIceCreamWindow(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        loadUi('../ui/order_ice_cream_window.ui', self)
        
        self.main_window = main_window
        self.Home_Button.clicked.connect(self.go_home)
    
    def go_home(self):
        if self.main_window:
            self.main_window.show()
        self.close()
