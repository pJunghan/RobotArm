# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ice_cream_window2.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1172, 722)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(10, 40, 1156, 471))
        self.tabWidget.setStyleSheet(u"")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.recommendView_1 = QGraphicsView(self.tab)
        self.recommendView_1.setObjectName(u"recommendView_1")
        self.recommendView_1.setGeometry(QRect(20, 20, 256, 191))
        self.recommendView_6 = QGraphicsView(self.tab)
        self.recommendView_6.setObjectName(u"recommendView_6")
        self.recommendView_6.setGeometry(QRect(300, 230, 256, 192))
        self.recommendView_2 = QGraphicsView(self.tab)
        self.recommendView_2.setObjectName(u"recommendView_2")
        self.recommendView_2.setGeometry(QRect(300, 20, 256, 192))
        self.recommendView_7 = QGraphicsView(self.tab)
        self.recommendView_7.setObjectName(u"recommendView_7")
        self.recommendView_7.setGeometry(QRect(580, 230, 256, 192))
        self.recommendView_3 = QGraphicsView(self.tab)
        self.recommendView_3.setObjectName(u"recommendView_3")
        self.recommendView_3.setGeometry(QRect(580, 20, 256, 192))
        self.recommendView_8 = QGraphicsView(self.tab)
        self.recommendView_8.setObjectName(u"recommendView_8")
        self.recommendView_8.setGeometry(QRect(870, 230, 256, 192))
        self.recommendView_4 = QGraphicsView(self.tab)
        self.recommendView_4.setObjectName(u"recommendView_4")
        self.recommendView_4.setGeometry(QRect(870, 20, 256, 192))
        self.recommendView_5 = QGraphicsView(self.tab)
        self.recommendView_5.setObjectName(u"recommendView_5")
        self.recommendView_5.setGeometry(QRect(20, 230, 256, 191))
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.tab_2.setMinimumSize(QSize(827, 0))
        self.gridLayoutWidget_2 = QWidget(self.tab_2)
        self.gridLayoutWidget_2.setObjectName(u"gridLayoutWidget_2")
        self.gridLayoutWidget_2.setGeometry(QRect(1133, 9, 16, 16))
        self.gridLayout_2 = QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayoutWidget_3 = QWidget(self.tab_2)
        self.gridLayoutWidget_3.setObjectName(u"gridLayoutWidget_3")
        self.gridLayoutWidget_3.setGeometry(QRect(1141, 9, 16, 16))
        self.gridLayout_3 = QGridLayout(self.gridLayoutWidget_3)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.graphicsView_3 = QGraphicsView(self.tab_2)
        self.graphicsView_3.setObjectName(u"graphicsView_3")
        self.graphicsView_3.setGeometry(QRect(800, 22, 321, 192))
        self.graphicsView_6 = QGraphicsView(self.tab_2)
        self.graphicsView_6.setObjectName(u"graphicsView_6")
        self.graphicsView_6.setGeometry(QRect(800, 220, 321, 192))
        self.graphicsView_5 = QGraphicsView(self.tab_2)
        self.graphicsView_5.setObjectName(u"graphicsView_5")
        self.graphicsView_5.setGeometry(QRect(420, 220, 321, 192))
        self.graphicsView_2 = QGraphicsView(self.tab_2)
        self.graphicsView_2.setObjectName(u"graphicsView_2")
        self.graphicsView_2.setGeometry(QRect(420, 20, 321, 192))
        self.graphicsView_4 = QGraphicsView(self.tab_2)
        self.graphicsView_4.setObjectName(u"graphicsView_4")
        self.graphicsView_4.setGeometry(QRect(40, 218, 321, 192))
        self.graphicsView_1 = QGraphicsView(self.tab_2)
        self.graphicsView_1.setObjectName(u"graphicsView_1")
        self.graphicsView_1.setGeometry(QRect(40, 20, 321, 192))
        self.tabWidget.addTab(self.tab_2, "")
        self.listView = QListView(self.centralwidget)
        self.listView.setObjectName(u"listView")
        self.listView.setGeometry(QRect(30, 550, 717, 131))
        self.Next_Button = QPushButton(self.centralwidget)
        self.Next_Button.setObjectName(u"Next_Button")
        self.Next_Button.setGeometry(QRect(910, 560, 139, 91))
        self.Home_Button = QPushButton(self.centralwidget)
        self.Home_Button.setObjectName(u"Home_Button")
        self.Home_Button.setGeometry(QRect(760, 560, 138, 91))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1172, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("MainWindow", u"\ucd94\ucc9c \uba54\ub274", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("MainWindow", u"\uc804\uccb4 \uba54\ub274", None))
        self.Next_Button.setText(QCoreApplication.translate("MainWindow", u"\uacb0\uc81c", None))
        self.Home_Button.setText(QCoreApplication.translate("MainWindow", u"\ud648", None))
    # retranslateUi

