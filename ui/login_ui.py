# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'login.ui'
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
        MainWindow.resize(1024, 768)
        MainWindow.setStyleSheet(u"")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.memberBtn = QPushButton(self.centralwidget)
        self.memberBtn.setObjectName(u"memberBtn")
        self.memberBtn.setGeometry(QRect(380, 620, 251, 71))
        font = QFont()
        font.setPointSize(13)
        self.memberBtn.setFont(font)
        self.graphicsView = QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName(u"graphicsView")
        self.graphicsView.setGeometry(QRect(170, 30, 661, 551))
        self.mileagebtn = QPushButton(self.centralwidget)
        self.mileagebtn.setObjectName(u"mileagebtn")
        self.mileagebtn.setGeometry(QRect(20, 267, 80, 31))
        self.orderbtn = QPushButton(self.centralwidget)
        self.orderbtn.setObjectName(u"orderbtn")
        self.orderbtn.setGeometry(QRect(20, 230, 81, 31))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1024, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"LoginWindow", None))
        self.memberBtn.setText("")
        self.mileagebtn.setText(QCoreApplication.translate("MainWindow", u"\ud68c\uc6d0 \uc801\ub9bd", None))
        self.orderbtn.setText(QCoreApplication.translate("MainWindow", u"\ube44\ud68c\uc6d0 \uc8fc\ubb38", None))
    # retranslateUi

