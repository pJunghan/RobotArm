# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import resources_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.setWindowModality(Qt.ApplicationModal)
        MainWindow.resize(1024, 768)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.orderButton = QPushButton(self.centralwidget)
        self.orderButton.setObjectName(u"orderButton")
        self.orderButton.setGeometry(QRect(410, 670, 171, 71))
        self.orderButton.setIconSize(QSize(50, 50))
        self.autoButton = QPushButton(self.centralwidget)
        self.autoButton.setObjectName(u"autoButton")
        self.autoButton.setGeometry(QRect(850, 0, 171, 71))
        self.autoButton.setAutoFillBackground(True)
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setEnabled(True)
        self.label.setGeometry(QRect(0, 0, 1024, 768))
        self.label.setScaledContents(True)
        self.label.setWordWrap(False)
        self.label.setMargin(0)
        self.label.setIndent(-1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.label.raise_()
        self.autoButton.raise_()
        self.orderButton.raise_()
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1024, 22))
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.orderButton.setText(QCoreApplication.translate("MainWindow", u"\uc8fc\ubb38\ud558\uae30", None))
        self.autoButton.setText("")
        self.label.setText("")
    # retranslateUi

