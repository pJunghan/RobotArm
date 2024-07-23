# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'check_login.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_check_login(object):
    def setupUi(self, check_login):
        if not check_login.objectName():
            check_login.setObjectName(u"check_login")
        check_login.resize(430, 590)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(check_login.sizePolicy().hasHeightForWidth())
        check_login.setSizePolicy(sizePolicy)
        check_login.setStyleSheet(u"")
        self.gridLayout = QGridLayout(check_login)
        self.gridLayout.setObjectName(u"gridLayout")
        self.frame = QFrame(check_login)
        self.frame.setObjectName(u"frame")
        self.frame.setStyleSheet(u"background-color: rgb(255, 255, 255);")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.label = QLabel(self.frame)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(140, 10, 131, 31))
        self.label.setStyleSheet(u"")
        self.user_photo = QGraphicsView(self.frame)
        self.user_photo.setObjectName(u"user_photo")
        self.user_photo.setGeometry(QRect(80, 50, 256, 221))
        self.user_photo.setStyleSheet(u"selection-color: rgb(53, 132, 228);")
        self.Name = QTextBrowser(self.frame)
        self.Name.setObjectName(u"Name")
        self.Name.setGeometry(QRect(80, 310, 251, 41))
        self.Name.setStyleSheet(u"")
        self.Birth = QTextBrowser(self.frame)
        self.Birth.setObjectName(u"Birth")
        self.Birth.setGeometry(QRect(80, 390, 251, 41))
        self.Birth.setStyleSheet(u"")
        self.yesBtn = QPushButton(self.frame)
        self.yesBtn.setObjectName(u"yesBtn")
        self.yesBtn.setGeometry(QRect(80, 470, 90, 50))
        self.yesBtn.setStyleSheet(u"")
        self.noBtn = QPushButton(self.frame)
        self.noBtn.setObjectName(u"noBtn")
        self.noBtn.setGeometry(QRect(250, 470, 90, 50))
        self.label_2 = QLabel(self.frame)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(80, 290, 67, 17))
        self.label_3 = QLabel(self.frame)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(80, 370, 67, 17))

        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)


        self.retranslateUi(check_login)

        QMetaObject.connectSlotsByName(check_login)
    # setupUi

    def retranslateUi(self, check_login):
        check_login.setWindowTitle(QCoreApplication.translate("check_login", u"Dialog", None))
        self.label.setText(QCoreApplication.translate("check_login", u"<html><head/><body><p align=\"center\"><span style=\" font-size:16pt; font-weight:600;\">\ud68c\uc6d0 \ud655\uc778</span></p></body></html>", None))
        self.Name.setHtml(QCoreApplication.translate("check_login", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>", None))
        self.Birth.setHtml(QCoreApplication.translate("check_login", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>", None))
        self.yesBtn.setText(QCoreApplication.translate("check_login", u"\uc608", None))
        self.noBtn.setText(QCoreApplication.translate("check_login", u"\uc544\ub2c8\uc694", None))
        self.label_2.setText(QCoreApplication.translate("check_login", u"\uc774\ub984", None))
        self.label_3.setText(QCoreApplication.translate("check_login", u"\uc5f0\ub77d\ucc98", None))
    # retranslateUi

