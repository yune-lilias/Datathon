# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Frame.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Frame(object):
    def setupUi(self, Frame):
        Frame.setObjectName("Frame")
        Frame.resize(720, 640)
        Frame.setMaximumSize(QtCore.QSize(1280, 1280))
        self.label = QtWidgets.QLabel(Frame)
        self.label.setGeometry(QtCore.QRect(160, 20, 421, 71))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Frame)
        self.label_2.setGeometry(QtCore.QRect(110, 120, 111, 41))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Frame)
        self.label_3.setGeometry(QtCore.QRect(310, 120, 111, 41))
        self.label_3.setStyleSheet("")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Frame)
        self.label_4.setGeometry(QtCore.QRect(480, 120, 201, 41))
        self.label_4.setStyleSheet("")
        self.label_4.setObjectName("label_4")
        self.Page2 = QtWidgets.QLineEdit(Frame)
        self.Page2.setGeometry(QtCore.QRect(310, 160, 113, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Page2.setFont(font)
        self.Page2.setObjectName("Page2")
        self.Top_k = QtWidgets.QLineEdit(Frame)
        self.Top_k.setGeometry(QtCore.QRect(520, 160, 113, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Top_k.setFont(font)
        self.Top_k.setObjectName("Top_k")
        self.label_5 = QtWidgets.QLabel(Frame)
        self.label_5.setGeometry(QtCore.QRect(60, 300, 301, 41))
        self.label_5.setObjectName("label_5")
        self.Result = QtWidgets.QLineEdit(Frame)
        self.Result.setGeometry(QtCore.QRect(370, 310, 291, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Result.setFont(font)
        self.Result.setAlignment(QtCore.Qt.AlignCenter)
        self.Result.setObjectName("Result")
        self.label_6 = QtWidgets.QLabel(Frame)
        self.label_6.setGeometry(QtCore.QRect(60, 380, 91, 31))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(Frame)
        self.label_7.setGeometry(QtCore.QRect(60, 470, 81, 31))
        self.label_7.setObjectName("label_7")
        self.Result1 = QtWidgets.QLineEdit(Frame)
        self.Result1.setGeometry(QtCore.QRect(210, 370, 451, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Result1.setFont(font)
        self.Result1.setObjectName("Result1")
        self.Result2 = QtWidgets.QLineEdit(Frame)
        self.Result2.setGeometry(QtCore.QRect(210, 460, 451, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Result2.setFont(font)
        self.Result2.setObjectName("Result2")
        self.Search = QtWidgets.QPushButton(Frame)
        self.Search.setGeometry(QtCore.QRect(225, 220, 81, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Search.setFont(font)
        self.Search.setObjectName("Search")
        self.Cancel = QtWidgets.QPushButton(Frame)
        self.Cancel.setGeometry(QtCore.QRect(430, 220, 81, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Cancel.setFont(font)
        self.Cancel.setObjectName("Cancel")
        self.Page1 = QtWidgets.QLineEdit(Frame)
        self.Page1.setGeometry(QtCore.QRect(110, 160, 113, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Page1.setFont(font)
        self.Page1.setObjectName("Page1")
        self.label_9 = QtWidgets.QLabel(Frame)
        self.label_9.setGeometry(QtCore.QRect(90, 10, 81, 91))
        self.label_9.setText("")
        self.label_9.setPixmap(QtGui.QPixmap(":/logo.jpg"))
        self.label_9.setScaledContents(True)
        self.label_9.setObjectName("label_9")
        self.label_8 = QtWidgets.QLabel(Frame)
        self.label_8.setGeometry(QtCore.QRect(-60, -10, 941, 651))
        self.label_8.setText("")
        self.label_8.setPixmap(QtGui.QPixmap(":/realback.png"))
        self.label_8.setScaledContents(True)
        self.label_8.setObjectName("label_8")
        self.label_8.raise_()
        self.label.raise_()
        self.label_2.raise_()
        self.label_3.raise_()
        self.label_4.raise_()
        self.Page2.raise_()
        self.Top_k.raise_()
        self.label_5.raise_()
        self.Result.raise_()
        self.label_6.raise_()
        self.label_7.raise_()
        self.Result1.raise_()
        self.Result2.raise_()
        self.Search.raise_()
        self.Cancel.raise_()
        self.Page1.raise_()
        self.label_9.raise_()

        self.retranslateUi(Frame)
        QtCore.QMetaObject.connectSlotsByName(Frame)

    def retranslateUi(self, Frame):
        _translate = QtCore.QCoreApplication.translate
        Frame.setWindowTitle(_translate("Frame", "Frame"))
        self.label.setText(_translate("Frame", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:20pt; font-weight:600;\">Link &amp; Recommendation</span></p></body></html>"))
        self.label_2.setText(_translate("Frame", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:600;\">Page1</span></p></body></html>"))
        self.label_3.setText(_translate("Frame", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:600;\">Page2</span></p></body></html>"))
        self.label_4.setText(_translate("Frame", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:600;\">Related Number</span></p></body></html>"))
        self.Page2.setWhatsThis(_translate("Frame", "<html><head/><body><p><br/></p></body></html>"))
        self.Top_k.setWhatsThis(_translate("Frame", "<html><head/><body><p><br/></p></body></html>"))
        self.label_5.setText(_translate("Frame", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Whether Link Exists?</span></p></body></html>"))
        self.Result.setWhatsThis(_translate("Frame", "<html><head/><body><p><br/></p></body></html>"))
        self.label_6.setText(_translate("Frame", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Page1</span></p></body></html>"))
        self.label_7.setText(_translate("Frame", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Page2</span></p></body></html>"))
        self.Result1.setWhatsThis(_translate("Frame", "<html><head/><body><p><br/></p></body></html>"))
        self.Result2.setWhatsThis(_translate("Frame", "<html><head/><body><p><br/></p></body></html>"))
        self.Search.setText(_translate("Frame", "search"))
        self.Cancel.setText(_translate("Frame", "cancel"))
        self.Page1.setWhatsThis(_translate("Frame", "<html><head/><body><p><br/></p></body></html>"))

import back1_rc
import back2_rc
import back_rc
import background_rc
import erback_rc
import kback_rc
import logo2_rc
import logo_rc
import realback_rc
import sign_rc
