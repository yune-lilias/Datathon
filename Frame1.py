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
        self.label.setGeometry(QtCore.QRect(270, 30, 181, 51))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Frame)
        self.label_2.setGeometry(QtCore.QRect(70, 110, 41, 9))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Frame)
        self.label_3.setGeometry(QtCore.QRect(70, 160, 41, 9))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Frame)
        self.label_4.setGeometry(QtCore.QRect(50, 250, 71, 20))
        self.label_4.setObjectName("label_4")
        self.Page2 = QtWidgets.QLineEdit(Frame)
        self.Page2.setGeometry(QtCore.QRect(170, 150, 113, 31))
        self.Page2.setObjectName("Page2")
        self.Top_k = QtWidgets.QLineEdit(Frame)
        self.Top_k.setGeometry(QtCore.QRect(170, 200, 113, 31))
        self.Top_k.setObjectName("Top_k")
        self.label_5 = QtWidgets.QLabel(Frame)
        self.label_5.setGeometry(QtCore.QRect(490, 100, 111, 31))
        self.label_5.setObjectName("label_5")
        self.Result = QtWidgets.QLineEdit(Frame)
        self.Result.setGeometry(QtCore.QRect(480, 130, 113, 31))
        self.Result.setObjectName("Result")
        self.label_6 = QtWidgets.QLabel(Frame)
        self.label_6.setGeometry(QtCore.QRect(60, 390, 91, 21))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(Frame)
        self.label_7.setGeometry(QtCore.QRect(60, 470, 81, 31))
        self.label_7.setObjectName("label_7")
        self.Result1 = QtWidgets.QLineEdit(Frame)
        self.Result1.setGeometry(QtCore.QRect(210, 370, 451, 51))
        self.Result1.setObjectName("Result1")
        self.Result2 = QtWidgets.QLineEdit(Frame)
        self.Result2.setGeometry(QtCore.QRect(210, 460, 451, 51))
        self.Result2.setObjectName("Result2")
        self.Search = QtWidgets.QPushButton(Frame)
        self.Search.setGeometry(QtCore.QRect(350, 120, 56, 31))
        self.Search.setObjectName("Search")
        self.Cancel = QtWidgets.QPushButton(Frame)
        self.Cancel.setGeometry(QtCore.QRect(350, 190, 56, 31))
        self.Cancel.setObjectName("Cancel")
        self.Page1 = QtWidgets.QLineEdit(Frame)
        self.Page1.setGeometry(QtCore.QRect(170, 100, 113, 31))
        self.Page1.setObjectName("Page1")
        self.label_8 = QtWidgets.QLabel(Frame)
        self.label_8.setGeometry(QtCore.QRect(-70, -30, 1001, 671))
        self.label_8.setText("")
        self.label_8.setPixmap(QtGui.QPixmap(":/back2.jpg"))
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

        self.retranslateUi(Frame)
        QtCore.QMetaObject.connectSlotsByName(Frame)

    def retranslateUi(self, Frame):
        _translate = QtCore.QCoreApplication.translate
        Frame.setWindowTitle(_translate("Frame", "Frame"))
        self.label.setText(_translate("Frame", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; font-weight:600;\">Link Recongnizer</span></p></body></html>"))
        self.label_2.setText(_translate("Frame", "<html><head/><body><p><span style=\" font-weight:600;\">Page1</span></p></body></html>"))
        self.label_3.setText(_translate("Frame", "<html><head/><body><p><span style=\" font-weight:600;\">Page2</span></p></body></html>"))
        self.label_4.setText(_translate("Frame", "<html><head/><body><p>Related Number</p></body></html>"))
        self.Page2.setWhatsThis(_translate("Frame", "<html><head/><body><p><br/></p></body></html>"))
        self.Top_k.setWhatsThis(_translate("Frame", "<html><head/><body><p><br/></p></body></html>"))
        self.label_5.setText(_translate("Frame", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Relationship</span></p></body></html>"))
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
import sign_rc
