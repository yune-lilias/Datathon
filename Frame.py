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
        Frame.resize(640, 480)
        Frame.setMaximumSize(QtCore.QSize(640, 480))
        self.label = QtWidgets.QLabel(Frame)
        self.label.setGeometry(QtCore.QRect(210, 30, 181, 41))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Frame)
        self.label_2.setGeometry(QtCore.QRect(80, 110, 41, 9))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Frame)
        self.label_3.setGeometry(QtCore.QRect(80, 160, 41, 9))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Frame)
        self.label_4.setGeometry(QtCore.QRect(60, 210, 71, 20))
        self.label_4.setObjectName("label_4")
        self.Page2 = QtWidgets.QLineEdit(Frame)
        self.Page2.setGeometry(QtCore.QRect(170, 150, 113, 20))
        self.Page2.setObjectName("Page2")
        self.Top_k = QtWidgets.QLineEdit(Frame)
        self.Top_k.setGeometry(QtCore.QRect(170, 210, 113, 20))
        self.Top_k.setObjectName("Top_k")
        self.label_5 = QtWidgets.QLabel(Frame)
        self.label_5.setGeometry(QtCore.QRect(460, 120, 81, 31))
        self.label_5.setObjectName("label_5")
        self.Result = QtWidgets.QLineEdit(Frame)
        self.Result.setGeometry(QtCore.QRect(430, 170, 113, 20))
        self.Result.setObjectName("Result")
        self.label_6 = QtWidgets.QLabel(Frame)
        self.label_6.setGeometry(QtCore.QRect(90, 300, 41, 9))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(Frame)
        self.label_7.setGeometry(QtCore.QRect(90, 370, 41, 9))
        self.label_7.setObjectName("label_7")
        self.Result1 = QtWidgets.QLineEdit(Frame)
        self.Result1.setGeometry(QtCore.QRect(270, 290, 113, 20))
        self.Result1.setObjectName("Result1")
        self.Result2 = QtWidgets.QLineEdit(Frame)
        self.Result2.setGeometry(QtCore.QRect(270, 370, 113, 20))
        self.Result2.setObjectName("Result2")
        self.Search = QtWidgets.QPushButton(Frame)
        self.Search.setGeometry(QtCore.QRect(330, 130, 56, 17))
        self.Search.setObjectName("Search")
        self.Cancel = QtWidgets.QPushButton(Frame)
        self.Cancel.setGeometry(QtCore.QRect(330, 190, 56, 17))
        self.Cancel.setObjectName("Cancel")
        self.Page1 = QtWidgets.QLineEdit(Frame)
        self.Page1.setGeometry(QtCore.QRect(170, 100, 113, 20))
        self.Page1.setObjectName("Page1")

        self.retranslateUi(Frame)
        QtCore.QMetaObject.connectSlotsByName(Frame)

    def retranslateUi(self, Frame):
        _translate = QtCore.QCoreApplication.translate
        Frame.setWindowTitle(_translate("Frame", "Frame"))
        self.label.setText(_translate("Frame", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Link Recongnizer</span></p></body></html>"))
        self.label_2.setText(_translate("Frame", "<html><head/><body><p><span style=\" font-weight:600;\">Page1</span></p></body></html>"))
        self.label_3.setText(_translate("Frame", "<html><head/><body><p><span style=\" font-weight:600;\">Page2</span></p></body></html>"))
        self.label_4.setText(_translate("Frame", "<html><head/><body><p>Related Number</p></body></html>"))
        self.Page2.setWhatsThis(_translate("Frame", "<html><head/><body><p><br/></p></body></html>"))
        self.Top_k.setWhatsThis(_translate("Frame", "<html><head/><body><p><br/></p></body></html>"))
        self.label_5.setText(_translate("Frame", "<html><head/><body><p>Relationship</p></body></html>"))
        self.Result.setWhatsThis(_translate("Frame", "<html><head/><body><p><br/></p></body></html>"))
        self.label_6.setText(_translate("Frame", "<html><head/><body><p><span style=\" font-weight:600;\">Page1</span></p></body></html>"))
        self.label_7.setText(_translate("Frame", "<html><head/><body><p><span style=\" font-weight:600;\">Page2</span></p></body></html>"))
        self.Result1.setWhatsThis(_translate("Frame", "<html><head/><body><p><br/></p></body></html>"))
        self.Result2.setWhatsThis(_translate("Frame", "<html><head/><body><p><br/></p></body></html>"))
        self.Search.setText(_translate("Frame", "search"))
        self.Cancel.setText(_translate("Frame", "cancel"))
        self.Page1.setWhatsThis(_translate("Frame", "<html><head/><body><p><br/></p></body></html>"))
