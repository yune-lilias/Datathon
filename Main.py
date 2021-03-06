from PyQt5 import QtWidgets
import pandas as pd
from PyQt5.QtWidgets import QWidget, QApplication, QMessageBox
import Frame
from gcn import Net
import torch
import torch_geometric
import numpy as np
import matplotlib.pyplot as plt
from Search import interact


class main_window(QWidget, Frame.Ui_Frame):
    def __init__(self):
        super(main_window, self).__init__()
        self.setupUi(self)
        # 信号连接到槽
        self.Search.clicked.connect(self.doSearch)
        self.Cancel.clicked.connect(self.doCancel)

    def doSearch(self):
        send = self.sender()
        if (len(self.Page1.text()) != 0 and
                len(self.Page2.text()) != 0 and
                len(self.Top_k.text()) != 0):
            print("search")
            page1 = int(self.Page1.text())
            page2 = int(self.Page2.text())
            self.page = [[page1, page2]]
            top_k = int(self.Top_k.text())
            Result = interact(self.page, 0)
            self.Result.setText("Yes" if Result[0]==1 else "No")
            r1 = interact(page1, top_k)
            r2 = interact(page2, top_k)
            self.Result1.setText(np.array2string(r1[:top_k], precision=2, separator=',', suppress_small=True))
            self.Result2.setText(np.array2string(r2[:top_k], precision=2, separator=',', suppress_small=True))
        else:
            print("error")
            QMessageBox.warning(self, "Warning", "Input Error！")

    def doCancel(self):
        reply = QMessageBox.information(self, "The message for the page", "Are you sure to cancel the search?",
                                        QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes: self.close()
        if reply == QMessageBox.No: return self


### 测试用
if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    u = main_window()
    u.show()
    sys.exit(app.exec_())
