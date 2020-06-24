import os
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
from deal import start

windowWidth = 700
windowHeight = 500
searchWidth = 200
searchHeight = 20
picWidth = 400
picHeight = 300
captionWidth = 400
captionHeight = 80
categoryWidth = 100
categoryHeight = 40
fileName = "Dic.txt"

class picture(QWidget):
    def __init__(self):
        super(picture, self).__init__()

        self.resize(windowWidth, windowHeight)
        self.setWindowTitle("Image Caption")
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap("./background.jpeg")))
        self.setPalette(palette)

        self.label = QLabel(self)
        self.label.setFixedSize(picWidth, picHeight)
        self.label.move((windowWidth-picWidth)/2, (windowHeight-picHeight)/2-30)
        self.label.setWindowOpacity(0)
        self.label.setStyleSheet(
                                 "QLabel{color:rgb(0.8,0.6,0.4);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )

        self.search = QLineEdit(self)
        self.search.setGeometry(QRect((windowWidth - searchWidth) / 2 - 110, 25, searchWidth, searchHeight))

        searchBtn = QPushButton(self)
        searchBtn.setText("search")
        searchBtn.move((windowWidth - searchWidth) / 2 - 90 + searchWidth, 22)
        searchBtn.clicked.connect(self.searchButtonDidClick)

        self.caption = QLabel(self)
        self.caption.setWindowOpacity(0)
        self.caption.setGeometry(QRect((windowWidth - captionWidth) / 2, (picHeight)+60, captionWidth, captionHeight))
        self.caption.setWordWrap(True)
        self.caption.setAlignment(QtCore.Qt.AlignCenter)

        self.category = QLabel(self)
        self.category.setWindowOpacity(0)
        self.category.setGeometry(QRect((windowWidth - categoryWidth) / 2, (picHeight) + 135, categoryWidth, categoryHeight))
        self.category.setWordWrap(True)
        self.category.setAlignment(QtCore.Qt.AlignCenter)

        btn = QPushButton(self)
        btn.setText("Open Picture")
        btn.move((windowWidth-btn.width())/2+140, 22)
        btn.clicked.connect(self.openimage)




    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "Open Picture", "", "*.jpg;*.jpeg")
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        label, caption = start(imgName)
        self.caption.setText("Caption: " + caption)
        self.category.setText("Label: " + label)
        self.label.setPixmap(jpg)
        with open(fileName, "a") as w:
            string = imgName + ";" + label + ";" + caption + "\n"
            w.write(string)

    def searchButtonDidClick(self):
        with open(fileName, "r") as r:
            content = r.readlines()
            for sentence in content:
                result = sentence.split(";", 3);
                if self.search.text() in result[1] or self.search.text() in result[2]:
                    os.system("open " + result[0]) # Mac open
                    #os.system("start explore " + result[0]) # Windows open


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = picture()
    my.show()
    sys.exit(app.exec_())
