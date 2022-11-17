import re
import sys
from random import randint

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QApplication, QDesktopWidget, QFileDialog,
                             QInputDialog, QLabel, QLineEdit, QMainWindow,
                             QMessageBox, QPushButton, QWidget)
from scipy import ndimage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.w = None
        self.inputWindow = None
        self.inputWindowOfRotation = None
        self.adjBrgWindow = None
        self.flipImage = None

        self.warning = None
        xtop = 10
        yleft = 10
        buttonWidth = 150
        buttonHeight = 50

        sizeObject = QDesktopWidget().screenGeometry()

        # access screen dimensions for image - screen accordance
        self.width = int(sizeObject.getRect()[2])
        self.height = int(sizeObject.getRect()[3])

        self.imgWidth = 0
        self.imgHeight = 0

        # set up message area widget
        self.message = QLabel(self)
        self.message.setFont(QFont("Arial", 12))
        self.message.setGeometry(920, 60, 700, 200)
        self.message.setStyleSheet("color: red")

        hFile = QLabel(self)
        hFile.setText("File")
        hFile.setFont(QFont("Arial", 20))
        hFile.move(xtop+int(buttonWidth/4), yleft)

        # load button widget
        # create new button
        loadButton = QPushButton(self)
        # set text of button
        loadButton.setText("Load Image")
        # set button coordinates and its width, height
        loadButton.setGeometry(xtop, yleft+buttonHeight, buttonWidth, buttonHeight)
        # runs function (show_new_window) when clicked button
        loadButton.clicked.connect(self.show_new_window)

        # save button widget
        saveButton = QPushButton(self)
        saveButton.setText("Save Image")
        saveButton.setGeometry(xtop, yleft+2*buttonHeight, buttonWidth, buttonHeight)
        saveButton.clicked.connect(self.save)

        # ------ EDIT AREA
        # Edit text area widget
        hEdit = QLabel(self)
        hEdit.setText("Edit")
        hEdit.setFont(QFont("Arial", 20))
        hEdit.move(buttonWidth+yleft+int(buttonWidth/4),xtop)

        # blur button widget
        blurButton = QPushButton(self)
        blurButton.setText("Blur Image")
        blurButton.setGeometry(xtop+buttonWidth, yleft+buttonHeight, buttonWidth, buttonHeight)
        blurButton.clicked.connect(self.blur)

        # deblur button widget
        deblurButton = QPushButton(self)
        deblurButton.setText("Deblur Image")
        deblurButton.setGeometry(xtop+buttonWidth, yleft+buttonHeight*2, buttonWidth, buttonHeight)
        deblurButton.clicked.connect(self.deblur)

        # reverse color button widget
        negativeButton = QPushButton(self)
        negativeButton.setText("Negative Image")
        negativeButton.setGeometry(xtop+buttonWidth*2, yleft+buttonHeight, buttonWidth, buttonHeight)
        negativeButton.clicked.connect(self.negative)

        # grayscale button widget
        grayScaleButton = QPushButton(self)
        grayScaleButton.setText("Grayscale Image")
        grayScaleButton.setGeometry(xtop+buttonWidth*2, yleft+buttonHeight*2, buttonWidth, buttonHeight)
        grayScaleButton.clicked.connect(self.grayscale)

        # flip button widget
        flipButton = QPushButton(self)
        flipButton.setText("Flip Image")
        flipButton.setGeometry(xtop+buttonWidth*3, yleft+buttonHeight, buttonWidth, buttonHeight)
        flipButton.clicked.connect(self.flip)

        # mirror button widget
        mirrorButton = QPushButton(self)
        mirrorButton.setText("Mirror Image")
        mirrorButton.setGeometry(xtop+buttonWidth*3, yleft+buttonHeight*2, buttonWidth, buttonHeight)
        mirrorButton.clicked.connect(self.mirror)

        # adjBrg: adjust brightness
        adjBrgButton = QPushButton(self)
        adjBrgButton.setText("Adjust Brightness")
        adjBrgButton.setGeometry(xtop+buttonWidth*4, yleft+buttonHeight, buttonWidth, buttonHeight)
        adjBrgButton.clicked.connect(self.adjBrg)

        # Detect Edges
        detectEdgesButton = QPushButton(self)
        detectEdgesButton.setText("Detect Edges")
        detectEdgesButton.setGeometry(xtop+buttonWidth*4, yleft+buttonHeight*2, buttonWidth, buttonHeight)
        detectEdgesButton.clicked.connect(self.detectEdges)

        # Add Noise
        addNoiseButton = QPushButton(self)
        addNoiseButton.setText("Add Noise")
        addNoiseButton.setGeometry(xtop+buttonWidth*5, yleft+buttonHeight, buttonWidth, buttonHeight)
        addNoiseButton.clicked.connect(self.addNoise)

        # Histogram
        PlotButton = QPushButton(self)
        PlotButton.setText("Histogram")
        PlotButton.setGeometry(xtop+buttonWidth*5, yleft+buttonHeight*2, buttonWidth, buttonHeight)
        PlotButton.clicked.connect(self.histogram)


        # Loaded Image widget
        self.loadedImage = QLabel(self)
        self.loadedImage.setScaledContents(True)
        self.loadedImage.setFixedHeight(int(self.height - 250))
        self.loadedImage.setFixedWidth(int(self.width/2))
        self.loadedImagePath = ""
        self.loadedImage.move( 0,250)

        # Processed Image widget
        self.manipulatedImage = QLabel(self)
        self.manipulatedImage.setScaledContents(True)
        self.manipulatedImage.setFixedHeight(int(self.height - 250))
        self.loadedImage.setFixedWidth(int(self.width / 2))
        self.manipulatedImage.move(
            int(self.width/2),250)

        # set coordinate and sizes of main screen of application
        self.setGeometry(0, 0, self.width, self.height)
        self.setWindowTitle("Pengolahan Citra")

    def show_new_window(self, checked):

        if self.w is None:
            self.w = QFileDialog.Options()
            # get filename of image
            fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
                                                      "All Files (*.jpg *.png *.jpeg)", options=self.w)

            # Original Image Widget
            pixmap = QPixmap(fileName)
            pixmap2 = pixmap.scaledToWidth(int(self.width / 2))
            self.loadedImage.setPixmap(pixmap2)
            self.loadedImage.adjustSize()
            self.loadedImagePath = fileName

        self.w = None


    def save(self):
        try:
            # if path is empty, raise FileNotFoundError
            if(len(self.loadedImagePath) == 0):
                raise FileNotFoundError
            self.manipulatedImage.pixmap().save("savedImage.jpg","JPG")
            self.message.setText("")

        # display error message
        except FileNotFoundError:
            self.message.setText("Belum terdapat gambar yang diproses!")
        except Exception as E:
            self.message.setText(str(E))

    def histogram(self):
        try:
            image1 = cv2.imread(self.loadedImagePath).mean(axis=2)
            image2 = cv2.imread("temp.jpg").mean(axis=2)
            
            # Separate Histograms for each color
            plt.subplot(3, 1, 1)
            plt.title("Original")
            plt.xlim([-10,265])
            plt.hist(image1)
            
            plt.subplot(3, 1, 2)
            plt.title("Processed")
            plt.xlim([-10,265])
            plt.hist(image2)

            plt.tight_layout()
            plt.show()
        except FileNotFoundError:
            self.message.setText("Belum terdapat gambar yang diproses!")
        except Exception as E:
            self.message.setText(str(E))

    def blur(self):

        # QCoreApplication.exit(0)

        try:
            # access loaded Image
            image = cv2.imread(self.loadedImagePath)
            if(image is None):
                raise FileNotFoundError
            # blur image
            blurImg = cv2.blur(image, (9, 9))

            # save blurred image temporarily
            cv2.imwrite("temp.jpg", blurImg)

            pixmap = QPixmap("./temp.jpg")
            pixmap2 = pixmap.scaledToWidth(int(self.width / 2))
            self.manipulatedImage.setPixmap(pixmap2)
            self.manipulatedImage.adjustSize()
            # set message text to empty, when process s successfull
            self.message.setText("")
        except FileNotFoundError:
            self.message.setText("Belum terdapat gambar yang diproses!")
        except Exception as E:
            self.message.setText(str(E))

    def deblur(self):
        try:
            image = cv2.imread(self.loadedImagePath)
            if(image is None):
                raise FileNotFoundError
            sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpen = cv2.filter2D(image, -1, sharpen_kernel)

            cv2.imwrite("temp.jpg", sharpen)

            pixmap = QPixmap("./temp.jpg")
            pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

            self.manipulatedImage.setPixmap(pixmap2)
            self.manipulatedImage.adjustSize()

            self.message.setText("")

        except FileNotFoundError:
            self.message.setText("Belum terdapat gambar yang diproses!")
        except Exception as E:
            self.message.setText(str(E))
            print(E)

    def negative(self):
        try:
            image = cv2.imread(self.loadedImagePath)
            # if image is None, raise FileNotFoundError
            if(image is None):
                raise FileNotFoundError

            # reverse color
            image = (255 - image)
            cv2.imwrite("temp.jpg", image)

            pixmap = QPixmap("./temp.jpg")
            pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

            self.manipulatedImage.setPixmap(pixmap2)
            self.manipulatedImage.adjustSize()
            self.message.setText("")

        #   display relevent error message in ui
        except FileNotFoundError:
            self.message.setText("Belum terdapat gambar yang diproses!")
        except Exception as E:
            self.message.setText(str(E))

    def grayscale(self):
        try:
            image = cv2.imread(self.loadedImagePath)
            if(image is None):
                raise FileNotFoundError

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            cv2.imwrite("temp.jpg", gray_image)

            pixmap = QPixmap("./temp.jpg")
            pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

            self.manipulatedImage.setPixmap(pixmap2)
            self.manipulatedImage.adjustSize()
            self.message.setText("")

        # display relevant error message
        except FileNotFoundError:
            self.message.setText("Belum terdapat gambar yang diproses!")
        except Exception as E:
            self.message.setText(str(E))
            print(E)

    def flip(self):
        if(self.flipImage is None):
            try:
                flipValue, okPressed = QInputDialog.getText(self, "Rotation", "Masukkan value\n"
                                                                              "0: Vertical Flip\n"
                                                                              "1: Horizontal Flip", QLineEdit.Normal, "",)
                image = cv2.imread(self.loadedImagePath)
                if(image is None):
                    raise FileNotFoundError
                elif(int(flipValue)>1):
                    raise Exception

                # second argument of cv2.flip is horizontal or vertical
                # 0 for vertical flip
                # 1 for horizontal flip
                flippedImage = cv2.flip(image,int(flipValue))

                cv2.imwrite("temp.jpg", flippedImage)

                pixmap = QPixmap("./temp.jpg")
                pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

                self.manipulatedImage.setPixmap(pixmap2)
                self.manipulatedImage.adjustSize()
                self.message.setText("")
            # display relevant error message
            except FileNotFoundError:
                self.message.setText("Belum terdapat gambar yang diproses!")
            except Exception as E:
                self.message.setText("Invalid Input")
                print(E)

    def mirror(self):
        try:
            image = cv2.imread(self.loadedImagePath)
            if(image is None):
                raise FileNotFoundError

            mirroredImage = cv2.flip(image,1)
            cv2.imwrite("temp.jpg", mirroredImage)

            pixmap = QPixmap("./temp.jpg")
            pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

            self.manipulatedImage.setPixmap(pixmap2)
            self.manipulatedImage.adjustSize()
            self.message.setText("")
        # display relevant error
        except FileNotFoundError:
            self.message.setText("Belum terdapat gambar yang diproses!")
        except Exception as E:
            self.message.setText(str(E))
            print(E)

    def adjBrg(self):
        if self.adjBrgWindow is None:
            try:
                if (len(self.loadedImagePath) == 0):
                    raise FileNotFoundError
                value, okPressed = QInputDialog. \
                    getText(self, "Adjust Brightness",
                            "Masukkan brightness value negatif atau positif:\n(default value is 0)", QLineEdit.Normal, "", )

                image = cv2.imread(self.loadedImagePath)

                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                h, s, v = cv2.split(hsv)

                # find value without its sign mark
                val = int(re.findall('\d+', value)[0])

                if(int(value)>0):
                    v = cv2.add(v,int(val))
                else:
                    v = cv2.subtract(v,int(val))

                v[v > 255] = 255
                v[v < 0] = 0
                final_hsv = cv2.merge((h, s, v))
                img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


                cv2.imwrite("temp.jpg", img)

                pixmap = QPixmap("./temp.jpg")
                pixmap2 = pixmap.scaledToWidth(int(self.width / 2))
                self.manipulatedImage.setPixmap(pixmap2)
                self.manipulatedImage.adjustSize()

            # display relevant error message
            except FileNotFoundError:
                self.message.setText("Belum terdapat gambar yang diproses!")
            except Exception as E:
                self.message.setText("invalid input")
                # self.message.setText(str(E))
                print(E)

    def detectEdges(self):

        try:
            img = cv2.imread(self.loadedImagePath)
            if(img is None):
                raise FileNotFoundError
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

            edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection


            edges = np.array(edges)
            cv2.imwrite("temp.jpg", edges)

            pixmap = QPixmap("./temp.jpg")
            pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

            self.manipulatedImage.setPixmap(pixmap2)
            self.manipulatedImage.adjustSize()
            self.message.setText("")

        # display relevant error message
        except FileNotFoundError:
            self.message.setText("Belum terdapat gambar yang diproses!")
        except Exception as E:
            self.message.setText(str(E))
            print(E)

    def addNoise(self):
        try:
            image = cv2.imread(self.loadedImagePath)
            if(image is None):
                raise FileNotFoundError

            gauss = np.random.normal(0, 1, image.size)
            gauss = gauss.reshape(image.shape[0], image.shape[1], image.shape[2]).astype('uint8')
            mode = "speckle"

            if mode == "gaussian":
                img_gauss = cv2.add(image, gauss)

                cv2.imwrite("./temp.jpg", img_gauss)
                pixmap = QPixmap("./temp.jpg")
                pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

                self.manipulatedImage.setPixmap(pixmap2)
                self.manipulatedImage.adjustSize()

            elif mode == "speckle":
                noise = image + image * gauss

            cv2.imwrite("./temp.jpg", noise)
            pixmap = QPixmap("./temp.jpg")
            pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

            self.manipulatedImage.setPixmap(pixmap2)
            self.manipulatedImage.adjustSize()
            # cv2.imshow('ab', noise)
            # cv2.waitKey()
            # return noise
            self.message.setText("")

        # display relevant error message
        except FileNotFoundError:
            self.message.setText("Belum terdapat gambar yang diproses!")
        except Exception as E:
            self.message.setText(str(E))
            print(E)

app = QApplication(sys.argv)
main = MainWindow()
main.show()
app.exec()