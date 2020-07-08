import sys
import cv2
import numpy as np
import argparse
import math
from PyQt5 import QtCore,QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QAction
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt

class ShowImage(QMainWindow) :
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('untitled.ui', self)
        self.Image = None
        self.contour.clicked.connect(self.contourrClicked)
        self.loadButton.clicked.connect(self.loadClicked)
        self.grayButton.clicked.connect(self.grayClicked)
        self.actionBrightness_3.triggered.connect(self.brightClicked)
        self.actionContrast.triggered.connect(self.contrastClicked)
        self.actionContrastStretching.triggered.connect(self.contrastStretchingClicked)
        self.actionNegative_Image.triggered.connect(self.negativeImageClicked)
        self.actionBiner_Image.triggered.connect(self.binerImageClicked)
        self.actionHistogram_Grayscale.triggered.connect(self.histogramGrayscaleClicked)
        self.actionHistogram_RGB.triggered.connect(self.histogramRGBClicked)
        self.actionHistogram_Equalization.triggered.connect(self.histogramEqualizationClicked)
        self.actionTranslasi.triggered.connect(self.translasiClicked)
        self.actionTranspose.triggered.connect(self.transposeClicked)
        self.action45.triggered.connect(self.rotasi45Clicked)
        self.action_45.triggered.connect(self.rotasimin45Clicked)
        self.action90.triggered.connect(self.rotasi90Clicked)
        self.action_90.triggered.connect(self.rotasimin90Clicked)
        self.action180.triggered.connect(self.rotasi180Clicked)
        self.actionzoomIn.triggered.connect(self.zoomInClicked)
        self.actionzoomout.triggered.connect(self.zoomOutClicked)
        self.actionSkewed.triggered.connect(self.skewedImageClicked)
        self.actionCrop.triggered.connect(self.cropClicked)
        self.actionaritmatika.triggered.connect(self.aritmatikaClicked)
        self.actionlogika.triggered.connect(self.logikaClicked)
        self.actionFiltering.triggered.connect(self.filteringClicked)
        self.actionMean.triggered.connect(self.meanClicked)
        self.actionSmoothmean.triggered.connect(self.SmoothmeanClicked)
        self.actionGaussian.triggered.connect(self.GaussianClicked)
        self.actionSharpening.triggered.connect(self.SharpeningClicked)
        self.actionMedian.triggered.connect(self.medianClicked)
        self.actionMax.triggered.connect(self.maxClicked)
        self.actionMin.triggered.connect(self.minClicked)
        self.actionSobel.triggered.connect(self.sobelClicked)
        self.actionRobert.triggered.connect(self.robertClicked)
        self.actionPrewitt.triggered.connect(self.prewittClicked)
        self.actionCanny.triggered.connect(self.cannyClicked)
        self.actionDilasi.triggered.connect(self.dilasiClicked)
        self.actionErosi.triggered.connect(self.erosiClicked)
        self.actionOpening.triggered.connect(self.openingClicked)
        self.actionClosing.triggered.connect(self.closingClicked)
        self.actionBinary.triggered.connect(self.binaryClicked)
        self.actionBinaryInvers.triggered.connect(self.binaryinvClicked)
        self.actionTrunc.triggered.connect(self.truncClicked)
        self.actionToZero.triggered.connect(self.tozeroClicked)
        self.actionToZeroInvers.triggered.connect(self.tozeroinvClicked)
        self.actionColorTracking.triggered.connect(self.colorClicked)
        self.actionObjectDetection.triggered.connect(self.objectDetectionClicked)
        self.actionColorPicker.triggered.connect(self.colorPickerClicked)
        self.actionOtsuThresholding.triggered.connect(self.otsuThresholdingClicked)
        self.actionGaussianThresholding.triggered.connect(self.gaussianThresholdingClicked)
        self.actionMeanThresholding.triggered.connect(self.meanThresholdingClicked)
        self.actionContour.triggered.connect(self.contourClicked)

    @pyqtSlot()
    def loadClicked(self):
        img= self.loadImage('2.1.jpg')
        print(img)

    @pyqtSlot()
    def nothing(x):
        pass

    @pyqtSlot()
    def objectDetectionClicked(self):
        cam = cv2.VideoCapture("video.mp4")
        car_cascade = cv2.CascadeClassifier("cars.xml")

        while True:
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cars = car_cascade.detectMultiScale(gray, 1.1, 3)

            for (x, y, w, h) in cars:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            cv2.imshow("video", frame)
            if cv2.waitKey(50) & 0xFF == ord("q"):
                break

        cam.release()
        cv2.destroyAllWindows()

    @pyqtSlot()
    def colorPickerClicked(self):
        cam = cv2.VideoCapture("video.mp4")
        cv2.namedWindow("Trackbars")
        cv2.createTrackbar("L-H", "Trackbars", 0, 179, self.nothing)
        cv2.createTrackbar("L-S", "Trackbars", 0, 255, self.nothing)
        cv2.createTrackbar("L-V", "Trackbars", 0, 255, self.nothing)
        cv2.createTrackbar("U-H", "Trackbars", 179, 179, self.nothing)
        cv2.createTrackbar("U-S", "Trackbars", 255, 255, self.nothing)
        cv2.createTrackbar("U-V", "Trackbars", 255, 255, self.nothing)

        while True:
            _, frame = cam.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            l_h = cv2.getTrackbarPos("L-H", "Trackbars")
            l_s = cv2.getTrackbarPos("L-S", "Trackbars")
            l_v = cv2.getTrackbarPos("L-V", "Trackbars")
            u_h = cv2.getTrackbarPos("U-H", "Trackbars")
            u_s = cv2.getTrackbarPos("U-S", "Trackbars")
            u_v = cv2.getTrackbarPos("U-V", "Trackbars")
            lower_color = np.array([l_h, l_s, l_v])
            upper_color = np.array([u_h, u_s, u_v])
            mask = cv2.inRange(hsv, lower_color, upper_color)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", mask)
            cv2.imshow("Result", result)

            key = cv2.waitKey(1)
            if key == 27:
                break
        cam.release()
        cv2.destroyAllWindows()

    @pyqtSlot()
    def colorClicked(self):
        cam = cv2.VideoCapture("hdcctv.mp4")
        while True:
            _,frame = cam.read()
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            lower_color = np.array([1, 1, 100])
            upper_color = np.array([80, 56, 255])
            mask = cv2.inRange(hsv, lower_color,upper_color)
            result = cv2.bitwise_and(frame,frame,mask=mask)
            cv2.imshow("frame",frame)
            cv2.imshow("mask",mask)
            cv2.imshow("result",result)
            key=cv2.waitKey(1)
            if key == 27:
                break
        cam.release()
        cv2.destroyAllWindows()

    @pyqtSlot()
    def gaussianThresholdingClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print(img)
        imgh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
        cv2.imshow("Gaussian Thresholding", imgh)
        print(imgh)

    @pyqtSlot()
    def meanThresholdingClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print(img)
        imgh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
        cv2.imshow("Mean Thresholding", imgh)
        print(imgh)

    @pyqtSlot()
    def otsuThresholdingClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print(img)
        T = 130
        max = 255
        ret, thresh = cv2.threshold(img, T, max, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow("Otsu thresholding", thresh)
        print(thresh)

    @pyqtSlot()
    def contourrClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        t = 127
        ret, thres = cv2.threshold(img, t, 255, 0)
        _,contours, hierarchy = cv2.findContours(thres, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        print("Number of contours = " + str(len(contours)))
        for cnt in contours:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(self.image, [approx], -1, (0, 255, 0), 3)
            x, y = approx[0][0]

            if len(approx) == 3:
                cv2.putText(self.image, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
            elif len(approx) == 4:
                cv2.putText(self.image, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
            elif len(approx) == 5:
                cv2.putText(self.image, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
            elif 6 < len(approx) < 15:
                cv2.putText(self.image, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
            else:
                cv2.putText(self.image, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
        cv2.imshow("image", self.image)
        cv2.imshow("approx", thres)

    @pyqtSlot()
    def contourClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        t = 127
        ret, thres = cv2.threshold(img, t, 255, 0)
        _,contours, hierarchy = cv2.findContours(thres, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        print("Number of contours = " + str(len(contours)))
        for cnt in contours:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(self.image, [approx], -1, (0, 255, 0), 3)
            x, y = approx[0],[0]

            if len(approx) == 3:
                cv2.putText(self.image, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
            elif len(approx) == 4:
                cv2.putText(self.image, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
            elif len(approx) == 5:
                cv2.putText(self.image, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
            elif 6 < len(approx) < 15:
                cv2.putText(self.image, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
            else:
                cv2.putText(self.image, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
        cv2.imshow("image", self.image)
        cv2.imshow("approx", thres)

    @pyqtSlot()
    def tozeroinvClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print(img)
        T = 127
        max = 255
        ret, thresh = cv2.threshold(img, T, max, cv2.THRESH_TOZERO_INV)
        cv2.imshow("To Zero Invers", thresh)
        print(thresh)
    @pyqtSlot()
    def tozeroClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print(img)
        T = 127
        max = 255
        ret, thresh = cv2.threshold(img, T, max, cv2.THRESH_TOZERO)
        cv2.imshow("To Zero", thresh)
        print(thresh)
    @pyqtSlot()
    def truncClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print(img)
        T = 127
        max = 255
        ret, thresh = cv2.threshold(img, T, max, cv2.THRESH_TRUNC)
        cv2.imshow("Trunc", thresh)
        print(thresh)
    @pyqtSlot()
    def binaryinvClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print(img)
        T = 127
        max = 255
        ret, thresh = cv2.threshold(img, T, max, cv2.THRESH_BINARY_INV)
        cv2.imshow("Binary Invers", thresh)
        print(thresh)
    @pyqtSlot()
    def binaryClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print(img)
        T = 127
        max = 255
        ret, thresh = cv2.threshold(img, T, max, cv2.THRESH_BINARY)
        cv2.imshow("Binary", thresh)
        print(thresh)
    @pyqtSlot()
    def closingClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print(img)
        ret, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        strel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        imgh = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, strel, iterations=1)
        cv2.imshow("Closing", imgh)
        print(imgh)

    @pyqtSlot()
    def openingClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print(img)
        ret, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        strel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        imgh = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, strel, iterations=1)
        cv2.imshow("Opening", imgh)
        print(imgh)

    @pyqtSlot()
    def erosiClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print(img)
        ret, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        imgh = cv2.erode(threshold, strel, iterations=1)
        cv2.imshow("Erosi", imgh)
        print(imgh)

    @pyqtSlot()
    def dilasiClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print(img)
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        strel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        imgh = cv2.dilate(img, strel, iterations=1)
        cv2.imshow("Dilasi", imgh)
        print(imgh)

    @pyqtSlot()
    def cannyClicked(self):
        # Gussian
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        H, W = img.shape[:2]
        gaussian = (1.0 / 345) * np.array(
            [[1, 5, 7, 5, 1],
             [5, 20, 33, 20, 5],
             [7, 33, 55, 33, 7],
             [5, 20, 33, 20, 5],
             [1, 5, 7, 5, 1]])
        img_outgaussian = self.convolve(img, gaussian)
        print(img)
        print(img_outgaussian)
        # Sobel
        Sx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
        Sy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])

        img_x = self.convolve(img_outgaussian, Sx)
        img_y = self.convolve(img_outgaussian, Sy)

        img_outsobel = np.sqrt(img_x * img_x + img_y * img_y)

        for i in np.arange(H):
            for j in np.arange(W):
                a = img_outsobel.item(i, j)
                if a > 255:
                    a = 255;
                elif a < 0:
                    a = 0
                else:
                    a = a

        theta = np.arctan2(img_y, img_x)
        cv2.imshow("Reduksi Noise ", img_outsobel)

        print(theta)



        # Gussian
        angle = theta * 180. / np.pi
        angle[angle < 0] += 180
        Z = np.zeros((H, W))

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                try:
                    q = 255
                    r = 255
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img_outsobel[i, j + 1]
                        r = img_outsobel[i, j - 1]
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = img_outsobel[i + 1, j - 1]
                        r = img_outsobel[i - 1, j + 1]
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img_outsobel[i + 1, j]
                        r = img_outsobel[i - 1, j]
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img_outsobel[i - 1, j - 1]
                        r = img_outsobel[i + 1, j + 1]

                    if (img_outsobel[i, j] >= q) and (img_outsobel[i, j] >= r):
                        Z[i, j] = img_outsobel[i, j]
                    else:
                        Z[i, j] = 0
                except IndexError as e:
                    pass

        img_N = Z.astype("uint8")
        cv2.imshow("Non maximum suppression", img_N)


        print(img_N)
        weak = 50
        strong = 70
        for i in np.arange(H):
            for j in np.arange(W):
                a = img_N.item(i, j)
                if (a > weak):
                    b = weak
                    if (a > strong):
                        b = 255
                else:
                    b = 0

                img_N.itemset((i, j), b)

        img_H1 = img_N.astype("uint8")
        cv2.imshow("hysteresis part 1", img_H1)

        print(img_H1)
        strong = 255
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if (img_H1[i, j] == weak):
                    try:
                        if ((img_H1[i + 1, j - 1] == strong) or (img_H1[i + 1, j] == strong) or
                                (img_H1[i + 1, j + 1] == strong) or (img_H1[i, j - 1] == strong) or
                                (img_H1[i, j + 1] == strong) or (img_H1[i - 1, j - 1] == strong) or
                                (img_H1[i - 1, j] == strong) or (img_H1[i - 1, j + 1] == strong)):
                            img_H1[i, j] = strong
                        else:
                            img_H1[i, j] = 0
                    except IndexError as e:
                        pass

        img_H2 = img_H1.astype("uint8")

        cv2.imshow("hysteresis part 2", img_H2)


    @pyqtSlot()
    def prewittClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        Px = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        Py = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        img_x = self.convolve(img, Px)
        img_y = self.convolve(img, Py)

        img_out = np.sqrt((img_x * img_x) + (img_y * img_y))
        img_output = (img_out / np.max(img_out)) * 255
        print(img)
        print(img_output)
        self.image = img
        self.displayImage(2)

        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    @pyqtSlot()
    def robertClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        Sx = np.array([[1, 0],
                       [0, -1]])
        Sy = np.array([[0, 1],
                       [-1, 0]])
        img_x = self.konvolusi(img, Sx)
        img_y = self.konvolusi(img, Sy)
        img_out = np.sqrt(img_x * img_x + img_y * img_y)
        img_output = (img_out / np.max(img_out)) * 255
        print(img)
        print(img_output)
        self.image = img
        self.displayImage(2)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    @pyqtSlot()
    def sobelClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        Sx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
        Sy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
        img_x = self.convolve(img, Sx)
        img_y = self.convolve(img, Sy)
        img_out = np.sqrt(img_x * img_x + img_y * img_y)
        img_output = (img_out / np.max(img_out)) * 255
        print(img)
        print(img_output)
        self.image = img
        self.displayImage(2)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    @pyqtSlot()
    def maxClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_out = img.copy()
        h, w = img.shape[:2]
        for i in np.arange(3, h - 3):
            for j in np.arange(3, w - 3):
                max = 0
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        if a > max:
                            max = a
                b = max
                img_out.itemset((i, j), b)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([], plt.yticks([]))
        plt.show()
        print(img)
        print(img_out)

    @pyqtSlot()
    def minClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_out = img.copy()
        h, w = img.shape[:2]
        for i in np.arange(3, h - 3):
            for j in np.arange(3, w - 3):
                min = 255
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        if a < min:
                            min = a
                b = min
                img_out.itemset((i, j), b)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([], plt.yticks([]))
        plt.show()
        print(img)
        print(img_out)

    @pyqtSlot()
    def medianClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_out = img.copy()
        h, w = img.shape[:2]

        for i in np.arange(3, h - 3):
            for j in np.arange(3, w - 3):
                neighbors = []
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        neighbors.append(a)
                neighbors.sort()
                median = neighbors[24]
                b = median
                img_out.itemset((i, j), b)

            plt.imshow(img_out, cmap='gray', interpolation='bicubic')
            plt.xticks([], plt.yticks([]))
            plt.show()
            print(img)
            print(img_out)

    @pyqtSlot()
    def konvolusi(self, X,F):
        X_height = X.shape[0]
        X_width = X.shape[1]

        F_height = F.shape[0]
        F_width = F.shape[1]

        H = (F_height) // 2
        W = (F_width) // 2

        out = np.zeros((X_height, X_width))

        for i in np.arange(H + 1, X_height - H):
            for j in np.arange(W + 1, X_width - W):
                sum = 0
                for k in np.arange(-H, H ):
                    for l in np.arange(-W, W ):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum += (w * a)
                out[i, j] = sum

        return out

    @pyqtSlot()
    def convolve(self, X,F):
        X_height = X.shape[0]
        X_width = X.shape[1]

        F_height = F.shape[0]
        F_width = F.shape[1]

        H = (F_height) // 2
        W = (F_width) // 2

        out = np.zeros((X_height, X_width))

        for i in np.arange(H + 1, X_height - H):
            for j in np.arange(W + 1, X_width - W):
                sum = 0
                for k in np.arange(-H, H + 1):
                    for l in np.arange(-W, W + 1):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum += (w * a)
                out[i, j] = sum

        return out

    def filteringClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[6, 0, -6],
                           [6, 1, -6],
                           [6, 0, -6]])
        img_out = self.konvolusi(img, kernel)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([], plt.yticks([]))
        plt.show()

    @pyqtSlot()
    def SharpeningClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        laplace = (1.0 / 16) *np.array(
             [[0, 0, -1, 0, 0],
                [0, -1, -2, -1, 0],
                          [-1, -2, 16, -2, -1],
                          [0, -1, -2, -1, 0],
                          [0, 0, -1, 0, 0]])

        img_out = self.convolve(img, laplace)

        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([], plt.yticks([]))
        plt.show()
        print(img)
        print(img_out)

    @pyqtSlot()
    def GaussianClicked(self):
        img = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)

        gauss =(1.0 / 345)* np.array(
            [[1, 5, 7, 5, 1],
             [5, 20, 33, 20, 5],
             [7, 33, 55, 33, 7],
             [5, 20, 33, 20, 5],
             [1, 5, 7, 5, 1]])

        img_out = self.convolve(img, gauss)

        plt.imshow(img_out, cmap='gray', interpolation ='bicubic')
        plt.xticks([], plt.yticks([]))
        plt.show()
        print(img)
        print(img_out)

    @pyqtSlot()
    def SmoothmeanClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        mean = (1.0 / 9)* np.array(
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]])

        img_out = self.konvolusi(img, mean)

        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([], plt.yticks([]))
        plt.show()
        print(img)
        print(img_out)

    @pyqtSlot()
    def meanClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        kernel = np.array(
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]])

        img_out = self.convolve(img, kernel)

        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([], plt.yticks([]))
        plt.show()
        print(img)
        print(img_out)

    @pyqtSlot()
    def aritmatikaClicked(self):
        img1 = cv2.imread('image5.jpg',0)
        img2 = cv2.imread('image3.jpg', 0)
        add_img = img1 + img2
        subtract = img1 - img2
        mul = img1 * img2
        div = img1 / img2

        cv2.imshow('image pertama', img1)
        cv2.imshow('image kedua', img2)
        cv2.imshow('add', add_img)
        cv2.imshow('subtractions', subtract)
        cv2.imshow('multiply', mul)
        cv2.imshow('divide', div)
        print(div)

    @pyqtSlot()
    def logikaClicked(self):
        img1 = cv2.imread('image5.jpg', 1)
        img2 = cv2.imread('image3.jpg', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        opand= cv2.bitwise_and(img1,img2)
        opor = cv2.bitwise_or(img1,img2)
        opxor = cv2.bitwise_xor(img1,img2)

        cv2.imshow('image pertama', img1)
        cv2.imshow('image kedua', img2)
        cv2.imshow('AND', opand)
        cv2.imshow('OR', opor)
        cv2.imshow('XOR', opxor)

    @pyqtSlot()
    def cropClicked(self):
        h,w = self.image.shape[:2]
        start_row, start_col=int(h*.1),int(w*.1)
        end_row, end_col=int(h*.5),int(w*.5)
        crop = self.image[start_row:end_row,start_col:end_col]
        cv2.imshow('Original',self.image)
        cv2.imshow('Crop Image',crop)

    @pyqtSlot()
    def skewedImageClicked(self):
        resize_img = cv2.resize(self.image, (900, 400), interpolation=cv2.INTER_AREA)
        self.image = resize_img
        cv2.imshow('', self.image)

    @pyqtSlot()
    def zoomInClicked(self):
        resize_img = cv2.resize(self.image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        self.image = resize_img
        cv2.imshow('', self.image)
        self.displayImage(2)

    @pyqtSlot()
    def zoomOutClicked(self):
        resize_img = cv2.resize(self.image, None, fx=0.50, fy=0.50)
        self.image = resize_img
        cv2.imshow('', self.image)
        self.displayImage(2)

    @pyqtSlot()
    def contrastStretchingClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        min = 0
        maks = 255
        for i in np.arange(h):
            for j in np.arange(w):
                a = img.item(i, j)
                if a > maks:
                    a = maks
                elif a < min:
                    a = min
        for i in np.arange(h):
            for j in np.arange(w):
                a = img.item(i, j)
                b = float(a - min) / (maks - min) * 255
                img.itemset((i, j), b)
        self.image = img
        self.displayImage(2)

    @pyqtSlot()
    def rotasi(self, degree):
        h, w = self.image.shape[:2]

        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, .7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2
        rot_image = cv2.warpAffine(self.image, rotationMatrix, (h, w))
        self.image = rot_image

    @pyqtSlot()
    def rotasi45Clicked(self):
        self.rotasi(45)
        self.displayImage(2)

    @pyqtSlot()
    def rotasimin45Clicked(self):
        self.rotasi(-45)
        self.displayImage(2)

    @pyqtSlot()
    def rotasi90Clicked(self):
        self.rotasi(90)
        self.displayImage(2)

    @pyqtSlot()
    def rotasimin90Clicked(self):
        self.rotasi(-90)
        self.displayImage(2)

    @pyqtSlot()
    def rotasi180Clicked(self):
        self.rotasi(180)
        self.displayImage(2)

    @pyqtSlot()
    def translasiClicked(self):
        h,w = self.image.shape[:2]
        quarter_h, quarter_w = h / 4, w / 4
        T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
        img=cv2.warpAffine(self.image, T, (w, h))
        self.image = img
        self.displayImage(2)

    @pyqtSlot()
    def histogramEqualizationClicked(self):
        hist, bins = np.histogram(self.image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        self.image = cdf[self.image]
        self.displayImage(2)
        plt.plot(cdf_normalized, color='b')
        plt.hist(self.image.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()

    @pyqtSlot()
    def histogramRGBClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histo = cv2.calcHist([self.image], [i], None, [256], [0, 256])
            plt.plot(histo, color=col)
            plt.xlim([0, 256])
        plt.show()
        self.image = img
        self.displayImage(2)

    @pyqtSlot()
    def histogramGrayscaleClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = img
        self.displayImage(2)
        plt.hist(img.ravel(), 255, [0, 255])
        plt.show()

    @pyqtSlot()
    def binerImageClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        h, w = self.image.shape[:2]
        threshold = 120
        for i in range(h):
            for j in range(w):
                a = img.item(i, j)
                if a > threshold:
                    a = 255
                elif a < threshold:
                    a = 0
                else:
                    a = a
                img.itemset((i, j), a)
        print(img)
        self.image = img
        self.displayImage(2)


    @pyqtSlot()
    def negativeImageClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        maximum_intensity = 255
        h, w = img.shape[:2]
        for i in np.arange(h):
            for j in np.arange(w):
                a = img.item(i, j)
                b = maximum_intensity - a
                img.itemset((i, j), b)
        self.image = img
        self.displayImage(2)

    @pyqtSlot()
    def transposeClicked(self):
        trans_img=cv2.transpose(self.image)
        self.image=trans_img
        self.displayImage(2)

    @pyqtSlot()
    def loadImage(self, flname):
        self.image = cv2.imread(flname)
        self.displayImage()

    @pyqtSlot()
    def grayClicked(self):
        h, w= self.image.shape[:2]
        gray = np.zeros((h,w), np.uint8)
        for i in range (h):
            for j in range (w):
                gray[i,j]=np.clip((0.333*self.image[i,j,0])+(0.333*self.image[i,j,1])+(0.333*self.image[i,j,2]),0,255)

        self.image = gray
        self.displayImage(2)

    @pyqtSlot()
    def contrastClicked(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        contrast = 1.6
        height = gray.shape [0]
        width = gray.shape [1]
        for i in np.arange(height):
            for j in np.arange(width):
                a = gray.item(i, j)
                b = math.ceil(a * contrast)
                if b > 255:
                    b = 255

                gray.itemset((i, j), b)
        self.image = gray
        self.displayImage(2)
        print(gray)

    @pyqtSlot()
    def brightClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        brightness = -50
        h, w=img.shape[:2]
        for i in np.arange(h):
            for j in np.arange(w):
                a = img.item(i,j)
                b = a + brightness
                if b > 255:
                    b = 255
                elif b < 0:
                    b=0
                else:
                    b = b

                img.itemset((i,j),b)
        self.image = img
        self.displayImage(2)

    @pyqtSlot()
    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:
            if (self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        img = img.rgbSwapped()

        if windows == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.imgLabel.setScaledContents(True)
        if windows == 2:
            self.grayLabel.setPixmap(QPixmap.fromImage(img))
            self.grayLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.grayLabel.setScaledContents(True)


app=QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Show Image GUI')
window.show()
sys.exit(app.exec_())