import math
import time

import cv2
import numpy
from cvzone.HandTrackingModule import HandDetector
import numpy as np  # 用来创建矩阵

cap = cv2.VideoCapture(0)  # 网络摄像头 起名为0
detector = HandDetector(maxHands=1)  # 追踪一个手

offset = 20  # 偏移量不仅仅是获取手的部分，防止损坏
imgSize = 300  # 定义展示图片的面积

counter = 0
folder = "Data/C"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]  # 因为只有一个手
        x, y, w, h = hand['bbox']  # 获取长宽高，存储为boding box

        imgWhite = np.ones((imgSize, imgSize, 3),
                           numpy.uint8) * 255  # 创建一个背景板矩阵300*300*3 ，颜色阈值设为0-255 即为8bit numpy.uint8 *255将背景从1变成255即为白色

        imgCrop = img[y - offset:y + h + offset,
                  x - offset:x + w + offset]  # 因为是矩阵，所以定义他的初始x，y和宽度 创建一个offset变量，获取比手部大一些的图案

        imgCropShape = imgCrop.shape  # 矩阵由长度，宽度和channel组成

        aspectRatio = h / w  # 判断捕捉图形的比例

        if aspectRatio > 1:
            k = imgSize / h  # 获取拉伸常数
            wCal = math.ceil(k * w)  # 始终将w宽度变大 即 3.2变为4 3.6变为4 等等
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape  # 矩阵由长度，宽度和channel组成
            wGap = math.ceil((imgSize - wCal) / 2)  # 将画面移至中心点，通过预先留出位置 即imgSize减去计算出的w宽度 获取最终的位置
            imgWhite[0:imgResizeShape[0], wGap:wCal + wGap] = imgResize  # 将imgResizeShape的手部图像 放在白色背景板上

        else:
            k = imgSize / w  # 获取拉伸常数
            hCal = math.ceil(k * h)  # 始终将w宽度变大 即 3.2变为4 3.6变为4 等等
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape  # 矩阵由长度，宽度和channel组成
            hGap = math.ceil((imgSize - hCal) / 2)  # 将画面移至中心点，通过预先留出位置 即imgSize减去计算出的w宽度 获取最终的位置
            imgWhite[hGap:hCal + hGap, ] = imgResize

        # cv2.imshow("ImageCrop", imgCrop)   #离镜头过近将会失去捕捉画面？  如何解决？
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord("s"):  # 按下s时保存
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)  # 通过时间来命名保存的学习文件
        print(counter)
