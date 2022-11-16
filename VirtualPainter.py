#####################
# Title:- Virtual-Sketchpad
# 1) Ujjwal Pathak 211105
# 2) Aamya Chauhan 211112
#####################

import cv2
import os
import numpy as np
import HandTrackingModule as handTracModule

# defining constants in (px)
brushThickness = 25
eraserThickness = 65

# catching assets
folderPath = "Header"
myList = os.listdir(folderPath)
# print(myList)

# list for assets
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(image)
    overlayList.append(image)

# initial header
header = overlayList[0]
# pink color (default)
drawColor = (255, 0, 255)

# video access
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# calling class
detector = handTracModule.handTracking(detectionCon=0.5, maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # 1. import video
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. find hand locations
    img = detector.findHands(img)
    # list with hand coordinates
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # index finger coordinated
        x1, y1 = lmList[8][1:]
        # middle finger coordinated
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # 4. If Selection Mode
        # if first index and secound index == 1 (i.e they are up)
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            # print("Selection Mode")
            # # Checking for the click
            if y1 < 125:
                # print(x1)
                if 245 < x1 < 390:
                    header = overlayList[0]
                    drawColor = (178, 102, 255)
                elif 391 < x1 < 536:
                    header = overlayList[1]
                    drawColor = (0, 0, 255)
                elif 537 < x1 < 682:
                    header = overlayList[2]
                    drawColor = (33, 239, 239)
                elif 683 < x1 < 828:
                    header = overlayList[3]
                    drawColor = (0, 165, 255)
                elif 829 < x1 < 974:
                    header = overlayList[4]
                    drawColor = (0, 255, 0)
                elif 975 < x1 < 1280:
                    header = overlayList[5]
                    drawColor = (0, 0, 0)
                    # to show selected color
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25),
                          drawColor, cv2.FILLED)

        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            # print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)

            # conditions to draw
            # condition to use eraser
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1),
                         drawColor, eraserThickness)

            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1),
                         drawColor, brushThickness)
            xp, yp = x1, y1

        # Clear Canvas when all fingers are up
        if all(x >= 1 for x in fingers):
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting the header image
    img[0:125, 0:1280] = header
    cv2.imshow("Image", img)
    cv2.waitKey(1)
