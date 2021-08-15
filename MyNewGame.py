import cv2
import mediapipe as mp
import time
import  HandTrackingModule as htm
ptime = 0
ctime = 0
cap = cv2.VideoCapture(0)
detector = htm.HandDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img,draw=False)
    lmlist = detector.findPosition(img,draw=False)
    if len(lmlist) != 0:
        print(lmlist[4])
    ctime = time.time()
    fp = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, str(int(fp)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)