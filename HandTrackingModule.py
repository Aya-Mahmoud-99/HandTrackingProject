import cv2
import mediapipe as mp
import time


cap=cv2.VideoCapture(0)
class HandDetector:
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackingCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackingCon=trackingCon
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackingCon)
        self.npDraw=mp.solutions.drawing_utils
    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.npDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosition(self,img,handNo=0,draw=True):
        lmList=[]
        if self.results.multi_hand_landmarks:
            myHnad=self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHnad.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
        return lmList



def main():
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(0)
    detector=HandDetector()

    while True:
        success, img = cap.read()
        img=detector.findHands(img)
        lmlist=detector.findPosition(img)
        if len(lmlist)!=0:
            print(lmlist[4])
        ctime=time.time()
        fp=1/(ctime-ptime)
        ptime=ctime
        cv2.putText(img,str(int(fp)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)

        cv2.imshow("Image",img)
        cv2.waitKey(1)
if __name__ == "__main__":
    main()