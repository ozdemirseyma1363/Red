import cv2
import numpy as np
kamera=cv2.VideoCapture(0)
dusuk=np.array([90,50,50],np.uint16)#hsv uzayından renk aralığı belirleme
yuksek=np.array([110,255,255],np.uint16)
while True:
    a,goruntu=kamera.read()
    hsv=cv2.cvtColor(goruntu,cv2.COLOR_BGR2HSV)
    blur=cv2.medianBlur(hsv,5)
    mask = cv2.inRange(blur,dusuk,yuksek)
    a, ıkı = cv2.threshold(mask, 160, 179, cv2.THRESH_BINARY)  # ikiliye dönüştürme
    kontur, b = cv2.findContours(ıkı, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = kontur[0]
    alan = cv2.contourArea(cnt)
    daire = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,goruntu.shape[0]/2,param1=200,param2=10,minRadius=1,maxRadius=50) # çözünürlük değreri,min mesafe,metoda özel değerler param1,param2,minimum yarçap ,mak yarı çap)
    if daire is not None:  # NONE DAN FARKLIYSA BOŞ DEĞİLSE
       daire = np.uint16(np.around(daire)) # değer yuvarlama
       for i in daire[0, :]:
          cv2.circle(goruntu,(i[0],i[1]),i[2],(0,255,0),2) # yuvarlak içine alma#merkezi# #yarıçap
    cv2.imshow("resim",goruntu)
    print(daire)
    print(alan)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
kamera.release()
cv2.destroyAllWindows()
