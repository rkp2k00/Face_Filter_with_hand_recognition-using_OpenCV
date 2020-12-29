import cv2
import numpy as np
import math as m

cap=cv2.VideoCapture(0)
count=0
while True:

  ret, frame=cap.read()
  blur = cv2.GaussianBlur(frame,(5,5),0)
  frame=cv2.resize(frame,(1000,680))
  roi=frame[150:550,50:440]
  cv2.rectangle(frame,(30,60),(450,560),(255,255,255),3)

  roi1=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
   
  lower_hsv=np.array([0,0,32])
  upper_hsv=np.array([192,106,255])
    
  mask=cv2.inRange(roi1,lower_hsv,upper_hsv)
  res=cv2.bitwise_and(roi,roi,mask=mask)
    
  contours,heirarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  

  for i in contours:

      if cv2.contourArea(i)>20000:
          epsilon = 0.001*cv2.arcLength(i,True)
          approx= cv2.approxPolyDP(i,epsilon,True)
          # cv2.drawContours(roi,contours,-1,(0,0,255),3)
          hull = cv2.convexHull(approx,returnPoints=False)
          defects = cv2.convexityDefects(approx,hull)
          count=0
          
          for j in range(defects.shape[0]):
            s,e,f,d=defects[j,0]
            f1_tip=tuple(approx[s][0])
            f2_tip=tuple(approx[e][0])
            j_pos=tuple(approx[f][0])
          
            # cv2.line(roi,f1_tip,f2_tip,[0,255,0],2) 
            # cv2.circle(roi,dip,5,[0,0,255],-1)
            
            a = m.sqrt((f2_tip[0] - f1_tip[0])**2 + (f2_tip[1] - f1_tip[1])**2)
            b = m.sqrt((j_pos[0] - f1_tip[0])**2 + (j_pos[1] - f1_tip[1])**2)
            c = m.sqrt((f2_tip[0] - j_pos[0])**2 + (f2_tip[1] - j_pos[1])**2)
            
            # apply cosine rule here
            angle = m.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            if angle<=90:
               count+=1
               
      if count==0:
        cv2.putText(frame,'1',(10,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),2)

      elif count==1:
        cv2.putText(frame,'2',(10,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),2)

      elif count==2:
        cv2.putText(frame,'3',(10,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),2)

      elif count==3:
        cv2.putText(frame,'4',(10,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),2)   
      
      elif count==4:
        cv2.putText(frame,'5',(10,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),2)  


  cv2.imshow('mask',mask)
  cv2.imshow('frame',frame)
   
  if cv2.waitKey(1) & 0xFF==ord('q'):
    
    break

cap.release()
cv2.destroyAllWindows()






