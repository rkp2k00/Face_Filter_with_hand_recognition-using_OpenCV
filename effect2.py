import cv2
import numpy as np
from math import hypot
import math as m
import dlib

count=0
# Loading Camera and dog image and Creating mask
cap = cv2.VideoCapture(0)
dog_image = cv2.imread("filter 1.png")
potter_image = cv2.imread("filter 2.png")

# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    frame=cv2.resize(frame,(1000,680))
 
    rows, cols, _ = frame.shape
    
    dog_mask = np.zeros((rows, cols), np.uint8)
    potter_mask = np.zeros((rows, cols), np.uint8)
    dog_mask.fill(0)
    potter_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)

    for face in faces:
        landmarks = predictor(gray_frame, face)

        # smoothing portion

        c1=landmarks.part(0).x
        c2=landmarks.part(16).x
        c3=landmarks.part(19).y-20
        c4=landmarks.part(8).y

        #dog filter coordinates
        top_dog = (landmarks.part(19).x, landmarks.part(19).y-70)
        center_dog = (landmarks.part(30).x, landmarks.part(30).y)
        left_dog = (landmarks.part(17).x, landmarks.part(17).y)
        right_dog = (landmarks.part(26).x, landmarks.part(26).y)

        dog_width = int(hypot(left_dog[0] - right_dog[0],
                           left_dog[1] - right_dog[1]) * 1.7)
        dog_height = int(dog_width * 0.77)

        # New dog position
        top_left = (int(center_dog[0] - dog_width / 2),
                              int(center_dog[1] - dog_height / 2))
        bottom_right = (int(center_dog[0] + dog_width / 2),
                       int(center_dog[1] + dog_height / 2))

        #potter filter coordinates
      
        top_potter = (landmarks.part(19).x, landmarks.part(19).y-50)
        center_potter = (landmarks.part(21).x+13, landmarks.part(21).y-10)
        left_potter = (landmarks.part(36).x, landmarks.part(36).y)
        right_potter = (landmarks.part(45).x, landmarks.part(45).y)

        potter_width = int(hypot(left_potter[0] - right_potter[0],
                           left_potter[1] - right_potter[1]) * 1.7)
        potter_height = int(potter_width * 0.77)

        # New dog position
        top_left_1 = (int(center_dog[0] - dog_width / 2),
                              int(center_dog[1] - dog_height / 2))
        bottom_right_1 = (int(center_dog[0] + dog_width / 2),
                       int(center_dog[1] + dog_height / 2))
               
        # New potter position
        top_left_2 = (int(center_potter[0] - potter_width / 2),
                              int(center_potter[1] - potter_height / 2))
        bottom_right_2 = (int(center_potter[0] + potter_width / 2),
                       int(center_potter[1] + potter_height / 2))
    
######################################################   hand recognition 
    
    roi=frame[150:550,50:440]
    cv2.rectangle(frame,(30,60),(450,560),(255,255,255),3)

    roi1=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
   
    lower_hsv=np.array([0,0,32])
    upper_hsv=np.array([192,106,255])
    
    mask=cv2.inRange(roi1,lower_hsv,upper_hsv)
    res=cv2.bitwise_and(roi,roi,mask=mask)
    
    contours,heirarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


############################################################################################## hand contour found out

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

        #    Adding the new dog filter
           img_gray=cv2.cvtColor(dog_image,cv2.COLOR_BGR2GRAY)
           _,thresh=cv2.threshold(img_gray,225,255,cv2.THRESH_BINARY_INV)
           dog_pig=cv2.bitwise_and(dog_image,dog_image,mask=thresh)
           dog_pig = cv2.resize(dog_pig, (dog_width, dog_height))
           dog_pig_gray = cv2.cvtColor(dog_pig, cv2.COLOR_BGR2GRAY)
           _, dog_mask = cv2.threshold(dog_pig_gray, 20, 255, cv2.THRESH_BINARY_INV)

           dog_area = frame[top_left_1[1]: top_left_1[1] + dog_height,
                    top_left_1[0]: top_left_1[0] + dog_width]
           dog_area_no_dog = cv2.bitwise_and(dog_area, dog_area, mask=dog_mask)
           final_dog = cv2.add(dog_area_no_dog, dog_pig)

           frame[top_left_1[1]: top_left_1[1] + dog_height,
                    top_left_1[0]: top_left_1[0] + dog_width] = final_dog

        #    cv2.imshow("dog area", dog_area_no_dog)
        #    cv2.imshow("dog pig", dog_pig)
        #    cv2.imshow("final dog", final_dog)


        elif count==1:
           cv2.putText(frame,'2',(10,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),2)

           # Adding the new potter filter
           img_gray=cv2.cvtColor(potter_image,cv2.COLOR_BGR2GRAY)
           _,thresh=cv2.threshold(img_gray,220,255,cv2.THRESH_BINARY_INV)
           potter_pig=cv2.bitwise_and(potter_image,potter_image,mask=thresh)
           potter_pig = cv2.resize(potter_pig, (potter_width, potter_height))
           potter_pig_gray = cv2.cvtColor(potter_pig, cv2.COLOR_BGR2GRAY)
           _, potter_mask = cv2.threshold(potter_pig_gray, 20, 255, cv2.THRESH_BINARY_INV)

           potter_area = frame[top_left_2[1]: top_left_2[1] + potter_height,
                    top_left_2[0]: top_left_2[0] + potter_width]
           potter_area_no_potter = cv2.bitwise_and(potter_area, potter_area, mask=potter_mask)
           final_potter = cv2.add(potter_area_no_potter, potter_pig)

           frame[top_left_2[1]: top_left_2[1] + potter_height,
                    top_left_2[0]: top_left_2[0] + potter_width] = final_potter
   
        elif count==2:
           cv2.putText(frame,'3',(10,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),2)
        #    port=frame[c1,c2-c1 : c3,c4-c3]
        #    gaussianblur = cv2.GaussianBlur(port, (5, 5), 0)
           
        #    frame[c1,c2-c1 : c3,c4-c3]=gaussianblur
           
           
        elif count==3:
           cv2.putText(frame,'4',(10,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),2)   
      
        elif count==4:
           cv2.putText(frame,'5',(10,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),2)  


    cv2.imshow("Frame", frame)
    cv2.imshow('mask',mask)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  