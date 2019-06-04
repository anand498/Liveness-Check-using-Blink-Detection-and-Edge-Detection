import cv2
import numpy as np
import imutils
import PIL
import pandas as pd
import logging as log
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream,VideoStream
import argparse
from imutils import face_utils
import time
import dlib
import sys
import datetime as dt
from time import sleep
from matplotlib import pyplot as plt
from eye_aspect import eye_aspect_ratio


 
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0
TOTAL = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# dat file should be in the same directory

# grab the indexes 
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


vs = FileVideoStream("videopath").start()# video file name should be in the same directory
fileStream = True
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
time.sleep(1.0)
ij=10
# loop over frames 
while ij>0:
	
	if fileStream and not vs.more():
		break

	frame = vs.read()
	frame = imutils.resize(frame, width=800)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	
	faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
	for (x, y, w, h) in faces:
    		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
    #calculating the eye aspect ratio
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
    # to draw the markings along the eye of the subject
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		if ear < EYE_AR_THRESH:
			COUNTER += 1
		else:
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1

			COUNTER = 0
		if TOTAL>0:
			cv2.putText(frame, "Live!", (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			cv2.putText(frame, "Fake!", (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		ij-=1
		
	#display the frame at every instant
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break



vidcap = cv2.VideoCapture(0)# for using webcam feed
#vidcap = cv2.VideoCapture('blink_detection_demo2.mp4')
success,image = vidcap.read()
count1 = 0
countchanges=0
count2=10
counter=0
while(count2>0):
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    cv2.imwrite("frame%d.jpg" % count1, image)
    
    img = cv2.imread('frame%d.jpg'%count1)
    if(count2==9):
       cv2.imshow("Frame", img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #convert grayscale into threshold image
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((15,15),np.uint8)
    dilate = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, 3)
    contours,hierarchy = cv2.findContours(dilate,2,1)
    cv2.drawContours(img, contours, -1, (0,255,0), 3)
    cv2.imwrite("thresh%d.jpg" % countchanges, thresh)
    countchanges+=1
    img = PIL.Image.open("thresh0.jpg").convert("L")
    imgarr = numpy.array(img)
   #normlaize the array values to 255 if >128 else set it to 0
    imgarr[imgarr > 128] = 255
    imgarr[imgarr < 128]= 0
     # checks for the middle row of the image for detecting the value changes
    a=imgarr[250,:]
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count2-=1


for index,k in enumerate(a):
       if index+1 < len(a):
         if k != a[index+1]:
          count1+=1

if(count1>5):# set the threshold to check the number of binks by the subject
  print("This is a Fake person")
else:
  print("This is a real person")


cv2.destroyAllWindows()
vs.stop()