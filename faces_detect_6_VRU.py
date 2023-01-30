# Include with VGE106 VRU. //
from ctypes.wintypes import RGB
import os
import numpy as np
import cv2
import time
import pickle
from datetime import datetime
import requests

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

# Set parameters for take photo
seconds_between_shots = 5 # Tak a shot send line beetween unknow person
timelapse_img_dir = 'imgsrc/captures/'
iImgTake = 0
if not os.path.exists(timelapse_img_dir):
    os.mkdir(timelapse_img_dir)

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

def linenotify(message, imgfile):
  url = 'https://notify-api.line.me/api/notify'
  token = 'lczHoxFWTwFC3vjJgNEUy1nPRGadvKmQAuLuvxRWQTr' # Line Notify Token SCS
  img = {'imageFile': open(imgfile,'rb')} #Local picture File
  data = {'message': message}
  headers = {'Authorization':'Bearer ' + token}
  session = requests.Session()
  session_post = session.post(url, headers=headers, files=img, data =data)
  print(session_post.text) 

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
    	#print(x,y,w,h)
    	roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
    	roi_color = frame[y:y+h, x:x+w]

    	# recognize? deep learned model predict keras tensorflow pytorch scikit learn
    	id_, conf = recognizer.predict(roi_gray)
    	font = cv2.FONT_HERSHEY_SIMPLEX
    	stroke = 2
    	if conf>=4 and conf <= 85:
    		#print(5: #id_)
    		#print(labels[id_])
    		color = (0, 128, 0)
    		name = labels[id_]
    		cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
    		filename        = f"{timelapse_img_dir}/{iImgTake}.jpg"
    		cv2.imwrite(filename, frame)
    		# time.sleep(seconds_between_shots)
    		#linenotify(name, filename)			
    		#time.sleep(seconds_between_shots)

    	else:
    		color = (255, 0, 0)
    		cv2.putText(frame, "Unknow", (x,y), font, 1, color, stroke, cv2.LINE_AA)
    		# filename = f"{timelapse_img_dir}/{iImgNo}.jpg"
    		# iImgNo += 1
    		# cv2.imwrite(filename, frame)
    		message = 'Found person unknow' #Set your message here!
    		# linenotify(message, filename)			
    		# time.sleep(seconds_between_shots)
    		
    	#img_item = "7.png"
    	#cv2.imwrite(img_item, roi_color)

    	color = (255, 255, 255) #BGR 0-255 
    	stroke = 2
    	end_cord_x = x + w
    	end_cord_y = y + h
    	cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    	
    	# subitems = smile_cascade.detectMultiScale(roi_gray)
    	# for (ex,ey,ew,eh) in subitems:
    	# 	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    	# subitems2 = eye_cascade.detectMultiScale(roi_gray)
    	# for (ex,ey,ew,eh) in subitems2:
    	# 	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
