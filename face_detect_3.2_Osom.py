from ctypes.wintypes import RGB
import os
import numpy as np
import cv2
import time
import pickle
from datetime import datetime
import requests

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

# Set parameters for take photo
seconds_between_shots = 10 # Tak a shot send line beetween unknow person
unknow_img_dir = 'imgsrc/captures/'
iImgNo = 0
if not os.path.exists(unknow_img_dir):
    os.mkdir(unknow_img_dir)

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

def linenotify(message, imgfile):
  url = 'https://notify-api.line.me/api/notify'
  token = 'AzJEkWTUfzlrShxvqhclkJOOJasds8nYpe6rKBhTf1c' # Line Notify Token OSom
  img = {'imageFile': open(imgfile,'rb')} #Local picture File
  data = {'message': message}
  headers = {'Authorization':'Bearer ' + token}
  session = requests.Session()
  session_post = session.post(url, headers=headers, files=img, data =data)
  print(session_post.text) 

while(True):
    duration = 0 #-----> ประกาศตัวแปนเอาไว้ บวก-ลบ เวลา
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    name_tmp = "" #-----> ประกาศตัวแปรเอาไว้ใช้ใส่ชื่อ
    for (x, y, w, h) in faces:
    	start = time.perf_counter_ns() #-----> เริ่มจับเวลา
    	#print(x,y,w,h)
    	roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
    	roi_color = frame[y:y+h, x:x+w]

    	# recognize? deep learned model predict keras tensorflow pytorch scikit learn
    	id_, conf = recognizer.predict(roi_gray)
    	font = cv2.FONT_HERSHEY_SIMPLEX
    	stroke = 2
    	if conf>=4 and conf <= 85:
    		duration = time.perf_counter_ns() - start #-----> ได้เวลาสำหรับคนที่บันทึก
    		#print(5: #id_)
    		#print(labels[id_])
    		color = (253, 2, 203)
    		name_tmp = labels[id_] 
    		#cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
    		print(time.strftime("%H"+":"+"%M"+":"+"%S."))
    		# filename        = f"{unknow_img_dir}/{iImgNo}.jpg"
    		# iImgNo               += 1
    		# cv2.imwrite(filename, frame)
    		# time.sleep(seconds_between_shots)
			
    	else:
			#-----> ได้เวลาสำหรับคนที่ ไม่ได้บันทึก
    		duration = time.perf_counter_ns() - start 
    		color = (4, 252, 27)
			#-----> ได้ Unknow สำหรับคนที่ไม่ได้บันทึก
			name_tmp = "Unknow" 
    		#cv2.putText(frame, "Unknow", (x,y), font, 1, color, stroke, cv2.LINE_AA)
    		filename = f"{unknow_img_dir}/{iImgNo}.jpg"
    		iImgNo += 1
    		cv2.imwrite(filename, frame)
    		message = 'Found person unknow' #Set your message here!
    		linenotify(message, filename)			
    		time.sleep(seconds_between_shots)
    		
    	#img_item = "7.png"
    	#cv2.imwrite(img_item, roi_color)
		#-----> ทำ If Else แล้วใส่ข้อความลงในรูป
    	if duration != 0:
    		txt_label = name_tmp + ":" + str(duration)
    		cv2.putText(frame, name_tmp, (x,y), font, 1, color, stroke, cv2.LINE_AA)
    	else:
    		cv2.putText(frame, name_tmp, (x,y), font, 1, color, stroke, cv2.LINE_AA)
    	color = (255, 255, 255) #BGR 0-255 
    	stroke = 2
    	end_cord_x = x + w
    	end_cord_y = y + h
    	cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    	
    	subitems = smile_cascade.detectMultiScale(roi_gray)
    	for (ex,ey,ew,eh) in subitems:
    		cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    	subitems2 = eye_cascade.detectMultiScale(roi_gray)
    	for (ex,ey,ew,eh) in subitems2:
    		cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
