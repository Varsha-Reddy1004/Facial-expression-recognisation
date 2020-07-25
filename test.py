from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier=cv2.CascadeClassifier(r'C:\Users\varsh\OneDrive\Desktop\facial-expression\haarcascade_frontalface_default.xml')
model=load_model(r'C:\Users\varsh\OneDrive\Desktop\facial-expression\Emotion_little_vgg.h5')

class_labels=['angry','disgust','fear','happy','neutral','sad','suprise']
##capturing the video
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    labels=[]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)##Covert it to gray scale
    faces=face_classifier.detectMultiScale(gray,1.3,5)##it is used to scale down the data
    
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)
            
            preds=model.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position=(x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            
        else:
            cv2.putText(frame,'NO FACE FOUND',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
         
    cv2.imshow("Emotion detector",frame)
    if cv2.waitKey(1)& 0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
            
            
