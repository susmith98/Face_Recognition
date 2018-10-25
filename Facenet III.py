#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import keras
from keras import applications
import os
from keras.models import load_model,model_from_json
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import dlib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import imutils
from imutils.face_utils import FaceAligner

# CONTINUED FROM FACENET II

# RECOGNIZING MULTIPLE FACES IN REAL TIME USING WEBCAM VIDEO LIVE 
cam=cv2.VideoCapture(0)
while True:
    ret,img=cam.read()
    temp1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=fd(img,1)
    x=[]
    y=[]
    w=[]
    h=[]
    F=[]
    Fimages=[]
    flag=False
    for i,f in enumerate(faces):
        flag=True
        a=f.left()
        b=f.top()
        x.append(a)
        y.append(b)
        w.append(f.right()-a)
        h.append(f.bottom()-b)
        F.append(f)
        
    #temp=img[y:y+h,x:x+w]
    if flag==True:
        for i in range(len(x)):
            temp=fa.align(temp1,temp1,F[i])
            temp=cv2.resize(temp,(160,160))
            cv2.imwrite(os.path.join(r'D:\ProjectML',r'temp.jpg'),temp)
            temp=cv2.imread(os.path.join(r'D:\ProjectML',r'temp.jpg'))
            os.remove(os.path.join(r'D:\ProjectML',r'temp.jpg'))
            Fimages.append(temp)
            
        Fimages=Standardize(np.array(Fimages))
        facenetpred=L2_Norm(model.predict(Fimages))
        pred=[]
        for i in range(len(facenetpred)):
            t=clf.predict(np.reshape(facenetpred[i],(1,len(facenetpred[i]))))
            #print(t)
            tt=clf.predict_proba(np.reshape(facenetpred[i],(1,len(facenetpred[i]))))
            # FMax and SMax ARE FIRST AND SECOND HIGHEST PROBALITIES OF PREDICTION RESPECTIVELY
            FMax=np.max(tt)
            SMax=np.max(np.delete(tt,tt.argmax()))
            # CONTIDIONS TO FILTER UNKNOWN FACES
                  # FROM THE EMBEDDING DISTANCE CALCULATIONS DISTANCE THRESHOLD PICKED IS 0.6
                  # THRESHOLD FOR THE DIFFERENCE OF FIRST AND SECOND HIGHTEST PREDICTED PROBABILITIES IS PICKED AS 0.3
            if Euclied_dist(labels_emb[t[0]],facenetpred[i])<0.6 and FMax-SMax>0.3:
            # IF BOTH THE ABOVE CONDITIONS ARE SATISFIED IT IS CONSIDERED AS A VALID PREDICTION    
                pred.append(le.inverse_transform(t)[0])
            else:
                pred.append('Unknown')
                
        c=0      
        for i in pred:
            cv2.putText(img,i,(x[c]-10,y[c]-10),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)
            cv2.rectangle(img,(x[c],y[c]),(x[c]+w[c],y[c]+h[c]),(255,0,255),2)
            c+=1
            
    cv2.imshow('Video',img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
           
    
cv2.destroyAllWindows() 
cam.release()








