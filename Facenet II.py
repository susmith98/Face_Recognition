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

# CONTINUED FROM FACENET I

# DETECTING FACES AND THEIR NAMES FROM AN IMAGE READ FROM DISK

img=cv2.imread(os.path.join(r'D:\ProjectML\FR',r'CC3.jpg'))
temp_img=img.copy()
t_images=[]
faces=fd(img,1)
Coord=[]
for i,f in enumerate(faces):
    a,b,w,h=f.left(),f.top(),(f.right()-f.left()),(f.bottom()-f.top())
    #img=cv2.resize(cv2.cvtColor(img[b:b+h,a:a+w],cv2.COLOR_BGR2GRAY),(160,160))
    cv2.rectangle(temp_img,(a,b),(a+w,b+h),(0,0,255),2)
    Coord.append((a-10,b-10))
    tep=cv2.resize(fa.align(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),f),(160,160))
    cv2.imshow('test',tep)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(base_dir,r'test.jpg'),tep)
    tep=cv2.imread(os.path.join(base_dir,r'test.jpg'))
    t_images.append(tep)
    os.remove(os.path.join(base_dir,r'test.jpg'))


t_images=Standardize(np.array(t_images))
facenetpred=L2_Norm(model.predict(t_images))
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
    cv2.putText(temp_img,i,Coord[c],cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)
    c+=1
        
cv2.imshow('test',imutils.resize(temp_img,height=600))
cv2.waitKey(0)
cv2.destroyAllWindows()


# PREDICTING PROBABILITIES(CONFIDENCE LEVEL) FOR EACH FACE
p=clf.predict_proba(facenetpred)


# CONTINUATION IN FACENET III
