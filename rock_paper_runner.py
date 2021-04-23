import tensorflow as tf
import keras
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.utils import to_categorical
import os
import seaborn as sns
# import cv2

print("imported")

model=keras.models.load_model(r"D:\Models\rock_paper_model_pract.h5")

print(model.summary())
print("done till here")
classes={0:"paper",1:"rock",2:"scissors"}
def preprocess_predict_video(img):
        # img=cv2.imread(img,0)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=cv2.resize(img,(50,50))
    img=np.expand_dims(img,0)
    img=np.expand_dims(img,3)
    img=img/255.0
    # img=load_img(img,target_size=(50,50),color_mode = "grayscale")
    # img=img_to_array(img)
    
    # print(img.shape)
    
    pred=np.argmax(model.predict([img]))
    pred=classes[pred]
    return pred
    
    
import cv2
vid=cv2.VideoCapture(0)
while True:
    ret,img=vid.read()
#     print(ret)
    if(ret==True):
        img=cv2.rectangle(img,(100,100),(400,400),0.3)
        
        # cv2.imshow("frame",img);
        newimg=img[100:400,100:400]
        
        prediction=preprocess_predict_video(newimg)
        print(prediction)
        cv2.putText(img,str(prediction),(300,450),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.imshow("camera",img);
     
        if cv2.waitKey(10) & 0xFF == ord('q'):
            
            break
    else:
        print("Can't open")
        break
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
