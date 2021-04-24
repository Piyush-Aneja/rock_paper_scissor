
import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import os
import time
import random

model=keras.models.load_model(r"D:\Models\rock_paper_model_pract.h5")
# print(model.summary())
print("model loaded")
classes={0:"Paper",1:"Rock",2:"Scissor"}

def preprocess_predict_video(img):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=cv2.resize(img,(50,50))
    img=np.expand_dims(img,0)
    img=np.expand_dims(img,3)
    # img=img/255.0
    pred=np.argmax(model.predict([img]))
    pred_class=classes[pred]
    return pred_class,pred
    
user_wins_conditions=["Rock>Scissor","Scissor>Paper","Paper>Rock"]
comp_score,player_score=0,0

def destroy_img_window():
    cv2.destroyWindow("Video")
    cv2.destroyWindow(f"Player({prediction})")
    cv2.destroyWindow("Computer")

def who_wins(user_img,comp_img):
    winner=""
    global player_score,comp_score
    if user_img==comp_img:
        print("Match Tie")
         
    else:
         strr=user_img+">"+comp_img
         if strr in user_wins_conditions:
            print("User Wins")
            player_score+=1
            winner="Player"
         else:
            print("Computer Wins")
            comp_score+=1
            winner="Computer"
    score_board=cv2.imread(r"D:\vs code files\python files\rock_paper_scissor\black.jpg",-1)
    score_board=score_board[0:70,0:500]
    cv2.putText(score_board,f"Player:{str(player_score)}",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    cv2.putText(score_board,f"Computer:{str(comp_score)}",(200,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    
    window_display("Score",score_board,550,100)
    return winner
        

import cv2
vid=cv2.VideoCapture(0)
j=20
f=1
user_img=[]
img=[]
guess_list=[]
comp_turn_path=r"D:\vs code files\python files\rock_paper_scissor\choices"
for img in os.listdir(comp_turn_path):
    # print(img)
    guess_list.append(cv2.imread(os.path.join(comp_turn_path,img),-1))


def window_display(wind_name,img,x,y):
    cv2.namedWindow(wind_name)        # Create a named window
    cv2.moveWindow(wind_name, x,y)  
    cv2.imshow(wind_name, img)

while True:
    
    if j<=100:
      
        ret,img=vid.read()
        if(ret==True):
            img=cv2.rectangle(img,(100,100),(400,400),0.3)
            cv2.putText(img,f"Get ready:{str(j//30)}",(300,450),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            window_display("Video",img,500,200)
            user_img=img[100:400,100:400]
        
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            print("Can't open")
            break
    else:
        cv2.destroyWindow("Video")
        prediction,user_pred_index=preprocess_predict_video(user_img)
        print(prediction)
        
        comp_index=random.choice([0,1,2])
        comp_display=classes[comp_index] #to store what computer is displaying(paper,rock,scissor)
        comp_img=guess_list[comp_index]
        
        winner=who_wins(prediction,comp_display)
        #display image
        if winner=="Player":
            cv2.putText(user_img,"Winner",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        
        if winner=="Computer":
            cv2.putText(comp_img,"Winner",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
           
        window_display(f"Player({prediction})",user_img,500,200)
        window_display("Computer",comp_img,500+user_img.shape[0],200)
        
      
        j=19
        if cv2.waitKey(2000) & 0xFF == ord('q'):    
            break
        else:
            destroy_img_window()
    
    j=j+1
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


