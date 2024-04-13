from operator import rshift
from time import sleep

import cv2 as cv 
import numpy as np
import mediapipe as mp 
import pyautogui
pyautogui.FAILSAFE = False
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
cords=np.load("cords.npy")
scroll_mode=False
screen_w, screen_h = pyautogui.size()
cap = cv.VideoCapture(0)
prev_x,prev_y=0,0
blink_c_r,blink_c_s,blink_c_l=0,0,0,
prevL=False
def left_blink(a,b):
    print("Left:",(b[1]-a[1]))
    if(b[1]-a[1] < 6):
        return True
    else:
        return False
def right_blink(a,b):
    
    print("Right:",(b[1]-a[1]))
    if(b[1]-a[1] < 6):
        return True
    else:
        return False
lastIteration=[True,True,True]
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        mouseCurX,mouseCurY=pyautogui.position()
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            
            


            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            for pt in mesh_points:
                cv.circle(frame, pt, int(1), (255,0,255), 1, cv.LINE_AA)
                print(pt)
            cv.circle(frame, mesh_points[94], int(1), (255,0,255), 1, cv.LINE_AA)
            cv.circle(frame, mesh_points[159], int(1), (255,0,255), 1, cv.LINE_AA)
            cv.circle(frame, mesh_points[145], int(1), (255,0,255), 1, cv.LINE_AA)
            cv.circle(frame, mesh_points[386], int(1), (255,0,255), 1, cv.LINE_AA)
            cv.circle(frame, mesh_points[374], int(1), (255,0,255), 1, cv.LINE_AA)
            marker=mesh_points[94]

            # print(prev_x,marker[0],prev_x<marker[0], mouseCurX,mouseCurY)
            x=marker[0]/img_w
            y=marker[1]/img_h
            
            width=cords[2]-cords[0]
            height=cords[1]-cords[3]
            x=(marker[0]-cords[0])/width
            y=(marker[1]-cords[3])/height
            # new_x=marker[0]-width
            # new_y=marker[1]-height
            
            screen_x = screen_w * (x -0.09)
            screen_y = screen_h * (y -0.09)
            pyautogui.moveTo(screen_x,screen_y)
                
            if not scroll_mode:

                if left_blink(mesh_points[159],mesh_points[145]):
                
                    if lastIteration[0]:
                        blink_c_l=blink_c_l+1
                        print("L++")
                        lastIteration[0]=True
           


                    
                    if blink_c_l>8:
                        pyautogui.click(button="left")
                        print("Left")
                        blink_c_l=0
                else:
                   
                    blink_c_l=0
        
                if right_blink(mesh_points[386],mesh_points[374]):
                    if lastIteration[1]:
                        blink_c_r=blink_c_r+1
                        print("R++")
                        lastIteration[1]=True
                    if blink_c_r>8:
                        pyautogui.click(button="right")
                        print("Right")
                        blink_c_r=0
                else:
                  
                    blink_c_r=0
            if left_blink(mesh_points[159],mesh_points[145]) and right_blink(mesh_points[386],mesh_points[374]):
                if lastIteration[2]:
                        blink_c_s=blink_c_s+1
                        print("D++")
                        lastIteration[2]=True
                if blink_c_s>4: 
                    pyautogui.doubleClick()
                    blink_c_s=0
            else:
                blink_c_s=0
            # if scroll_mode:
            #     if(marker[1]<prev_y):
            #         pyautogui.scroll(40)
            #     if(marker[1]>prev_y):
            #         pyautogui.scroll(-40)
            prev_y=marker[1]
        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key ==ord('q'):
            break
cap.release()
cv.destroyAllWindows()