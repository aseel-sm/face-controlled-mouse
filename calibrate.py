from time import sleep
import cv2 as cv 
import numpy as np
import mediapipe as mp 


mp_face_mesh = mp.solutions.face_mesh

cap = cv.VideoCapture(0)
x,y=[],[]
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
     
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            # for pt in mesh_points:
            #     cv.circle(frame, pt, int(1), (255,0,255), 1, cv.LINE_AA)
            #     print(pt)
            cv.circle(frame, mesh_points[94], int(1), (255,0,255), 1, cv.LINE_AA)
            marker=mesh_points[94]
            x.append(marker[0])
            y.append(marker[1])
          
            text=f"Max X:{max(x)} Min X:{min(x)} Max y:{max(y)} Min X:{min(y)}"

            cv.putText(frame,text,(10,20),cv.FONT_HERSHEY_SIMPLEX,0.5,(209, 80, 0, 255),1)
            cv.rectangle(frame,(min(x),max(y)),(max(x),min(y)),(209, 80, 0, 255),1)
            sleep(0.1)
        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key ==ord('q'):
            np.save("cords.npy",[min(x),max(y),max(x),min(y)])
            break
cap.release()
cv.destroyAllWindows()