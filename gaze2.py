
import cv2
import dlib
import pyautogui
import numpy as np
# Initialize the video stream
cap = cv2.VideoCapture(0)

# Load the pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define the range of head movement
MAX_X = 200  # Maximum horizontal movement (pixels)
MAX_Y = 200  # Maximum vertical movement (pixels)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    
    # Resize the frame and convert it to grayscale
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect the face in the frame
    rects = detector(gray, 0)
    
    # Loop over the detected faces
    for rect in rects:
        # Get the facial landmarks for the face
        landmarks = predictor(gray, rect)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        # Compute the head position based on the landmarks
        nose_point = landmarks[30]
        chin_point = landmarks[8]
        head_position = chin_point - nose_point
        
        # Convert the head position to mouse movement
        dx = int(head_position[0] / MAX_X * pyautogui.size()[0])
        dy = int(head_position[1] / MAX_Y * pyautogui.size()[1])
        
        # Move the mouse cursor
        pyautogui.move(dx, dy)
    
    # Display the frame
    cv2.imshow("Head Tracking", frame)
    
    # Exit the program when the "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
