import cv2
import pyautogui

# Initialize the camera
cap = cv2.VideoCapture(0)

# Set the mouse sensitivity
sensitivity = 20

while True:
    # Capture the video frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Move the mouse pointer based on face position
    for (x, y, w, h) in faces:
        # Calculate the center of the face
        face_center_x = x + w/2
        face_center_y = y + h/2

        # Move the mouse pointer based on face position
        pyautogui.moveRel((face_center_x - sensitivity)/sensitivity, (face_center_y - sensitivity)/sensitivity)

    # Display the image
    cv2.imshow('frame', frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
