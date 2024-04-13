import cv2
import mediapipe as mp

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Create a Mediapipe face detection object
mp_face_detection = mp.solutions.face_detection

# Create a Mediapipe drawing object
mp_drawing = mp.solutions.drawing_utils 

while True:
    # Read the frame from the video stream
    ret, frame = cap.read()

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(frame_rgb)
        
        # If a face is detected, draw a bounding box around it and crop the image to only show the face
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)
                x, y, w, h = int(detection.location_data.relative_bounding_box.xmin * frame.shape[1]), \
                             int(detection.location_data.relative_bounding_box.ymin * frame.shape[0]), \
                             int(detection.location_data.relative_bounding_box.width * frame.shape[1]), \
                             int(detection.location_data.relative_bounding_box.height * frame.shape[0])
                frame = frame[y:y+h, x:x+w]

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
