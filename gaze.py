import cv2
import mediapipe as mp

# Initialize the MediaPipe face mesh and gaze estimation models
mp_face_mesh = mp.solutions.face_mesh
mp_gaze = mp.solutions.gaze

# Initialize the video capture device
cap = cv2.VideoCapture(0)

# Initialize the MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh, \
    mp_gaze.Gaze(min_detection_confidence=0.5,
                 min_tracking_confidence=0.5) as gaze:
    while cap.isOpened():
        # Read a frame from the video capture device
        success, image = cap.read()
        if not success:
            break

        # Flip the image horizontally for a more natural viewing experience
        image = cv2.flip(image, 1)

        # Convert the image to RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect the face landmarks in the image
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            # Get the facial landmarks for the first face in the image
            face_landmarks = results.multi_face_landmarks[0]

            # Draw the face landmarks on the image
            mp_drawing.draw_landmarks(
                image, face_landmarks, mp_face_mesh.FACE_CONNECTIONS)

            # Get the gaze direction vector for the first eye in the image
            left_eye_landmarks = face_landmarks.landmark[mp_face_mesh.FaceLandmark.LEFT_EYE]
            right_eye_landmarks = face_landmarks.landmark[mp_face_mesh.FaceLandmark.RIGHT_EYE]
            left_eye_coords = (left_eye_landmarks.x, left_eye_landmarks.y, left_eye_landmarks.z)
            right_eye_coords = (right_eye_landmarks.x, right_eye_landmarks.y, right_eye_landmarks.z)
            gaze_direction = gaze.process(
                image_rgb,
                left_eye_coords=left_eye_coords,
                right_eye_coords=right_eye_coords).gaze_vector

            # Display the gaze direction vector on the image
            x, y = int(gaze_direction[0] * 100), int(-gaze_direction[1] * 100)
            cv2.arrowedLine(image, (320, 240), (320 + x, 240 + y), (0, 0, 255), 2)

        # Display the image
        cv2.imshow('MediaPipe Gaze Estimation', image)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
cv2.destroyAllWindows()
