import cv2
import dlib
import time
import numpy as np
from scipy.spatial import distance as dist
from playsound import playsound
import threading

alarm_playing = False 

# Function to play alarm sound in a separate thread
def play_alarm():
    global alarm_playing
    playsound("alarm_sound.mp3")  # Play sound
    alarm_playing = False  # Reset flag after sound ends



# Eye Aspect Ratio (EAR) Calculation
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical
    B = dist.euclidean(eye[2], eye[4])  # Vertical
    C = dist.euclidean(eye[0], eye[3])  # Horizontal
    ear = (A + B) / (2.0 * C)
    return ear


detector = dlib.get_frontal_face_detector()  # Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Pretrained landmarks model

# Define eye landmarks (left and right)
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))


EYE_CLOSURE_THRESHOLD = 0.25  # EAR below this value indicates eye closure
EYE_CLOSURE_TIME = 3  # Time (in seconds) to trigger the alert

start_time = None  # Track when eyes were first detected as closed

cap = cv2.VideoCapture(0)  # Start video capture

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Extract left and right eye landmarks
        left_eye = landmarks[LEFT_EYE]
        right_eye = landmarks[RIGHT_EYE]

        # Calculate EAR for both eyes
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Check if eyes are closed
        if ear < EYE_CLOSURE_THRESHOLD:
            if start_time is None:
                start_time = time.time()
            else:
                duration = time.time() - start_time
                if duration > EYE_CLOSURE_TIME:
                    cv2.putText(frame, "ALERT! Eyes closed!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

                    # Start alarm if not already playing
                    if not alarm_playing:
                        alarm_playing = True
                        threading.Thread(target=play_alarm, daemon=True).start()
        else:
            start_time = None  # Reset if eyes are open
            alarm_playing = False  # Stop alarm

        # Draw eye landmarks
        for (x, y) in np.concatenate((left_eye, right_eye)):
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Eye Closure Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
