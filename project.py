import cv2
import mediapipe as mp
import time
import math
import pyttsx3
import threading  # 🔸 For background voice speaking

# Initialize speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Function to speak number in background (so it doesn’t block video)
def speak_number(num):
    threading.Thread(target=lambda: (engine.say(str(num)), engine.runAndWait())).start()

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Start camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

pTime = 0
last_spoken = -1

def get_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        total_fingers = 0
        tip_ids = [4, 8, 12, 16, 20]

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                fingers = []

                # Thumb (X-coordinate logic)
                thumb_tip = hand_landmarks.landmark[tip_ids[0]]
                thumb_ip = hand_landmarks.landmark[tip_ids[0] - 2]
                if thumb_tip.x < thumb_ip.x:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Other 4 fingers (Y-coordinate logic)
                for i in range(1, 5):
                    fingertip = hand_landmarks.landmark[tip_ids[i]]
                    pip_joint = hand_landmarks.landmark[tip_ids[i] - 2]
                    if fingertip.y < pip_joint.y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                total_fingers += fingers.count(1)

        # 🔊 Speak when number changes
        if total_fingers != last_spoken:
            if total_fingers > 0:
                print(f"Detected Fingers: {total_fingers}")  # Debug info
                speak_number(total_fingers)
            last_spoken = total_fingers

        # Display
        cv2.putText(frame, f'Total Fingers: {total_fingers}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime else 0
        pTime = cTime
        cv2.putText(frame, f'FPS: {int(fps)}', (50, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Tracking with Voice", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()