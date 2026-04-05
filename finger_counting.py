import cv2
import mediapipe as mp
import time
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# FPS timing
pTime = 0

# Function to calculate 2D Euclidean distance between two landmarks
def get_distance(point1, point2):
    return math.hypot(point1.x - point2.x, point1.y - point2.y)

# Start MediaPipe Hands
with mp_hands.Hands(max_num_hands=4, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Convert back to BGR for OpenCV display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        total_fingers = 0
        tip_ids = [4, 8, 12, 16, 20]  # Thumb tip and other fingertips

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                fingers = []

                # ✅ Thumb Detection (distance-based)
                thumb_tip = hand_landmarks.landmark[tip_ids[0]]
                thumb_ip = hand_landmarks.landmark[3]  # Joint before thumb tip
                thumb_distance = get_distance(thumb_tip, thumb_ip)

                if thumb_distance > 0.04:  # This threshold works well in most cases
                    fingers.append(1)
                else:
                    fingers.append(0)

                # ✅ Other 4 Fingers (Y-coordinate logic)
                for i in range(1, 5):
                    fingertip = hand_landmarks.landmark[tip_ids[i]]
                    pip_joint = hand_landmarks.landmark[tip_ids[i] - 2]

                    if fingertip.y < pip_joint.y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Add this hand's finger count to total
                total_fingers += fingers.count(1)

        # Display total finger count
        cv2.putText(image, f'Total Fingers: {total_fingers}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        # FPS display
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, f'FPS: {int(fps)}', (50, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the output
        cv2.imshow("Hand Tracking", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()