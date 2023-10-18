import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe hand solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Capture video from the first camera device
cap = cv2.VideoCapture(0)

def calculate_angle(point1, point2):
    """
    Calculate the angle between the line from point1 to point2 and the horizontal line from the frame center.
    """
    # Calculate the angle of the line with the y-axis
    angle = math.degrees(math.atan2(point1.y - point2.y, point1.x - point2.x))

    # You might need to adjust the returned angle here depending on how you wish to measure the angles
    return angle

while cap.isOpened():
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get the hand landmarks
    results = hands.process(rgb_image)

    # Draw the hand annotations on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Assuming you wish to measure the angle of the line from the wrist (landmark 0) to the middle finger MCP joint (landmark 9)
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

            # Calculate the angle
            angle = calculate_angle(wrist, middle_finger_mcp) - 90
            # print(angle)
            # Set theta_referencia based on the angle
            if angle > 20:
                theta_referencia = 25.0
            elif angle < -20:
                theta_referencia = -25.0
            else:
                theta_referencia = 0

            # Here, integrate the code to send theta_referencia to your system
            # For example, you might use a function like this (you'll need to define it according to your setup):
            # set_theta_reference(theta_referencia)

    # Display the frame
    cv2.imshow('MediaPipe Hands', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
