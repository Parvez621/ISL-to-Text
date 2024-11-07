import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands   #initialization mediapipe hands
hands = mp_hands.Hands(static_image_mode = False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Press 'q' to quit.")

while cap.isOpened():
    ret,frame = cap.read()

    if ret:
        mirrored_frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB) #convert frame to rgb
        result = hands.process(rgb_frame) # detects hands 

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                #draw hand landmarks ont he frame
                mp_drawing.draw_landmarks(mirrored_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                #get coordinates of the wrist landmark(landmark 0)
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                h, w, _ = mirrored_frame.shape
                wrist_x, wrist_y = int(wrist.x*w), int(wrist.y*h)

                #draw a circle on the wrist position
                cv2.circle(mirrored_frame, (wrist_x, wrist_y), 10, (255,0,0),-1)

        cv2.imshow('Hand Tracking', mirrored_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print("Error: could not read frame.")
        break

cap.release()
cv2.destroyAllWindows()
hands.close()