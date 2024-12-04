import cv2
import mediapipe as mp
import json
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# File for storing gesture database
GESTURE_FILE = "gesture_database.json"

# Function to save a gesture to the database
def save_gesture_to_database(label, landmarks, file_name):
    # Normalize landmarks
    normalized_landmarks = normalize_landmarks(landmarks)
    # Prepare gesture entry
    gesture_entry = {"gesture": label, "landmarks": normalized_landmarks}

    # Handle missing or invalid JSON file
    try:
        with open(file_name, "r") as f:
            database = json.load(f)
            if not isinstance(database, list):  # Ensure the file contains a list
                raise ValueError("Invalid database format")
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        database = []  # Initialize as an empty list if file is missing or invalid

    # Append new gesture and save
    database.append(gesture_entry)
    with open(file_name, "w") as f:
        json.dump(database, f, indent=4)
    print(f"Gesture '{label}' saved to database.")


# Function to normalize landmarks
def normalize_landmarks(landmarks):
    x_coords = [lm.x for lm in landmarks.landmark]
    y_coords = [lm.y for lm in landmarks.landmark]
    z_coords = [lm.z for lm in landmarks.landmark]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    min_z, max_z = min(z_coords), max(z_coords)
    return [{"x": (lm.x - min_x) / (max_x - min_x),
             "y": (lm.y - min_y) / (max_y - min_y),
             "z": (lm.z - min_z) / (max_z - min_z)}
            for lm in landmarks.landmark]

# Function to calculate Euclidean distance
def calculate_distance(coord1, coord2):
    return math.sqrt((coord1["x"] - coord2["x"])**2 +
                     (coord1["y"] - coord2["y"])**2 +
                     (coord1["z"] - coord2["z"])**2)

# Function to match detected landmarks with database gestures
def find_matching_gesture(detected_landmarks, database, threshold=0.1):
    detected_landmarks = normalize_landmarks(detected_landmarks)
    best_match = None
    min_distance = float('inf')
    for entry in database:
        distances = [calculate_distance(d, s) for d, s in zip(detected_landmarks, entry["landmarks"])]
        avg_distance = sum(distances) / len(distances)
        if avg_distance < min_distance:
            min_distance = avg_distance
            best_match = entry["gesture"]
    if min_distance < threshold:
        return best_match
    return "No match found"

# Real-time hand gesture recognition
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Press 's' to save a gesture, 'c' to compare gestures, or 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Mirror the frame for natural interaction
    mirrored_frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(mirrored_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # Save gesture
                label = input("Enter gesture label: ")
                save_gesture_to_database(label, hand_landmarks, GESTURE_FILE)
            elif key == ord('c'):  # Compare gesture
                try:
                    with open(GESTURE_FILE, "r") as f:
                        database = json.load(f)
                    match = find_matching_gesture(hand_landmarks, database)
                    print(f"Detected Gesture: {match}")
                except FileNotFoundError:
                    print(f"No gesture database found at {GESTURE_FILE}.")
            elif key == ord('q'):  # Quit
                break

    cv2.imshow("Hand Gesture Recognition", mirrored_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
