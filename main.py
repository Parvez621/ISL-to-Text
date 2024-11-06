import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Press 'q' to quit.")

while cap.isOpened():
    ret,frame = cap.read()

    if ret:
        mirrored_frame = cv2.flip(frame, 1)
        cv2.imshow('Camera', mirrored_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print("Error: could not read frame.")
        break

cap.release()
cv2.destroyAllWindows()