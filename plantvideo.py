import numpy as np
import cv2

# Open the video file
cap = cv2.VideoCapture("C:/Users/suchi/OneDrive/Desktop/Plant-Height-Detection-master/Vid1.mp4")

# Read a frame to get the frame size
ret, frame = cap.read()
if not ret:
    print("Error reading video file")
    exit()

# Get the frame size
height, width = frame.shape[0:2]

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break  # End of video, exit the loop

    # Resize the frame
    frame = cv2.resize(frame, (int(width/2), int(height/2)))

    # Convert to HSV and apply color thresholding
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_color = np.array([28, 70, 133])
    upper_color = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Apply bitwise AND to segment the plant
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Find contours and draw bounding rectangles
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if(area > 1000):
            cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame, (int(x+w/2), y), 5, (0, 255, 255), -1)
            cv2.circle(frame, (int(x+w/2), y+h), 5, (0, 0, 255), -1)
            cv2.putText(frame, " Height:" + str(h), (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the output
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)

    # Exit on key press
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # Esc key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()