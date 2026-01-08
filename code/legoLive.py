import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Kan ikke åbne kamera")

range_color = {
    'red': [(0, 100, 100), (10, 255, 255)],
    'dark_green': [(30, 80, 80), (75, 255, 255)],
    'yellow': [(20, 100, 100), (30, 255, 255)],
    'blue': [(90, 100, 100), (140, 255, 255)],
}

min_area = 500
TARGET_WIDTH = 600

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize (meget vigtigt for performance)
    scale = TARGET_WIDTH / frame.shape[1]
    frame = cv2.resize(frame, None, fx=scale, fy=scale)

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_comb = np.zeros(frame.shape[:2], dtype=np.uint8)

    for lower, upper in range_color.values():
        mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
        mask_comb |= mask
    
    contours, _ = cv2.findContours(
        mask_comb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:
        if cv2.contourArea(c) > min_area:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Lego detection (live)", frame)

    # ESC for exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
