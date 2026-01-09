import cv2
import numpy as np

cap = cv2.VideoCapture("http://192.168.87.15:8080/video")

if not cap.isOpened():
    raise RuntimeError("Kan ikke åbne telefonkamera-stream")

range_color = {
    'red1': [(0, 150, 60), (8, 255, 255)],
    'red2': [(175, 100, 100), (185, 255, 255)],
    'dark_green': [(50, 60, 10), (80, 255, 255)],
    'yellow': [(18, 40, 150), (25, 255, 255)],
    'blue': [(85, 50, 50), (135, 255, 255)],
}

min_area = 500
TARGET_WIDTH = 600

def show_hsv_on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = param
        print("HSV:", hsv[y, x])

while True:
    ret, frame = cap.read()
    if not ret:
        break
# Resize
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

    img_area = frame.shape[0] * frame.shape[1]
    max_area = img_area * 0.12
    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Lego detection", frame)
    cv2.setMouseCallback("Lego detection", show_hsv_on_click, hsv_img)


    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()