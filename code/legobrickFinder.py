import cv2
import numpy as np

img = cv2.imread('C:/Users/ulrik/Desktop/cvmaterial/c1.jpg')
width = 600
scale = width / img.shape[1]
img = cv2.resize(img, None, fx=scale, fy=scale)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

range_color = {
    'red': [(0, 100, 100), (10, 255, 255)],
    'dark_green': [(30, 80, 80), (75, 255, 255)], 
    'yellow': [(20, 100, 100), (30, 255, 255)],
    'blue': [(90, 100, 100), (140, 255, 255)], 
}
mask_comb = np.zeros(img.shape[:2], dtype=np.uint8)

for color, (lower, upper) in range_color.items():
    mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
    mask_comb |= mask

contours, _ = cv2.findContours(mask_comb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_area = 500
contour_filter = [c for c in contours if cv2.contourArea(c) > min_area]

for contour in contour_filter:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('DataFlair', img)
cv2.waitKey(0)
cv2.destroyAllWindows()