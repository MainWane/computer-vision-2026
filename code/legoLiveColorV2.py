import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Kan ikke åbne kamera")

# Angiver HSV ranges
range_color = {
    'red': [(170, 100, 100), (176, 255, 255)],
    'green': [(40, 80, 50), (85, 255, 255)],
    'yellow': [(14, 100, 100), (30, 255, 255)],
    'blue': [(90, 100, 100), (140, 255, 255)],
}

# BGR farver til firkanter
draw_colors = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'yellow': (0, 255, 255),
    'blue': (255, 0, 0),
}

min_area = 500
TARGET_WIDTH = 600

# Nye parametre for validering
MIN_ASPECT_RATIO = 0.4  # Tillader rektangulære klodser
MAX_ASPECT_RATIO = 2.5
MIN_EXTENT = 0.6  # Contour-fylde (area/boundingRect-area)
MIN_EDGE_DENSITY = 0.15  # Procent af pixels i ROI med edges

def show_hsv_on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = param
        print("HSV:", hsv[y, x])

def validate_lego_shape(contour, roi_gray):
    """
    Validerer om konturen ligner en legoklods baseret på:
    - Aspect ratio (bredde/højde forhold)
    - Extent (hvor meget konturen fylder i bounding box)
    - Edge density (kantdetektering i ROI)
    """
    x, y, w, h = cv2.boundingRect(contour)
    
    # 1. Aspect ratio tjek
    aspect_ratio = w / float(h)
    if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
        return False
    
    # 2. Extent tjek (hvor 'fyldt' er konturen?)
    area = cv2.contourArea(contour)
    rect_area = w * h
    extent = area / float(rect_area)
    if extent < MIN_EXTENT:
        return False
    
    # 3. Edge density tjek (legoklodser har skarpe, uniforme kanter)
    edges = cv2.Canny(roi_gray, 50, 150)
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.size
    edge_density = edge_pixels / float(total_pixels)
    
    if edge_density < MIN_EDGE_DENSITY:
        return False
    
    return True

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize
    scale = TARGET_WIDTH / frame.shape[1]
    frame = cv2.resize(frame, None, fx=scale, fy=scale)
    
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    mask_comb = np.zeros(frame.shape[:2], dtype=np.uint8)
    for lower, upper in range_color.values():
        mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
        mask_comb |= mask
    
    contours, _ = cv2.findContours(
        mask_comb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    img_area = frame.shape[0] * frame.shape[1]
    max_area = img_area * 0.08
    
    for color_name, (lower, upper) in range_color.items():
        mask = cv2.inRange(
            hsv_img,
            np.array(lower),
            np.array(upper)
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for c in contours:
            area = cv2.contourArea(c)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(c)
                
                # Udsnit ROI til validering
                roi_gray = gray[y:y+h, x:x+w]
                
                # Valider om det er en legoklods
                if validate_lego_shape(c, roi_gray):
                    # Tegner farvet firkant for validerede klodser
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        draw_colors[color_name],
                        2
                    )
                    # Tekstlabel
                    cv2.putText(
                        frame,
                        color_name,
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        draw_colors[color_name],
                        2,
                        cv2.LINE_AA
                    )
    
    cv2.imshow("Lego detection", frame)
    cv2.setMouseCallback("Lego detection", show_hsv_on_click, hsv_img)
    
    # ESC for exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()