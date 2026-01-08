import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

TARGET_WIDTH = 800

# BGR preprocess
def preprocess_image(path):
    bgr = cv.imread(path)
    if bgr is None:
        raise IOError(f"Billedet findes ikke: {path}")
    # Resize 
    h, w = bgr.shape[:2]
    scale = TARGET_WIDTH / w
    new_size = (int(w * scale), int(h * scale))
    bgr = cv.resize(bgr, new_size, interpolation=cv.INTER_AREA)
    # Smoothing 
    bgr = cv.GaussianBlur(bgr, (5, 5), 0)
    return bgr

img = preprocess_image('C:/Users/ulrik/Desktop/cvmaterial/l1.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
img2 = img.copy()

template = preprocess_image('../images/legocrop.jpg')
assert template is not None, "file could not be read, check with os.path.exists()"
w, h = template.shape[1], template.shape[0]

# All the 6 methods for comparison in a list
methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR',
            'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = getattr(cv, meth)

    # Apply template Matching
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
     
     # Konverter BGR til RGB for korrekt farve display
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_rgb)
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()