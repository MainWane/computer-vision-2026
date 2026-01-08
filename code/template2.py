import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

TARGET_WIDTH_IMG = 800       # Billede
TARGET_WIDTH_TEMPLATE = 400  # Template (juster efter behov)

def preprocess_image(path, target_width):
    bgr = cv.imread(path)
    if bgr is None:
        raise IOError(f"Billedet findes ikke: {path}")
    # Resize 
    h, w = bgr.shape[:2]
    scale = target_width / w
    new_size = (int(w * scale), int(h * scale))
    bgr = cv.resize(bgr, new_size, interpolation=cv.INTER_AREA)
    # Smoothing 
    bgr = cv.GaussianBlur(bgr, (3, 3), 0)
    return bgr

# Stort billede hvor du søger
img = preprocess_image('C:/Users/ulrik/Desktop/cvmaterial/b1.jpg', TARGET_WIDTH_IMG)
assert img is not None, "file could not be read, check with os.path.exists()"
img2 = img.copy()

# Lille template du søger efter
template = preprocess_image('../images/legocrop.jpg', TARGET_WIDTH_TEMPLATE)
assert template is not None, "file could not be read, check with os.path.exists()"

# Debug information
print(f"Image shape: {img.shape}")
print(f"Template shape: {template.shape}")
w, h = template.shape[1], template.shape[0]
print(f"Template w={w}, h={h}")

# Tjek om template er for stort
if w > img.shape[1] or h > img.shape[0]:
    print("ADVARSEL: Template er større end billedet!")
    print(f"Image: {img.shape[1]}x{img.shape[0]}, Template: {w}x{h}")
else:
    print(f"✓ Template størrelse OK - kan scanne {img.shape[1]-w+1} x {img.shape[0]-h+1} positioner")

methods = ['TM_CCOEFF_NORMED','TM_CCORR_NORMED', 'TM_SQDIFF_NORMED']

THRESHOLD = 0.85

for meth in methods:
    img = img2.copy()
    method = getattr(cv, meth)
    
    res = cv.matchTemplate(img, template, method)
    print(f"\n{meth}: Result shape: {res.shape}")
    
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
        score = min_val
        print(f"Match score (min): {score}")
        # For SQDIFF: lavere er bedre
        if score > (1 - THRESHOLD):  # Inverteret threshold
            print("Match score for dårlig - springer over")
            continue
    else:
        top_left = max_loc
        score = max_val
        print(f"Match score (max): {score}")
        # For andre metoder: højere er bedre
        if score < THRESHOLD:
            print("Match score for dårlig - springer over")
            continue
    
    print(f"✓ God match! Location: {top_left}")
    
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
    
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    plt.subplot(121)
    plt.imshow(res, cmap='gray', aspect='auto')
    plt.title(f'Matching Result (score: {score:.3f})')
    plt.xticks([]), plt.yticks([])
    plt.colorbar()
    
    plt.subplot(122)
    plt.imshow(img_rgb)
    plt.title('Detected Point')
    plt.xticks([]), plt.yticks([])
    
    plt.suptitle(meth)
    plt.tight_layout()
    plt.show()


# Se hvor i billedet der er høje match scores
threshold = 0.92

for meth in methods:
    img = img2.copy()
    method = getattr(cv, meth)
    
    res = cv.matchTemplate(img, template, method)
    
    # Normaliser resultat til 0-1 range
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        # Inverter så høje værdier = gode matches
        res_normalized = 1 - cv.normalize(res, None, 0, 1, cv.NORM_MINMAX)
    else:
        res_normalized = cv.normalize(res, None, 0, 1, cv.NORM_MINMAX)
    
    # Find alle matches over threshold
    loc = np.where(res_normalized >= threshold)
    
    print(f"\n{meth}: Fundet {len(loc[0])} matches over {threshold}")
    
    # Tegn alle gode matches
    for pt in zip(*loc[::-1]):
        cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
    
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    plt.subplot(121)
    plt.imshow(res_normalized, cmap='hot', aspect='auto')
    plt.title('Matching Heatmap')
    plt.colorbar()
    
    plt.subplot(122)
    plt.imshow(img_rgb)
    plt.title(f'All Matches (threshold={threshold})')
    
    plt.suptitle(meth)
    plt.tight_layout()
    plt.show()