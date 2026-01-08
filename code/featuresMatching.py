import cv2  
from matplotlib import pyplot as plt 

TARGET_WIDTH = 800

def preprocess_image(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise IOError(f"Billedet findes ikke: {path}")

    # Resize 
    h, w = bgr.shape[:2]
    scale = TARGET_WIDTH / w
    new_size = (int(w * scale), int(h * scale))
    bgr = cv2.resize(bgr, new_size, interpolation=cv2.INTER_AREA)

    # Grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Smoothing 
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return gray


# Preprocess begge billeder
gray = preprocess_image("C:/Users/ulrik/Desktop/cvmaterial/b1.jpg")
grayComparison = preprocess_image("../images/lego.jpg")

# Background subtraction: https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html


#bgr = cv2.imread("../images/zebra.jpg")
#hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

# Ikke nødvendigt i keypoint detection
# gauss = cv2.GaussianBlur(gray, (11, 11), 0)
# bilat = cv2.bilateralFilter(gray, 11, sigmaColor=75, sigmaSpace=75)


def orbKeypoints(grayimg):
    orb = cv2.ORB.create()
    keypoints, destination = orb.detectAndCompute(grayimg, None)
    return keypoints, destination
    

def siftKeypoints(grayimg):
    sift = cv2.SIFT.create()
    keypoints, destination = sift.detectAndCompute(grayimg, None)
    return keypoints, destination

# Kun for ORB keypoints
def bruteforceMatching(img1, img2, keypoints1, keypoints2, description1, description2):
    # Hamming distance som metric er nødvendig for orb keypoints
    brute = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute.match(description1, description2)
    # Viser kun 10 bedste matches, ændre matches[:100] for flere/færre
    imageWithMatches = cv2.drawMatches(img1, keypoints1, 
                                       img2, keypoints2, 
                                       matches[:100], None, 
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(imageWithMatches), plt.show()


def knnDistanceMatch(img1, img2, keypoints1, keypoints2, description1, description2):
    brute = cv2.BFMatcher()
    matches = brute.knnMatch(description1, description2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    imageWithMatches = cv2.drawMatchesKnn(img1, keypoints1, 
                                          img2, keypoints2,
                                          good, None,
                                          flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(imageWithMatches), plt.show()


def flannKnnMatching(img1, img2, keypoints1, keypoints2, description1, description2):
    index = dict(algorithm = 1, trees = 15)
    search = dict(checks = 60)
    flann = cv2.FlannBasedMatcher(indexParams=index, searchParams=search)
    matches = flann.knnMatch(description1, description2, k=2)
    matchesMask = [[0,0] for i in range(len(matches))]
    
    for i, (m,n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            matchesMask[i] = [1.0]

    drawParams = dict(matchColor = (0, 255, 0),
                     singlePointColor = (255, 0, 0),
                     matchesMask = matchesMask,
                     flags = cv2.DrawMatchesFlags_DEFAULT)
    imageWithMatches = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches, None, **drawParams)
    plt.imshow(imageWithMatches,),plt.show()


k1, d1 = siftKeypoints(gray)
k2, d2 = siftKeypoints(grayComparison)

#flannKnnMatching(gray, grayComparison, k1, k2, d1, d2)
#print("Kører SIFT med FlannKNN...")
#knnDistanceMatch(gray, grayComparison, k1, k2, d1, d2)


# === 2) ORB med Bruteforce ===
print("Kører ORB med Bruteforce...")
k1_orb, d1_orb = orbKeypoints(gray)
k2_orb, d2_orb = orbKeypoints(grayComparison)
bruteforceMatching(gray, grayComparison, k1_orb, k2_orb, d1_orb, d2_orb)