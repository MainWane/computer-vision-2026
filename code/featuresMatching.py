import cv2  
import numpy as np 
from matplotlib import pyplot as plt 


gray = cv2.imread("../images/pills.jpg", 0)
grayComparison = cv2.imread("../images/stars.jpg", 0)
bgr = cv2.imread("../images/zebra.jpg")
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

gauss = cv2.GaussianBlur(gray, (11, 11), 0)
bilat = cv2.bilateralFilter(gray, 11, sigmaColor=75, sigmaSpace=75)


def orbKeypoints(grayimg):
    orb = cv2.ORB.create()
    keypoints, destination = orb.detectAndCompute(grayimg, None)
    

def siftKeypoints(grayimg):
    sift = cv2.SIFT.create()
    keypoints, destination = sift.detectAndCompute(grayimg, None)


def bruteforceMatching(img1, img2, keypoints1, keypoints2, destination1, destination2):
    brute = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute.match(destination1, destination2)
    # Viser kun 10 bedste matches, ændre matches[:10] for flere/færre
    imageWithMatches = cv2.drawMatches(img1, keypoints1, 
                                       img2, keypoints2, 
                                       matches[:10], None, 
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(imageWithMatches), plt.show()


def knnDistanceMatch(img1, img2, keypoints1, keypoints2, destination1, destination2):
    brute = cv2.BFMatcher()
    matches = brute.knnMatch(destination1, destination2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    imageWithMatches = cv2.drawMatchesKnn(img1, keypoints1, 
                                          img2, keypoints2,
                                          good, None,
                                          flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(imageWithMatches), plt.show()


def flannKnnMatching(img1, img2, keypoints1, keypoints2, destination1, destination2):
    index = dict(algorithm = 1, trees = 5)
    search = dict(checks = 50)
    flann = cv2.flannBasedMatcher(indexParams=index, searchParams=search)
    matches = flann.knnMatch(destination1, destination2, k=2)
    matchesMask = [[0,0] for i in range(len(matches))]
    
    for i, (m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    drawParams = dict(matchColor = (0, 255, 0),
                     singlePointColor = (255, 0, 0),
                     matchesMask = matchesMask,
                     flags = cv2.DrawMatchesFlags_DEFAULT)
    imageWithMatches = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches, None, **drawParams)
    plt.imshow(imageWithMatches,),plt.show()
