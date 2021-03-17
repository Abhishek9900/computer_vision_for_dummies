import cv2
import matplotlib.pyplot as plt
import numpy as np
import random



# Call function ORB
def ORB():
    # Initiate ORB detector
    ORB = cv2.ORB_create()

    return ORB

imgpath1 =  "image1.jpg"
imgpath2 =  "image2.jpg"

# Open and Convert the input image from BGR to GRAYSCALE
image1 = cv2.imread(filename = imgpath1,
                   flags = cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(filename = imgpath2,
                   flags = cv2.IMREAD_GRAYSCALE)

normType = cv2.NORM_L2
orb_object = ORB()

kp1, des1 = orb_object.detectAndCompute(image1,None)
kp2, des2 = orb_object.detectAndCompute(image2,None)

# Create BFMatcher object
"""
Brute-force descriptor matcher.

For each descriptor in the first set, this matcher finds the closest descriptor in the second set by trying each one. This descriptor matcher supports masking permissible matches of descriptor sets.

"""
BFMatcher = cv2.BFMatcher(normType = normType,
                            crossCheck = True)

# Matching descriptor vectors using Brute Force Matcher
matches = BFMatcher.match(queryDescriptors = des1,
                            trainDescriptors = des2)

# Sort them in the order of their distance
matches = sorted(matches, key = lambda x: x.distance)

output = cv2.drawMatches(img1 = image1,
                        keypoints1 = kp1,
                        img2 = image2,
                        keypoints2 = kp2,
                        matches1to2 = matches[:30],
                        outImg = None,
                        flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


cv2.imwrite('orb_output.jpg', output)