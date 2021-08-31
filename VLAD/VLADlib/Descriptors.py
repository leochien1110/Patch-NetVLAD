#surf descriptor
# Jorge Guevara

import cv2
import numpy as np 


#doc at http://docs.opencv.org/master/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html#gsc.tab=0
def describeSURF( image):
    surf = cv2.xfeatures2d.SURF_create()
    # it is better to have this value between 300 and 500
    surf.setHessianThreshold(400)
    kp, des = surf.detectAndCompute(image,None)
    return kp,des

#doc at http://docs.opencv.org/master/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html#gsc.tab=0
def describeSIFT( image):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)
    kp, des = sift.detectAndCompute(image,None)
    #draw keypoints
    # img2 = cv2.drawKeypoints(image,kp,None,(255,0,0),4)
    # img2 = cv2.resize(img2, (1280,960))
    # cv2.imshow('SIFT keypoints',img2)
    # cv2.waitKey(0)
    return kp,des

def describeORB( image):
    #An efficient alternative to SIFT or SURF
    #doc http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_orb/py_orb.html
    #ORB is basically a fusion of FAST keypoint detector and BRIEF descriptor 
    #with many modifications to enhance the performance
    orb=cv2.ORB_create()
    kp, des=orb.detectAndCompute(image,None)
    return kp,des





 

