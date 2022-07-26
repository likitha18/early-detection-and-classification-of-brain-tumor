#............................segmentation.....................................#

import os
import cv2
import numpy as np


def main(path):
    img = cv2.imread(path)
    

def segment(path):
    img = cv2.imread(path)
    imgbkp = img
    gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((10,10),np.float32)/64
    st = cv2.filter2D(gry,-1,kernel)
    Avg = cv2.blur(st,(8,8))
    thresh = 80
    img_th = cv2.threshold(Avg, thresh, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    erodeim = cv2.erode(img_th,kernel,iterations = 1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    DilImg = cv2.dilate(erodeim,kernel2,iterations = 1)
    msk = cv2.bitwise_and(gry,gry,mask = DilImg)
    bwi = cv2.threshold(msk, 180, 255, cv2.THRESH_BINARY)[1]
    erodebw = cv2.erode(bwi,kernel,iterations = 1)
    DilBW = cv2.dilate(erodebw,kernel2,iterations = 1)
    contours,val = cv2.findContours(DilBW, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    largest_areas = sorted(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(largest_areas[len(contours)-1])
    finalimg = cv2.rectangle(imgbkp,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('img',finalimg)
    cv2.waitKey(5000)

segment('g8.jpg')
