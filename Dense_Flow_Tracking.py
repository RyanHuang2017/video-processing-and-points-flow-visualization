#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sichuan Huang
"""


import numpy as np
import cv2 as cv
from optical_flow_utility import draw_contourmap, draw_vector, draw_flow
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

case='_DSC0777'
cap = cv.VideoCapture(case+'.MOV')

fps = cap.get(cv.CAP_PROP_FPS)

# set up the frame size for displaying image
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) # in pixel
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) # in pixel
T, B, L, R = int(0.01*height), int(0.99*height), int(0.01*width), int(0.99*width)

# set up the figure size of the output image
figw= 3.24 # in inch
figh= figw*(B-T)/(R-L) # in inch

scaling_factor = 3.24/(R-L)
#these define the starting and ending frame numbers of interest
frm1, frm2 =100,600

# read the frame info of the video
ret, frame1 = cap.read()

# resize the displaying frame according to the aspect ratio of original video
frame1 = cv.resize(frame1, (width, height))
# display only the desired range of the first frame
frame1 = frame1[T:B, L:R]
# convert the initial first image from BGR color space to gray color space
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame1,dtype=np.uint8)
hsv[...,1] = 255
frameid = []
time = []

while True:
    ret, frame2 = cap.read()
    
    if not ret:
        break
    
    # resize the frame size to the user-defined frame size
    frame2 = cv.resize(frame2, (width, height))
    frame2 = frame2[T:B, L:R]

    # dense flow tracking through comparing two neighbouring frames
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 20, 3, 5, 1.1, 0)

    # get the time info for each frame
    frmid = cap.get(cv.CAP_PROP_POS_FRAMES); 
    frameid.append(frmid)
    time = np.dot(frameid, 1/fps)
    t=time[-1]

    # print frames within the desired range
    if (frmid >= frm1) and (frmid < frm2):
        fig = plt.figure(figsize=[figw, figh])
        vis = draw_vector(prvs,flow, scaling_factor)
        plt.axis('off')
        plt.text(0.05, 0.9, "Time: " + str(round(t, 3)) + "s, Frame id: " + str(int(frmid)), transform=fig.transFigure, color = 'w', fontsize=9)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
        fig1 = plt.gcf()
        plt.imshow(next)       
        fig1.savefig(case+'_'+str(frmid)+'_'+'contourmap.png', dpi=600)
        # plt.show()
        plt.close('all')
        
    elif frmid==frm2:
        break
    prvs = next

cap.release()
cv.destroyAllWindows()
cv.waitKey(1)