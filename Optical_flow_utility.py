# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 21:26:08 2019

@author: Sichuan Huang
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def draw_flow(img, flow):
    step=16
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx*100, y+fy*100]).T.reshape(-1, 2, 2)
    print(x.shape)
    print(y.shape)
    print(fx.shape)
    print(fy.shape)
    print(lines.shape)
    lines = np.int32(lines)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0),2)
    return vis

def draw_contourmap(img,flow):
    h, w = flow.shape[:2]
    y = np.arange(h-1,-1,-1)
    x = np.arange(1,w+1,1)
    fx, fy = flow[:,:,0], flow[:,:,1]
    z = np.sqrt(fx*fx+fy*fy)
    z = np.where(fy<0, -z, z)
    d = 0.1
    lev=np.arange(0,1.0,d)
    plt.set_cmap('viridis')
    vct = plt.contourf(x, y, z,levels=lev)
    return vct

def draw_vector(img, flow, scaling_factor):
    step=20
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    vis = plt.quiver(x,y,fx,fy, scale = 8*scaling_factor, scale_units='xy')
    return vis