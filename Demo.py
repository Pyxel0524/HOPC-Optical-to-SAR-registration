# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 09:24:53 2022

Optical-SAR registration demo

@author: zpy
"""
import cv2
import numpy as np
from HOPC import *

def NCC(v1, v2):
    return np.mean(np.multiply((v1-np.mean(v1)),(v2-np.mean(v2))))/(np.std(v1)*np.std(v2))

    
if __name__ == '__main__':
    img2 = cv2.imread('D:/Study/Code/Image_Processing/Feature/Descriptor/HOPC/SAR.tif',0);
    HOPC2 = HOPC_descriptor(img2, cell_size=12, bin_size=8)
    vector2, image_hopc = HOPC2.extract()
    H,W = img2.shape
    img1 = cv2.imread('D:/Study/Code/Image_Processing/Feature/Descriptor/HOPC/Optical.tif',0);

    # plt.figure()
    # plt.imshow(img2)
    # plt.figure()
    # plt.imshow(img1[240 : 240 + H, 240 : 240 + W])
    
    
    map_size = 30
    NCC_map = np.zeros([map_size,map_size])
    for i in range(map_size):
        for j in range(map_size):
            HOPC1 = HOPC_descriptor(img1[217 + i : 217 + H + i, 222 + j : 222 + W + j], cell_size=12, bin_size=8)
            vector1, image_hopc = HOPC1.extract()
            NCC_map[i,j] = NCC(vector1,vector2)