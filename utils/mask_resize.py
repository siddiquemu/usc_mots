from __future__ import division
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
#from t_SNE_plot import*
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
import glob
import math
from matplotlib.patches import Polygon
from scipy.spatial import distance
import scipy.io as sio
from utils.utils_data import extract_patch_RGB
import time
from utils.utils_data import resize_image


def extract_from_dict(box_all):
    if box_all is not None:
        box_list = [b.get('box') for b in box_all if len(b) > 0]
        if(len(box_list)>0):
            box_all = np.array(box_list)
        else:
            box_all = None
    return box_all
#
# read all box_mask feature frame by frame
#
def masks_resize(boxs_raw,masks_raw,max_dim,min_dim,vis=1):
    for i, box in enumerate(boxs_raw):
        # by using box coordinate, remap 28*28 mask from mrcnn head on original image (1920*1080)
        box_ref =boxs_raw
        w = box_ref[2]
        h = box_ref[3]
        mask = resize_image(masks_raw,box_ref,min_dim=max_dim,max_dim=min_dim,min_scale=1,mode='square')
        #patch_rgb = resize_image(patch_rgb,min_dim=128,max_dim=128,min_scale=1,mode='square')
        if vis and i<50:
            #plt.imshow(patch_rgb)
            #plt.ion()
            #plt.pause(0.01)
            plt.imshow(mask)
            #plt.imsave(vis_path+str(i)+'.png',mask,dpi=200)
            plt.ion()
            plt.pause(0.01)
            #plt.close()
            #time.sleep(0.1)
        #patches.append(patch_rgb)
    return mask


def masks_resize_batch(boxs_raw,masks_raw,min_dim,max_dim,vis=1):
    mask_all = []
    for i, box in enumerate(boxs_raw):
        # by using box coordinate, remap 28*28 mask from mrcnn head on original image (1920*1080)
        #box_ref =boxs_raw[i]
        #w = box_ref[2]
        #h = box_ref[3]
        #mask = resize_image(masks_raw[i],box_ref,min_dim=min_dim,max_dim=max_dim,min_scale=1)#mode='square'
        mask = cv2.resize(masks_raw[0], (min_dim, max_dim), interpolation=cv2.INTER_AREA)
        #patch_rgb = resize_image(patch_rgb,min_dim=128,max_dim=128,min_scale=1,mode='square')
        if vis and i<50:
            #plt.imshow(patch_rgb)
            #plt.ion()
            #plt.pause(0.01)
            plt.imshow(mask)
            #plt.imsave(vis_path+str(i)+'.png',mask,dpi=200)
            plt.pause(0.01)
            plt.close()
            #time.sleep(0.1)
        mask_all.append(mask)
        #patches.append(patch_rgb)
    mask_all = np.array(mask_all)
    assert boxs_raw.shape[0]==mask_all.shape[0],'pose and shape instances must be equal'
    return np.array(mask_all)