#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 21:24:30 2020

@author: wuhang
"""

### pcd_file_name: 3d_vision/nn_input/pcd/object_objnum_f/v_degree/res_resolution

import numpy as np
import pcl
import cv2
import os
import math
from collections import  Counter

def save_img(img2,filename):
    i1 = np.shape(img2)[0]
    i2 = np.shape(img2)[1]         
    img3 = img2.astype(np.uint8)
    img3 = np.reshape(img3,[i1,i2,1])
    img3 = cv2.resize(img3,(256,256))
    print np.shape(img3)
    #cv2.imshow("new image", img3)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite(filename, img3)
    return img3
    

def refine_img(img2):
    i1 = np.shape(img2)[0]
    i2 = np.shape(img2)[1]
    num = 0
    for i in range (2,i1-3):
        for j in range (1,i2-2):
            if ( (img2[i,j]==0)&(img2[i-1,j]!=0)&(img2[i+1,j]!=0) ):
                img2[i,j] = (img2[i+1,j]+img2[i-1,j])/2.0
                num = num +1
            if ( (img2[i,j]==0)&(img2[i,j-1]!=0)&(img2[i,j+1]!=0) ):
                img2[i,j] = (img2[i,j-1]+img2[i,j+1])/2.0
                num = num +1
            if ( (img2[i,j]==0)&(img2[i,j-2]!=0)&(img2[i,j+1]!=0) ):
                img2[i,j] = (img2[i,j-2]+img2[i,j+1])/2.0
                num = num +1
            if ( (img2[i,j]==0)&(img2[i,j-1]!=0)&(img2[i,j+2]!=0) ):
                img2[i,j] = (img2[i,j-1]+img2[i,j+2])/2.0
                num = num +1
    if (num!=0):
        img2 = refine_img(img2)
    return img2

def load_pcd(loadpath='/home/wuhang/sofa.pcd'):
    cloud = pcl.load(loadpath)
    print('Loaded ' + loadpath)
    cloud_np = np.zeros([cloud.size,3])
    for i in range (0,cloud.size):
        cloud_np[i][0] = cloud[i][0]
        cloud_np[i][1] = cloud[i][1]
        cloud_np[i][2] = cloud[i][2]

    cloud_x = cloud_np[:,0]
    cloud_y = cloud_np[:,1]
    cloud_z = cloud_np[:,2]

    x_min = np.min(cloud_x)-((2.2-(np.max(cloud_x)-np.min(cloud_x)))/2)
    y_min = np.min(cloud_y)-((2.2-(np.max(cloud_y)-np.min(cloud_y)))/2)
    z_min = np.min(cloud_z)

    cloud_mod = cloud_np-np.hstack(( x_min*(np.ones([cloud.size,1])),
                                    y_min*(np.ones([cloud.size,1])),
                                    z_min*(np.ones([cloud.size,1])) ))
    return cloud_mod

def make_depth(cloud_mod,i1=100.0,i2=100.0):
    det1 = 2.2/i1
    det2 = 2.2/i2
    print 'x: inteval '+str(det2)+', y: inteval '+str(det1)+', pixel: '+str(i1)

    img = np.zeros([int(i1),int(i2),3])
    for i in range (0,cloud_mod.shape[0]):
        row = int(cloud_mod[i][1]/det1)
        col = int(cloud_mod[i][0]/det2)
        img[row,col,0] += cloud_mod[i][2]
        img[row,col,1] += 1.0
        img[row,col,2] = float(img[row,col,0]/img[row,col,1])

    #img2 = np.round(img[:,:,2])
    img2 = img[:,:,2]
    return img2

def read_img_pcd(mode, path):
    if mode == 'img':
        out = cv2.imread(path,cv2.IMREAD_GRAYSCALE).astype(float)
    elif mode == 'pcd':
        cloud = pcl.load(path)
        out = np.zeros([cloud.size,3])
        for i in range (0,cloud.size):
            out[i][0] = cloud[i][0]
            out[i][1] = cloud[i][1]
            out[i][2] = cloud[i][2]
    else:
        print 'Error: Input a img or pcd'
    return out
    
def resize_pixel(cloud_mod, i1, i2, r0=0):
    img2 = make_depth(cloud_mod,i1,i2)
    img2 = img2[::-1,:]/np.max(img2)*255    
    img2 = refine_img(img2)
    t0 = 0.0
    t = 0.0
    for i in range(0,i1):
        a = img2[i,:]
        if np.sum(a) > 0:
            b = a[np.min(np.argwhere(a!=0)):np.max(np.argwhere(a!=0))+1]
            t0 = t0 + Counter(b)[0]
            t = t + np.size(b)
    r = 1.0-t0/t
    print r
    if (r<0.99):
        r0 = r
        i1 = int(i1 * math.exp(-0.05))
        i2 = int(i2 * math.exp(-0.05))
        img2,r = resize_pixel(cloud_mod, i1, i2, r0)
    return img2,r

def file_name(file_dir):   
    L=[]
    l=[]
    d=[]
    if not os.path.exists(file_dir):
        print 'File '+file_dir+' not exists!'
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.pcd': 
                l.append(file)
                L.append(os.path.join(root, file))
        for dir in dirs:
                d.append(dir)           
    return L,l,d

if __name__ == '__main__':
    pcd_base = '/home/wuhang/3d_vision/nn_input/pcd/'
    png_base = '/home/wuhang/3d_vision/nn_input/img/'
    
    _,_,d = file_name(pcd_base)
    if len(d) == 0:
        print 'Base directory is empty!'
    
    for n in range(0,len(d)):
        L,l,_ = file_name(pcd_base + d[n])
        if not os.path.exists(png_base + d[n]):
            os.makedirs(png_base + d[n])
    
        for i in range (len(L)):
            pcd_dir = L[i]
            pcd_file = l[i]
            save_dir = png_base + d[n] + '/' + pcd_file
            save_dir = save_dir[:-4] + '.png'
            print 'Load path is '+pcd_dir
            print 'Save path is '+save_dir
            cloud_mod = load_pcd(loadpath=pcd_dir)
            img2,_ = resize_pixel(cloud_mod, i1 = 400, i2 = 400)
            img3 = save_img(img2,save_dir)
            img3 = np.reshape(img3,[np.shape(img3)[0],np.shape(img3)[1]])
    print 'Done!'

