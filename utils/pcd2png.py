import numpy as np
import pcl
import matplotlib.pyplot as plt
import cv2
import os
import math
from collections import  Counter

def save_img(img2,filename):
    i1 = np.shape(img2)[0]
    i2 = np.shape(img2)[1]         
    img3 = img2.astype(np.uint8)
    img3 = np.reshape(img3,[i1,i2,1])
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

def plot_cloud_2d(cloud_mod):
    fig = plt.figure(dpi=72)
    plt.axis( [0, 2.2, 0, 2.2] )
    plt.gca().set_aspect('equal', adjustable='box') 
    sc1 = fig.add_subplot(111) 
    sc1.scatter(cloud_mod[:,0], cloud_mod[:,1], marker=',',s = 1)
    plt.show()
    
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

for i in range(20,171,20):
    pcd_dir = '/home/wuhang/3d_vision/nn_input/sofa7v'+str(i)+'.pcd'
    save_dir = '../nn_input/img/sofa7v'+str(i)+'.png'
    cloud_mod = load_pcd(loadpath='/home/wuhang/3d_vision/nn_input/pcd/sofa7f10/res0.1.pcd')
    i1 = 300
    i2 = 300
    img2,_ = resize_pixel(cloud_mod, i1 = 400, i2 = 400)
    plot_cloud_2d(cloud_mod)
    img3 = save_img(img2,save_dir)
    img3 = np.reshape(img3,[np.shape(img3)[0],np.shape(img3)[1]])
