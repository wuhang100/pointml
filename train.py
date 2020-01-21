#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 11:00:01 2019

@author: wuhang
"""

import numpy as np
import tensorflow as tf
import cv2
import sys
import os
import pcl
import pprint
import json
import io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'tfmodel'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, './external/structural_losses'))
try:
    from tf_nndistance import nn_distance
    from tf_approxmatch import approx_match, match_cost
    print('External Losses (Chamfer-EMD) successfully loaded.')
except:
    print('External Losses (Chamfer-EMD) cannot be loaded. Please install them first.')
import pcd2png
import model

obj_name = 'sofa'
obj_num = 7
img_path = './nn_input/img/'+obj_name+str(obj_num)+'.png'
pcd_path_v = './nn_input/'+obj_name+str(obj_num)+'v.pcd'
pcd_path_f = './nn_input/'+obj_name+str(obj_num)+'f.pcd'

img = pcd2png.read_img_pcd('img', img_path)
pc_f = pcd2png.read_img_pcd('pcd', pcd_path_f)

def unicode_convert(input):
    if isinstance(input, dict):
        return {unicode_convert(key): unicode_convert(value) for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [unicode_convert(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

with io.open('./param.json','r') as f:
  param = json.load(f)
param = unicode_convert(param)

#120x200->
#120x200x32->60*100*64->30*50*128->15*25*128->15*25*1
conv_net = param["img_encoder_conv"]

#15*25*1->375 ->...-> 64
fully_con = param["img_encoder_full"]

# 64 ->...-> 2048*3
fully_dec = param["decoder"]

def get_loss(img, pc_in, is_training=True, reuse_mode=False):
    logits = model.img_encoder(img, conv_net, fully_con, is_training=is_training, reuse_mode=reuse_mode)
    pc_out = model.pc_decoder(logits, fully_dec, is_training=is_training, reuse_mode=reuse_mode)
    cost_p1_p2, _, cost_p2_p1, _ = nn_distance(pc_in, pc_out)
    cd_loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
    return cd_loss,pc_out

def network_opt(cd_loss, step):
    lr = tf.train.exponential_decay(0.0001,step,1000,0.9,staircase=True)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        opt_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(cd_loss)
    return opt_step

def train():
    batch_size = 1
    epoch = 1000
    step = tf.Variable(0, trainable = False)
    
    pc_in = tf.placeholder(dtype=tf.float32, shape=[None, 2048, 3], name='pc_input')    
    deepimg = tf.placeholder(dtype=tf.float32, shape=[None, 120, 200], name='img_input')      
    cd_loss,_ = get_loss(deepimg, pc_in)
    opt_step = network_opt(cd_loss, step)
    
    pc_input = np.reshape(pc_f,[batch_size,2048,3])
    img_input = np.reshape(img,[batch_size,120,200])
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tvars = tf.trainable_variables()
        pp = pprint.PrettyPrinter()
        pp.pprint(tvars)
        for e in range(epoch):
            loss,_ = sess.run([cd_loss,opt_step],feed_dict={deepimg:img_input,pc_in:pc_input,step:e})
            print 'Step '+str(e)+': loss is '+str(loss)
        _,pc_gen = get_loss(deepimg, pc_in, is_training=False, reuse_mode=True)
        outputs = sess.run(pc_gen,feed_dict={deepimg:img_input,pc_in:pc_input,step:e})
    return outputs
        
if __name__=='__main__':
    outputs = train()
    cloud = pcl.PointCloud()
    cloud.from_array(outputs[0,])
    pcl.save(cloud,'./test.pcd')
    print (np.shape(outputs))
    print ('OK!')
