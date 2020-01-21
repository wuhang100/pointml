#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 21:28:27 2019

@author: wuhang
"""

import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util2

activation = {"relu":tf.nn.relu,"sigmoid":tf.nn.sigmoid,"none":None}

def img_encoder(image, conv_net, fully_con, is_training=True, reuse_mode=False): 
    layer = tf.reshape(image, [-1,np.shape(image)[1],np.shape(image)[2],1])
    with tf.variable_scope(conv_net['name'], reuse=reuse_mode):
        for i in range (len(conv_net['layers'])):
            layer = tf_util2.conv_2d(conv_net['layers'][i],layer,conv_net['outs'][i],
                                     is_training,reuse_mode,activation[conv_net['activation'][i]],          
                                     conv_net['padding'][i],conv_net['kenel'][i],conv_net['stride'][i],
                                     0.001,'xaiver',use_bn = conv_net['bn'][i],bn_momentum = 0.9)        
        layer = tf.layers.Flatten()(layer)
    with tf.variable_scope(fully_con['name'], reuse=reuse_mode):
        for i in range (len(fully_con['layers'])):
            layer = tf_util2.fully_connection(fully_con['layers'][i],layer,fully_con['outs'][i],
                                              is_training,reuse_mode,activation[fully_con['activation'][i]],
                                              0.001,'xaiver',use_bn = fully_con['bn'][i],bn_momentum = 0.9)
    return layer

def pc_decoder(logits, fully_con, is_training=True, reuse_mode=False):
    layer = tf.reshape(logits, [-1,np.shape(logits)[1]])
    with tf.variable_scope(fully_con['name'], reuse=reuse_mode):
        for i in range (len(fully_con['layers'])):
            layer = tf_util2.fully_connection(fully_con['layers'][i],layer,fully_con['outs'][i],
                                              is_training,reuse_mode,activation[fully_con['activation'][i]],
                                              0.001,'xaiver',use_bn = fully_con['bn'][i],bn_momentum = 0.9)
    layer = tf.reshape(layer,[-1,np.shape(layer)[1]/3,3])
    return layer

def pc_vae_encoder(pc_in, is_training, reuse_mode = False, bn_mode = True):
    with tf.variable_scope ('vae_encoder', reuse=reuse_mode):
        # b*2048*3*1 -> b*2048*1*64 -> b*2048*1*128 -> b*2048*1*128 -> b*2048*1*256 -> b*2048*1*256
        inputs = tf.reshape(pc_in, [-1,2048,3,1])
        layer1 = tf_util2.conv_2d ('e1',inputs,64,is_training,reuse_mode,tf.nn.relu,'valid',[1,3],(1,1),use_bn = bn_mode)
        layer2 = tf_util2.conv_2d ('e2',layer1,128,is_training,reuse_mode,tf.nn.relu,'valid',[1,1],(1,1),use_bn = bn_mode)
        layer3 = tf_util2.conv_2d ('e3',layer2,128,is_training,reuse_mode,tf.nn.relu,'valid',[1,1],(1,1),use_bn = bn_mode)
        layer4 = tf_util2.conv_2d ('e4',layer3,256,is_training,reuse_mode,tf.nn.relu,'valid',[1,1],(1,1),use_bn = bn_mode)
        layer5 = tf_util2.conv_2d ('e5',layer4,256,is_training,reuse_mode,tf.nn.relu,'valid',[1,1],(1,1),use_bn = bn_mode)
        # b*2048*1*256 -> b*1*1*256 -> b*256
        layer6 = tf.layers.max_pooling2d(layer5,[2048,1],[1,1],padding='valid')
        layer7 = tf.squeeze(layer6)
        # b*256 -> (z_sigma, z_mu)
        z_sigma = tf_util2.fully_connection ('es',layer7,256,is_training,reuse_mode,None,0.001,use_bn = False)
        z_mu = tf_util2.fully_connection ('em',layer7,256,is_training,reuse_mode,None,0.001,use_bn = False)
        epsion = tf.random_normal(shape=tf.shape(z_sigma),mean=0,stddev=1, dtype=tf.float32)
        latent = z_mu + tf.sqrt(tf.exp(z_sigma)) * epsion
    return latent, z_sigma, z_mu

def pc_vae_decoder(latent, is_training, reuse_mode = False, bn_mode = False, pc_cat = 1):
    with tf.variable_scope ('vae_decoder', reuse=reuse_mode):
        #latent2: b*1*256, ones*latent2: b*2048*256, points: b*2048*3 -> inputs: b*2048*259
        latent2 = tf.reshape(latent, [tf.shape(latent)[0],1,256])
        ones = tf.ones(shape=[tf.shape(latent)[0],2048,1])
        points = tf.random_normal(shape=[tf.shape(latent)[0],2048,3],mean=0,stddev=1, dtype=tf.float32, seed = pc_cat)
        inputs0 = tf.concat([points, tf.matmul(ones,latent2)], axis=2)
        # b*2048*259*1 -> b*2048*1*256 -> b*2048*1*256 -> b*2048*1*128 -> b*2048*1*128 -> b*2048*1*3 -> b*2048*3
        inputs = tf.reshape(inputs0,[-1,2048,259,1])
        layer1 = tf_util2.conv_2d ('g1',inputs,256,is_training,reuse_mode,tf.nn.relu,'valid',[1,259],(1,1),use_bn = bn_mode)
        layer2 = tf_util2.conv_2d ('g2',layer1,256,is_training,reuse_mode,tf.nn.relu,'valid',[1,1],(1,1),use_bn = bn_mode)
        layer3 = tf_util2.conv_2d ('g3',layer2,128,is_training,reuse_mode,tf.nn.relu,'valid',[1,1],(1,1),use_bn = bn_mode)
        layer4 = tf_util2.conv_2d ('g4',layer3,128,is_training,reuse_mode,tf.nn.relu,'valid',[1,1],(1,1),use_bn = bn_mode)
        layer5 = tf_util2.conv_2d ('g5',layer4,3,is_training,reuse_mode,None,'valid',[1,1],(1,1),use_bn = False)
        recon = tf.squeeze(layer5)
    return recon

def pc_gan_discriminator(pc_in, is_training, reuse_mode = False, bn_mode = False):
    with tf.variable_scope ('gan_discriminator', reuse=reuse_mode):
        # b*2048*3*1 -> b*2048*1*64 -> b*2048*1*128 -> b*2048*1*256 -> b*2048*1*1024
        inputs = tf.reshape(pc_in, [-1,2048,3,1])
        layer1 = tf_util2.conv_2d ('d1',inputs,64,is_training,reuse_mode,tf.nn.relu,'valid',[1,3],(1,1),use_bn = bn_mode)
        layer2 = tf_util2.conv_2d ('d2',layer1,128,is_training,reuse_mode,tf.nn.relu,'valid',[1,1],(1,1),use_bn = bn_mode)
        layer3 = tf_util2.conv_2d ('d3',layer2,256,is_training,reuse_mode,tf.nn.relu,'valid',[1,1],(1,1),use_bn = bn_mode)
        layer4 = tf_util2.conv_2d ('d4',layer3,1024,is_training,reuse_mode,tf.nn.relu,'valid',[1,1],(1,1),use_bn = bn_mode)
        # b*2048*1*1024 -> b*1*1*1024 -> b*1024
        layer5 = tf.layers.max_pooling2d(layer4,[2048,1],[1,1],padding='valid')
        layer6 = tf.squeeze(layer5)
        # b*1024 -> b*256 -> b*256 -> b*1
        layer7 = tf_util2.fully_connection ('d7',layer6,256,is_training,reuse_mode,tf.nn.relu,0.001,use_bn = False)
        layer8 = tf_util2.fully_connection ('d8',layer7,256,is_training,reuse_mode,tf.nn.relu,0.001,use_bn = False)
        logits = tf_util2.fully_connection ('logits',layer8,1,is_training,reuse_mode,None,0.001,use_bn = False)
        outputs = tf.sigmoid(logits)
    return outputs, logits
           
