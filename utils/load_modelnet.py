#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 12:28:03 2019

@author: wuhang
"""

import openmesh as om
import numpy as np
import os

if not os.path.exists('../data/ply_modelnet/airplane/train/'):
    os.makedirs('../data/ply_modelnet/airplane/train/')

mesh = om.read_trimesh("../data/ModelNet40/airplane/train/airplane_0001.off")
om.write_mesh('../data/ply_modelnet/airplane/train/airplane_0001.ply', mesh)
print "OK!"