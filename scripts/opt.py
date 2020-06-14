#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from util.depth_map_utils import *
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import cv2
from pyquaternion import Quaternion
import time
from torch.utils.data import DataLoader
from util.data import *
from tqdm import tqdm

# load dataset
upb = UPBDataset(root_dir='../dataset', train=True, normalize=False)

mean = 0.
std = 0.

for i in tqdm(range(len(upb))):
    img = upb[i]['gt']
    img = img.view(img.size(0), -1)
    
    mean += img.mean(1)
    std += img.std(1)
    
mean /= len(upb)
std /= len(upb)


print("Mean", mean)
print("Std", std)




