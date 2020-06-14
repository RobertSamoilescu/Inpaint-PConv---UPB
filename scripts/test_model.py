#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from util.depth_map_utils import *
import PIL.Image as pil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import util.opt as opt
import os
import cv2
from tqdm import tqdm
from pyquaternion import Quaternion
from util.inverse_warp import *
from util.image import *
from util.io import *
from util.data import *

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from network.net import PConvUNet, VGG16FeatureExtractor, NLayerDiscriminator
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
args = parser.parse_args()


batch_size = 1
n_threads = 4

dataset = UPBDataset('../dataset/', train=False, random_mask=True)
iterator = iter(data.DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=InfiniteSampler(len(dataset)),
    num_workers=n_threads
))

# load vanilla model
modelV = PConvUNet()
start_iter = load_ckpt(
    "../snapshots/ckpt/%s" % (args.model), [('model', modelV)],
    None
)
modelV.eval()
models = [modelV]


# In[3]:


def save(img, idx):
    npimg = img.numpy().transpose(1, 2, 0)
    npimg = np.clip((255 * npimg), 0, 255).astype(np.uint8)
    pil.fromarray(npimg).save("../results/" + str(idx) + ".png")


# In[12]:


def test(idx):
    # extract sample
    sample = next(iterator)
    gt, img, mask = sample['gt'], sample['img'], sample['mask']
    outputs = []

    for model in models:
        # run image through model
        with torch.no_grad():
            output, _ = model(img.float(), mask.float())

        # compute output
        output = mask.float() * img.float() + (1 - mask.float()) * output

        # unnormalize
        output = unnormalize(output).double()
        outputs.append(output)

    img = unnormalize(img.float()).double()
    gt = unnormalize(gt.float()).double()

    # concatenate input, output, gt
    results = torch.cat([img] + outputs + [gt], dim=3)

    # plot image
    save(make_grid(results, nrow=1), idx)


# From left to right: input, output, ground truth

# In[13]:


for i in tqdm(range(10)):
    test(i)


# In[ ]:





# In[ ]:




