import os
import numpy as np
import pandas as pd
import re

# read files
files = os.listdir('dataset/imgs')
files = [file[:-4] for file in files]

# read train - test split
with open("scene_splits/train_scenes.txt") as fout:
    train_split = fout.read()
    
with open("scene_splits/test_scenes.txt") as fout:
    test_split = fout.read() 
    
train_split = set(train_split.split("\n"))
test_split = set(test_split.split("\n"))

train_files = []
test_files = []

for file in files:
    scene, frame_idx = file.split(".")
    
    if scene in train_split:
        train_files.append(file)
    else:
        test_files.append(file)
        
train_csv = pd.DataFrame(train_files, columns=["name"])
test_csv = pd.DataFrame(test_files, columns=["name"])

train_csv.to_csv("dataset/train.csv", index=False)
test_csv.to_csv("dataset/test.csv", index=False)
