#!/usr/bin/env python
# coding: utf-8
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import cv2
import numpy as np
import os
import PIL.Image as pil
import glob
from tqdm import tqdm
import pandas as pd
import pickle as pkl

from pipeline.utils import *
import imgaug.augmenters as iaa
from multiprocessing import Pool, cpu_count
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src_folder", type=str, help="raw dataset directory")
parser.add_argument("--dst_folder", type=str, help="destination directory")
args = parser.parse_args()

monodepth = Monodepth("monodepth", root_dir="./pipeline")

def get_camera_intrinsic():
	camera_intrinsic = np.array([
		[0.61, 0, 0.5],   # width
		[0, 1.22, 0.5],   # height
		[0, 0, 1]], dtype=np.float32)
		
	camera_intrinsic[0, :] *= 256
	camera_intrinsic[1, :] *= 128
	return camera_intrinsic


def get_camera_extrinsic():
	camera_extrinsic = np.zeros((3, 4), dtype=np.float32)
	camera_extrinsic[:3, :3] = np.eye(3)
	return camera_extrinsic


def get_depth_map(img: np.array):
	global monodepth
	
	# transform image to tensor
	img = img.transpose(2, 0, 1)
	timg = torch.tensor(img).unsqueeze(0).float()

	# predict depth
	tdisp = monodepth.forward(timg)
	tdepth = monodepth.get_depth(tdisp)
	depth = tdepth.view(img.shape[1], img.shape[2]).numpy()
	
	return depth


def random_transformation(imgs, intrinsics, extrinsics, depths):
	"""
	Generate random transformation
	@param imgs:         [B, 3, H, W]
	@param depths:       [B, H, W]
	@param intrinsics:   [B, 3, 3]
	@param extriniscs:   [B, 3, 4]
	"""
	# sample random transformation
	B = imgs.shape[0]
	poses = torch.zeros(B, 6).double()
	
	tx = torch.zeros(B)
	ry = torch.zeros(B)
	
	if np.random.rand() < 0.33:
		tx = .75 * 2 * (torch.rand(B) - 0.5)
		ry = .10 * 2 * (torch.rand(B) - 0.5)
	else:
		if np.random.rand() < 0.5:
			tx = 1.5 * 2 * (torch.rand(B) - 0.5)
		else:
			ry = .20 * 2 * (torch.rand(B) - 0.5) 
		
	poses[:, 0], poses[:, 4] = tx, ry
	
	# apply transformation - faster inverse-warp
	projected_imgs, valid_points = inverse_warp(
		img=imgs, 
		depth=depths, 
		pose=poses, 
		intrinsics=intrinsics,
		extrinsics=None
	)
	
	# mask of valid points
	valid_points = valid_points.double()
	projected_imgs = projected_imgs * valid_points.unsqueeze(1)
	return projected_imgs, valid_points



def read_video(file: str, src_folder: str, dst_folder: str, verbose: bool = False):
	seq = iaa.Sequential([
		iaa.CoarseDropout((0.0, 0.10), size_percent=(0.02, 0.25))
	])
	
	# Create a VideoCapture object and read from input file
	# If the input is the camera, pass 0 instead of the video file name
	src_path = os.path.join(src_folder, file)
	dst_path = dst_folder
	cap = cv2.VideoCapture(src_path)
	
	# Check if camera opened successfully
	if (cap.isOpened()== False): 
		print("Error opening video stream or file")

	# make destination folder
	if not os.path.exists(dst_path):
		os.makedirs(dst_path)
		
	# frame index    
	frame_idx = 0    
	
	# Read until video is completed
	while(cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()

		if ret == True:
			frame = frame[:320, ...]
			
			# Display the resulting frame
			if verbose:
				cv2.imshow('Frame',frame)
			
			# save frame image
			img_path = os.path.join(dst_path, "imgs", file[:-6] + "." + str(frame_idx) + ".png")
			img = pil.fromarray(frame[..., ::-1])
			resized_img = img.resize((256, 128))
			resized_img.save(img_path, 'png')
			
			# predict depth and save it
			resized_img = img.resize((512, 256))
			resized_img = np.asarray(resized_img)
			depth = get_depth_map(resized_img)
			resized_depth = cv2.resize(depth, dsize=(256, 128))
			depth_path = os.path.join(dst_path, "depths", file[:-6] + "." + str(frame_idx) + ".pkl")
			with open(depth_path, "wb") as fout:
				pkl.dump(resized_depth, fout)
				
			#save camera intrinsic
			camera_intrinsic = get_camera_intrinsic()
			camera_intrinsic_path = os.path.join(dst_path, "intrinsics", file[:-6] + "." + str(frame_idx) + ".pkl")
			with open(camera_intrinsic_path, "wb") as fout:
				pkl.dump(camera_intrinsic, fout)
				
			# save camera extrinsic
			camera_extrinsic = get_camera_extrinsic()
			camera_extrinsic_path = os.path.join(dst_path, "extrinsics", file[:-6] + "." + str(frame_idx) + ".pkl")
			with open(camera_extrinsic_path, "wb") as fout:
				pkl.dump(camera_extrinsic, fout)
				
			# generate mask
			timg = torch.tensor(resized_img.transpose(2, 0, 1)).unsqueeze(0).double()
			tdepth = torch.tensor(resized_depth).unsqueeze(0).double()
			tintrinsic = torch.tensor(camera_intrinsic).unsqueeze(0).double()
			textrinsic = torch.tensor(camera_extrinsic).unsqueeze(0).double()
			
			# projection
			tprojected_img, tvalid_points = random_transformation(timg, tintrinsic, textrinsic, tdepth)
			
			# mask
			mask = 255 * tvalid_points.to(torch.uint8).squeeze(0).numpy()
			mask = seq(images=mask)
			mask_path = os.path.join(dst_path, "masks", file[:-6] + "." + str(frame_idx) + ".png")
			cv2.imwrite(mask_path, mask)
				
			# increment number of frame
			frame_idx += 1 
				
			# Press Q on keyboard to  exit
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
				
		# Break the loop
		else:
			break

	# When everything done, release the video capture object
	cap.release()

	# Closes all the frames
	cv2.destroyAllWindows()



if __name__ == "__main__":
	if not os.path.exists("dataset"):
		os.makedirs("dataset/imgs")
		os.makedirs("dataset/masks")
		os.makedirs("dataset/depths")
		os.makedirs("dataset/extrinsics")
		os.makedirs("dataset/intrinsics")

	videos = os.listdir(args.src_folder)
	videos = [v for v in videos if v.endswith('.mov')]

	for v in tqdm(videos):
		read_video(v, args.src_folder, args.dst_folder, False)

