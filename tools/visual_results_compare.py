"""
Tools to visualize and compare range images
"""

import numpy as np 
import cv2 
import time
import sys
import yaml
import os
from auxiliary_SK.laserscan import LaserScan, SemLaserScan

def get_semantic_RI(lidar,label,i_width,CFG):
	color_dict = CFG["color_map"]
	learning_map = CFG["learning_map"]

	nclasses = len(color_dict)
	scan = SemLaserScan(nclasses, map_dict = learning_map, sem_color_dict=color_dict,project=True ,W=i_width)
	scan.open_scan(lidar)	
	scan.open_label(label)
	scan.colorize()
	print(scan.proj_sem_label.shape)
	return scan.proj_sem_label


def rgb_semantics(ri,CFG):

	# map to original:
	ri = np.vectorize(CFG["learning_map_inv"].get)(ri)
	img_color = np.zeros((ri.shape[0],ri.shape[1],3),dtype=np.uint8)
	for r in range (ri.shape[0]):
		for c in range (ri.shape[1]):
			img_color[r,c,:] =  CFG["color_map"][ri[r,c]]

	return img_color

def load_and_compare():
	############ CONFIG
	root_bin = "/media/daniel/FILES/UB/Data/data_odometry_velodyne/dataset/sequences/08/velodyne"
	lidar_file = "000002.bin"
	gt_label = "/media/daniel/FILES/UB/Data/data_odometry_velodyne/dataset/sequences/08/labels"
	gt_file = "000002.label"
	pred_label = "/home/daniel/Documents/LU_Net_TFRecord_tempora/LU-Net/lunet/sequences/08/predictions"
	pred_file = "000002.label"
	i_width = 1024  #Image hight 64 always
	config = "/home/daniel/Documents/SemanticKITTI/semantic-kitti-api/config/semantic-kitti.yaml"
	############
	# open config file
	try:
		print("Opening config file %s" % config)
		CFG = yaml.safe_load(open(config, 'r'))
	except Exception as e:
		print(e)
		print("Error opening yaml file.")
		quit()
	
	ri_gt = rgb_semantics(get_semantic_RI(os.path.join(root_bin,lidar_file),os.path.join(gt_label,gt_file),i_width,CFG),CFG)
	rgb_pred = rgb_semantics(get_semantic_RI(os.path.join(root_bin,lidar_file),os.path.join(pred_label,pred_file),i_width,CFG),CFG)

	cv2.namedWindow("Groundtruth")
	cv2.imshow('Groundtruth', ri_gt)

	cv2.namedWindow("Prediction")
	cv2.imshow('Prediction', rgb_pred)
	c = cv2.waitKey(0)
	if 'q' == chr(c & 255):
		quit()
		#print("finish")
if __name__ == "__main__":
	#Function to take .bin lidar and display ground truth and predicted segmentation 
	load_and_compare()
	#normal_display()
	#comprate_ruttine()