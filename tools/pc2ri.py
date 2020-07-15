"""
Old version of pointcloud to range image tranformation
This algorithm is deprecated and user should use the SemanticKITTI APIs
"""

import numpy as np 
import os
import cv2 
import time
from datetime import datetime
import pandas as pd

def lidar_to_2d_front_view_3(points, v_res=26.9/64,
	#h_res=0.17578125
	h_res=0.08
	):

	x_lidar = points[:, 0]  # -71~73
	y_lidar = points[:, 1]  # -21~53
	z_lidar = points[:, 2]  # -5~2.6
	r_lidar = points[:, 3]  # Reflectance  0~0.99

	# Distance relative to origin
	d = np.sqrt(x_lidar ** 2 + y_lidar ** 2 + z_lidar ** 2)

	# Convert res to Radians
	v_res_rad = np.radians(v_res)
	h_res_rad = np.radians(h_res)

	# PROJECT INTO IMAGE COORDINATES

	# -1024~1024   -3.14~3.14  ;
	x_img_2 = np.arctan2(-y_lidar, x_lidar)# 

	# x_img_2 = -np.arcsin(y_lidar/r)  #   -1.57~1.57

	#identify points whith angles less than a threshold
	#angle_diff = np.abs(np.diff(x_img_2))
	#threshold_angle = np.radians(250)  #
	#angle_diff = np.hstack((angle_diff, 0.001)) # 
	#angle_diff_mask = angle_diff > threshold_angle
	# print('angle_diff_mask',np.sum(angle_diff_mask), threshold_angle)


	x_img = np.floor((x_img_2 / h_res_rad)).astype(int)  #
	x_img -= np.min(x_img)  # 
	x_img[x_lidar < 0] = 0  # 
	# 

	# -52~10  -0.4137~0.078
	y_img_2 = -np.arctan2(z_lidar, r_lidar) #
	# 

	y_img_2 = -np.arcsin(z_lidar/d) # 
	y_img = np.round((y_img_2 / v_res_rad)).astype(int)  # 
	y_img -= np.min(y_img) # 
	y_img[y_img >= 64] = 63 # 



	y_img[y_img >= 64] = 63 #


	x_max = int(360.0 / h_res) + 1  # 
	# x_max = int(180.0 / h_res) + 1  # 

	# 
	depth_map = np.zeros((64, x_max, 5))#+255
	depth_map[y_img, x_img, 0] = x_lidar
	depth_map[y_img, x_img, 1] = y_lidar
	depth_map[y_img, x_img, 2] = z_lidar
	depth_map[y_img, x_img, 3] = r_lidar
	depth_map[y_img, x_img, 4] = d

	# 
	start_index = int(x_max/2 - 256)
	result = depth_map[:, start_index:(start_index+512), :]

	return result

def load_velodyne_binary_labels(velodyne_bin_path, labels_path):
	"""Decode a binary Velodyne example (of the form '<timestamp>.bin') **** From Oxfort Robot Car
	Args:
	    example_path (AnyStr): Oxford Radar RobotCar Dataset binary Velodyne pointcloud example path
	Returns:
	    ptcld (np.ndarray): XYZI pointcloud from the binary Velodyne data Nx4
	Notes:
	    - The pre computed points are *NOT* motion compensated.
	    - Converting a raw velodyne scan to pointcloud can be done using the
	        `velodyne_ranges_intensities_angles_to_pointcloud` function.
	"""
	ext = os.path.splitext(velodyne_bin_path)[1]
	if ext != ".bin":
		raise RuntimeError("Velodyne binary pointcloud file should have `.bin` extension but had: {}".format(ext))
	if not os.path.isfile(velodyne_bin_path):
		raise FileNotFoundError("Could not find velodyne bin example: {}".format(velodyne_bin_path))
	data = np.fromfile(velodyne_bin_path, dtype=np.float32)
	ptcld = data.reshape(-1,4)
	ptcld = np.transpose(ptcld)

	#Read labels
	ext = os.path.splitext(labels_path)[1]
	if ext != ".label":
		raise RuntimeError("label file should have `.label` extension but had: {}".format(ext))
	if not os.path.isfile(velodyne_bin_path):
		raise FileNotFoundError("Could not find label file: {}".format(labels_path))

	# if all goes well, open label
	label = np.fromfile(labels_path, dtype=np.uint32)
	label_ID = label >> 16    # instance id in upper half
	label= label & 0xFFFF #Following SemanticKITTI example (laser_scan.py ln:247)
	
	label = label.reshape(1,-1)
	np.set_printoptions(edgeitems=15)
	label = map_labels(label)
	print(label_ID)
	#ptcld = np.array([[ptcld],[labels_path]])
	#ptcld = np.concatenate((ptcld,label),axis = 0)
	return ptcld,label

def map_labels(label):
	return


def load_velodyne_binary(velodyne_bin_path):
    """Decode a binary Velodyne example (of the form '<timestamp>.bin') **** From Oxfort Robot Car
    Args:
        example_path (AnyStr): Oxford Radar RobotCar Dataset binary Velodyne pointcloud example path
    Returns:
        ptcld (np.ndarray): XYZI pointcloud from the binary Velodyne data Nx4
    Notes:
        - The pre computed points are *NOT* motion compensated.
        - Converting a raw velodyne scan to pointcloud can be done using the
            `velodyne_ranges_intensities_angles_to_pointcloud` function.
    """
    ext = os.path.splitext(velodyne_bin_path)[1]
    if ext != ".bin":
        raise RuntimeError("Velodyne binary pointcloud file should have `.bin` extension but had: {}".format(ext))
    if not os.path.isfile(velodyne_bin_path):
        raise FileNotFoundError("Could not find velodyne bin example: {}".format(velodyne_bin_path))
    data = np.fromfile(velodyne_bin_path, dtype=np.float32)
    ptcld = data.reshape(-1,4)
    ptcld = np.transpose(ptcld)
    return ptcld

def load_velodyne_txt(velodyne_txt_path):
    ext = os.path.splitext(velodyne_txt_path)[1]
    if ext != ".txt":
        raise RuntimeError("Velodyne txt pointcloud file should have `.txt` extension but had: {}".format(ext))
    if not os.path.isfile(velodyne_txt_path):
        raise FileNotFoundError("Could not find velodyne bin example: {}".format(velodyne_txt_path))

    ptcld = pd.read_csv(velodyne_txt_path,sep=' ')
    ptcld = ptcld.to_numpy()
    print(ptcld.shape)
    ptcld = np.transpose(ptcld)
    return ptcld

def pc2ri_pw_360(pc, v_FOV_degrees = 26.9, v_beams=64.0, h_res = 0.08):
	"""
	HDL-64E parameters by default
	"""
	#vertical parameters
	#v_FOV_degrees = np.array([-v_FOV_degrees/2,v_FOV_degrees/2])
	v_FOV_degrees = np.array([-23.9,3])
	v_res = (v_FOV_degrees[1]-v_FOV_degrees[0])/v_beams
	#horizontal parameters
	#h_res = 0.17578125
	horizontal_grids = 1024.0
	h_FOV_degrees = np.array([-180,180]) #degrees
	h_res = (h_FOV_degrees[1]-h_FOV_degrees[0])/horizontal_grids
	
	range_image = np.zeros([int(v_beams),int(horizontal_grids),5])
	counter = 0

	x = pc[0,:]
	y = pc[1,:]
	z = pc[2,:]
	d = np.sqrt((x ** 2) + (y ** 2) + (z ** 2))
	angle_vertical = np.rad2deg((np.arcsin(z / d)))
	angle_azimuth = np.rad2deg(np.arctan2(x,y)) 

	print(pc.shape[1])
	for index in range (pc.shape[1]):
		point = pc[:,index]

		if(h_FOV_degrees[0]<angle_azimuth[index] and h_FOV_degrees[1]>angle_azimuth[index] and (v_FOV_degrees[0])<angle_vertical[index] and (v_FOV_degrees[1])>angle_vertical[index]):

			r_index = np.round((angle_vertical[index]+(-v_FOV_degrees[0])) / v_res).astype(int)
			if (r_index > int(v_beams-1)):
				r_index = int(v_beams-1)

			c_index = np.round((angle_azimuth[index]-h_FOV_degrees[0]) / h_res).astype(int)
			if (c_index > (horizontal_grids-1)):
				c_index = int(horizontal_grids-1)

			if (range_image[r_index,c_index,4]!=0):
				counter += 1
			range_image[r_index,c_index,0:4] =  point
			range_image[r_index,c_index,4] =  d[index]
	range_image = np.flip(range_image,0)
	print(counter)
	return range_image   

def pc2ri_pw(pc, v_FOV_degrees = 26.9, v_beams=64.0, h_res = 0.08):
	"""
	HDL-64E parameters by default
	"""
	#vertical parameters
	#v_FOV_degrees = np.array([-v_FOV_degrees/2,v_FOV_degrees/2])
	v_FOV_degrees = np.array([-23.9,3])
	v_res = (v_FOV_degrees[1]-v_FOV_degrees[0])/v_beams
	#horizontal parameters
	#h_res = 0.17578125
	horizontal_grids = 512.0
	h_FOV_degrees = np.array([45,135]) #degrees
	h_res = (h_FOV_degrees[1]-h_FOV_degrees[0])/horizontal_grids
	
	range_image = np.zeros([int(v_beams),int(horizontal_grids),5])
	counter = 0

	x = pc[0,:]
	y = pc[1,:]
	z = pc[2,:]
	d = np.sqrt((x ** 2) + (y ** 2) + (z ** 2))
	angle_vertical = np.rad2deg((np.arcsin(z / d)))
	angle_azimuth = np.rad2deg(np.arctan2(x,y)) 


	for index in range (pc.shape[1]):
		point = pc[:,index]

		if(h_FOV_degrees[0]<angle_azimuth[index] and h_FOV_degrees[1]>angle_azimuth[index] and (v_FOV_degrees[0])<angle_vertical[index] and (v_FOV_degrees[1])>angle_vertical[index]):

			r_index = np.round((angle_vertical[index]+(-v_FOV_degrees[0])) / v_res).astype(int)
			if (r_index > 63):
				r_index = 63

			c_index = np.round((angle_azimuth[index]-h_FOV_degrees[0]) / h_res).astype(int)
			if (c_index > 511):
				c_index = 511


			range_image[r_index,c_index,0:4] =  point
			range_image[r_index,c_index,4] =  d[index]
	range_image = np.flip(range_image,0)
	return range_image



if __name__ == "__main__":
	
	"""
	#pc = load_velodyne_binary("./2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000000.bin")
	pc_path = "/home/daniel/Documents/pointCloud_RangeImage/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000010.bin"
	pc = load_velodyne_binary(pc_path)
	label_path = "/home/daniel/Documents/SemanticKITTI/data_odometry_labels/dataset/sequences/00/labels/000010.label" 
	#pc = load_velodyne_txt("/home/daniel/Documents/pointCloud_RangeImage/2011_09_26_drive_0048_extract_city/2011_09_26/2011_09_26_drive_0048_extract/velodyne_points/data/0000000000.txt")
	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	#np.savetxt(current_time + '_mini_pc.txt', np.float32(np.transpose(pc)),fmt='%1.6e')	
	
	compare_pc = np.load("/home/daniel/Documents/LU_Net_Original/lidar_2d/2011_09_26_0001_0000000010.npy")

	#ri = pc2ri(pc)
	image = np.uint8(255*compare_pc[:,:,4]/np.max(compare_pc[:,:,4]))
	compare_pc_colors = cv2.applyColorMap(image, cv2.COLORMAP_JET)
	cv2.imshow('ground_truth',compare_pc_colors )
	
	ri = pc2ri_pw(pc)
	color_image = np.uint8(255*ri[:,:,4]/np.max(ri[:,:,4]))
	color_image = cv2.applyColorMap(color_image, cv2.COLORMAP_JET)
	cv2.imshow('my_version', color_image)
	cv2.imwrite("GTcolor_image.png",compare_pc_colors)
	cv2.imwrite("color_image.png",color_image)
	c = cv2.waitKey(0)
	if 'q' == chr(c & 255):
		print("finish")
	"""


	

	pc_path = "/home/daniel/Documents/pointCloud_RangeImage/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000010.bin"
	pc = load_velodyne_binary(pc_path)
	label_path = "/home/daniel/Documents/SemanticKITTI/data_odometry_labels/dataset/sequences/00/labels/000010.label" 

	load_velodyne_binary_labels(pc_path,label_path)

	ri = pc2ri_pw_360(pc)
	color_image = np.uint8(255*ri[:,:,4]/np.max(ri[:,:,4]))
	color_image = cv2.applyColorMap(color_image, cv2.COLORMAP_JET)
	cv2.imshow('my_version', color_image)
	c = cv2.waitKey(0)
	if 'q' == chr(c & 255):
		print("finish")
