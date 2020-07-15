"""
This file generates a dataset of range images from the KITTI odometry scans
"""

import numpy as np 
import cv2 
import yaml
import os
from auxiliary_SK.laserscan import LaserScan, SemLaserScan
"""
*******************************************************
This file generates a dataset using the Tools provided by SemanticKITTI image size 64x2048
*******************************************************
"""

def get_range_image(CFG,bin_file,label_names=None):

	color_dict = CFG["color_map"]
	learning_map = CFG["learning_map"]

	nclasses = len(color_dict)
	scan = SemLaserScan(nclasses, map_dict = learning_map, sem_color_dict=color_dict,project=True ,W=2048)
	scan.open_scan(bin_file)	
	if (label_names != None):
		scan.open_label(label_names)
		scan.colorize()
		label_channel = scan.proj_sem_label_map.reshape((scan.proj_H,scan.proj_W,1))
	else:
		label_channel = np.zeros((scan.proj_H,scan.proj_W,1))
	#Construct tensor with x,y,z,i,d,l
	range_image = np.concatenate( (scan.proj_xyz,   \
								scan.proj_remission.reshape((scan.proj_H,scan.proj_W,1)), \
								scan.proj_range.reshape((scan.proj_H,scan.proj_W,1)), \
								label_channel), \
								axis=2)
	return range_image

def visualize(range_image):
		
	# TRANSFORM RANGE IMAGE INTO COLOR MAP FOR DISPLAY
	
	ri = range_image[:,:,4]
	color_image = np.uint8(255*ri/np.max(ri))
	data = cv2.applyColorMap(color_image, cv2.COLORMAP_JET)
	

	# TRANSFORM RANGE IMAGE GRAY SCALE FOR DISPLAY
	"""
	power = 16
	data = np.copy(range_image[:,:,5])
	data[data > 0] = data[data > 0]**(1 / power)
	data[data < 0] = data[data > 0].min()
	data = (data - data[data > 0].min()) / \
		(data.max() - data[data > 0].min())
	"""
	# FOR DISPLAYING ONLY THE SEMANTICS
	#data = ri
	#data = scan.proj_sem_color[..., ::-1]

	return data


if __name__ == "__main__":
	
	config = "G:\\UB\\Documents\\SemanticKITTI\\semantic-kitti-api\\config\\semantic-kitti.yaml"
	dataset = "G:\\UB\\Data\\data_odometry_velodyne\\dataset"
	root_path = "G:\\UB\\Data\\Generated_Datasets\\20200228"

	# open config file
	try:
		print("Opening config file %s" % config)
		CFG = yaml.safe_load(open(config, 'r'))
	except Exception as e:
		print(e)
		print("Error opening yaml file.")
		quit()

	#Creating test set only
	Train_Dataset = False 
	Test_Dataset = True

	if (Train_Dataset == True):
		print(" *** GENERATING TRAIN DATASET ***")
		save_path = root_path + "train/"
		max_sequence = 10
		for s_index in range (max_sequence+1):
			file_name = root_path + "semantic_%02d.txt" % (s_index)
			name_list = open(file_name,"w")
			sequence =  "%02d" % (s_index)

			# List the pointcloud files 
			scan_paths = os.path.join(dataset, "sequences",
										sequence, "velodyne")
			scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
				os.path.expanduser(scan_paths)) for f in fn]
			scan_names.sort()
			# List the label files
			label_paths = os.path.join(dataset, "sequences",
		                                 sequence, "labels")
			label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
				os.path.expanduser(label_paths)) for f in fn]
			label_names.sort()

			for file_index in range (len(scan_names)):
				file = "%06d" % (file_index)
				save_name = save_path + "%s_%s.npy" % (sequence, file)
				print("Starting processing sequence ", sequence, " file " , file)

				range_image = get_range_image(CFG,scan_names[file_index],label_names[file_index])	
				np.save(save_name,range_image)
				seq =  "%02d" % (s_index)
				file = "%06d" % (file_index)
				line = "%s_%s\n" % (seq, file)
				name_list.write(line) 
				
				################### For visualizing purposes
				"""
				data = visualize(range_image)
				cv2.imshow('my_version', data)
				c = cv2.waitKey(0)
				if 'n' == chr(c & 255):
					pass
				if 's' == chr(c & 255):
					break
				if 'q' == chr(c & 255):
					quit()
				"""
			name_list.close()

	if (Test_Dataset == True):
		save_path = os.path.join(root_path , "test")
		print(" *** GENERATING TEST DATASET ***")
		print("  No label file provided, filling last channel with zeros")
		sequences = [16,17,18,19,20,21]  
		#for s_index in range (11,22):
		for s_index in sequences:    
			sequence =  "%02d" % (s_index)

			file_name = root_path + "semantic_%02d.txt" % (s_index)
			#name_list = open(file_name,"w")
			sequence =  "%02d" % (s_index)

			# List the pointcloud files 
			scan_paths = os.path.join(dataset, "sequences",
										sequence, "velodyne")
			scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
				os.path.expanduser(scan_paths)) for f in fn]
			scan_names.sort()
			# No labels for test set
			print( scan_paths )
			for file_index in range (len(scan_names)):
				file = "%06d" % (file_index)
				save_name = os.path.join(save_path, "%s_%s" % (sequence, file))
				print("Starting processing sequence ", sequence, " file " , file)

				range_image = get_range_image(CFG,scan_names[file_index])					
				np.save(save_name,range_image)

				seq =  "%02d" % (s_index)
				file = "%06d" % (file_index)
				line = "%s_%s\n" % (seq, file)
				#name_list.write(line) 
			#name_list.close()
			





