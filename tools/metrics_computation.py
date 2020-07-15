"""
Script with different tools for point cloud and range image analysis
It includes the generation of .txt pointcloud, for visualization in CloudCompare 
"""

import tensorflow as tf
import numpy as np
import cv2
import yaml
#import shutil
import os
import sys
from matplotlib import pyplot as plt

from auxiliary_SK.laserscan import LaserScan, SemLaserScan

def open_label_and_map(filename,CFG):
    """ Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in '.label'):
      raise RuntimeError("Filename extension is not valid label file.")

    # if all goes well, open label
    label = np.fromfile(filename, dtype=np.uint32)
    label = label.reshape(-1)

    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16    # instance id in upper half

    #Map labels
    sem_label_map = np.vectorize(CFG["learning_map"].get)(sem_label)
    return sem_label, sem_label_map

def semi_propagate(RI,scan,CFG):
	"""
	Inputs a labels range (H,W,1) image and its corresponding pointcloud and returns a labeled list of points, some of them as 'unlabeled'
	"""
	# order in decreasing depth as in semantic KITTI API
	"""
	num_points = scan.size()
	depth = np.linalg.norm(scan.points, 2, axis=1)
	ref_RI = np.zeros(RI.shape) # to keep only the one closer to the lidar. 0 if no depth has been assigned 
	l_pc = np.zeros((num_points,6)) #xyzrgb
	labels = np.zeros(num_points)
	for i in range (num_points):
		color = CFG["color_map"][0]
		if (ref_RI[scan.proj_y[i],scan.proj_x[i]] == 0) or (depth[i]<ref_RI[scan.proj_y[i],scan.proj_x[i]]):
			labels[i] = RI[scan.proj_y[i],scan.proj_x[i]] 
			label = CFG["learning_map_inv"][RI[scan.proj_y[i],scan.proj_x[i]]]
			color = CFG["color_map"][label]
			ref_RI[scan.proj_y[i],scan.proj_x[i]] = depth[i]
		l_pc[i,0:3] = scan.points[i,:]
		l_pc[i,3:6] = color
	"""
	RI = RI * scan.proj_mask
	num_points = scan.size()
	labels = np.zeros(num_points)
	l_pc = np.zeros((num_points,6)) #xyzrgb
	l_pc[:,0:3] = scan.points
	for r in range (RI.shape[0]):
		for c in range (RI.shape[1]):
			label = CFG["learning_map_inv"][RI[r,c]]
			color = CFG["color_map"][label]
			idx = scan.proj_idx[r,c]
			labels[idx] = label
			l_pc[idx,3:6] = color
	
	return l_pc, labels

def fully_propagate(RI,scan,CFG):
	"""
	Inputs a labels range (H,W,1) image and its corresponding pointcloud and returns a labeled list of points. Assign label to all points following the simplest propagation method
	"""
	RI = RI * scan.proj_mask
	num_points = scan.size()
	l_pc = np.zeros((num_points,6)) #xyzrgb
	for i in range (num_points):
		label = CFG["learning_map_inv"][RI[scan.proj_y[i],scan.proj_x[i]]]
		color = CFG["color_map"][label]
		l_pc[i,0:3] = scan.points[i,:]
		l_pc[i,3:6] = color
	return l_pc

def groundtruth_pointcloud(scan,label,CFG):
	"""
	Returns a np array of the pointcloud in format (num_points,6) where the last dimension is xyzrgb
	"""
	label, _ = open_label_and_map(label,CFG)
	num_points = scan.size()
	l_pc = np.zeros((num_points,6)) #xyzrgb
	for i in range (num_points):
		color = CFG["color_map"][label[i]]
		l_pc[i,0:3] = scan.points[i,:]
		l_pc[i,3:6] = color
	return l_pc

# Compute scores for a single image
def compute_iou_per_class(pred, groundtruth, classes):
	n_class = len(classes)
	ious = np.zeros(n_class)
	acc  = np.zeros(n_class)
	tps  = np.zeros(n_class)
	fns  = np.zeros(n_class)
	fps  = np.zeros(n_class)

	for cls_id in range(n_class):
		tp = np.sum(pred[groundtruth == classes[cls_id]] == classes[cls_id])
		fp = np.sum(groundtruth[pred == classes[cls_id]] != classes[cls_id])
		fn = np.sum(pred[groundtruth == classes[cls_id]] != classes[cls_id])
		tn = np.sum(pred[groundtruth != classes[cls_id]] != classes[cls_id])

		ious[cls_id] = tp/(tp+fn+fp+0.00000001)
		acc[cls_id] = tp/(tp+fn+fp+tn+0.00000001)
		tps[cls_id] = tp
		fps[cls_id] = fp
		fns[cls_id] = fn

	return ious, acc, tps, fps, fns, tn


def plot_fist_frames(seq,num_frames):
	num_frames = 50
	metrics_ious = np.zeros((n_classes,num_frames))
	iou_semi = np.zeros((n_classes,num_frames))
	sequence =  "%02d" % seq
	for f in range (0,num_frames):
		
		frame = "%06d" % f
		_,label_pred = open_label_and_map("/home/daniel/Documents/LU_Net_TFRecord_tempora/LU-Net/lunet/sequences/"+sequence+"/predictions/"+frame+".label",CFG)
		_,label_gt = open_label_and_map("/home/daniel/Documents/SemanticKITTI/data_odometry_labels/dataset/sequences/"+sequence+"/labels/"+frame+".label",CFG)
		metrics_ious[:,f],_, _, _, _,t_ = compute_iou_per_class(label_pred,label_gt,CFG["learning_map_inv"].keys())

		#Compute semipropagate labels to compute metris before and after
		RI = "/home/daniel/Documents/LU_Net_TFRecord_tempora/LU-Net/lunet/sequences/"+sequence+"/predictions/"+frame+".npy"
		pc = "/media/daniel/FILES/UB/Data/data_odometry_velodyne/dataset/sequences/"+sequence+"/velodyne/"+frame+".bin"
		scan = LaserScan(project=True ,W=1024)
		scan.open_scan(pc)
		RI = np.load(RI)
		_,sp_label = semi_propagate(RI,scan,CFG)
		#sp_label = np.vectorize(CFG["learning_map_inv"].get)(sp_label)
		iou_semi[:,f],_, _, _, _,_ = compute_iou_per_class(sp_label,label_gt,CFG["learning_map_inv"].keys())

	
	fig, ax = plt.subplots()
	ax.plot(metrics_ious[1,:])
	ax.set(xlabel='frames (scans)', ylabel=' IoU',
			title="Average IoU CAR CLASS along the scans, SEQUENCE %s"% sequence)
	ax.grid()
	
	
	_, ax2 = plt.subplots()
	ax2.plot(np.mean(metrics_ious,axis=0))
	ax2.set(xlabel='frames (scans)', ylabel=' IoU',
			title="Average IoU ALL CLASSES along the lidar scans, SEQUENCE %s"% sequence)
	ax2.grid()
	
	
	_, ax3 = plt.subplots()
	ax3.plot(iou_semi[1,:])
	ax3.set(xlabel='frames (scans)', ylabel=' IoU',
			title="Average IoU semipropagate vs groundtruth CAR CLASS SEQUENCE %s"% sequence)
	
	_, ax4 = plt.subplots()
	t = np.linspace(0,num_frames,endpoint=False)
	ax4.plot(t,metrics_ious[1,:],t,iou_semi[1,:])
	ax4.set(xlabel='frames (scans)', ylabel=' IoU',
			title="Mean IoU before and after propagation CAR CLASS seq %s"% sequence)

	_, ax4 = plt.subplots()
	t = np.linspace(0,num_frames,endpoint=False)
	ax4.plot(t,metrics_ious[9,:],t,iou_semi[9,:])
	ax4.set(xlabel='frames (scans)', ylabel=' IoU',
			title="Mean IoU before and after propagation ROAD seq %s"% sequence)
	plt.show()


def read_print_metrics(seq,CFG):
	tps_sum = np.load("tps_sum.npy")
	#tns_sum = np.load("tns_sum.npy")
	fps_sum = np.load("fps_sum.npy")
	fns_sum = np.load("fns_sum.npy")

	tps_sum2 = np.load("tps_sum2.npy")
	#tns_sum2 = np.load("tns_sum2.npy")
	fps_sum2 = np.load("fps_sum2.npy")
	fns_sum2 = np.load("fns_sum2.npy")
	acc = np.load("acc_sum.npy")

	ious = tps_sum.astype(np.float)/(tps_sum + fns_sum + fps_sum + 0.000000001)
	pr   = tps_sum.astype(np.float)/(tps_sum + fps_sum + 0.000000001)
	re   = tps_sum.astype(np.float)/(tps_sum + fns_sum + 0.000000001)

	output = "Sequence [{}] stats:\n".format(seq)
	output += "Accuracy of all sequence = {} :\n".format(acc)
	for i in range(0, len(CFG["learning_map_inv"].keys())):
		output += "\tPixel-seg: P: {:.3f}, R: {:.3f}, IoU: {:.5f}\n".format(pr[i], re[i], ious[i])
	output += "\n"
	print(output)

	ious = tps_sum2.astype(np.float)/(tps_sum2 + fns_sum2 + fps_sum2 + 0.000000001)
	pr   = tps_sum2.astype(np.float)/(tps_sum2 + fps_sum2 + 0.000000001)
	re   = tps_sum2.astype(np.float)/(tps_sum2 + fns_sum2 + 0.000000001)

	output = "Sequence [{}] stats BEFORE PROPAGATION:\n".format(seq)
	for i in range(0, len(CFG["learning_map_inv"].keys())):
		output += "\tPixel-seg: P: {:.3f}, R: {:.3f}, IoU: {:.5f}\n".format(pr[i], re[i], ious[i])
	output += "\n"
	print(output)


	return output


def compute_metrics(seq,CFG):
	sequence =  "%02d" % seq

	# Get number of files 
	gt_label = "/home/daniel/Documents/SemanticKITTI/data_odometry_labels/dataset/sequences/"+sequence+"/labels/"
	num_frames = len([name for name in os.listdir(gt_label) if os.path.isfile(os.path.join(gt_label, name))])
	print("Number of frames = ",num_frames)

	metrics_ious = np.zeros((n_classes,num_frames))
	iou_semi = np.zeros((n_classes,num_frames))
	sequence =  "%02d" % seq
	tps_sum, fns_sum, fps_sum,tps_sum2, fns_sum2, fps_sum2 = 0,0,0,0,0,0
	acc_sum = 0
	for f in range (0,num_frames):
		print("Processing frame number = ", f)
		frame = "%06d" % f
		_,label_pred = open_label_and_map("/home/daniel/Documents/LU_Net_TFRecord_tempora/LU-Net/lunet/sequences/"+sequence+"/predictions/"+frame+".label",CFG)
		_,label_gt = open_label_and_map(gt_label+frame+".label",CFG)
		_,_, tps, fps, fns, _ = compute_iou_per_class(label_pred,label_gt,CFG["learning_map_inv"].keys())
		acc = np.sum(tps)/label_pred.size
		#print(acc)
		tps_sum += tps
		fns_sum += fns
		fps_sum += fps
		#tns_sum += tns

		#Compute semipropagate labels to compute metris before and after
		RI = "/home/daniel/Documents/LU_Net_TFRecord_tempora/LU-Net/lunet/sequences/"+sequence+"/predictions/"+frame+".npy"
		pc = "/media/daniel/FILES/UB/Data/data_odometry_velodyne/dataset/sequences/"+sequence+"/velodyne/"+frame+".bin"
		scan = LaserScan(project=True ,W=1024)
		scan.open_scan(pc)
		RI = np.load(RI)
		_,sp_label = semi_propagate(RI,scan,CFG)
		#sp_label = np.vectorize(CFG["learning_map_inv"].get)(sp_label)
		_,_, tps2, fps2, fns2,_ = compute_iou_per_class(sp_label,label_gt,CFG["learning_map_inv"].keys())

		tps_sum2 += tps2
		fns_sum2 += fns2
		fps_sum2 += fps2
		#tns_sum2 += tns2


		acc_sum += acc


	acc_sum = acc_sum/num_frames
	print(acc_sum)
	np.save("acc_sum",acc_sum)
	np.save("tps_sum",tps_sum)
	#np.save("tns_sum",tns_sum)
	np.save("fps_sum",fps_sum)
	np.save("fns_sum",fns_sum)

	np.save("tps_sum2",tps_sum2)
	#np.save("tns_sum2",tns_sum2)
	np.save("fps_sum2",fps_sum2)
	np.save("fns_sum2",fns_sum2)

	read_print_metrics(8,CFG)


def generate_pointcloud_txt(CFG,display=False):
	sequence =  "%02d" % 8
	frame = "%06d" % 0
	#RI = "/home/daniel/Documents/LU_Net_TFRecord_tempora/LU-Net/lunet/sequences/"+sequence+"/predictions/"+frame+".npy"
	#pc = "/media/daniel/FILES/UB/Data/data_odometry_velodyne/dataset/sequences/"+sequence+"/velodyne/"+frame+".bin"
	#label_gt = "/home/daniel/Documents/SemanticKITTI/data_odometry_labels/dataset/sequences/"+sequence+"/labels/"+frame+".label"

	pc = "G:\\UB\\Data\\data_odometry_velodyne\\dataset\\sequences\\" + sequence + "\\velodyne\\"+frame+".bin"
	label_gt = "G:\\UB\\Data\\data_odometry_velodyne\\dataset\\sequences\\" + sequence + "\\labels\\" + frame + ".label"
	RI = "G:\\UB\\Data\\Results\\1X_0022\\range_image\\" + frame + ".npy"

	scan = LaserScan(project=True ,W=1024)
	scan.open_scan(pc)
	RI = np.load(RI)


	sp_pc,_ = semi_propagate(RI,scan,CFG)
	np.savetxt("semi_propagate.txt", np.float32(sp_pc),fmt='%1.6e')
	fp_pc = fully_propagate(RI,scan,CFG)
	np.savetxt("fully_propagate.txt", np.float32(fp_pc),fmt='%1.6e')
	gt_pc = groundtruth_pointcloud(scan,label_gt,CFG)
	np.savetxt("groundtruth_pointcloud.txt", np.float32(gt_pc),fmt='%1.6e')

	
	if display:
		nclasses = len(CFG["color_map"].keys())
		scan = SemLaserScan(nclasses, sem_color_dict=CFG["color_map"],project=True ,W=1024)
		scan.open_scan(pc)	
		scan.open_label(label_gt)
		scan.colorize()


		labels_ri = RI

		img_color = np.zeros((labels_ri.shape[0],labels_ri.shape[1],3),dtype=np.uint8)
		for r in range (labels_ri.shape[0]):
			for c in range (labels_ri.shape[1]):
				img_color[r,c,:] =  CFG["color_map"][CFG["learning_map_inv"][labels_ri[r,c]]]
				#sclass = [0,10]
				#img_color[r,c,:] = paint_single_class(labels_ri[r,c],sclass)

		plt.subplot(2,1,1)
		plt.imshow(scan.proj_sem_color)
		plt.title("Groundtruth")
		plt.subplot(2,1,2)
		plt.imshow(img_color)
		plt.title("Prediction")
		plt.show()
	
def pc_bin_to_txt(CFG):
	pc = os.path.join("G:","\\UB","Data","data_odometry_velodyne","dataset","sequences","08","velodyne","000000.bin")
	scan = LaserScan(project=True, W=1024)
	scan.open_scan(pc)
	label = os.path.join("G:","\\UB","Data","data_odometry_velodyne","dataset","sequences","08","labels","000000.label")

	gt_pc = groundtruth_pointcloud(scan, label, CFG)
	np.savetxt("groundtruth_pointcloud.txt", np.float32(gt_pc), fmt='%1.6e')


def count_points_pc():
	pc = "G:\\UB\\Data\\data_odometry_velodyne\\dataset\\sequences\\08\\velodyne\\"
	for i in range(20):
		scan_num = "%06d.bin" % i
		scan = LaserScan(project=True, W=1024)
		scan.open_scan(pc+scan_num)
		print(scan.size())

if __name__ == "__main__":
	n_classes = 20
	# open config file
	# config = "/home/daniel/Documents/SemanticKITTI/semantic-kitti-api/config/semantic-kitti.yaml"
	#config = os.path.join("G:","\\UB","Documents","SemanticKITTI","semantic-kitti-api","config","semantic-kitti.yaml")
	config = "C:/Users/Daniel/Google Drive/Masters/Intership/Code/SemanticKITTI/semantic-kitti-api/config/semantic-kitti.yaml"

	try:
		print("Opening config file %s" % config)
		CFG = yaml.safe_load(open(config, 'r'))
	except Exception as e:
		print(e)
		print("Error opening yaml file.")
		quit()
	
	### Funciton to generate the txt of the pointcloud for visualization
	#generate_pointcloud_txt(CFG,display=True)
	
	#sequence = 8
	### Function to generate metrings along several frames and sequences
	#plot_fist_frames(sequence,50) #Plot the metrics of the first 50 frames of sequence 8

	### Calculate numerical statistics for all the sequence
	#compute_metrics(sequence,CFG)
	#read_print_metrics(sequence,CFG)

	#pc_bin_to_txt(CFG)
	count_points_pc()

	