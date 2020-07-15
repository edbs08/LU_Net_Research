"""
Script to study the impact of K-Nearest Neighbor algorithm when apply in it to the ground-truth
"""

import sys
import numpy as np
import cv2
import yaml
import os
import matplotlib.pyplot as plt
import itertools
import shutil
import copy
sys.path.append('./')
from sklearn.neighbors import KNeighborsClassifier
#import numpy as np

from auxiliary.laserscan import LaserScan, SemLaserScan

label_names = ["unlabeled","car","bicicle","motorcycle","truck","other-veh","person","bicyclist","motocyclist","road","parking","sidewalk","other-ground","building","fence","vegetation","trunk","terrain","pole","traffic-sign"]
label_number = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]


GT_dataset_path = os.path.join("..","..","20200228")
GT_velodyne_path = os.path.join("..","..","data_odometry_velodyne","dataset","sequences")
config = "./../../tools/semantic-kitti-api/config/semantic-kitti.yaml"
pred_data = "xyz"
reasign_RI = True
#pred_data = "di" #distance and intensity
#pred_data = "xyzi" #xyz and intensity (reflectance)
#pred_data = "simple"




def get_trainig_data_xyzi(RI,scan,CFG,rem_scaled = False):
    
        RI_points = (scan.proj_H*scan.proj_W)
        RI = RI * scan.proj_mask
        labels = np.zeros(RI_points)
        l_pc = np.zeros((RI_points,4)) #xyzi
        counter = 0
        for r in range (RI.shape[0]):
                for c in range (RI.shape[1]):
                        #label = CFG["learning_map_inv"][RI[r,c]]
                        label = RI[r,c]
                        idx = scan.proj_idx[r,c]
                        labels[counter] = label
                        l_pc[counter,0:3] = scan.points[idx]
                        l_pc[counter,3] = scan.remissions[idx]
                        counter += 1

        if (rem_scaled == True):
            l_pc[:,3]= (l_pc[:,3]*100)-50
        return l_pc, labels
    
def get_trainig_data_D_R(RI,scan,CFG):
        """
        Inputs a labels range (H,W,1) image and its corresponding pointcloud and returns the list of distance and reflectance  per point and the corresponding label from the RI
        Returns vectors of size 1x(H*W)
    """

        RI_points = (scan.proj_H*scan.proj_W)
        RI = RI * scan.proj_mask
        labels = np.zeros(RI_points)
        l_pc = np.zeros((RI_points,2)) #distance and intensity
        counter = 0
        for r in range (RI.shape[0]):
                for c in range (RI.shape[1]):
                        #label = CFG["learning_map_inv"][RI[r,c]]
                        label = RI[r,c]
                        idx = scan.proj_idx[r,c]
                        labels[counter] = label
                        l_pc[counter,0] = scan.unproj_range[idx]
                        l_pc[counter,1] = scan.remissions[idx]
                        counter += 1

        return l_pc, labels


def get_trainig_data(RI,scan,CFG):
        """
        Inputs a labels range (H,W,1) image and its corresponding pointcloud and returns the list of points and the corresponding label from the RI
        """
        # order in decreasing depth as in semantic KITTI API

        RI_points = (scan.proj_H*scan.proj_W)
        RI = RI * scan.proj_mask
        labels = np.zeros(RI_points)
        l_pc = np.zeros((RI_points,3)) #xyz
        counter = 0
        for r in range (RI.shape[0]):
                for c in range (RI.shape[1]):
                        #label = CFG["learning_map_inv"][RI[r,c]]
                        label = RI[r,c]
                        idx = scan.proj_idx[r,c]
                        labels[counter] = label
                        l_pc[counter,:] = scan.points[idx]
                        counter += 1

        return l_pc, labels


def reasign_RI_values(RI,scan,y_test): 
        print("reasign values")      
        for r in range (RI.shape[0]):
                for c in range (RI.shape[1]):
                        label = RI[r,c]
                        idx = scan.proj_idx[r,c]
                        y_test[idx]=label
        return y_test
    
    
def pc_propagation_Knn(sequence,frame,pred,CFG,save_ri = False):
        sequence_str = "%02d" % (sequence)
        frame_str = "%06d" % (frame)

        dirpath = os.path.join("sequences",sequence_str,"predictions")

        scan_name = os.path.join(GT_velodyne_path, sequence_str , "velodyne" , "%06d" % (frame)+".bin")

        color_dict = CFG["color_map"]
        learning_map = CFG["learning_map"]
        nclasses = len(color_dict)
        scan = SemLaserScan(nclasses, map_dict = learning_map, sem_color_dict=color_dict,project=True ,W=2048)
        scan.open_scan(scan_name)
        label = np.zeros(scan.points.shape[0])

        if (pred_data == "xyz"):
                X_train, y_train = get_trainig_data(pred,scan,CFG)
                knn = KNeighborsClassifier(n_neighbors = 3,weights='distance')
                knn.fit(X_train,y_train)
                label = knn.predict(scan.points)
                if (reasign_RI == True):
                    label = reasign_RI_values(pred,scan,label)
        if (pred_data == "di"):
                X_train, y_train = get_trainig_data_D_R(pred,scan,CFG)
                #print(X_train.shape)
                knn = KNeighborsClassifier(n_neighbors = 3)
                knn.fit(X_train,y_train)
                array_=np.transpose(np.array((scan.unproj_range,scan.remissions)))
                #print(array_.shape)
                label = knn.predict(array_)
        if (pred_data == "xyzi"):
                rem_scaled = True
                X_train, y_train = get_trainig_data_xyzi(pred,scan,CFG,rem_scaled)
                knn = KNeighborsClassifier(n_neighbors = 3)
                knn.fit(X_train,y_train)
                X_test = np.zeros((scan.points.shape[0],4))
                X_test[:,0:3]=scan.points
                remiss = (scan.remissions)
                if rem_scaled == True:
                    remiss = (scan.remissions*100)-50
                X_test[:,3]=remiss
                #print(array_.shape)
                label = knn.predict(X_test)
        if (pred_data == "simple"):
            	for i in range (scan.points.shape[0]):
            	    	label[i] = pred[scan.proj_y[i],scan.proj_x[i]]   
                        
                        
        #Translate to original learning map
        for i in range (scan.points.shape[0]):
                label[i] = CFG["learning_map_inv"][label[i]]


        print(os.path.join(dirpath,frame_str) + ".label")
        labels = label.astype(np.uint32)
        labels.tofile(os.path.join(dirpath,frame_str) + ".label")
        if save_ri == True:
                np.save(os.path.join(dirpath,frame_str),pred)


def predict_from_GT():
    #Create output directory if it doesnt exist
    dirpath = "sequences"
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)


    # open config file
    try:
        print("Opening config file %s" % config)
        CFG = yaml.safe_load(open(config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()

    sequences = [5] #small test
    for seq in sequences:
        sub_path="train"
        if (seq > 10):
            sub_path="test"
        sequence_str = "%02d" % (seq)
        test_path = os.path.join(GT_dataset_path ,sub_path)

        dirpath = os.path.join("sequences",sequence_str)
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
                shutil.rmtree(dirpath)
        os.mkdir(dirpath)
        dirpath = os.path.join(dirpath,"predictions")
        os.mkdir(dirpath)

        # List the Range Images files
        scan_paths = test_path
        scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                                os.path.expanduser(scan_paths)) for f in fn]
        scan_names.sort()

        print("found this number of files ", len(scan_names))

        frame = 0
        for file_index in range (len(scan_names)):
            seq_prefix = sequence_str + "_"
            if (seq_prefix in scan_names[file_index]):
                print("Processing file \"{}\" ".format(scan_names[file_index]))
                RI = np.load(scan_names[file_index])
                RI = RI[:,:,5]
                pc_propagation_Knn(seq,frame,RI,CFG,save_ri = False)
                frame +=1


if __name__ == "__main__":
    predict_from_GT()