import numpy as np
import cv2
import yaml
import os
from semantic_kitti_api.auxiliary.laserscan import LaserScan, SemLaserScan

"""
*******************************************************
This file generates a dataset using the API tools provided by SemanticKITTI 
*******************************************************
"""
def get_range_image(bin_file, CONFIG):
    config = os.path.join("semantic_kitti_api", "config", "semantic-kitti.yaml")
    # open config file
    try:
        print("Opening config file %s" % config)
        CFG = yaml.safe_load(open(config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()
    color_dict = CFG["color_map"]
    learning_map = CFG["learning_map"]

    nclasses = len(color_dict)
    scan = SemLaserScan(nclasses, map_dict = learning_map, sem_color_dict=color_dict,project=True ,W=CONFIG.IMAGE_WIDTH)
    scan.open_scan(bin_file)

    #Get labels
    st = os.path.normpath(bin_file).split(os.path.sep)
    scan_num = st[-1].split('.')[0]
    sequence_num = st[-3]
    label_path = os.path.join(CONFIG.LABELS_DATA, sequence_num, "labels", scan_num+".label")

    if label_path != None:
        scan.open_label(label_path)
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