import numpy as np
import cv2
import yaml
import os
import data_loader
from semantic_kitti_api.auxiliary.laserscan import LaserScan, SemLaserScan


def get_range_image(bin_file, CONFIG, CFG):
    """
    *******************************************************
    This function generates range image from pointcloud using the API tools provided by SemanticKITTI
    *******************************************************
    """

    config = os.path.join("semantic_kitti_api", "config", "semantic-kitti.yaml")
    color_dict = CFG["color_map"]
    learning_map = CFG["learning_map"]

    nclasses = len(color_dict)
    scan = SemLaserScan(nclasses, map_dict = learning_map, sem_color_dict=color_dict,project=True ,W=CONFIG.IMAGE_WIDTH)
    scan.open_scan(bin_file)

    #Get labels
    st = os.path.normpath(bin_file).split(os.path.sep)
    scan_num = st[-1].split('.')[0]
    sequence_num = st[-3]
    label_path = os.path.join(CONFIG.LABELS_DATA, str(sequence_num), "labels", str(scan_num)+".label")

    #label_path = label_path.encode('utf8', 'replace')

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


def range_image_processing(data,CONFIG):

    # Load labels
    mask = data[:, :, 0] != 0

    # data = data_loader.interp_data(data[:,:,0:5], mask)

    p, n = data_loader.pointnetize(data[:, :, 0:5], n_size=CONFIG.N_SIZE)

    groundtruth = data_loader.apply_mask(data[:, :, 5], mask)

    # Compute weigthed mask
    contours = np.zeros((mask.shape[0], mask.shape[1]), dtype=bool)

    if np.amax(groundtruth) > CONFIG.N_CLASSES - 1:
        print("[WARNING] There are more classes than expected !")

    for c in range(1, int(np.amax(groundtruth)) + 1):
        channel = (groundtruth == c).astype(np.float32)
        gt_dilate = cv2.dilate(channel, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        gt_dilate = gt_dilate - channel
        contours = np.logical_or(contours, gt_dilate == 1.0)

    contours = contours.astype(np.float32) * mask

    dist = cv2.distanceTransform((1 - contours).astype(np.uint8), cv2.DIST_L2,
                                 cv2.DIST_MASK_PRECISE)

    # Create output label for training
    label = np.zeros((groundtruth.shape[0], groundtruth.shape[1], CONFIG.N_CLASSES + 2))
    for y in range(groundtruth.shape[0]):
        for x in range(groundtruth.shape[1]):
            label[y, x, int(groundtruth[y, x])] = 1.0

    label[:, :, CONFIG.N_CLASSES] = dist
    label[:, :, CONFIG.N_CLASSES + 1] = mask

    return p, n, label



