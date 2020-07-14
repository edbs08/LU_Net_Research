import sys
import tensorflow as tf
import numpy as np
import cv2
import random
import yaml
import os
from auxiliary.laserscan import LaserScan, SemLaserScan

# imports Settings manager
sys.path.append('./')
import data_loader
from settings import Settings

CONFIG = Settings(required_args=["config"])

# Channels description:
# 0: X
# 1: Y
# 2: Z
# 3: REFLECTANCE
# 4: DEPTH
# 5: LABEL

"""
*******************************************************
Generates tfrecords with training data with chance of data augmentation
*******************************************************
"""

"""
    Consult auciliary/semantic-kitti.yaml for further information about the classes in SemanticKITTI dataset
"""
clases_names = ["unlabeled", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist",
                "motorcyclist", "road", "parking", "sidewalk", "other-ground", "building", "fence", "vegetation",
                "trunk", "terrain", "pole", "traffic-sign"]

MIN_THRESHOLD = 10

CLASS_BICYCLE = 2
CLASS_MOTORCYCLE = 3
CLASS_TRUCK = 4
CLASS_OTHER_VEH = 5
CLASS_BICYCLIST = 7
CLASS_MOTORCYCLIST = 8
CLASS_OTHER_GROUND = 12

NUM_SEQUENCES = 21
TEST_SEQ = 10
#cd lsemantic_base = CONFIG.LIDAR_2D
frequence = 2
random_flip = False  # Flip randomly or every 'frequency' times

valid_data_aug = ["original", "flip_x", "flip_y", "freq_flip_x", "freq_flip_y"]

def get_range_image(CFG, bin_file, label_names=None):
    color_dict = CFG["color_map"]
    learning_map = CFG["learning_map"]

    nclasses = len(color_dict)
    scan = SemLaserScan(nclasses, map_dict=learning_map, sem_color_dict=color_dict, project=True, W=2048)
    scan.open_scan(bin_file)
    if label_names is not None:
        scan.open_label(label_names)
        scan.colorize()
        label_channel = scan.proj_sem_label_map.reshape((scan.proj_H, scan.proj_W, 1))
    else:
        label_channel = np.zeros((scan.proj_H, scan.proj_W, 1))
    # Construct tensor with x,y,z,i,d,l
    range_image = np.concatenate((scan.proj_xyz, \
                                  scan.proj_remission.reshape((scan.proj_H, scan.proj_W, 1)), \
                                  scan.proj_range.reshape((scan.proj_H, scan.proj_W, 1)), \
                                  label_channel), \
                                 axis=2)
    return range_image


def get_data(data, aug_type, line_num):
    continue_ = True
    if aug_type == "original":
        continue_ = True
    elif aug_type == "flip_x":
        if random_flip:
            continue_ = bool(random.randint(0, 1))
        else:
            continue_ = line_num % frequence
        if continue_:
            data = np.fliplr(data)
    elif aug_type == "flip_y":
        if random_flip:
            continue_ = bool(random.randint(0, 1))
        else:
            continue_ = line_num % frequence
        if continue_:
            pc_flip = np.zeros(data.shape)
            division = int(data.shape[1] / 2)
            pc_flip[:, 0:division, :] = data[:, division:data.shape[1], :]
            pc_flip[:, division:data.shape[1], :] = data[:, 0:division, :]
            data = pc_flip
    elif aug_type == "freq_flip_x":
        labels = list(np.reshape(data[:, :, -1], data.shape[0] * data.shape[1]).tolist())

        count = np.zeros(CONFIG.N_CLASSES)
        for c in range(CONFIG.N_CLASSES):
            count[c] = labels.count(c)

        continue_ = False
        if (count[CLASS_BICYCLE] > MIN_THRESHOLD) or (count[CLASS_MOTORCYCLE] > MIN_THRESHOLD) or (
                count[CLASS_TRUCK] > MIN_THRESHOLD) or (
                count[CLASS_OTHER_VEH] > MIN_THRESHOLD) or (count[CLASS_BICYCLIST] > MIN_THRESHOLD) or (
                count[CLASS_MOTORCYCLIST] > MIN_THRESHOLD) or (
                count[CLASS_OTHER_GROUND] > MIN_THRESHOLD):
            continue_ = True

        if continue_:
            data = np.fliplr(data)

    elif aug_type == "freq_flip_y":
        labels = list(np.reshape(data[:, :, -1], data.shape[0] * data.shape[1]).tolist())

        count = np.zeros(CONFIG.N_CLASSES)
        for c in range(CONFIG.N_CLASSES):
            count[c] = labels.count(c)

        continue_ = False
        if (count[CLASS_BICYCLE] > MIN_THRESHOLD) or (count[CLASS_MOTORCYCLE] > MIN_THRESHOLD) or (
                count[CLASS_TRUCK] > MIN_THRESHOLD) or (
                count[CLASS_OTHER_VEH] > MIN_THRESHOLD) or (count[CLASS_BICYCLIST] > MIN_THRESHOLD) or (
                count[CLASS_MOTORCYCLIST] > MIN_THRESHOLD) or (
                count[CLASS_OTHER_GROUND] > MIN_THRESHOLD):
            continue_ = True
        if continue_:
            pc_flip = np.zeros(data.shape)
            division = int(data.shape[1] / 2)
            pc_flip[:, 0:division, :] = data[:, division:data.shape[1], :]
            pc_flip[:, division:data.shape[1], :] = data[:, 0:division, :]
            data = pc_flip
    else:
        print("Unknown augmentation config")
        quit()
    return data, continue_


def get_name(aug_type):
    name = aug_type
    if aug_type == "original":
        name = ""
    return name


config = CONFIG.SEMKITTI_CFG
dataset = CONFIG.LIDAR_DATASET

# Generates tfrecords for training
def make_tfrecord():
    # open config file
    try:
        print("Opening config file %s" % config)
        CFG = yaml.safe_load(open(config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()

    if not set(CONFIG.AUGMENTATION).issubset(valid_data_aug):
        print("Unknown augmentation config")
        quit()
    # Creates each tfrecord (each of the sequences)
    for sequence in range(NUM_SEQUENCES + 1):

        seq_type = "test/"
        if sequence <= TEST_SEQ:
            seq_type = "train/"

        # Augmentation settings
        for aug_type in CONFIG.AUGMENTATION:
            # Augmentation is only for training and validation data
            if seq_type == "test/" and aug_type != "original":
                continue
            name = get_name(aug_type)
            dataset_output = "%02d" % (sequence)
            dataset_output = "data/seq_" + dataset_output + name + ".tfrecord"

            with tf.python_io.TFRecordWriter(dataset_output) as writer:
                seq_string = "%02d" % (sequence)
                # List the pointcloud files
                scan_paths = os.path.join(dataset, "sequences", seq_string, "velodyne")
                scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                    os.path.expanduser(scan_paths)) for f in fn]
                scan_names.sort()
                # List the label files
                label_paths = os.path.join(dataset, "sequences", seq_string, "labels")
                label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                    os.path.expanduser(label_paths)) for f in fn]
                label_names.sort()

                # Going through each example
                line_num = 0
                for index in range(len(scan_names)):

                    data = get_range_image(CFG, scan_names[index], label_names[index])
                    data, continue_ = get_data(data, aug_type, line_num)
                    if continue_:
                        print("Processing file \"{}\", Data augmentation {} ".format(scan_names[index], aug_type))
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

                        # Serialize example
                        n_raw = n.astype(np.float32).tostring()
                        p_raw = p.astype(np.float32).tostring()
                        label_raw = label.astype(np.float32).tostring()

                        # Create tf.Example
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
                            'neighbors': tf.train.Feature(bytes_list=tf.train.BytesList(value=[n_raw])),
                            'points': tf.train.Feature(bytes_list=tf.train.BytesList(value=[p_raw]))}))

                        # Adding Example to tfrecord
                        writer.write(example.SerializeToString())

                        #file_list_name.write(semantic_base + file[:-1] + ".npy\n")

                    line_num += 1

                print("Process finished, stored {} entries in \"{}\"".format(line_num - 1, dataset_output))

                #file_list_name.close()

    print("All files created.")


if __name__ == "__main__":
    make_tfrecord()
