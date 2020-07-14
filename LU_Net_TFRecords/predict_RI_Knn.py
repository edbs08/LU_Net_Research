"""
File used to label the full point cloud using the range image prediction
Specify the checkpoint to use

"""

import tensorflow as tf
import numpy as np
import cv2
import yaml
import shutil
import os
import sys
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

sys.path.append('./')
import data_loader
from settings import Settings
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

CONFIG = Settings(required_args=["gpu", "config", "checkpoint"])
label_names = ["unlabeled", "car", "bicicle", "motorcycle", "truck", "other-veh", "person", "bicyclist", "motocyclist",
               "road", "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain",
               "pole", "traffic-sign"]
label_number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

from auxiliary.laserscan import LaserScan, SemLaserScan


# Computes softmax
def softmax(x):
    e_x = np.exp(x)
    return e_x / np.expand_dims(np.sum(e_x, axis=2), axis=2)


# Erase line in stdout
def erase_line():
    sys.stdout.write("\033[F")

# Takes a sequence of channels and returns the corresponding indices in the rangeimage
def seq_to_idx(seq):
    idx = []
    if "x" in seq:
        idx.append(0)
    if "y" in seq:
        idx.append(1)
    if "z" in seq:
        idx.append(2)
    if "r" in seq:
        idx.append(3)
    if "d" in seq:
        idx.append(4)

    return np.array(idx, dtype=np.intp)


# Read a single file
def read_example(string_record):
    # Create example
    example = tf.train.Example()
    example.ParseFromString(string_record)

    features = example.features.feature

    points_lin = np.fromstring(features["points"].bytes_list.value[0], dtype=np.float32)
    neighbors_lin = np.fromstring(features["neighbors"].bytes_list.value[0], dtype=np.float32)
    label_lin = np.fromstring(features["label"].bytes_list.value[0], dtype=np.float32)

    points = np.reshape(points_lin, (CONFIG.IMAGE_HEIGHT * CONFIG.IMAGE_WIDTH, 1, 5))
    neighbors = np.reshape(neighbors_lin, (CONFIG.IMAGE_HEIGHT * CONFIG.IMAGE_WIDTH, CONFIG.N_LEN, 5))

    points = np.take(points, seq_to_idx(CONFIG.CHANNELS), axis=2)
    neighbors = np.take(neighbors, seq_to_idx(CONFIG.CHANNELS), axis=2)

    label = np.reshape(label_lin, (CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.N_CLASSES + 2))
    groundtruth = np.argmax(label[:, :, 0:CONFIG.N_CLASSES], axis=2)
    mask = label[:, :, CONFIG.N_CLASSES + 1] == 1

    return points, neighbors, groundtruth, label[:, :, 0:CONFIG.N_CLASSES], mask


def pc_propagation(sequence, frame, pred, CFG, save_ri=False):
    sequence_str = "%02d" % (sequence)
    frame_str = "%06d" % (frame)

    dirpath = os.path.join("sequences", sequence_str, "predictions")

    scan_name = CONFIG.LIDAR_DATASET + "/sequences/" + sequence_str + "/velodyne/" + "%06d" % (frame) + ".bin"

    color_dict = CFG["color_map"]
    learning_map = CFG["learning_map"]
    nclasses = len(color_dict)
    scan = SemLaserScan(nclasses, map_dict=learning_map, sem_color_dict=color_dict, project=True, W=1024)
    scan.open_scan(scan_name)
    label = np.zeros(scan.points.shape[0])

    for i in range(scan.points.shape[0]):
        label[i] = pred[scan.proj_y[i], scan.proj_x[i]]
        # Translate to original learning map
        label[i] = CFG["learning_map_inv"][label[i]]
    print(os.path.join(dirpath, frame_str) + ".label")
    labels = label.astype(np.uint32)
    labels.tofile(os.path.join(dirpath, frame_str) + ".label")
    if save_ri:
        np.save(os.path.join(dirpath, frame_str), pred)


def get_trainig_data(RI, scan):
    """
	Inputs a labels range (H,W,1) image and its corresponding pointcloud and returns a labeled list of points, some of them as 'unlabeled'
	"""
    # order in decreasing depth as in semantic KITTI API

    RI_points = (scan.proj_H * scan.proj_W)
    RI = RI * scan.proj_mask
    labels = np.zeros(RI_points)
    l_pc = np.zeros((RI_points, 3))  # xyz
    # l_pc[:,0:3] = scan.points
    counter = 0
    for r in range(RI.shape[0]):
        for c in range(RI.shape[1]):
            # label = CFG["learning_map_inv"][RI[r,c]]
            label = RI[r, c]
            idx = scan.proj_idx[r, c]
            labels[counter] = label
            l_pc[counter, :] = scan.points[idx]
            counter += 1

    return l_pc, labels


def pc_propagation_Knn(sequence, frame, pred, CFG, save_ri):
    sequence_str = "%02d" % sequence
    frame_str = "%06d" % frame

    dirpath = os.path.join("sequences", sequence_str, "predictions")

    scan_name = CONFIG.LIDAR_DATASET + "/sequences/" + sequence_str + "/velodyne/" + "%06d" % (frame) + ".bin"

    color_dict = CFG["color_map"]
    learning_map = CFG["learning_map"]
    nclasses = len(color_dict)
    scan = SemLaserScan(nclasses, map_dict=learning_map, sem_color_dict=color_dict, project=True, W=CONFIG.IMAGE_WIDTH)
    scan.open_scan(scan_name)
    label = np.zeros(scan.points.shape[0])

    X_train, y_train = get_trainig_data(pred, scan)
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn.fit(X_train, y_train)
    label = knn.predict(scan.points)

    # Translate to original learning map
    for i in range(scan.points.shape[0]):
        label[i] = CFG["learning_map_inv"][label[i]]

    print(os.path.join(dirpath, frame_str) + ".label")
    labels = label.astype(np.uint32)
    labels.tofile(os.path.join(dirpath, frame_str) + ".label")
    if save_ri == True:
        np.save(os.path.join(dirpath, frame_str), pred)


# Run test routine
def predict_ri(checkpoint=None, display=False, save_ri = False):
    # Which checkpoint should be tested
    if checkpoint is not None:
        CONFIG.TEST_CHECKPOINT = checkpoint

    # Create output directory if it doesnt exist
    dirpath = "sequences"
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    config = CONFIG.SEMKITTI_CFG
    # open config file
    try:
        print("Opening config file %s" % config)
        CFG = yaml.safe_load(open(config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()

    sequences = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    # sequences = [8] #small test
    for seq in sequences:
        test_path = "data/seq_%02d.tfrecord" % seq
        sequence_str = "%02d" % seq

        dirpath = os.path.join("sequences", sequence_str)
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        os.mkdir(dirpath)
        dirpath = os.path.join(dirpath, "predictions")
        os.mkdir(dirpath)

        print("Processing dataset file \"{}\" for checkpoint {}:".format(test_path, str(CONFIG.TEST_CHECKPOINT)))

        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            loader = tf.train.import_meta_graph(CONFIG.OUTPUT_MODEL + "-" + str(CONFIG.TEST_CHECKPOINT) + ".meta")
            loader.restore(sess, CONFIG.OUTPUT_MODEL + "-" + str(CONFIG.TEST_CHECKPOINT))

            points = graph.get_tensor_by_name("points_placeholder:0")
            neighbors = graph.get_tensor_by_name("neighbors_placeholder:0")
            train_flag = graph.get_tensor_by_name("flag_placeholder:0")
            y = graph.get_tensor_by_name("net/y:0")

            # Dataset iterator
            record_iterator = tf.python_io.tf_record_iterator(path=test_path)

            # Running network on each example
            line_num = 0
            for string_record in record_iterator:

                CONFIG.BATCH_SIZE = 1

                points_data, neighbors_data, _, label, mask = read_example(string_record)

                ref = np.reshape(points_data, (CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.IMAGE_DEPTH))
                img = ref

                # Inference
                data = sess.run(y, feed_dict={points: [points_data], neighbors: [neighbors_data], train_flag: False})
                pred = softmax(data[0, :, :, :])
                pred = np.argmax(pred, axis=2) * mask

                if display:
                    plt.subplot(2, 1, 1)
                    plt.imshow(ref[:, :, 3] * mask)
                    plt.title("Reflectance (for visualization)")
                    plt.subplot(2, 1, 2)
                    plt.imshow(pred)
                    plt.title("Prediction")
                    plt.show()

                print(" >> Prediction made for file {} frame {}".format(test_path, line_num))

                frame = line_num
                # For getting the labels of the whole pointcloud
                pc_propagation_Knn(seq, frame, pred, CFG, save_ri)
                line_num += 1


if __name__ == "__main__":
    predict_ri(display=False,save_ri=False)

