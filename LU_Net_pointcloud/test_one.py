"""
File used to test one checkpoint using the full validation set.
specify checkpoint to use by instruction --checkpoint
"""

import tensorflow as tf
import numpy as np
import cv2

import os
import sys
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

sys.path.append('./')
import data_loader
from settings import Settings

CONFIG = Settings(required_args=["gpu", "config", "checkpoint"])
label_names = ["unlabeled", "car", "bicicle", "motorcycle", "truck", "other-veh", "person", "bicyclist", "motocyclist",
               "road", "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain",
               "pole", "traffic-sign"]
label_number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


# Computes softmax
def softmax(x):
    e_x = np.exp(x)
    return e_x / np.expand_dims(np.sum(e_x, axis=2), axis=2)


# Erase line in stdout
def erase_line():
    sys.stdout.write("\033[F")


# Compute scores for a single image
def compute_iou_per_class(pred, label, mask, n_class):
    pred = np.argmax(pred[..., 0:n_class], axis=2) * mask
    label = label * mask

    ious = np.zeros(n_class)
    tps = np.zeros(n_class)
    fns = np.zeros(n_class)
    fps = np.zeros(n_class)

    for cls_id in range(n_class):
        tp = np.sum(pred[label == cls_id] == cls_id)
        fp = np.sum(label[pred == cls_id] != cls_id)
        fn = np.sum(pred[label == cls_id] != cls_id)

        ious[cls_id] = tp / (tp + fn + fp + 0.00000001)
        tps[cls_id] = tp
        fps[cls_id] = fp
        fns[cls_id] = fn

    return ious, tps, fps, fns


# Create a colored image with depth or label colors
def label_to_img(label_sm, depth, mask):
    img = np.zeros((label_sm.shape[0], label_sm.shape[1], 3))

    colors = np.array([[0, 0, 0], [78, 205, 196], [199, 244, 100], [255, 107, 107]])

    label = np.argmax(label_sm, axis=2)
    label = np.where(mask == 1, label, 0)

    for y in range(0, label.shape[0]):
        for x in range(0, label.shape[1]):
            if label[y, x] == 0:
                img[y, x, :] = [depth[y, x] * 255.0, depth[y, x] * 255.0, depth[y, x] * 255.0]
            else:
                img[y, x, :] = colors[label[y, x], :]

    return img / 255.0


# Export pointcloud with colored labels
def label_to_xyz(label_sm, data, mask, file):
    colors = np.array([[100, 100, 100], [78, 205, 196], [199, 244, 100], [255, 107, 107]])

    ys, xs = np.where(mask == 1)
    label = np.argmax(label_sm, axis=2)

    file = open(file, "w")
    for p in range(0, ys.shape[0]):
        x = xs[p]
        y = ys[p]
        l = label[y, x]
        file.write("{} {} {} {} {} {}\n".format(data[y, x, 0], data[y, x, 1], data[y, x, 2], colors[l, 0], colors[l, 1],
                                                colors[l, 2]))

    file.close()


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


def plot_confusion_matrix(cm, class_names):
    """
	Returns a matplotlib figure containing the plotted confusion matrix.

	Args:
	cm (array, shape = [n, n]): a confusion matrix of integer classes
	class_names (array, shape = [n]): String names of the integer classes
	"""

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color, fontsize=6)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


# Run test routine
def test(checkpoint=None, display=False):
    # Which checkpoint should be tested
    if checkpoint is not None:
        CONFIG.TEST_CHECKPOINT = checkpoint

    # Create output dir if needed
    if not os.path.exists(CONFIG.TEST_OUTPUT_PATH):
        os.makedirs(CONFIG.TEST_OUTPUT_PATH)

    print("Processing dataset file \"{}\" for checkpoint {}:".format(CONFIG.TFRECORD_VAL, str(CONFIG.TEST_CHECKPOINT)))

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        loader = tf.train.import_meta_graph(CONFIG.OUTPUT_MODEL + "-" + str(CONFIG.TEST_CHECKPOINT) + ".meta")
        loader.restore(sess, CONFIG.OUTPUT_MODEL + "-" + str(CONFIG.TEST_CHECKPOINT))

        points = graph.get_tensor_by_name("points_placeholder:0")
        neighbors = graph.get_tensor_by_name("neighbors_placeholder:0")
        train_flag = graph.get_tensor_by_name("flag_placeholder:0")
        y = graph.get_tensor_by_name("net/y:0")

        # Dataset iterator
        record_iterator = tf.python_io.tf_record_iterator(path=CONFIG.TFRECORD_VAL)

        # Running network on each example
        line_num = 1

        tps_sum = 0
        fns_sum = 0
        fps_sum = 0
        acc_sum = 0
        cm = 0
        forward_time = []
        for string_record in record_iterator:

            CONFIG.BATCH_SIZE = 1

            points_data, neighbors_data, groundtruth, label, mask = read_example(string_record)

            ref = np.reshape(points_data, (CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.IMAGE_DEPTH))
            groundtruth = data_loader.apply_mask(groundtruth, mask)

            # Inference
            t = time.time()
            data = sess.run(y, feed_dict={points: [points_data], neighbors: [neighbors_data], train_flag: False})
            forward_time.append(time.time() - t)
            pred = softmax(data[0, :, :, :])

            if display:
                plt.subplot(4, 1, 1)
                plt.imshow(ref[:, :, 3] * mask)
                plt.title("Reflectance (for visualization)")
                plt.subplot(4, 1, 2)
                plt.imshow(pred[:, :, 1] * mask)
                plt.title("Car prob")
                plt.subplot(4, 1, 3)
                plt.imshow(np.argmax(pred, axis=2) * mask)
                plt.title("Prediction")
                plt.subplot(4, 1, 4)
                plt.imshow(groundtruth)
                plt.title("Label")
                plt.show()

            iou, tps, fps, fns = compute_iou_per_class(pred, groundtruth, mask, CONFIG.N_CLASSES)
            acc = np.sum(tps) / (CONFIG.IMAGE_HEIGHT * CONFIG.IMAGE_WIDTH)

            tps_sum += tps
            fns_sum += fns
            fps_sum += fps
            acc_sum += acc
            # Confusion matrix computation

            pred = np.argmax(pred[..., 0:CONFIG.N_CLASSES], axis=2) * mask
            pred = list(pred.reshape((-1))) + label_number

            groundtruth = groundtruth * mask
            groundtruth = list(groundtruth.reshape((-1))) + label_number

            cm = cm + confusion_matrix(groundtruth, pred)
            cm = cm - confusion_matrix(label_number, label_number)
            print(" >> Processed file {}, time {} : IoUs {} , ".format(line_num, forward_time[line_num - 1], iou))

            line_num += 1

        cm_np = np.array(cm)
        np.save("training_pnl2/Confusion_matrix" + str(checkpoint), cm_np)

        # plt.show(plot_confusion_matrix(cm,label_names))

        ious = tps_sum.astype(np.float) / (tps_sum + fns_sum + fps_sum + 0.000000001)
        pr = tps_sum.astype(np.float) / (tps_sum + fps_sum + 0.000000001)
        re = tps_sum.astype(np.float) / (tps_sum + fns_sum + 0.000000001)
        iou = np.mean(ious)
        print(ious)
        print(np.mean(ious))
        print(ious[1:None])
        print(np.mean(ious[1:None]))
        iou_0 = np.mean(ious[1:None])
        acc = acc_sum / (line_num - 1)

        output = "[{}] Accuracy:\n".format(checkpoint)
        output += "mean IoU: {:.3f} , mean acc: {:.3f} \n".format(iou, acc)
        output += "with class 0 excluded: mean IoU: {:.3f}  \n".format(iou_0)
        output += "mean forward time per scan: {.6f} \n".format(np.mean(np.array(forward_time[2:])) / CONFIG.BATCH_SIZE)
        for i in range(1, CONFIG.N_CLASSES):
            output += "\tPixel-seg: P: {:.3f}, R: {:.3f}, IoU: {:.3f}\n".format(pr[i], re[i], ious[i])
        output += "\n"
        output += "checkpoint: [{}], IoU: {} , Accuracy: {} \n".format(checkpoint, iou, acc)
        return output


def ckpt_exists(ckpt):
    return os.path.isfile(CONFIG.OUTPUT_MODEL + "-" + str(ckpt) + ".meta")


def ckpt_exists(ckpt):
    return os.path.isfile(CONFIG.OUTPUT_MODEL + "-" + str(ckpt) + ".meta")


if __name__ == "__main__":
    file = open("results_one" + os.path.basename(CONFIG.CONFIG_NAME)[:-4] + ".txt", "w")

    ckpt = CONFIG.NUM_ITERS

    # This file was modified to test only on the last checkpoint
    # use --checkpoint
    output = test()

    print(output)
    file.write(output)
    file.flush()

    """ 
	while not ckpt_exists(ckpt + CONFIG.SAVE_INTERVAL) and ckpt < CONFIG.NUM_ITERS:
		print("Waiting for the next checkpoint ...")
		time.sleep(60)
	
		ckpt += CONFIG.SAVE_INTERVAL
	"""
    file.close()
