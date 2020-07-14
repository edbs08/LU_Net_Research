# Remove deprecated messages
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import os
import numpy as np
import cv2
import time
import glob
import math
import yaml
import sys
import tensorflow as tf
import pre_processing as prep

sys.path.append('./')
from settings import Settings

CONFIG = Settings()

# load data configuration file
config = os.path.join("semantic_kitti_api", "config", "semantic-kitti.yaml")
try:
    print("Opening config file %s" % config)
    CFG = yaml.safe_load(open(config, 'r'))
except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()

####################################### LOAD DATA #############################
def get_list_pc_files_train():
    files_all = []
    for seq in CONFIG.TRAIN_SEQ:
        seq_i = "%02d" % seq
        print("searching files seq %d", seq_i)

        pathIn = os.path.join(CONFIG.LIDAR_DATA, seq_i, "velodyne/")
        files_all = files_all + [pathIn + f for f in os.listdir(pathIn) if f.endswith('.bin')]
    files_all.sort()
    return files_all


def get_list_pc_files_val():
    files_all = []
    for seq in CONFIG.VALIDATION_SEQ:
        seq_i = "%02d" % seq
        print("searching files seq %d", seq_i)

        pathIn = os.path.join(CONFIG.LIDAR_DATA, seq_i, "velodyne/")
        files_all = files_all + [pathIn + f for f in os.listdir(pathIn) if f.endswith('.bin')]
    files_all.sort()
    return files_all

"""
def get_list_pc_files():
    files_all = []
    for seq in CONFIG.TRAIN_SEQ:
        seq_i = "%02d" % seq
        print("searching files seq %d", seq_i)

        pathIn = os.path.join(CONFIG.LIDAR_DATA, seq_i, "velodyne/")
        files_all = files_all + [pathIn + f for f in os.listdir(pathIn) if f.endswith('.bin')]
    files_all.sort()
    return files_all
"""

def get_next_batch(ds, batch_size):
    ds = ds.shuffle(buffer_size=100 * batch_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    return ds.take(batch_size)


def range_images_from_file(filename):
    #filename = tf.read_file(filename)heriWITy

    #print(type(filename))
    #filename = filename.encode('ascii','ignore') #= bytes.encode(filename) #filename.decode("utf-8") #= bytes.decode(filename)
    filename_ = filename.decode("utf-8")
    #filename_ = filename.encode('utf8', 'replace')
    #print(filename)
    #print(type(filename))
    #print(type(str(filename)))
    ri = prep.get_range_image(filename_, CONFIG, CFG)
    points, neighbours, label = prep.range_image_processing(ri, CONFIG)
    #print("*********************************************************")
    #print(neighbours.shape)
    #print(points.shape)
    #print(label.shape)
    return np.float32(points), np.float32(neighbours), np.float32(label)

"""
def range_images_from_files(batch):
    # returns list of range images
    neighbours = np.zeros(
        [CONFIG.BATCH_SIZE, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.N_LEN, CONFIG.IMAGE_DEPTH])
    points = np.zeros([CONFIG.BATCH_SIZE, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, 1, CONFIG.IMAGE_DEPTH])
    label = np.zeros([CONFIG.BATCH_SIZE, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.N_CLASSES + 2])
    count = 0
    neighbours_batch = np.array([CONFIG.BATCH_SIZE, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, ])
    batch_count = 0
    batch = batch.numpy().decode("utf-8")
    for elem in batch:
        #print(elem.numpy()[0].decode("utf-8"))
        print(elem)
        ri = prep.get_range_image(elem.numpy()[0].decode("utf-8"), CONFIG)
        print(ri.shape)
        neighbours[batch_count, :, :, :, :], points[batch_count, :, :, :, :], label[batch_count, :, :, :] = \
            prep.range_image_processing(ri, CONFIG)

        batch_count += 1

    print("*********************************************************")
    print(neighbours.shape)
    print(points.shape)
    print(label.shape)
    return points, neighbours, label

"""
def read_example(dataset, batch_size):
    #batch_ = get_next_batch(dataset, batch_size)
    #dataset = dataset.map(range_images_from_file)
    dataset = dataset.map(lambda x: tf.py_func(range_images_from_file, [x], [tf.float32, tf.float32, tf.float32]), num_parallel_calls=3)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    #batch_points, batch_neighbors, batch_label = iterator.get_next()
    return iterator.get_next()


################3


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


############################ NETWORK ARCHITECTURE ############################

# Components
def maxpool_layer(x, size, stride, name):
    with tf.name_scope(name):
        x = tf.layers.max_pooling2d(x, size, stride, padding='SAME')

    return x


def conv_layer(x, kernel, depth, train_logical, name):
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, depth, kernel, padding='SAME',
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                             bias_initializer=tf.zeros_initializer())
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x, training=train_logical, momentum=0.99, epsilon=0.001, center=True,
                                          scale=True)

    return x


def conv_transpose_layer(x, kernel, depth, stride, name):
    with tf.variable_scope(name):
        x = tf.contrib.layers.conv2d_transpose(x, depth, kernel, stride=stride, scope=name)

    return x


# U-net architecture
# - x: input tensor
# - train_logical: boolean indicating if training or not
def u_net(points, neighbors, train_logical):
    with tf.variable_scope('net'):

        # If pointnet is used as embedding extractor
        if CONFIG.POINTNET:
            neighbors_chan = 5  # xyzrd
            points_chan = 5  # xyzrd
            # num_channels = points_chan + CONFIG.N_LEN*neighbors_chan  # Same as input dimension (45)
            num_channels = 20  # Number of classes
            # with tf.variable_scope('pt_concat'):
            # 	x = tf.concat([neighbors[...,0:3], points[...,0:3]], axis=2, name='pn_concat')
            #
            # print(x.shape)

            # x = conv_layer(x, (1, 1), 16, train_logical, 'pn_conv_1')
            # x - feature

            p = tf.reshape(points[..., 0:points_chan], (-1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, points_chan))
            x = tf.reshape(neighbors[..., 0:neighbors_chan], (-1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.N_LEN*neighbors_chan))
            x = tf.concat([x, p], axis=3, name='pn_concat')
            # Starting from 64 channels since input is 45
            x = conv_layer(x, (3, 3), 64, train_logical, 'pn_conv_1')
            x = conv_layer(x, (3, 3), 128, train_logical, 'pn_conv_2')
            x = conv_layer(x, (3, 3), 256, train_logical, 'pn_conv_3')

            #with tf.variable_scope('pt_concat'):
                #points_chan = 4
                #p = tf.reshape(points[..., 0:points_chan], (-1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, points_chan))
                #x = tf.concat([x, p], axis=3, name='pn_concat')

            x = conv_layer(x, (3, 3), 128, train_logical, 'pn_conv_4')
            x = conv_layer(x, (3, 3), 64, train_logical, 'pn_conv_5')
            x = conv_layer(x, (3, 3), num_channels, train_logical, 'pn_conv_6')

            #x = tf.reshape(x, (-1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, num_channels))

            tf.summary.image('pn_embeddings',
                             tf.reshape(tf.transpose(x, perm=[0, 3, 1, 2]),
                                        (CONFIG.BATCH_SIZE * num_channels, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, 1)),
                             max_outputs=CONFIG.BATCH_SIZE * num_channels)
        else:
            x = tf.reshape(points, (-1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.IMAGE_DEPTH))

            # # Image pointnet
            # num_channels = 3
            # x = conv_layer(x[...,0:4], (3, 3), 4 * (CONFIG.N_LEN + 1), train_logical, 'neighbors_conv_1')
            #
            # x = tf.reshape(x, (-1, CONFIG.IMAGE_HEIGHT * CONFIG.IMAGE_WIDTH, CONFIG.N_LEN + 1, 4))
            #
            #
            # x = conv_layer(x, (1, 1), 16, train_logical, 'pn_conv_1')
            # x = conv_layer(x, (1, 1), 64, train_logical, 'pn_conv_2')
            # x = conv_layer(x, (1, 1), 128, train_logical, 'pn_conv_3')
            #
            # x = maxpool_layer(x, (1, 1), (1, CONFIG.N_LEN + 1), 'pn_maxpool_1')
            #
            # #
            # # with tf.variable_scope('pt_concat'):
            # # 	x = tf.concat([x, points[...,3:4]], axis=3, name='pn_concat')
            #
            # x = conv_layer(x, (1, 1), 64, train_logical, 'pn_conv_4')
            # x = conv_layer(x, (1, 1), 16, train_logical, 'pn_conv_5')
            # x = conv_layer(x, (1, 1), num_channels, train_logical, 'pn_conv_6')
            #
            # x = tf.reshape(x, (-1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, num_channels))

            # Saves input image in tensorboard for checking
            tf.summary.image('img_init',
                             tf.reshape(tf.transpose(x, perm=[0, 3, 1, 2]),
                                        (CONFIG.BATCH_SIZE * num_channels, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, 1)),
                             max_outputs=CONFIG.BATCH_SIZE)

        # Encoder
        input_neurons = 64
        levels = []
        for level in range(0, CONFIG.UNET_DEPTH - 1):
            neurons = input_neurons * (2 ** level)
            x = conv_layer(x, (3, 3), neurons, train_logical, 'conv' + str(level) + '_1')
            levels.append(conv_layer(x, (3, 3), neurons, train_logical, 'conv' + str(level) + '_2'))
            x = maxpool_layer(levels[level], (1, 2), (1, 2), 'maxpool' + str(level))

        # Decoder
        for level in range(CONFIG.UNET_DEPTH - 2, -1, -1):
            neurons = input_neurons * (2 ** level)
            x = conv_layer(x, (3, 3), neurons * 2, train_logical, 'conv' + str(level) + '_3')
            x = conv_layer(x, (3, 3), neurons, train_logical, 'conv' + str(level) + '_4')
            x = conv_transpose_layer(x, (1, 2), neurons, (1, 2), 'deconv' + str(level))
            x = tf.concat((levels[level], x), axis=3)

        # Final layers (convolution at input scale and fully connected)
        x = conv_layer(x, (3, 3), input_neurons, train_logical, 'conv_final_1')
        x = conv_layer(x, (3, 3), input_neurons, train_logical, 'conv_final_2')
        x = conv_layer(x, (1, 1), CONFIG.N_CLASSES, train_logical, 'fully_connected1')

        y = tf.identity(x, name='y')

        return y


############################ NETWORK LOSS ############################

# Returns slices of a tensor
def slice_tensor(x, start, end=None):
    if end < 0:
        y = x[..., start:]
    else:
        if end is None:
            end = start
        y = x[..., start:end + 1]

    return y

def gen_dice_class_mean(y_true, y_pred, eps=1e-6):
    # [b, h, w, classes]
    pred_tensor = tf.nn.softmax(y_pred)
    y_true_shape = tf.shape(y_true)

    # [b, h*w, classes]
    y_true = tf.reshape(y_true, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])
    y_pred = tf.reshape(pred_tensor, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])

    # [b, classes]
    # count how many of each class are present in
    # each image, if there are zero, then assign
    # them a fixed weight of eps
    counts = tf.reduce_sum(y_true, axis=1)
    weights = 1. / (counts ** 2)
    weights = tf.where(tf.is_finite(weights),weights, eps*tf.ones_like(weights))

    multed = weights*tf.reduce_sum(y_true * y_pred, axis=1)
    summed = weights*tf.reduce_sum(y_true + y_pred, axis=1)

    # [classes]
    numerators = tf.reduce_sum(weights * multed, axis=0)
    denom = tf.reduce_sum(weights * summed, axis=0)
    dices = 1. - 2. * numerators / denom
    dices = tf.where(tf.is_finite(dices), dices, tf.zeros_like(dices))

    # scalar
    return tf.reduce_mean(dices)

# U-net loss
# - pred: output of u_net(...)
# - label: a [BATCH, HEIGHT, WIDTH, N_CLASSES+1] tensor where last channel is
#          a weight map
def u_net_loss(pred, label, freq):
    with tf.variable_scope('loss'):
        # Retrieve mask on last channel of the label
        mask = slice_tensor(label, CONFIG.N_CLASSES + 1, -1)
        dist = slice_tensor(label, CONFIG.N_CLASSES, CONFIG.N_CLASSES)
        label = slice_tensor(label, 0, CONFIG.N_CLASSES - 1)
        epsilon = 1.e-9
        weight_norm = 2.0 * 3.0 ** 2.0

        weights_ce = 0.1 + 1.0 * tf.exp(- dist / weight_norm)
        weights_ce = weights_ce * mask
        weight_c = freq+epsilon
        weight_c = tf.convert_to_tensor(weight_c, dtype=tf.float32)
        weight_c = -tf.log(weight_c)

        weight_c_sum = tf.reduce_sum(weight_c*label)

        # Compute the cross entropy
        if CONFIG.FOCAL_LOSS:
            with tf.name_scope('focal_loss'):

                gamma = 2.
                pred_softmax = tf.nn.softmax(pred)
                cross_entropy = tf.multiply(label, -tf.log(pred_softmax + epsilon))
                weights_fl = tf.multiply(label, tf.pow(tf.subtract(1., pred_softmax), gamma))
                weigths_total = weights_fl * weights_ce
                loss = tf.reduce_sum(weights_ce * weight_c * weights_fl * cross_entropy) / (tf.reduce_sum(weights_ce) + weight_c_sum)

                loss = gen_dice_class_mean(label, pred) + loss
                tf.summary.scalar("global_loss", loss)
        else:
            with tf.name_scope('loss'):

                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=pred,
                                                                           name="cross_entropy")
                loss = tf.reduce_sum(tf.reshape(cross_entropy,
                                                [CONFIG.BATCH_SIZE, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH,
                                                 1]) * weights_ce) / tf.reduce_sum(weights_ce)

                tf.summary.scalar("global_loss", loss)

        # Compute average precision
        with tf.name_scope('average_precision'):
            softmax_pred = tf.nn.softmax(pred)

            argmax_pred = tf.math.argmax(softmax_pred, axis=3)
            mask_bin = tf.squeeze(tf.math.greater(mask, 0))
            for c in range(1, CONFIG.N_CLASSES):
                p = tf.math.equal(argmax_pred, c)
                l = tf.squeeze(tf.math.equal(slice_tensor(label, c, c), 1.0))

                intersection = tf.logical_and(p, l)
                union = tf.logical_or(p, l)

                iou = tf.reduce_sum(tf.cast(tf.logical_and(intersection, mask_bin), tf.float32)) / (
                            tf.reduce_sum(tf.cast(tf.logical_and(union, mask_bin), tf.float32)) + 0.00000001)
                tf.summary.scalar("iou_class_0" + str(c), iou)

    # Display prediction and groundtruth for object of class 1
    with tf.variable_scope('predictions'):
        for i in range(0, CONFIG.N_CLASSES):
            tf.summary.image('pred_class_0' + str(i),
                             tf.reshape(tf.transpose(slice_tensor(pred, i, i) * mask, perm=[0, 3, 1, 2]),
                                        (CONFIG.BATCH_SIZE, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, 1)),
                             max_outputs=CONFIG.BATCH_SIZE)

    with tf.variable_scope('labels'):
        for i in range(0, CONFIG.N_CLASSES):
            tf.summary.image('label_class_0' + str(i),
                             tf.reshape(tf.transpose(slice_tensor(label, i, i) * mask, perm=[0, 3, 1, 2]),
                                        (CONFIG.BATCH_SIZE, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, 1)),
                             max_outputs=CONFIG.BATCH_SIZE)

    return loss


############################ TRAINING MANAGER ############################

# Displays configuration
def print_config():
    print("\n----------- RIU-NET CONFIGURATION -----------")
    print("input channels     : {}".format(CONFIG.CHANNELS.upper()))
    print("input dims         : {}x{}x{}".format(CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.IMAGE_DEPTH))
    print("pointnet embeddings: {}".format("yes" if CONFIG.POINTNET == True else "no"))
    print("focal loss         : {}".format("yes" if CONFIG.FOCAL_LOSS == True else "no"))
    print(
        "# of parameters    : {}".format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
    print("---------------------------------------------\n")


# Compute the average example processing time
def time_to_speed(batch_time, batch_size):
    return round(float(batch_size) / batch_time, 2)


def time_to_speed_2(batch_time, batch_size):
    return batch_time / float(batch_size)


# Pretty obvious
def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Returns last saved checkpoint index
def get_last_checkpoint(output_model):
    files = glob.glob(output_model + "-*.index")
    checkpoints = [-1]
    for checkpoint in files:
        checkpoint = checkpoint.replace(output_model + "-", "")
        checkpoint = checkpoint.replace(".index", "")
        checkpoints.append(int(checkpoint))

    return max(checkpoints)


# Computes current learning rate given the decay settings
def get_learning_rate(iteration, start_rate, decay_interval, decay_value):
    """
        decay_interval should be less than total number of iterations to have an actual impact
        set decay_interval to something around 100000
        """
    #rate = start_rate * (decay_value ** math.floor(iteration / decay_interval))
    # return rate
    return 0


# Training routine
def train():
    freq = np.zeros(CONFIG.N_CLASSES)
    sum_ = 0
    for k in CFG["learning_map"].keys():
        sum_ += CFG["content"][k]
        freq[CFG["learning_map"][k]] += CFG["content"][k]
    #print(sum_)
    #print(np.sum(freq))

    # Remove deprecated messages
    tf.logging.set_verbosity(tf.logging.ERROR)

    ds_train = tf.data.Dataset.list_files(get_list_pc_files_train())
    ds_train.shuffle(buffer_size=100 * CONFIG.BATCH_SIZE)

    ds_val = tf.data.Dataset.list_files(get_list_pc_files_val())
    ds_val.shuffle(buffer_size=100 * CONFIG.BATCH_SIZE)

    # Reading TFRecords
    with tf.device('/cpu:0'):
        with tf.name_scope('train_batch'):
            batch_points, batch_neighbors, batch_label = read_example(ds_train, CONFIG.BATCH_SIZE)
        with tf.name_scope('val_batch'):
            val_points, val_neighbors, val_label = read_example(ds_val, CONFIG.BATCH_SIZE)
        with tf.name_scope('learning_rate'):
            learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate_placeholder")

    # Save learning rate
    # tf.summary.scalar("learning_rate", learning_rate, family="learning_rate")

    # Creating input placeholder
    points = tf.placeholder(shape=[None, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, 1, CONFIG.IMAGE_DEPTH],
                            dtype=tf.float32, name='points_placeholder')
    neighbors = tf.placeholder(shape=[None, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.N_LEN, CONFIG.IMAGE_DEPTH],
                               dtype=tf.float32, name='neighbors_placeholder')
    label = tf.placeholder(shape=[None, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.N_CLASSES + 2],
                           dtype=tf.float32, name='label_placeholder')

    train_flag = tf.placeholder(dtype=tf.bool, name='flag_placeholder')

    # Create Network and Loss operator
    y = u_net(points, neighbors, train_flag)
    loss = u_net_loss(y, label, freq)

    # Creating optimizer
    opt = tf.train.AdamOptimizer(learning_rate=CONFIG.LEARNING_RATE)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = opt.minimize(loss)

    # Merging summaries
    summary_op = tf.summary.merge_all()

    # Saver for checkpoints
    saver = tf.train.Saver(max_to_keep=1000)

    # Starting session
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Check if a checkpoint already exists and load it
    start_step = 0
    if get_last_checkpoint(CONFIG.OUTPUT_MODEL) >= 0:
        start_step = get_last_checkpoint(CONFIG.OUTPUT_MODEL)
        print("[From checkpoint] Restoring checkpoint {}".format(start_step))
        saver.restore(sess, CONFIG.OUTPUT_MODEL + "-" + str(start_step))

    # Creating threads for batch sampling
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Creating output logs
    train_summary_writer = tf.summary.FileWriter(CONFIG.OUTPUT_LOGS + "train/", tf.get_default_graph())
    val_summary_writer = tf.summary.FileWriter(CONFIG.OUTPUT_LOGS + "val/", tf.get_default_graph())

    # Display config
    print_config()

    # Starting iterations
    for i in range(start_step, CONFIG.NUM_ITERS):

        # Retrieving current batch
        points_data, neighbors_data, label_data = sess.run([batch_points, batch_neighbors, batch_label])

        # Computing learning rate
        rate = get_learning_rate(i, CONFIG.LEARNING_RATE, 0, 0)

        # Training Network on the current batch
        t = time.time()
        _, loss_data, data = sess.run([train_step, loss, y],
                                      feed_dict={train_flag: True, points: points_data, neighbors: neighbors_data,
                                                 label: label_data})

        print('[Training] Iteration: {}, loss: {}, {} e/s '.format(int(i), loss_data,
                                                                   time_to_speed(time.time() - t, CONFIG.BATCH_SIZE)))

        # If summary has to be saved
        if i % 100 == 0:
            summary_str = sess.run(summary_op,
                                   feed_dict={train_flag: True, points: points_data, neighbors: neighbors_data,
                                              label: label_data})
            train_summary_writer.add_summary(summary_str, i)
        # If checkpoint has to be saved
        if (i + 1) % CONFIG.SAVE_INTERVAL == 0:
            saver.save(sess, CONFIG.OUTPUT_MODEL, global_step=i + 1)
        # If validation should be done
        if i % CONFIG.VAL_INTERVAL == 0:
            # Retrieving validation batch
            points_data, neighbors_data, label_data = sess.run([val_points, val_neighbors, val_label])

            # Running summaries and saving
            summary_str = sess.run(summary_op,
                                   feed_dict={train_flag: False, points: points_data, neighbors: neighbors_data,
                                              label: label_data})
            val_summary_writer.add_summary(summary_str, i)

            print("[Validation] Done on one batch")

    # Saving final network configuration
    saver.save(sess, CONFIG.OUTPUT_MODEL, global_step=i + 1)


if __name__ == "__main__":
    train()
