"""
Tools for visualize the Range Images, compare and save. 
"""

import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import yaml
import os
import sys

# np.set_printoptions(threshold=sys.maxsize)
# pc = np.load('lidar_2d/2011_09_26_0001_0000000000.npy')
# pc = np.load('lidar_2d/2011_09_26_0001_0000000001.npy')
labels = {
    0: "unlabeled",
    1: "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle",
}
color_map = {
    0: [0, 0, 0],
    1: [0, 0, 0],
    10: [245, 150, 100],
    11: [245, 230, 100],
    13: [250, 80, 100],
    15: [150, 60, 30],
    16: [255, 0, 0],
    18: [180, 30, 80],
    20: [255, 0, 0],
    30: [30, 30, 255],
    31: [200, 40, 255],
    32: [90, 30, 150],
    40: [255, 0, 255],
    44: [255, 150, 255],
    48: [75, 0, 75],
    49: [75, 0, 175],
    50: [0, 200, 255],
    51: [50, 120, 255],
    52: [0, 150, 255],
    60: [170, 255, 150],
    70: [0, 175, 0],
    71: [0, 60, 135],
    72: [80, 240, 150],
    80: [150, 240, 255],
    81: [0, 0, 255],
    99: [255, 255, 50],
    252: [245, 150, 100],
    256: [255, 0, 0],
    253: [200, 40, 255],
    254: [30, 30, 255],
    255: [90, 30, 150],
    257: [250, 80, 100],
    258: [180, 30, 80],
    259: [255, 0, 0],
}

small_color_map = {
    0: [0, 0, 0],
    1: [0, 0, 255],
    2: [245, 150, 100],
    3: [245, 230, 100],
}

learning_map_inv = {  # inverse of previous map
    0: 0,  # "unlabeled", and others ignored
    1: 10,  # "car"
    2: 11,  # "bicycle"
    3: 15,  # "motorcycle"
    4: 18,  # "truck"
    5: 20,  # "other-vehicle"
    6: 30,  # "person"
    7: 31,  # "bicyclist"
    8: 32,  # "motorcyclist"
    9: 40,  # "road"
    10: 44,  # "parking"
    11: 48,  # "sidewalk"
    12: 49,  # "other-ground"
    13: 50,  # "building"
    14: 51,  # "fence"
    15: 70,  # "vegetation"
    16: 71,  # "trunk"
    17: 72,  # "terrain"
    18: 80,  # "pole"
    19: 81,  # "traffic-sign"
}
use_color_map_SemanticKITTI = True


def comprate_ruttine():
    path_A = np.array(["/media/daniel/FILES/UB/Data/Generated_Datasets/All_labels/00_",
                       "/media/daniel/FILES/UB/Data/Generated_Datasets/All_labels/01_"])
    img_color_A = np.zeros((64, 512, 3), dtype=np.uint8)

    path_B = np.array(["/home/daniel/Documents/LU_Net_Original/lidar_2d/2011_09_26_0001_",
                       "/home/daniel/Documents/LU_Net_Original/lidar_2d/2011_09_26_0020_"])
    img_color_B = np.zeros((64, 512, 3), dtype=np.uint8)

    for ii in range(2):
        print(path_A[ii])
        print(path_B[ii])
        for i in range(50):
            nameA = "%06d.npy" % i
            nameA = path_A[ii] + nameA
            pc = np.load(nameA)
            labels_ri = pc[:, :, 5]
            for r in range(labels_ri.shape[0]):
                for c in range(labels_ri.shape[1]):
                    # print(img_color_A[r,c,:])
                    # print(labels_ri[r,c])
                    # print(color_map[labels_ri[r,c]])
                    img_color_A[r, c, :] = color_map[labels_ri[r, c]]

            nameB = "%010d.npy" % i
            nameB = path_B[ii] + nameB
            pc = np.load(nameB)
            labels_ri = pc[:, :, 5]
            for r in range(labels_ri.shape[0]):
                for c in range(labels_ri.shape[1]):
                    # print (img_color_A[r,c,:])
                    # print(labels_ri[r,c])
                    # print(color_map[labels_ri[r,c]])
                    img_color_B[r, c, :] = small_color_map[labels_ri[r, c]]

            cv2.namedWindow("image Semantic KITTI")
            cv2.imshow('image Semantic KITTI', img_color_A)

            cv2.namedWindow("image SqueezeSeg")
            cv2.imshow('image SqueezeSeg', img_color_B)
            # cv2.imshow('image', rgb_img)
            c = cv2.waitKey(0)
            if 'q' == chr(c & 255):
                # print("finish")
                break

def list_files2(directory, extension):
    for (dirpath, dirnames, filenames) in os.walk(directory):
        return (f for f in filenames if f.endswith('.' + extension))

def colorize_range_image(labels_ri):
    color_image = np.zeros((labels_ri.shape[0], labels_ri.shape[1], 3), dtype=np.uint8)
    for r in range(labels_ri.shape[0]):
        for c in range(labels_ri.shape[1]):
            color_image[r, c, :] = color_map[learning_map_inv[labels_ri[r, c]]]
    return color_image

def RI_comprate_ruttine():
    extention = 'npy'

    directory_gt = "G:\\UB\\Data\\Generated_Datasets\\20200312\\train\\08_"
    directory_pred = "G:\\UB\\Data\\Predictions\\20200524\\test\\sequences\\08\\predictions"
    files_pred = list(list_files2(directory_pred,extention))

    for i in range(0, len(files_pred)):
        num = "%06d" % i
        files_gt = "{}.npy".format(num)
        print(directory_gt+files_gt)
        gt = np.load(directory_gt+files_gt)
        gt = gt[:,:,5]
        gt = colorize_range_image(gt)

        print(os.path.join(directory_pred,files_pred[i]))
        pred = np.load(os.path.join(directory_pred,files_pred[i]))
        pred = colorize_range_image(pred)

        plt.subplot(2, 1, 1)
        plt.imshow(gt)
        plt.title("Groundtruth")

        plt.subplot(2, 1, 2)
        plt.imshow(pred)
        plt.title("Prediction")


        plt.subplots_adjust(top=0.9, bottom=0.0, left=0.01, right=0.995)
        figure = plt.gcf()
        figure.set_size_inches(9,2.5)
        #plt.savefig("images\\{}.jpg".format(num),dpi=400)
        plt.show()
        plt.cla()
        plt.clf()






def paint_single_class(actualClass, sclass):
    color = [0, 0, 0]
    for sclass_ in sclass:
        if (actualClass == sclass_):
            color = color_map[actualClass]
    return color


def get_inverse_label(pc):
    for r in range(pc.shape[0]):
        for c in range(pc.shape[1]):
            pc[r, c, 5] = learning_map_inv[pc[r, c, 5]]
    return pc


def pc_to_image_color(pc,learning_inv):
    if (learning_inv):
        pc = get_inverse_label(pc)
    # print(pc.shape)
    labels_ri = pc[:, :, 5]
    img_color = np.zeros((labels_ri.shape[0], labels_ri.shape[1], 3), dtype=np.uint8)
    if (use_color_map_SemanticKITTI == True):
        for r in range(labels_ri.shape[0]):
            for c in range(labels_ri.shape[1]):
                img_color[r, c, :] = color_map[labels_ri[r, c]]
                sclass = [0, 10]
            # img_color[r,c,:] = paint_single_class(labels_ri[r,c],sclass)

    else:
        img = pc[:, :, 5]
        img = img / np.amax(img)
        img = 255 * img
        img = img.astype(np.uint8)
        img_color = img_color.astype(np.uint8)
        img_color[:, :, 0] = img
        img_color[:, :, 1] = 255
        img_color[:, :, 2] = 255
        img_color = cv2.cvtColor(img_color, cv2.COLOR_HSV2BGR)
        img_color.astype(np.uint8)
    return img_color


def normal_display():
    for i in range(1):
        ###############
        # name = "lidar_2d/2011_09_26_0001_%010d.npy" % i
        # pc = np.load('lidar_2d/2011_09_26_0070_0000000336.npy')
        # print(name)
        ###############
        #name = "/media/daniel/FILES/UB/Data/Generated_Datasets/20200228/08_000000.npy"
        learning_inv = True
        # name = "/media/daniel/FILES/UB/Data/Generated_Datasets/All_labels/08_000000.npy"
        ############## WINDOWS
        name = "G:\\UB\\Data\\Generated_Datasets\\20200312\\train\\08_000005.npy"

        pc = np.load(name)
        pc_flip = np.zeros(pc.shape)
        division = int(pc.shape[1]/2)
        pc_flip[:, 0:division, :] = pc[:, division:pc.shape[1], :]
        pc_flip[:, division:pc.shape[1], :] = pc[:, 0:division, :]



        plt.subplot(2,1, 1)
        img_color = pc_to_image_color(pc,learning_inv)
        plt.imshow(img_color)
        plt.subplot(2, 1, 2)
        img_color = pc_to_image_color(pc_flip,learning_inv)
        plt.imshow(img_color)
        plt.show()

        #cv2.namedWindow("image")
        #cv2.imshow('image', img_color)

        #c = cv2.waitKey(0)
        #if 'q' == chr(c & 255):
        #    print("finish")
        # do nothing

        # count number of classes
        print("******************** ")
        print(np.amax(labels_ri))




def visual_data_function(range_img):
    pc = range_img
    # pc = np.load('lidar_2d/2011_09_26_0070_0000000336.npy')
    img = pc[:, :, 5]
    img = img / np.amax(img)
    img = 255 * img
    img = img.astype(np.uint8)

    img_color = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img_color = img_color.astype(np.uint8)
    img_color[:, :, 0] = img
    img_color[:, :, 1] = 255
    img_color[:, :, 2] = 255

    rgb_img = cv2.cvtColor(img_color, cv2.COLOR_HSV2BGR)
    img_color.astype(np.uint8)
    # print(img_color.shape)

    cv2.namedWindow("image")
    # cv2.imshow('image', img_color)
    cv2.imshow('image', rgb_img)
    c = cv2.waitKey(0)
    if 'q' == chr(c & 255):
        print("finish")
    time.sleep(5)


def analyze_cfg():
    # open config file
    config = "G:\\UB\\Documents\\SemanticKITTI\\semantic-kitti-api\\config\\semantic-kitti.yaml"
    try:
        print("Opening config file %s" % config)
        CFG = yaml.safe_load(open(config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()

    freq = np.zeros(20)

    for k in CFG["learning_map"].keys():
        freq[CFG["learning_map"][k]] += CFG["content"][k]
    print(freq)
    weight_c = -np.log(freq)
    print(weight_c)


if __name__ == "__main__":
#    normal_display()
    RI_comprate_ruttine()
# comprate_ruttine()
    #analyze_cfg()