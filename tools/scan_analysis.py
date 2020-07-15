"""
Script to count the number of elements of each class throughout all the scans 
Use for a quantified analysis of the scans
"""
import numpy as np
import csv
import cv2
import yaml
import os

from pandas import read_csv

from auxiliary_SK.laserscan import LaserScan, SemLaserScan

clases_names =  ["unlabeled","car","bicycle","motorcycle","truck","other-vehicle","person","bicyclist","motorcyclist","road","parking","sidewalk","other-ground","building","fence","vegetation","trunk","terrain","pole","traffic-sign"]
def open_label(filename):
    """ Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
        raise TypeError("Filename should be string type, "
                        "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in ['.label']):
        raise RuntimeError("Filename extension is not valid label file.")

    # if all goes well, open label
    label = np.fromfile(filename, dtype=np.uint32)
    label = label.reshape((-1))


    sem_label = label & 0xFFFF  # semantic label in lower half

    #self.inst_label = label >> 16  # instance id in upper half
    # sanity check
    #assert ((self.sem_label + (self.inst_label << 16) == label).all())

    return sem_label


if __name__ == "__main__":

    config = "G:\\UB\\Documents-untilmay20th\\SemanticKITTI\\semantic-kitti-api\\config\\semantic-kitti.yaml"
    dataset = "G:\\UB\\Data\\data_odometry_velodyne\\dataset"
    # root_path = "G:\\UB\\Data\\Generated_Datasets\\20200228"

    # open config file
    try:
        print("Opening config file %s" % config)
        CFG = yaml.safe_load(open(config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()
    NUM_CLASSES = 20
    max_sequence = 10
    for s_index in range(max_sequence + 1):
        sequence = "%02d" % s_index

        root_path = os.path.join(dataset, "sequences", sequence)

        file_name = os.path.join(root_path, "pc_content.csv")
        file = open(file_name, 'w+', newline='')

        label_paths = os.path.join(root_path, "labels")
        label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(label_paths)) for f in fn]
        label_names.sort()

        data = []
        data_total = []
        counter = 0

        for label_file in label_names:
            counter_ = "%06d" % counter
            labels = open_label(label_file)
            data = []
            print(counter)
            #map to 20 classes
            for i in range(len(labels)):
                labels[i] = CFG["learning_map"][labels[i]]

            count = np.zeros(NUM_CLASSES)
            for c in range(NUM_CLASSES):
                count[c] = list(labels).count(c)

            data=count

            print(data)

            data_total.append(data)
            counter += 1
            if counter == 6:
                break
        a = np.asarray(data_total)
        np.savetxt(file_name, data_total, delimiter=",",fmt='%i')
        file.close()

        df = read_csv(file_name)
        df.columns = clases_names
        df.to_csv(file_name)
