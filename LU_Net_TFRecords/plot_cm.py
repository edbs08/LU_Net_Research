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

label_names = ["unlabeled","car","bicicle","motorcycle","truck","other-veh","person","bicyclist","motocyclist","road","parking","sidewalk","other-ground","building","fence","vegetation","trunk","terrain","pole","traffic-sign"]
label_number = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]


def plot_confusion_matrix(cm, class_names):
	"""
	Returns a matplotlib figure containing the plotted confusion matrix.

	Args:
	cm (array, shape = [n, n]): a confusion matrix of integer classes
	class_names (array, shape = [n]): String names of the integer classes
	"""
	# Normalize the confusion matrix.
	cm = np.around(cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis]+0.000001), decimals=2)

	figure = plt.figure(figsize=(8, 8))
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title("Confusion matrix")
	plt.colorbar()
	tick_marks = np.arange(len(class_names))
	plt.xticks(tick_marks, class_names, rotation=45)
	plt.yticks(tick_marks, class_names)


	# Use white text if squares are dark; otherwise black.
	#threshold = cm.max() / 2.
	threshold = 0.5
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		color = "white" if cm[i, j] > threshold else "black"
		plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	return figure

if __name__ == "__main__":
	cm = np.load("Confusion_matrix.npy")
	plt.show(plot_confusion_matrix(cm,label_names))
