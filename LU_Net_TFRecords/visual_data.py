import numpy as np 
import cv2 
import time
import sys

#np.set_printoptions(threshold=sys.maxsize)
#pc = np.load('lidar_2d/2011_09_26_0001_0000000000.npy')
#pc = np.load('lidar_2d/2011_09_26_0001_0000000001.npy')
####


def visual_data_function(range_img):

	pc = range_img
	#pc = np.load('lidar_2d/2011_09_26_0070_0000000336.npy')
	img = pc[:,:,5]
	img = img / np.amax(img)
	img = 255*img
	img = img.astype(np.uint8)

	img_color = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
	img_color = img_color.astype(np.uint8)
	img_color[:,:,0] = img
	img_color[:,:,1] = 255
	img_color[:,:,2] = 255

	rgb_img = cv2.cvtColor(img_color,cv2.COLOR_HSV2BGR)
	img_color.astype(np.uint8)
	print(img_color.shape)
	print(img_color.dtype)
	cv2.namedWindow("image")
	#cv2.imshow('image', img_color)
	cv2.imshow('image', rgb_img)
	if(cv2.waitKey(10)=='q'):
		return
	time.sleep(0.1)
