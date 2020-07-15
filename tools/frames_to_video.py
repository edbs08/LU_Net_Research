"""
File used to create a .avi video from a list of range images
"""
import cv2
import numpy as np
import os
from os.path import isfile, join

#########################  CONFIG ################################################
# Path Images / Frames
pathIn = "C:\\Users\\Daniel\\Google Drive\\Masters\\Intership\\Code\\pointCloud_RangeImage\\images\\"
# Output path for video
pathOut = 'video.avi'
#frame rate
fps = 5
###################################################################################

frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
files.sort()

for i in range(3000, len(files)):#(len(files)):
    filename = pathIn + files[i]
    print(filename)
    # reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)

    # inserting the frames into an image array
    frame_array.append(img)

out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()