import os
import glob
import numpy as np
from optparse import OptionParser
import configparser

class Settings(object):
	LIDAR_DATA = None
	LABELS_DATA = None
	CONFIG_NAME = None
	TRAIN_SEQ = None
	VALIDATION_SEQ = None
	TEST_SEQ = None

	N_CLASSES = None
	IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH = None, None, None

	UNET_DEPTH        = None
	BATCH_SIZE        = None
	LEARNING_RATE     = None
	NUM_ITERS         = None
	FOCAL_LOSS        = None
	SAVE_INTERVAL     = None
	VAL_INTERVAL      = None

	OUTPUT_PATH  = None
	OUTPUT_MODEL = None
	OUTPUT_LOGS  = None

	AUGMENTATION   = None
	N_SIZE         = None
	N_LEN          = None
	CHANNELS       = ""
	POINTNET       = False

	TEST_CHECKPOINT     = None
	TEST_OUTPUT_PATH    = None

	# Constructor
	def __init__(self, required_args = ["config", "dataset", "gpu"]):
		parser = OptionParser()

		if "config" in required_args:
			parser.add_option("-c", "--config", dest="config", help="Configure configuration")
		if "gpu" in required_args:
			parser.add_option("-g", "--gpu", dest="gpu", help="Configure visible GPU")
		if "checkpoint" in required_args:
			parser.add_option("-m", "--checkpoint", dest="checkpoint", help="Configure checkpoint to load")


		(options, args) = parser.parse_args()


		if "gpu" in required_args and options.gpu:
		    os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu
		elif "gpu" in required_args:
			print('Please specify --gpu to select a GPU. You can run "gpustat" in order to see which GPU is available on the machine.')
			exit(0)

		config = None
		if "config" in required_args and options.config:
			config = configparser.ConfigParser()
			config.read(options.config)
		elif "config" in required_args:
			print('Please specify --config to select a configuration file.')
			exit(0)

		self.CONFIG_NAME = options.config

		# SETTING VALUES
		if "config" in required_args:
			self.LIDAR_DATA       = config["DATA"]["lidar_dataset"]
			self.LABELS_DATA      = config["DATA"]["labels_path"]
			self.TRAIN_SEQ = eval(config["DATA"]["train"])
			self.VALIDATION_SEQ   = eval(config["DATA"]["validation"])
			self.TEST_SEQ = eval(config["DATA"]["test"])
			self.AUGMENTATION   = eval(config["DATA"]["augmentation"])
			self.N_SIZE         = eval(config["DATA"]["n_size"])
			self.N_LEN          = self.N_SIZE[0] * self.N_SIZE[1] - 1
			self.CHANNELS       = config["DATA"]["channels"]
			self.POINTNET       = eval(config["DATA"]["pointnet"])

			self.N_CLASSES    = int(config["NETWORK"]["n_classes"])
			self.IMAGE_HEIGHT = int(config["NETWORK"]["img_height"])
			self.IMAGE_WIDTH  = int(config["NETWORK"]["img_width"])
			self.IMAGE_DEPTH  = len(self.CHANNELS)

			self.UNET_DEPTH        = int(config["TRAINING"]["unet_depth"])
			self.BATCH_SIZE        = int(config["TRAINING"]["batch_size"])
			self.LEARNING_RATE     = float(config["TRAINING"]["learning_rate"])
			self.NUM_ITERS         = int(config["TRAINING"]["num_iterations"])
			self.FOCAL_LOSS        = eval(config["TRAINING"]["focal_loss"])
			self.SAVE_INTERVAL     = int(config["TRAINING_OUTPUT"]["save_interval"])
			self.VAL_INTERVAL      = int(config["TRAINING"]["val_interval"])

			self.OUTPUT_PATH  = config["TRAINING_OUTPUT"]["path"]
			self.OUTPUT_MODEL = self.OUTPUT_PATH + config["TRAINING_OUTPUT"]["model"]
			self.OUTPUT_LOGS  = self.OUTPUT_PATH + config["TRAINING_OUTPUT"]["logs"]

			self.TEST_OUTPUT_PATH    = config["TEST"]["output_path"]

		if "checkpoint" in required_args and options.checkpoint:
			self.TEST_CHECKPOINT = int(options.checkpoint)
		elif "checkpoint" in required_args:
			files = glob.glob(self.OUTPUT_MODEL + "-*.index")
			checkpoints = []
			for checkpoint in files:
				checkpoint = checkpoint.replace(self.OUTPUT_MODEL + "-", "")
				checkpoint = checkpoint.replace(".index", "")
				checkpoints.append(int(checkpoint))

			self.TEST_CHECKPOINT = max(checkpoints)
