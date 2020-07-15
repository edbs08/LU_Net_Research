# Deep learning based point-cloud Semantic Segmentation 

We use LU-Net as backbone, adapting it to SemanticKITTI dataset. We also update different areas of the architecture. 
Further details can be found in the thesis document in the root path. 

The implementation is done in Python and Tensorflow.

## Requirements
The instructions have been tested on Ubuntu 16.04 with Python 3.6 and TensorFlow 1.6 with GPU support.

First, clone the repository:
```bash
git clone https://github.com/edbs08/LU_Net_Research.git
```

Then download the odometry data, calibration and labels for the KITTI lidar scans. Detail information can be found in:
http://www.semantic-kitti.org/dataset.html in the "Download" section

Manualy modify the file config/lunet.cfg to specify the location of the dataset and the semantic-kitti configuration file.

Make sure you are in the LU_Net_TFRecords folder.
## Train and validate
You can generate the _training_ and _validation_ **TFRecords** by running the following command:
```bash
python3 make_tfrecord_pn.py --config=config/lunet.cfg
```
Once done, you can start training the model by running:
```bash
python train_V4.py --config=config/lunet.cfg --gpu=0
```
You can set which GPU is being used by tuning the `--gpu` parameter.

During or after training, you can run the validation as following:
```bash
python test_all.py --config=config/lunet.cfg --gpu=1
```
This script test on a range of checkpoints that were empirically selected. To modify this range change the values at the end of the file.
Results will be stored in a .txt file in the training folder. 

For a detailed result on each class for a specific checkpoint use:
```bash
python test_one.py --config=config/lunet.cfg --gpu=1 --checkpoint=500000
```
Change the parameter --checkpoint to the one of interest. 

## Customize the settings
You can easily create different configuration settings by editing the configuration file located in **config/lunet.cfg**.


## TO Do 
Auxiliary folder is actually a modification of semanticKITTI API. In the future, this should be adapted to use directly the official API