# Deep learning-based LiDAR Point-cloud Semantic Segmentation using 2D Range Image

This project uses LU-Net as starting point and adapts it to the SemanticKITTI dataset. 
Throughout an exploratory research, we developed improvements to the original architecture to achieve higher score (mean Intersection over union).
We offer two versions of the training process:

### LU_Net_TFRecords
This version requires the generation of TFRecords for faster and independent training on a range image level. 
The user should run a script to generate the records that performs all the preprocessing and binarize the training examples. 

This version is recommended for faster training. 

### LU_Net_pointcloud
This version works directly from the pointcloud, performing all the preprocessing for each instance at each iteration. It is suitable for testing different configurations at any point of the process without the creation of TFRecords 
This version is able to do proper training but it hasn’t been optimized and takes around 2 seconds per batch at each iteration. 

Details about the research process are explained in the thesis master document


### References
LU-Net paper:
https://arxiv.org/pdf/1908.11656.pdf

LU-Net github:
https://github.com/pbias/lunet




