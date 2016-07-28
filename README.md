# Geometry-Based Region Proposals for Accelerated Image-Based Detection of 3D Objects


**Geom R-CNN** is an efficient framework for 3D object detection in Robotics applications. It is specifically designed for detecting objects in table-top scenes (or similar environments with objects sitting on a dominant plane).  The output of the system is a class label and 3D position for each object in the scene.  

Geom R-CNN
 - capitalizes on known geometric relationships to develop region proposals and localize each object
 - uses a CNN for object recognition
 - can be used to produce novel datasets, large enough for training deep neural net models
 - runs at an average of 12 hz on a Core i7 laptop
 - released as <img src="./ros.png" width="15"> ROS package 

[//]: # (For more details on Geom R-CNN, read the paper presented at the [Deep Learning Workshop at RSS 2016](http://juxi.net/workshop/deep-learning-rss-2016/).)


### Installation

1. Create a ROS workspace
```Shell
  mkdir -p ~/geom_rcnn_ws/src
  cd ~/geom_rcnn_ws/src
  catkin_init_workspace
  ```
2. Clone the Geom R-CNN repository in your ROS workspace
```Shell
  git clone https://github.com/asbroad/geom_rcnn.git
  ```
3. From the base directory of the workspace, build the code
```Shell
  cd ~/geom_rcnn_ws
  catkin_make
  ```

### Requirements

Software
1. [ROS](http://www.ros.org/) (Tested with ROS Indigo and Ubuntu 14.04)
2. [Theano](http://deeplearning.net/software/theano/) (Tested with Theano 0.8)
3. OpenCV
4. PCL 1.2 or greater

Additional requirements for training new CNNs (Neither is strictly necessary. sklearn is used to create a training and testing split for the model development.  matplotlib is used to plot results of model training).
1. sklearn
2. matplotlib

Hardware
1. A GPU.  If you choose to use the network described in the paper (and provided in code in this ), a mid-tier GPU is fine (e.g. nVidia GeForce 860M).  For training larger networks (e.g. VGG_16, GoogLeNet, etc...) a better GPU (e.g. TitanX) will be necessary.

### Running the System

**Object Detection Pipeline**: To run the object detection pipeline, open a terminal and run
```Shell
  roslaunch geom_rcnn full_pipeline.launch
  ```
Important parameters that can be set the 'full_pipeline.launch' file
* run_recognition
	* false (default): run full system *without* object recognition (you will still get object positions in 3D and bounding boxes displayed in the 2D image) 
	* true: run full system with object recognition (you will need to have trained a recognition model)
* xmin, xmax, ymin, ymax, zmin, zmax : parameters for point cloud passthrough filters.
* the following are necssary for object recognition
  * model_file : the location of the stored model weights 
  * category_file : the location of the stored object category dictionary

**Training a new CNN model**: To train a new CNN model for use in the Object Detection pipeline, open a terminal and run
```Shell
  roslaunch geom_rcnn train_cnn.launch
  ```
Important parameters that can be set the 'train_cnn.launch' file
* model_file : the location where the trained model will be stored  
* category_file : the location where the object category dictionary will be stored
* data_dir : the location of the base directory of the training data. The structure of the dataset expected by the code is as follows.  Within the base directory, each object class is broken out into its own subdirectory (the name of the subdirectory will be the class name). For example, for an apple class, there will be an 'apple' subdirectory within the data_dir, which itself contains all examples of that class (in .jpg format).  this is the structure that will be automatically created if using this system to create a new dataset.

**Creating a new dataset**: To create a new dataset using the object detection pipeline, open a terminal and run
```Shell
  roslaunch geom_rcnn dataset_acquisition.launch
  ```
* category : the current object category label.  that is, if you are capturing data related to the object category 'apple', this parameter would be 'apple'
* xmin, xmax, ymin, ymax, zmin, zmax : parameters for point cloud passthrough filters.
* data_dir : the same basedirectory directory described in the training instructions.  this code will automatically create the structure described in that section.
* init_idx : stored images will be labeled by their index.  each sequential image will be labeled as such numerically.  if you have already stored some data for a given object class, it is a good idea to set this value to a high number so as not to overwrite previous data
* rate : the number of images stored per second.

To create a multi-object dataset, you should collect data on each object seperately.  For best results, you'll want to capture many images of each object class at different positions, from different orientations and under different lighting conditions.  You can choose to move the object between runs of this launch file, or you can move the object while the code runs and retroactively remove images that includes poor data (such as your arm).

### Important ROS topics 
There are a number of topics used in the Geom R-CNN pipeline to pass messages between nodes, however the most imporant one for adding Geom R-CNN into your system is the '/detections' topic.  Each message includes a string representing the object class and a geometry_msgs/Point representing the centroid of the object in 3D.  To see the output of the system, open a terminal and run
 ```Shell
  rostopic echo /detections
  ```
To see how often messages are being published on the '/detections' topic, open a shell and run
 ```Shell
  rostopic hz /detections
  ```

[//]: # ()

### Citing Geom R-CNN

If you find Geom R-CNN useful in your research, please consider citing:

    @article{broad2016geometry,
        Author = {Alexander Broad and Brenna Argall},
        Title = {Geometry-Based Region Proposals for Accelerated Image-Based Detection of 3D Objects},
        booktitle={RSS Workshop Deep Learning}
        Year = {2016}
    }
)

### License

Geom R-CNN is released under the MIT License (refer to the LICENSE file for details).
