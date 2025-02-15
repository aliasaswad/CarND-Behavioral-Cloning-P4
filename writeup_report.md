# **CarND-Behavioral Cloning Project** 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="./readme_writeup/second_track.gif" width="700" height="400" align="center"/>

---
Overview
---
This repository contains the Behavioral Cloning Project for [Udacity Self Driving Car Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013).
In this project, I used deep neural networks and convolutional neural networks to clone driving behavior. I trained, validated and tested a model using Keras. The model output a steering angle to an autonomous vehicle.

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

#### 1. All required files used to run the simulator in autonomous mode

My project includes the following files:
* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **model.h5** containing a trained convolution neural network 
* **visualizer.ipynb** for visuaizing the model
* **clone.py** containing the script to test different models
* **writeup_report.md** : Summarizing the results

At the beginning, I worked on [`clone.py`](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/clone.py) file to test two different models and compared loss/valid. loss values. [LeNet](http://yann.lecun.com/exdb/lenet/) and [nVidia Autonomous](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) models structures were used for that test to create and train model. Also, I visualized my final results using a script in [`visualizer.ipynb`](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/visualizer.ipynb).

---
#### 2. Functional code
Using Udacity provided [simulator](https://github.com/aliasaswad/Self-Driving-Car-Simulator) and my [`drive.py`](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/drive.py) file, the car can be driven autonomously around the track by executing the command below. Within `drive.py` file, I changed the car speed from 9 to 15 by modifing the `set_speed` variable on code_line [47](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/drive.py#L47).

```sh
python drive.py model.h5
```

#### 3. Code is usable and readable

The [`model.py`](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My first work was using LeNet model, but the model didn't able to drive the car inside the street with three epochs (The model in  the `clone.py` could be found [here](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/clone.py#L49-L60)). Then, I tried another model which is [nVidia Autonomous Car Group model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). The nVidia model was able to drive the car and completed the track after just with three training epochs. My model consists of five convolution neural networks (`model.py` could be found [here](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/model.py#L72-L84)).

The model in `model.py` file includes RELU layers to introduce nonlinearity (code line [74-78](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/model.py#L74-L78)), and the data is normalized in the model using a Keras lambda layer (code line [88](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/model.py#L88)).

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. The below table summarize the nVidia model that I used in my project:

| Layer(type)                		|      Output Shape 	|Param #	|  	Connected to	  | 
|:---------------------------------:|:---------------------:|:---------:|:-------------------:|
|lambda_1 (Lambda)               	|(None, 160, 320, 3)	|0      	|lambda_input_2[0][0] |
|cropping2d_1 (Cropping2D)        	|(None, 90, 320, 3)    	|0      	|lambda_1[0][0]       |
|convolution2d_1 (Convolution2D)  	|(None, 43, 158, 24)   	|1824   	|cropping2d_1[0][0]   |
|convolution2d_2 (Convolution2D)  	|(None, 20, 77, 36)    	|21636  	|convolution2d_1[0][0]|
|convolution2d_3 (Convolution2D)  	|(None, 8, 37, 48)     	|43248  	|convolution2d_2[0][0]|
|convolution2d_4 (Convolution2D)  	|(None, 6, 35, 64)     	|27712  	|convolution2d_3[0][0]|
|convolution2d_5 (Convolution2D)  	|(None, 4, 33, 64)     	|36928  	|convolution2d_4[0][0]|
|flatten_1 (Flatten)              	|(None, 8448)          	|0      	|convolution2d_5[0][0]|
|dense_1 (Dense)                  	|(None, 100)           	|844900 	|flatten_1[0][0]      |
|dense_2 (Dense)                  	|(None, 50)            	|5050   	|dense_1[0][0]        |
|dense_3 (Dense)                  	|(None, 10)            	|510    	|dense_2[0][0]        |
|dense_4 (Dense)                  	|(None, 1)             	|11     	|dense_3[0][0]        |
```sh
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
```

#### 2. Attempts to reduce overfitting in the model

To reduce overfitting in the model I kept the number of the training epochs low. I just used 3 epochs to train the model. I shuffled the sample data and splitted off 20% of the data to use for a validation set and 80% for the train set (`model.py` [line 98](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/model.py#L98)). I used mean square error loss function to minimize the error between steering measurement that the network predicts and the ground truth steering measurement (`model.py` [line 108](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/model.py#L108)).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`model.py` [line 108](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/model.py#L108)).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the [data](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/tree/master/data) that I got from the [simulator](https://github.com/aliasaswad/Self-Driving-Car-Simulator) for the two tracks to train the model. For each track the simulator provides two type of information. First, [`driving_log.csv`](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/data/driving_log.csv) that contains the information about the steering angle. Second, three different view camera images ([`IMG`](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/tree/master/data/IMG)): center, left and right. Each one of these image was used to train the model.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to normalize the input images by adding a [lambda](https://keras.io/api/layers/core_layers/) layers can be used to create arbitrary functions that operate on each image as it passes through the layer. The lambda layer (`model.py` [line 88](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/model.py#L88)) is a convenient way to parallelize image normalization. The lambda layer will also ensure that the model will normalize input images when making predictions in `drive.py`.
The top portion of the image captures trees and hills and sky, and the bottom portion of the image captures the hood of the car. So, to make the model train faster, I cropped each image and focused on only the portion of the image that is useful for predicting a steering angle. I used [Cropping2D Layer](https://keras.io/api/layers/convolution_layers/) provided by keras for image cropping within the model.

Here is an example of an input image and its cropped version after passing through a Cropping2D layer:

|<img src="./readme_writeup/original_image_from_sim.jpg" width="450" height="300" align="center"/> <img src="./readme_writeup/cropped_image.jpg" width="400" height="200" align="center"/> 
|:--:| 
|*Original Image from Simulator ____________________Cropped Image after passing through a Cropping2D Layer*|
 

**My first** step was to use a convolution neural network model similar to the [LeNet](http://yann.lecun.com/exdb/lenet/) model. In order to gauge how well the model was working, I splitted my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. I thought this model might be appropriate solution but the model was unable to drive the car smoothly on the road and specially when I increased the speed (`drive.py` [line 47](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/drive.py#L47)).  

**The second** step was to use a more powerfull network architecture model. I used [nVidia](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) model. I modified the model by adding extra new layer at the end to have a single output as it was required. The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I augmented the data by adding the same image flipped with a negative angle (`model.py` [lines 68](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/model.py#L68)). I also used the left and right camera images as a correction factor on the steering angle to help the car stay in the center of the road (`model.py` [line 14-25](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/model.py#L14-L25)). 


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (`model.py` [lines 72-84](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/model.py#L72-L84)) consisted of normalization layer followed by a convolution neural network.
I used [Keras](https://keras.io/) to visualize my network. Keras provides a function to create a plot of the network neural network graph that can make more complex models easier to understand. The [`plot_model()`](https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model) function in Keras will create a plot of your network.
The code I used to visualize the model could be found in the [`visualizer.ipynb`](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/visualizer.ipynb) file.

```python
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
```

Here is a visualization of the final model architecture:


|<img src="./readme_writeup/model_plot.png" width="500" height="1100" align="center"/> 
|:--:| 
|*Final Model Visualization*|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I collected more [data](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/tree/master/data) to train the model. First, recorded one lap driving forward on track. I then recorded another lap driving backward. Then I repeated this process on track two in order to get more data points. To augment the data set, I also flipped images and angles thinking that this would help to add more data to train the model. I finally shuffled the data set and put 80% of the data into a validation set. 

I used this training data for training the model with three epochs ([`model.py`](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/model.py#L109)). The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The following plot shows the error loss for the trained model:

|<img src="./readme_writeup/loss_plot.png" width="600" height="400" align="center"/> 
|:--:| 
|*Trained Model Loss Error*|

With this trained model architecture, the car was autonomously driven on the road all the time on both the [first](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/tree/master/output_video/output_video_1track.mp4) and [second](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/tree/master/output_video/output_video_2track.mp4) track.






