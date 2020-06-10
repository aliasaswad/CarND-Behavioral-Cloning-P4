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

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

My first work was using LeNet model, but the dodel didn't able to drive the car inside the street with three epochs (The model in  the `clone.py ` could be found [here](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/clone.py#L49-L60)). Then, I tried another model which is [nVidia Autonomous Car Group model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). The nVidia model was able to drive the car and completed the track after just three training epochs (The model in the `clone.py ` could be found [here](https://github.com/aliasaswad/CarND-Behavioral-Cloning-P4/blob/master/clone.py#L62-L75)).

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. The below table summarize the nVidia model that I used in my project:

| Layer(type)     		|     Output Shape	 |Param #	| 	   Connected to 				| 
|:---------------------:|:--------------------:|:--------:|:---------------:|
lambda_1 (Lambda)	|(None, 160, 320, 3)	|0	|lambda_input_2[0][0]|


| Input         		| 32x32x3 RGB image   						| 
| Convolution 1     	| 1x1 stride, same padding, outputs 28x28x16|
| RELU					|											|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 			|
| Convolution 2 	    | 1x1 stride, same padding, outputs 10x10x64|
| RELU					|											|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64   			|
| Fatten					|										|
| Fully-connected 1		| 1600 										|
| RELU					|											|
| Dropout					|										|
| Fully-connected 2		| 240										|
| RELU					|											|
| Dropout				|											|
| Fully connected 3		| 43										|



#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
