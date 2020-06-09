# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


<!-- ![Alt Text](./readme_writeup/first_track.gif) -->

<img src="./readme_writeup/first_track.gif" width="700" height="400" align="center"/>


Overview
---
This repository contains the Behavioral Cloning Project.

In this project, I used deep neural networks and convolutional neural networks to clone driving behavior. I trained, validated and tested a model using Keras. The model output a steering angle to an autonomous vehicle.

I used a [simulator](https://github.com/aliasaswad/Self-Driving-Car-Simulator) provided by [Udacity](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) where I steered a car around two different tracks for data collection. Then, I used image data and steering angles to train a neural network and then used this model to drive the car autonomously around the tracks.

To meet specifications, I included the project requirement files in this repo. as below: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Writeup
---
In the writeup, I included a description of how I addressed each point.  Also, I included a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references. 


The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)


The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from [here](https://github.com/aliasaswad/Self-Driving-Car-Simulator). Udacity have also provided sample data that you can optionally use to help train model.


## Details About Files In This Directory

### `drive.py`

I used this command to load the trained model (that I trained and saved previously) and used the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Using `drive.py` requires to have saved trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```



**Note:** There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 track_1
```

The fourth argument, `track_1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2020-01-09 15:09:20 EST]  12KiB 2020_06_09_20_09_30_336.jpg
[2020-01-09 15:09:20 EST]  12KiB 2020_06_09_20_09_30_351.jpg
[2020-01-09 15:09:20 EST]  12KiB 2020_06_09_20_09_30_377.jpg
[2020-01-09 15:09:20 EST]  12KiB 2020_06_09_20_09_30_428.jpg
[2020-01-09 15:09:20 EST]  12KiB 2020_06_09_20_09_30_473.jpg
[2020-01-09 15:09:20 EST]  12KiB 2020_06_09_20_09_30_518.jpg
[2020-01-09 15:09:20 EST]  12KiB 2020_06_09_20_09_30_597.jpg
[2020-01-09 15:09:20 EST]  12KiB 2020_06_09_20_09_30_623.jpg
[2020-01-09 15:09:20 EST]  12KiB 2020_06_09_20_09_30_649.jpg
[2020-01-09 15:09:20 EST]  12KiB 2020_06_09_20_09_30_717.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

I created a video based on images found in the `track_1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `track_1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.


### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

#### How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

