# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 118-126) 

The model includes RELU layers to introduce nonlinearity (code line 118), and the data is normalized in the model using a Keras lambda layer (code line 116). 


#### 2. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 3. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to look for industry giants.

My first step was to use a convolution neural network model similar to NVIDIA's autonomous vehicle training.  I thought this model might be appropriate because NVIDIA is a reputed firm with proven track record.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

The final step was to run the simulator to see how well the car was driving around track one.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 103-138) consisted of a convolution neural network with the following layers and layer sizes:
5 convolutional layers each followed by RELU layer.
5 Dense layers with RELU follow the convolutional layers.
Layer 1: Input 66x200x3, Strides 2x2, Kernel 5x5, Padding valid, Activation Relu, Output 31x98x24 
Layer 2: Input 31x98x24, Strides 2x2, Kernel 5x5, Padding valid, Activation Relu,  Output 14x47x36 
Layer 3: Input 14x47x36, Strides 2x2, Kernel 5x5, Padding valid, Activation Relu,  Output 5x22x48 
Layer 4: Input 5x22x48, Strides 1x1, Kernel 3x3, Padding valid, Activation Relu,  Output 3x20x64 
Layer 5: Input 3x20x64, Strides 1x1, Kernel 3x3, Padding valid, Activation Relu,  Output 1x18x64

Flatten: 1x18x64 = 1152

Layer 6: Input 1152, Activation Relu,  Output 1164 
Layer 7: Input 1164, Activation Relu,  Output 100 
Layer 8: Input 100, Activation Relu,  Output 10

Layer 9: Input 100, Activation Tanh,  Output 1 


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. I went the other way on the track.


I then recorded the vehicle recovering from the left side and right sides of the road back to center.

Then I went around on track two in order to get more data points.

To augment the data sat, I also flipped images and angles.

After the collection process, I preprocessed this data by using normalization and cropping.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.
