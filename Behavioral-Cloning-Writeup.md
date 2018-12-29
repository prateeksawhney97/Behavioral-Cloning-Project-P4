# **Behavioral Cloning** 

## Writeup

### The goals / steps of this project are the following:

##### * Use the simulator to collect data of good driving behavior
##### * Build, a convolution neural network in Keras that predicts steering angles from images
##### * Train and validate the model with a training and validation set
##### * Test that the model successfully drives around track one without leaving the road
##### * Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
##### * model.py containing the script to create and train the model. It contains the main code necessary for the project to train.
##### * drive.py for driving the car in autonomous mode. It is used along with model.h5 to drive the car in the simulator autonomously.
##### * model.h5 containing a trained convolution neural network. 
##### * writeup_report.md or writeup_report.pdf summarizing the results as Behavioral-Cloning-Writeup.md. 
##### * an output video - 'run_video.mp4' of the car driving successfully in the simulator.

#### 2. Submission includes functional code

First of all, the model is trained to generate the model.h5 file with the help of following command. model.py file contains the code to train the model.
```sh
python model.py
```
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing: 
```sh
python drive.py model.h5
```
After the car successfully steers through the track, the video of the driving behavior can be formed by producing various frames and saving that frames in the run_video folder, by executing the following command. The fourth argument, run_video, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.
```sh
python drive.py model.h5 run_video
```
After all the frames of the car driving in the simulator are saved in the run_video folder, the video can be made by combining all the frames with the use of following command. It creates a video based on images found in the run_video directory. The name of the video will be the name of the directory followed by '.mp4'.
```sh
python video.py run_video
```
Optionally, we can specify the FPS (frames per second) of the video. The default is 60 fps.
```sh
python video.py run_video --fps 48
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Further, I have also included the code to plot the mean squared error loss for the training set and the validation set per epoch. It helps us to get a better idea of the scenario and gives us a lot of information on the various parameters to tune to get a much better result. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network which is implemented with the help of keras in a much easier way. The model is like the NVIDIA model, and contains five Convolutional layers and four Dense layers. The model also contains a Dropout layer, a Flatten layer and one Cropping2D layer. The data is normalized in the model using a Keras lambda layer.

The model architecture is depicted below with the help of table:


| Layer         		|     Output Shape	        					| Param |
|:---------------------:|:---------------------------------------------:|:---------------------------------------------:| 
| Cropping2D Layer         		| (None, 65, 320, 3)   							|   0   |
| Lambda Layer         		| (None, 65, 320, 3 )   							|  0  |
| Convolution Layer 1   	| (None, 31, 158, 24) 	|  1824 |
| Convolution Layer 2	    | (None, 14, 77, 36)    									|  21636 |
| Convolution Layer 3		| (None, 5, 37, 48) |      43248 									|
| Convolution Layer 4				| (None, 3, 35, 64)      |  		27712							|
|	Convolution Layer 5					|		(None, 1, 33, 64)			|					36928		|
| Dropout Layer 1              |     (None, 1, 33, 64)   |   0    |
| Flatten Layer 1              |        (None, 2112) |  0    |
|	Dense Layer 1				|		(None, 100)		|		211300	|
| Dense Layer 2		| (None, 50)     |   	5050								|
|	Dense Layer 3				|		(None, 10)						|   510  |
| Dense Layer 4		| (None, 1)   |     				11					|
 



#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Architecture and Training Documentation


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


### Simulation

#### 1. Car able to navigate correctly on test data
