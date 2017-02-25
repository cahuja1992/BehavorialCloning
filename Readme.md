# Use Deep Learning to Clone Driving Behavior

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

This project is aimed to train the neural network that will output a steering angle to an autonomous vehicle.

A simulator has been used for data collectionand thenn the image data and steering angles are being used to train a neural network and then use this model to drive the car autonomously around the track.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


The project repository contains:
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)


## 1. Dataset Collection

The model was trained by driving a car using the Udactity Car simulator, and the following features were captured.
* Image taken from the center camera on the car 
* Left camera Image
* Right camera Image
* Throttle
* Speed
* Brake
* Steering angle


The model was trained using the center image as input (X).
the steering angle as the variable to predict (y).

## 2. Model Design

### Pre-processing of input data
Apart from  inputing the original images we ,prefered to input the processsed image instead. Though, we could very well input the unprocessed image, but 

* Read Image
* Convert to RGB
* Resize to (80,18,1)
* Gray Scale
* Normalize Image
* Flatten Array


	
    def read_image(image_path,dim):
        img = cv2.imread(data_dir+"/"+image_path.strip())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dim)
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)[:,:,0]
        norm_image = gray_image/255.
        flatten_image = norm_image.flatten().tolist()

        return norm_image.astype(np.float32)


### Training the model

Because of resource constraint, not much data points has been used while training the model. Neither we could train our model with more iterations.

While modelling with limited resources, we could train our model 70% accuracy only. Which might have been better , if we could use more data and the iterations.

#### Convolutions
Convolution neural networks has been proven to be amongst the best architectures while working with images and thus computer vision problems. Therefore, they become as obvious choice for this problem.
Thus, the following types of layers has been used in the overall architecture.

#### Activations and Dropout
* Dropouts has been added to prevent the network from overfitting.
* In order to indroduce non-linearalites activation function like ReLU (Rectified Linear Units has been introduced).

#### Fully connected layer
* In order to model the high level features a fully connected layer has been used after convolution layers.

#### Final layer
* I contains a single node because being a regression problem , we just want our output ranges from -1.0 to +1.0

## 3. Model architecture

The specifications of the model are below:
______________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    convolution2d_1 (Convolution2D)  (None, 78, 38, 16)    160         convolution2d_input_1[0][0]      
    ____________________________________________________________________________________________________
    activation_1 (Activation)        (None, 78, 38, 16)    0           convolution2d_1[0][0]            
    ____________________________________________________________________________________________________
    convolution2d_2 (Convolution2D)  (None, 76, 36, 8)     1160        activation_1[0][0]               
    ____________________________________________________________________________________________________
    activation_2 (Activation)        (None, 76, 36, 8)     0           convolution2d_2[0][0]            
    ____________________________________________________________________________________________________
    convolution2d_3 (Convolution2D)  (None, 74, 34, 4)     292         activation_2[0][0]               
    ____________________________________________________________________________________________________
    activation_3 (Activation)        (None, 74, 34, 4)     0           convolution2d_3[0][0]            
    ____________________________________________________________________________________________________
    convolution2d_4 (Convolution2D)  (None, 72, 32, 2)     74          activation_3[0][0]               
    ____________________________________________________________________________________________________
    activation_4 (Activation)        (None, 72, 32, 2)     0           convolution2d_4[0][0]            
    ____________________________________________________________________________________________________
    maxpooling2d_1 (MaxPooling2D)    (None, 36, 16, 2)     0           activation_4[0][0]               
    ____________________________________________________________________________________________________
    dropout_1 (Dropout)              (None, 36, 16, 2)     0           maxpooling2d_1[0][0]             
    ____________________________________________________________________________________________________
    flatten_1 (Flatten)              (None, 1152)          0           dropout_1[0][0]                  
    ____________________________________________________________________________________________________
    dense_1 (Dense)                  (None, 16)            18448       flatten_1[0][0]                  
    ____________________________________________________________________________________________________
    activation_5 (Activation)        (None, 16)            0           dense_1[0][0]                    
    ____________________________________________________________________________________________________
    dense_2 (Dense)                  (None, 16)            272         activation_5[0][0]               
    ____________________________________________________________________________________________________
    activation_6 (Activation)        (None, 16)            0           dense_2[0][0]                    
    ____________________________________________________________________________________________________
    dense_3 (Dense)                  (None, 16)            272         activation_6[0][0]               
    ____________________________________________________________________________________________________
    activation_7 (Activation)        (None, 16)            0           dense_3[0][0]                    
    ____________________________________________________________________________________________________
    dropout_2 (Dropout)              (None, 16)            0           activation_7[0][0]               
    ____________________________________________________________________________________________________
    dense_4 (Dense)                  (None, 1)             17          dropout_2[0][0]                  
    ====================================================================================================
    
    
---