# coding: utf-8
import json
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
import tensorflow as tf
import os
import sys
import pandas as pd
import cv2
from sklearn.cross_validation import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
import json
import os
import h5py


# Read logs of the data

data_dir = "data"
data_csv = "{}/driving_log.csv".format(data_dir)
seed = 7
np.random.seed(seed)
image_dim = (80,18)


# # Read Images and pre process it

# Read Image
# Convert to RGB
# Gray Scale
# Normalize Image
# Flatten Array
def read_image(image_path,dim):
    img = cv2.imread(data_dir+"/"+image_path.strip())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dim)
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)[:,:,0]
    norm_image = gray_image/255.
    flatten_image = norm_image.flatten().tolist()
    
    return norm_image.astype(np.float32)

def prepare_features(data_csv,dim,samples):
    features = ()
    labels = ()
    
    data = pd.read_csv(data_csv)
    sample_df = data.ix[:samples-1,:]
    sample_df['center'] = sample_df['center'].apply(lambda x :read_image(x,dim))
    sample_df['left'] = sample_df['left'].apply(lambda x :read_image(x,dim))
    sample_df['right'] = sample_df['right'].apply(lambda x :read_image(x,dim))
    
    for i,r in sample_df.iterrows():
        center = r['center']
        left = r['left']
        right = r['right']

        features+= (center,left,right)
        labels+= (r['steering'],r['steering'],r['steering'])
    
    assert (sample_df.shape[0]*3 ==len(features)), "Dimensions didn't match"
    assert (sample_df.shape[0]*3 ==len(labels)), "Dimensions didn't match"
    features = np.array(features).reshape(len(features), dim[0], dim[1], 1)
    labels = np.array(labels)
    
    input_shape = features.shape[1:]
    print(features.shape)
    print(labels.shape)
    print('Image Shape {0}x{1}x{2}'.format(input_shape[0],input_shape[1],input_shape[2]))
    
    return (features,labels,input_shape)




features,labels,input_shape = prepare_features(data_csv,image_dim,100)
print(input_shape)

# # CNN Training
# Convolutions followed by Max Pooling , followed by MaxPooling, followed by fully connected layers
def create_model(dropout_rate=0.0):
    
    nb_filters1 = 16
    nb_filters2 = 8
    nb_filters3 = 4
    nb_filters4 = 2

    pool_size = (2, 2)
    kernel_size = (3, 3)

    model = Sequential()
    model.add(Convolution2D(nb_filters1, kernel_size[0], kernel_size[1],border_mode='valid',input_shape=input_shape))

    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1]))

    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters3, kernel_size[0], kernel_size[1]))

    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters4, kernel_size[0], kernel_size[1]))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    # Compile model
    model.compile(loss='mean_squared_error',optimizer=Adam(),metrics=['accuracy'])
    return model


def train_and_evaluate_model(model, features, labels):
    # Test, Train, Valid Split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.10,random_state=832289)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train,test_size=0.25,random_state=832289)
    # Fit the model
    history = model.fit(X_train, y_train,batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_valid, y_valid))
    # evaluate the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Test Accuracy : %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# # Find Best Dropout rate using GridCV Search

model = KerasClassifier(build_fn=create_model, nb_epoch=10, batch_size=5, verbose=1)
dropout_rate = [0.0, 0.2, 0.5]
param_grid = dict(dropout_rate=dropout_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_result = grid.fit(features, labels)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))


# # Training the model using the dropouts , discovered by GridCV Search

best_params = grid_result.best_params_
dropout_keep_prob = best_params['dropout_rate']


batch_size = 64
nb_epoch = 150
model = create_model(dropout_keep_prob)
model.summary()
train_and_evaluate_model(model,features,labels)


# # Save Model

model_file='model.json'
model_weights='model.h5'

def save_model(model_file,model_weights):
    json_string = model.to_json()
    with open(model_file, 'w') as outfile:
        json.dump(json_string, outfile)
        model.save_weights(model_weights)
        print("Completed... Model Saved")

if model_file in os.listdir():
    print("The file already exists")
    print("Want to overwite? y or n")
    is_overwrite = input()
    if is_overwrite.lower() == "y":
        save_model(model_file,model_weights)
    else:
        print("the model is not saved")
else:
    save_model(model_file,model_weights)