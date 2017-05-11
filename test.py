# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 17:53:23 2017

@author: dyson
"""
import csv
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Load lines in the CSV file: containing training data and labels
samples = []
with open('../recData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split samples
train_samples, valid_samples = train_test_split(samples, test_size = 0.2)

# Define the coroutine    
def generator(samples, batch_size = 100):
    num_samples = len(samples)
    while 1:
#        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset+batch_size]

            img_center_array = []
            img_left_array = []
            img_right_array = []
            car_images = []
        
            steering_center_array = []
            steering_left_array = []
            steering_right_array = []
            steering_angles = []
    #        images = []
    #        angles = []
            
            path = '../recData/IMG/'
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                steering_center_array.append(steering_center)
                correction = 0.07
                steering_left = steering_center + correction
                steering_left_array.append(steering_left)
                steering_right = steering_center - correction
                steering_right_array.append(steering_right)
                
                img_center = np.asarray(Image.open(path + batch_sample[0].split('\\')[-1]))
                img_center_array.append(img_center)
                img_left = np.asarray(Image.open(path + batch_sample[1].split('\\')[-1]))
                img_left_array.append(img_left)
                img_right = np.asarray(Image.open(path + batch_sample[2].split('\\')[-1]))
                img_right_array.append(img_right)
            
            car_images.extend(img_center_array)
            car_images.extend(img_left_array)
            car_images.extend(img_right_array)
            steering_angles.extend(steering_center_array)
            steering_angles.extend(steering_left_array)
            steering_angles.extend(steering_right_array)
            
            augmented_images, augmented_angles  = [], []
            for image, angle in zip(car_images, steering_angles):
                augmented_images.append(image)
                augmented_angles.append(float(angle))
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(float(angle) * -1.)
                
            x_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
                
            yield sklearn.utils.shuffle(x_train, y_train)
        
            
            
    
    
##lines = []
#img_center_array = []
#img_left_array = []
#img_right_array = []
#car_images = []
#
#steering_center_array = []
#steering_left_array = []
#steering_right_array = []
#steering_angles = []
#
#with open('../recData/driving_log.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    for element in reader:
#        # Obtain steering angle
#        steering_center = float(element[3])
#        steering_center_array.append(steering_center)
#        # Create adjusted steering measurements for the side camera images
#        correction = 0.05
#        steering_left = steering_center + correction
#        steering_left_array.append(steering_left)
#        steering_right = steering_center - correction
#        steering_right_array.append(steering_right)
#        
#        # Read in images from center, left and right
#        path = '../recData/IMG/'
#        img_center = np.asarray(Image.open(path + element[0].split("\\")[-1]))
#        img_center_array.append(img_center)
#        img_left = np.asarray(Image.open(path + element[1].split("\\")[-1]))
#        img_left_array.append(img_left)
#        img_right = np.asarray(Image.open(path + element[2].split("\\")[-1]))
#        img_right_array.append(img_right)
#        
#        
#    # Enrich the data set
#    car_images.extend(img_center_array)
#    car_images.extend(img_left_array)
#    car_images.extend(img_right_array)
#    steering_angles.extend(steering_center_array)
#    steering_angles.extend(steering_left_array)
#    steering_angles.extend(steering_right_array)
##        lines.append(line)
#
#augmented_images, augmented_measurements = [], []
#for image, measurement in zip(car_images, steering_angles):
#    augmented_images.append(image)
#    augmented_measurements.append(measurement)
#    augmented_images.append(cv2.flip(image, 1))
#    augmented_measurements.append(measurement * -1.0)
##    car_images.append(cv2.flip(image, 1))
##    steering_angles.append(measurement * -1.0)
#    
##plt.imshow(images[5])
#samples = np.array(augmented_images)
#labels = np.array(augmented_measurements)

# Use generated sample batches
train_gen = generator(train_samples)
valid_gen = generator(valid_samples)

model = Sequential()
#model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160, 320, 3)))
#model.add(Dropout(0.2))
model.add(Convolution2D(10,5,5, activation='relu'))
model.add(MaxPooling2D())
#model.add(Dropout(0.2))
model.add(Convolution2D(20,5,5, activation='relu'))
model.add(MaxPooling2D())
#model.add(Dropout(0.2))
model.add(Convolution2D(40,5,5, activation='relu'))
model.add(MaxPooling2D())
#model.add(Dropout(0.2))
model.add(Convolution2D(60,3,3, activation='relu'))
model.add(MaxPooling2D())
#model.add(Dropout(0.2))
model.add(Convolution2D(80,2,2, activation='relu'))
model.add(Flatten())

model.add(Dropout(0.2))
model.add(Dense(500))
model.add(Dropout(0.2))
model.add(Dense(300))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#obj = model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)
history_object = model.fit_generator(train_gen, samples_per_epoch= 
            len(train_samples), validation_data=valid_gen, 
            nb_val_samples=len(valid_samples), nb_epoch=100)

model.save('model.h5')

#plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()