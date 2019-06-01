# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 22:56:45 2019

@author: 92567
"""


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32,(3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 4, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dd/training',
                                                 target_size = (64, 64),
                                                 batch_size = 64,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dd/testing',
                                            target_size = (64, 64),
                                            batch_size = 64,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch = 949,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 235)

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
image_path = 'C:/Users/92567/Desktop/Convolutional_Neural_Networks/Convolutional_Neural_Networks/data2/testing/sad/tammo_left_sad_open_2.jpg'
test_image = image.load_img(image_path, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
 
print(result)

image_path3 = 'C:/Users/92567/Desktop/Convolutional_Neural_Networks/Convolutional_Neural_Networks/data2/training/neutral2/boland_straight_neutral_sunglasses_2.jpg'
test_image = image.load_img(image_path, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
print(result)

image_path = r'C:\Users\92567\Desktop\Convolutional_Neural_Networks\Convolutional_Neural_Networks\single_prediction\saavik_left_angry_open_4.jpg'
test_image = image.load_img(image_path, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
print(result)
