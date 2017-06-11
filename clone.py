import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Cropping2D, Dropout, Convolution2D, Activation, Lambda
from keras.preprocessing.image import ImageDataGenerator

# IMPORT AND PREPROCESSING

lines = []
with open ('train_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'train_data/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.resize(image, None, fx=0.5, fy=0.5)
    measurement = float(line[3])
    images.append(image)
    measurements.append(measurement)

# reading validation data -- this could be refactored to avoid code repetition

lines = []
with open ('val_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

val_images = []
val_measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'val_data/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.resize(image, None, fx=0.5, fy=0.5)
    measurement = float(line[3])
    images.append(image)
    measurements.append(measurement)

# GENERATORS

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    horizontal_flip=False)

valgen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    horizontal_flip=False)

data_dir = 'train_data'

X_train = np.array(images)
y_train = np.array(measurements)

val_dir = 'val_data'

X_val = np.array(val_images)
y_val = np.array(val_measurements)

print(X_train.shape)
print(X_val.shape)

# MODEL ARCHITECTURE

model = Sequential()

batch_size = 20

# 'Weird lame' model architecture of Cipher from the discussions forum. Does surprisingly well!
# Outperforms VGG and NVIDIA from functional perspective, I wasn't able to find or come up with a better model for this task

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(80,160,3)))
model.add(Cropping2D( cropping=((25, 15), (0, 0)) ))
model.add(Convolution2D(16,1,1))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(16))
model.add(Dropout(0.7))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# MODEL TRAINING

# Why don't we use train/validation/test split? The model is 'fine-tuned' by repeating parts of the track where it misperformed.
# We would need to collect much more data to afford discarding data from training and putting it in train/validation/test

model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), samples_per_epoch=len(X_train) / batch_size, nb_epoch=25, 
        validation_data=valgen.flow(X_val, y_val, batch_size=batch_size))

model.save('model.h5')

