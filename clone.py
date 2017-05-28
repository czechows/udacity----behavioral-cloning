import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Cropping2D, Dropout
from keras.preprocessing.image import ImageDataGenerator

lines = []
with open ('data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data1/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.resize(image, None, fx=0.5, fy=0.5)
    image = image / 255.0 - 0.5;

    measurement = 4*float(line[3])

    if not(measurement < 0.2):
        images.append(image)
        measurements.append(measurement)

for line in lines:
    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path = 'data1/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.resize(image, None, fx=0.5, fy=0.5)
    image = image / 255.0 - 0.5;
    images.append(image)
    measurement = float(line[3]) + 0.5
    measurements.append(measurement)

for line in lines:
    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = 'data1/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.resize(image, None, fx=0.5, fy=0.5)
    image = image / 255.0 - 0.5;
    images.append(image)
    measurement = float(line[3]) - 0.5
    measurements.append(measurement)

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    horizontal_flip=True)

data_dir = 'data1'


X_train = np.array(images)
y_train = np.array(measurements)

X_train = X_train.astype('float32')
#datagen.fit(X_train)

model = Sequential()

batch_size = 50



# NVIDIA
#model.add(Conv2D(24, 5, 5, activation='relu', name="conv1", input_shape=(80,160,3), border_mode='same'))
#model.add(Conv2D(36, 5, 5, activation='relu', name="conv2", border_mode='same'))
#model.add(Conv2D(48, 5, 5, activation='relu', name="conv3", border_mode='same'))
#model.add(Conv2D(64, 3, 3, activation='relu', name="conv4", border_mode='same'))
#model.add(Conv2D(64, 3, 3, activation='relu', name="conv5", border_mode='same'))

#model.add(Flatten(name='flatten'))
#model.add(Dense(100, name='fc1'))
#model.add(Dense(50, name='fc2'))
#model.add(Dense(10, name='fc3'))
#model.add(Dense(1, name='predictions'))

# Preprocessing

#model.add(Cropping2D( cropping((70,25), (1,1)), input_shape=(32,6,3) ))
#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3) )
        


#model.add(Conv2D(24, 5, 5, activation='relu', input_shape=(80,160,3), border_mode='same'))


# VGG: Block 1
model.add(Conv2D(64, 3, 3, activation='relu', input_shape=(80,160,3), border_mode='same', name='block1_conv1'))
model.add(Conv2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

# Block 2
model.add(Conv2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1'))
model.add(Conv2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

# Block 3
model.add(Conv2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1'))
model.add(Conv2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2'))
model.add(Conv2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

# Block 4
model.add(Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1'))
model.add(Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2'))
model.add(Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

# Block 5
model.add(Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1'))
model.add(Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2'))
model.add(Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

model.add(Flatten(name='flatten'))
model.add(Dropout(0.6))
model.add(Dense(1000, activation='relu', name='fc1'))
model.add(Dropout(0.6))
model.add(Dense(500, activation='relu', name='fc2'))
model.add(Dropout(0.6))
model.add(Dense(50, activation='relu', name='fc3'))
model.add(Dropout(0.6))
model.add(Dense(1, name='predictions'))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, batch_size=batch_size, validation_split=0.2, shuffle=True, nb_epoch=7)
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), samples_per_epoch=len(X_train) / batch_size, nb_epoch=5)

model.save('model.h5')

