import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Cropping2D
from keras.preprocessing.image import ImageDataGenerator

lines = []
with open ('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    image = image / 255.0 - 0.5;
    print(min(np.array(image)))
    print(max(np.array(image)))
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

X_train = np.array(images)
y_train = np.array(measurements)

#X_train = X_train.astype('float32')
#datagen.fit(X_train)

print("NORM OK")

model = Sequential()

batch_size = 16

# Preprocessing
#model.add(Cropping2D( cropping((70,25), (1,1)), input_shape=(160,320,3) ))

#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3) )
        


model.add(Conv2D(24, 5, 5, activation='relu', input_shape=(160,320,3), border_mode='same'))


# VGG: Block 1
#model.add(Conv2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2'))
#model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

# Block 2
#model.add(Conv2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1'))
#model.add(Conv2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2'))
#model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

# Block 3
#model.add(Conv2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1'))
#model.add(Conv2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2'))
#model.add(Conv2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3'))
#model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

# Block 4
#model.add(Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1'))
#model.add(Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2'))
#model.add(Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3'))
#model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

# Block 5
#model.add(Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1'))
#model.add(Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2'))
#model.add(Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3'))
#model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

model.add(Flatten(name='flatten'))
model.add(Dense(16, activation='relu', name='fc1'))
model.add(Dense(8, activation='relu', name='fc2'))
model.add(Dense(1, name='predictions'))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, batch_size=batch_size, validation_split=0.2, shuffle=True, nb_epoch=7)
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), samples_per_epoch=len(X_train) / batch_size, nb_epoch=5)

model.save('model.h5')

