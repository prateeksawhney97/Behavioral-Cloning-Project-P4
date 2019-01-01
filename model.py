import csv
import cv2
import os
import sklearn
import numpy as np
import ast
import pandas as pd

lines = []
image_path = 'data/IMG/'

#data_df = pd.read_csv(os.path.join(data, 'driving_log.csv'))

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def random_flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle
'''
def random_shadow(image):
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
'''
def random_brightness(image):
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def random_translate(image, steering_angle, range_x, range_y):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle
'''
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)
'''

def generator(lines, batch_size):
    num_samples = len(lines)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]

            images = []
            angles = []
            '''
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            '''

            for batch_sample in batch_samples:
                angle = float(batch_sample[3])
                image_choice = np.random.randint(3)
                if image_choice == 0:
                    name = image_path+batch_sample[1].split('/')[-1]
                    if abs(angle) > 1:
                        angle = angle + 0.25
                    else:
                        angle = angle + 0.18
                elif image_choice == 1:
                    name = image_path+batch_sample[0].split('/')[-1]
                elif image_choice == 2:
                    name = image_path+batch_sample[2].split('/')[-1]
                    if abs(angle) > 1:
                        angle = angle - 0.25
                    else:
                        angle = angle - 0.18
                image = cv2.imread(name)
                images.append(image)
                angles.append(angle)

            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                image = rgb2yuv(image)
                augmented_images.append(image)
                augmented_angles.append(angle)
                aug_image, aug_angle = random_flip(image, angle)
                image, angle = random_translate(aug_image, aug_angle, range_x=100, range_y=10)
                #image = random_shadow(image)
                image = random_brightness(image)
                augmented_images.append(image)
                augmented_angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, ELU, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.layers import BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#ch, row, col = 3, 80, 320  # Trimmed image format

'''
model.add(Lambda(lambda x: x/ 255.0 - 0.5, input_shape=(160, 320, 3)))
'''
'''
#model.add(Lambda(resize))

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3)))
model.add(Convolution2D(3, (1, 1), padding='same'))
model.add(ELU())
model.add(BatchNormalization())
model.add(Convolution2D(16, (5, 5), strides=(2, 2), padding='same'))
model.add(ELU())
model.add(BatchNormalization())
model.add(Convolution2D(32, (5, 5), strides=(2, 2), padding='same'))
model.add(ELU())
model.add(BatchNormalization())
model.add(Convolution2D(64, (3, 3), strides=(2, 2), padding='same'))
model.add(ELU())
model.add(BatchNormalization())
model.add(Convolution2D(128, (3, 3), strides=(2, 2), padding='same'))
model.add(ELU())
model.add(Flatten())
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(0.2))
model.add(ELU())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(ELU())
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(ELU())
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer=Adam(lr=1e-5))
'''

model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x/127.5) - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="elu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="elu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="elu"))
model.add(Convolution2D(64, 3, 3, activation="elu"))
model.add(Convolution2D(64, 3, 3, activation="elu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-4))
model.summary()

'''
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)
'''
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)

'''
history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
validation_data=validation_generator, validation_steps=len(validation_samples), epochs=10, verbose = 1)
'''
model.save('model.h5')

print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


