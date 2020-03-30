from keras.layers.pooling import MaxPooling2D
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import sklearn
import csv
import cv2
import matplotlib.image as mpimg
import numpy as np

datadir = 'data/data/'
csvfile = datadir + 'driving_log.csv'

lines = []
with open(csvfile) as input:
    reader = csv.reader(input)
    for line in reader:
        lines.append(line)

lines = lines[1:]

# 80% of the data will be used for training.
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)

    # Loop forever
    while 1:
        sklearn.utils.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            # Output batches (which will be of size 4*batch_size)
            # are allocated and filled on demand.
            images = []
            angles = []
            for batch_sample in batch_samples:

                filename_center = batch_sample[0].split('/')[-1]
                filename_left = batch_sample[1].split('/')[-1]
                filename_right = batch_sample[2].split('/')[-1]

                path_center = 'data/data/IMG/' + filename_center
                path_left = 'data/data/IMG/' + filename_left
                path_right = 'data/data/IMG/' + filename_right

                # A quick print statement of the top left pixel of an image being passed to
                # model.predict() in drive.py indicates that drive.py passes images to
                # to model.predict in RGB form, so we should train on data in RGB form.
                image_center = mpimg.imread(path_center)
                image_left = mpimg.imread(path_left)
                image_right = mpimg.imread(path_right)

                image_flipped = np.copy(np.fliplr(image_center))

                images.append(image_center)
                images.append(image_left)
                images.append(image_right)
                images.append(image_flipped)
                correction = 0.06  # trial and error
                angle_center = float(batch_sample[3])
                angle_left = angle_center + correction
                angle_right = angle_center - correction
                angle_flipped = -angle_center

                angles.append(angle_center)
                angles.append(angle_left)
                angles.append(angle_right)
                angles.append(angle_flipped)

            # Return a training batch of size 4*batch_size to model.fit_generator
            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)


# print(len(train_samples))
# print(len(validation_samples))

# Define generators for training and validation data
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


model = Sequential()
# Crop the hood of the car and the higher parts of the images
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
# Normalize the data.
model.add(Lambda(lambda x: x/255. - 0.5))
# Nvidia Network
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='elu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='elu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='elu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
#model.add( Dropout(0.5) )
model.add(Dense(10))
#model.add( Dropout(0.5) )
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

train_steps = np.ceil(len(train_samples)/32).astype(np.int32)
validation_steps = np.ceil(len(validation_samples)/32).astype(np.int32)

model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs=5, verbose=1, callbacks=None, validation_data=validation_generator,
                    validation_steps=validation_steps, class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0)

model.save('model.h5')
