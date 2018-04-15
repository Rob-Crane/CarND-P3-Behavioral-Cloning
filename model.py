import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from scipy import misc
from datetime import datetime
from math import ceil

CAM_OFFSET_FACTOR = 0.2
CSV_PATH = 'data/driving_log.csv'
DIR_PATH = '/home/robert/projects/bc_proj/data/IMG/'
IMG_SHAPE = (160, 320, 3)

num_images = sum(3 for line in open(CSV_PATH))

def _image_gen():
    with open(CSV_PATH) as f:
        reader = csv.reader(f)
        get_fname = lambda path : path.split('/')[-1]
        for line in reader:
            yield (misc.imread(DIR_PATH + get_fname(line[0])), 
                    float(line[3]))
            yield (misc.imread(DIR_PATH + get_fname(line[1])), 
                    float(line[3]) + CAM_OFFSET_FACTOR)
            yield (misc.imread(DIR_PATH + get_fname(line[2])), 
                    float(line[3]) - CAM_OFFSET_FACTOR)

image_source = _image_gen()

def get_batch(batch_size, randomize):

    trim_image = lambda image : image # TODO trim portion of sky
    image_adjuster = ImageDataGenerator(
            samplewise_center = True,
            samplewise_std_normalization = True,
            horizontal_flip = True,
            preprocessing_function = trim_image)
    if randomize:
        def preprocess(img_batch):
            normalized = image_adjuster.standardize(img_batch)
            return image_adjuster.random_transform(normalized)
    else:
        def preprocess(img_batch):
            normalized = image_adjuster.standardize(img_batch)
            return normalized

    while True:
        images = np.empty(shape=(batch_size,) + IMG_SHAPE)
        labels = np.empty(shape=(batch_size,))
        for i in range(batch_size):
            try:
                (image, label) = next(image_source)
                images[i] = image
                labels[i] = label
            except StopIteration:
                # trim off empty part of the batch and yield smaller last batch
                images = images[:i]
                labels = labels[:i]
                yield images, labels
                raise # and now we're done
        yield preprocess(images), labels

model = Sequential()

model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu', input_shape=IMG_SHAPE))
model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop',
                  loss='mse')

BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
EPOCHS = 4

steps_per_epoch = steps_per_epoch=ceil(num_images / BATCH_SIZE)
validation_steps = int(VALIDATION_SPLIT * steps_per_epoch)
train_steps = steps_per_epoch - validation_steps

train_source = get_batch(BATCH_SIZE, True)
test_source = get_batch(BATCH_SIZE, False)
model.fit_generator(
        train_source, 
        train_steps, 
        epochs = EPOCHS, 
        validation_data = test_source, 
        validation_steps = validation_steps)

timestamp = str(datetime.now())
model.save(timestamp + '.h5')
