import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, BatchNormalization
from scipy import misc

CAM_OFFSET_FACTOR = 0.2
CSV_PATH = 'data/driving_log.csv'
DIR_PATH = '/home/robert/projects/bc_proj/data/IMG/'
IMG_SHAPE = (160, 320, 3)

num_images = sum(3 for line in open(CSV_PATH))

def getTrainBatches(batch_size):
    def imageGen():
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

    image_dataset = imageGen()
    while True:
        images = np.empty(shape=(batch_size,) + IMG_SHAPE)
        labels = np.empty(shape=(batch_size,))
        for i in range(batch_size):
            try:
                (image, label) = next(image_dataset)
                images[i] = image
                labels[i] = label
            except StopIteration:
                # trim off empty part of the batch and yield last batch
                images = images[:i]
                labels = labels[:i]
                yield images, labels
                raise # and now we're done
        yield images, labels

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
                  loss='mse',
                  metrics=['accuracy'])

batch_size = 16
model.fit_generator(getTrainBatches(batch_size), steps_per_epoch=int(num_images / batch_size))
model.save('fitted.h5')

# from keras.utils import plot_model
# plot_model(model, to_file = 'plot.png')
