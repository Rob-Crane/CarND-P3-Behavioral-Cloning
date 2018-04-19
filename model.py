import os
import csv
from random import randint
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Lambda, Cropping2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adadelta
from scipy import misc
from time import strftime
from math import ceil
import bc_config

csv_file = bc_config.DATA_DIR + '/' + bc_config.CSV_FILE
img_dir = bc_config.DATA_DIR + '/' + bc_config.IMAGE_DIRNAME + '/'

num_images = sum(3 for line in open(csv_file))

def get_fname(path):
    split = path.split('/')
    if(len(split) > 1):
        return split[-1]
    else:
        return path.split('\\')[-1]

def get_batch(batch_size, h_flip):

    # provide iterator over all images in CSV file
    def _image_gen():
        with open(csv_file) as f:
            reader = csv.reader(f)
            for line in reader:
                yield (misc.imread(img_dir + get_fname(line[0])), 
                        float(line[3]))
                yield (misc.imread(img_dir + get_fname(line[1])), 
                        float(line[3]) + bc_config.CAM_OFFSET_FACTOR)
                yield (misc.imread(img_dir + get_fname(line[2])), 
                        float(line[3]) - bc_config.CAM_OFFSET_FACTOR)

    image_source = _image_gen()

    while True:
        images = np.empty(shape=(batch_size,) + bc_config.IMG_SHAPE)
        labels = np.empty(shape=(batch_size,))
        try:
            for i in range(batch_size):
                (image, label) = next(image_source)
                if h_flip and randint(0,1):
                    image = np.flip(image, 1)
                    label = -label
                images[i] = image
                labels[i] = label
        except StopIteration:
            images = images[:i]
            labels = labels[:i]
            if len(labels) == 0: # if num_examples % batch_size is 0
                continue
            yield images, labels
            image_source = _image_gen() # reset generator to loop back over
            continue
        yield images, labels


def normalize(x):
    from keras.backend import mean, std
    return (x - mean(x)) / std(x)


model = Sequential()
model.add(Lambda(normalize, input_shape=bc_config.IMG_SHAPE))
top_crop = int(bc_config.TOP_CROP * bc_config.IMG_SHAPE[0])
model.add(Cropping2D(((top_crop, 0), (0,0))))
model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
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

steps_per_epoch = steps_per_epoch=ceil(num_images / bc_config.BATCH_SIZE)
validation_steps = int(bc_config.VALIDATION_SPLIT * steps_per_epoch)
train_steps = steps_per_epoch - validation_steps

train_source = get_batch(bc_config.BATCH_SIZE, True)
test_source = get_batch(bc_config.BATCH_SIZE, True)

print('exp total:', str(steps_per_epoch))
print('train_steps: ', str(train_steps))
print('validation_steps: ', str(validation_steps))

optimizer = Adadelta()
model.compile(optimizer=optimizer,
                  loss='mse')

history = model.fit_generator(
        train_source, 
        train_steps, 
        epochs = bc_config.EPOCHS, 
        validation_data = test_source, 
        validation_steps = validation_steps)

timestamp = strftime('%d-%b-%y_%H:%M:%S')

try:
    os.mkdir(bc_config.MODEL_DIR)
except FileExistsError:
    pass

model.save(bc_config.MODEL_DIR + '/' + timestamp + '.h5')

if bc_config.PLOT_LOSSES:
    import matplotlib.pyplot as plt

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Loss', 'Val. Loss'], loc='upper left')
    try:
        os.mkdir(bc_config.PLOT_DIR)
    except FileExistsError:
        pass
    plt.savefig(bc_config.PLOT_DIR + '/' + timestamp + '.png')
print('done')
