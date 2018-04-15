import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from scipy import misc
from datetime import datetime
from math import ceil
import matplotlib.pyplot as plt

CAM_OFFSET_FACTOR = 0.2
CSV_PATH = 'data/driving_log.csv'
DIR_PATH = '/home/robert/projects/bc_proj/data/IMG/'
IMG_SHAPE = (160, 320, 3)

num_images = sum(3 for line in open(CSV_PATH))

def get_batch(batch_size, randomize):
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

    # Image pre-processing pipeline

    trim_image = lambda image : image # TODO trim portion of sky
    image_adjuster = ImageDataGenerator(
            samplewise_center = True,
            samplewise_std_normalization = True,
            vertical_flip = True,
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
        try:
            for i in range(batch_size):
                (image, label) = next(image_source)
                images[i] = image
                labels[i] = label
        except StopIteration:
            images = images[:i]
            labels = labels[:i]
            if len(labels) == 0: # if num_examples % batch_size is 0
                continue
            yield preprocess(images), labels
            image_source = _image_gen() # reset generator to loop back over
            continue
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



BATCH_SIZE = 16
VALIDATION_SPLIT = 0.1
EPOCHS = 1
LEARN_RATES = [0.00001, 0.0001, 0.001, 0.01]

steps_per_epoch = steps_per_epoch=ceil(num_images / BATCH_SIZE)
validation_steps = int(VALIDATION_SPLIT * steps_per_epoch)
train_steps = steps_per_epoch - validation_steps

train_source = get_batch(BATCH_SIZE, True)
test_source = get_batch(BATCH_SIZE, True)

print('exp total:', str(steps_per_epoch))
print('train_steps: ', str(train_steps))
print('validation_steps: ', str(validation_steps))

val_losses = []

for lr in LEARN_RATES:
    optimizer = RMSprop(lr = lr)
    model.compile(optimizer=optimizer,
                      loss='mse')

    history = model.fit_generator(
            train_source, 
            train_steps, 
            epochs = EPOCHS, 
            validation_data = test_source, 
            validation_steps = validation_steps)

    timestamp = str(datetime.now())
    model.save('models/' + timestamp + '.h5')
    val_losses.append(history.history['val_loss'][0])

plt.plot(np.log10(LEARN_RATES), val_losses)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Learn Rate')
timestamp = str(datetime.now())
plt.savefig('figs/' + timestamp + '.png')
