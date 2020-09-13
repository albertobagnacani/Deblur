import os
import pickle
import random
import shutil
from copy import deepcopy
from os import listdir
from os.path import isfile, join
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_core.python.keras import Input
from tensorflow_core.python.keras.models import Model
from tensorflow_core.python.keras.optimizer_v2.learning_rate_schedule import PolynomialDecay
from tensorflow_core.python.training.adam import AdamOptimizer

cifar_path = '../res/datasets/cifar-10/'
cifar_path_modified = cifar_path + 'modified/'
cifar_train_path = cifar_path_modified+'data_batch_unified'
cifar_test_path = cifar_path_modified+'test_batch'
cifar_blurred_train_path = cifar_path_modified+'data_batch_unified_blurred'
cifar_blurred_test_path = cifar_path_modified+'test_batch_blurred'
cifar_saved_paths = {'train': cifar_path+'saved/train/original/', 'train_b': cifar_path+'saved/train/blurred/',
                     'test': cifar_path+'saved/test/original/', 'test_b': cifar_path+'saved/test/blurred/'}
# cifar_val_path
# cifar_blurred_val_path

reds_path = '../res/datasets/REDS/'
reds_train_sharp = reds_path + 'train/train_sharp/'
reds_train_blur = reds_path + 'train/train_blur/'
reds_val_sharp = reds_path + 'val/val_sharp/'
reds_val_blur = reds_path + 'val/val_blur/'
# reds_test_blur = reds_path + 'test/test_sharp/'
reds_test_blur = reds_path + 'test/test_blur/'

min_sigma = 0
max_sigma = 3
channel = 1024
image_size = 32

seed = 42

rescale = 1./255
validation_split = 0.2
target_size = (256, 256)
input_shape = (256, 256, 3)
'''
# Depending on the backend, the channel can be the first or the last parameter of a tuple
if tf.keras.backend.image_data_format() == 'channels_first': 
    INPUT_SHAPE = (INPUT_SHAPE[-1], INPUT_SHAPE[0], INPUT_SHAPE[1])
# Input shape representing the replicated gray-level images (1 channel) on the 3 RGB channels for pre-trained networks:
# from (256, 256, 1) to (256, 256, 3)
INPUT_SHAPE3 = INPUT_SHAPE[0:2]+(3,)
if tf.keras.backend.image_data_format() == 'channels_first':
    INPUT_SHAPE3 = (INPUT_SHAPE3[-1], INPUT_SHAPE3[0], INPUT_SHAPE3[1])
'''
batch_size = 16
class_mode = None

steps_per_epoch = 2000
epochs = 50
validation_steps = 800

'''
# Number of images in each folder
image_count = {"train": len(train_generator.filenames), "val": len(val_generator.filenames), 
               "test": len(test_generator.filenames)}
# Number of steps per epoch
steps_per_epoch = {"train": np.ceil(image_count["train"]/BATCH_SIZE), "val": np.ceil(image_count["val"]/BATCH_SIZE), 
                   "test": np.ceil(image_count["test"]/BATCH_SIZE)}'''


def unpickle(file):
    """
    :param file:
    :return:
    """
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def load_cifar(path):
    """
    The cifar-10 files contains a dictionary with the following elements:
    - data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
    The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
    The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the
    first row of the image.
    - labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in
    the array data.

    The dataset contains another file, called batches.meta. It too contains a Python dictionary object.
    It has the following entries:
    - label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array described
    above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.

    :param path:
    :return:
    """
    files = [f for f in listdir(path) if isfile(join(path, f)) and 'batch' in f]
    files.sort()

    train = []
    test = []
    for file in files:
        filename = os.path.join(path, file)
        batch = unpickle(filename)

        if 'data_batch' in filename:
            train.append(batch)
        elif 'test_batch' in filename:
            test.append(batch)

    return {'train': train, 'val': [], 'test': test}


def blur_cifar(ds):  # TODO5 can be optimized (e.g. multiprocessing)
    """
    Applies gaussian blurring with random stdev between 0 and 3 (included).

    The saved dataset was created with seed = 42.

    :param ds:
    :return:
    """
    print('Blurring')
    result = {key: [] for key in ds}

    for key in ds:
        for entry in ds[key]:
            tmp = []

            for image in entry[b'data']:
                blurred = []  # 0 = red, 1 = green, 2 = blue
                sigma = random.randint(min_sigma, max_sigma)

                for i in range(3):
                    img_c = np.reshape(image[channel*i:channel*(i+1)], (image_size, image_size))

                    blurred_c = np.reshape(gaussian_filter(img_c, sigma), (channel,))
                    blurred.extend(blurred_c)

                tmp.append(blurred)

            new_entry = deepcopy(entry)  # deepcopy is heavy
            new_entry[b'data'] = np.array(tmp)
            result[key].append(new_entry)

    return result


def reshape_cifar(ds):  # TODO5 can be optimized (e.g. multiprocessing)
    """

    :param ds:
    :return:
    """
    print('Reshaping')
    result = []

    for entry in ds:
        for image in entry[b'data']:
            img = []

            for i in range(3):
                img_c = np.reshape(image[channel * i:channel * (i + 1)], (image_size, image_size))

                img.append(img_c)

            img_m = np.array(img).swapaxes(0, 1).swapaxes(1, 2)  # (3, 32, 32) -> (32, 32, 3)
            result.append(img_m)

    return np.array(result)


def save_cifar(ds, path):
    """

    :param ds:
    :param path:
    :return:
    """
    count = 0

    for image in ds:
        cv2.imwrite(path+str(count)+'.png', image)
        count += 1


def reds_merge(input_path):
    """

    :param input_path:
    :return:
    """
    print('Merging {}'.format(input_path))

    root, dirs, files = next(os.walk(input_path))
    dirs.sort()

    count = 0
    for dir_ in dirs:
        root_n, dirs_n, files_n = next(os.walk(os.path.join(root, dir_)))
        files_n.sort()

        for file in files_n:
            filename = os.path.join(root_n, file)
            # or move/copy to a 'merged' folder
            # shutil.move(filename, os.path.join(input_path, str(count)+".png"))
            shutil.copyfile(filename, os.path.join(input_path, str(count)+".png"))

            count += 1

        # shutil.rmtree(root_n)


# TODO1 should do directly on reds_merge and save_cifar
def keras_folder(paths):
    """

    :param paths:
    :return:
    """
    print('Moving')

    res = {key: '' for key in paths}

    for key in paths:
        p = paths[key]
        new_p = p + 'folder/'

        Path(new_p).mkdir(parents=True, exist_ok=True)

        files = [f for f in listdir(p) if isfile(join(p, f))]
        for f in files:
            shutil.move(p + f, new_p)

        res[key] = new_p

    return res


'''
def avg_metric(original_path, test_path):
    sum_psnr = 0
    sum_mse = 0
    sum_ssim = 0

    files_orig = [f for f in listdir(original_path) if isfile(join(original_path, f))]
    files_deb = [f for f in listdir(test_path) if isfile(join(test_path, f))]

    count = 0
    for orig, deb in zip(files_orig, files_deb):
        orig_fn = join(original_path, orig)
        deb_fn = join(test_path, deb)
        orig_img = cv2.imread(orig_fn)
        deb_img = cv2.imread(deb_fn)

        sum_psnr += peak_signal_noise_ratio(orig_img, deb_img)
        sum_mse += mean_squared_error(orig_img, deb_img)
        sum_ssim += structural_similarity(orig_img, deb_img, multichannel=True)

        count += 1
        print('Analyzed: {}/{}'.format(count, len(files_orig)))

    avg_psnr = sum_psnr/len(files_orig)
    avg_mse = sum_mse/len(files_orig)
    avg_ssim = sum_ssim/len(files_orig)

    return avg_mse, avg_psnr, avg_ssim
'''

'''
def load_save(name, model):
    callbacks = []
    reduce_learning_rate = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, cooldown=2, min_lr=1e-5, verbose=1)
    callbacks.append(reduce_learning_rate)

    es = EarlyStopping(patience=5)
    # callbacks.append(es)

    if SAVE_BEST:
        filepath = MODEL_PARAM_DIR + "model_" + NAME + "-{epoch:02d}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks.append(checkpoint)

    if LOAD:
        if LOAD_MODEL:
            model = load_model(MODEL_PARAM_DIR + "model_" + name + '.h5')
        else:
            model.load_weights(MODEL_PARAM_DIR + "weights_" + name + '.h5')

        model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    else:
        model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
        history = model.fit_generator(train_generator, validation_data=val_generator, epochs=EPOCHS,
                                      steps_per_epoch=steps_per_epoch["train"], validation_steps=steps_per_epoch["val"],
                                      callbacks=callbacks)

        plot_learning(name, history)

        model.save(MODEL_PARAM_DIR + "model_" + name + '.h5')
        model.save_weights(MODEL_PARAM_DIR + "weights_" + name + '.h5')

    return model
'''


random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

if not os.path.exists(cifar_train_path):
    dataset = load_cifar(cifar_path)
    ds_blurred = blur_cifar(dataset)

    for k in ['train', 'test']:
        reshaped_ds = reshape_cifar(dataset[k])
        reshaped_ds_b = reshape_cifar(ds_blurred[k])

        path_c = cifar_train_path if k == 'train' else cifar_test_path
        path_c_b = cifar_blurred_train_path if k == 'train' else cifar_blurred_test_path

        Path(cifar_path_modified).mkdir(parents=True, exist_ok=True)

        with open(path_c, 'wb') as ff:
            pickle.dump(reshaped_ds, ff)
        with open(path_c_b, 'wb') as ff:
            pickle.dump(reshaped_ds_b, ff)

# Those are data (n_images, 32, 32, 3)
cifar = {'train': unpickle(cifar_train_path), 'test': unpickle(cifar_test_path),
         'train_b': unpickle(cifar_blurred_train_path), 'test_b': unpickle(cifar_blurred_test_path)}

'''
# Save CIFAR-10 images
for key in cifar_saved_paths:
    Path(cifar_saved_paths[key]).mkdir(parents=True, exist_ok=True)
    save_cifar(cifar[key], cifar_saved_paths[key])

# Look at images
for i in range(10):
    cv2.imwrite('/home/alby/Downloads/train_'+str(i)+'.png', cifar_train[i])
    cv2.imwrite('/home/alby/Downloads/test_'+str(i)+'.png', cifar_test[i])
    cv2.imwrite('/home/alby/Downloads/train_b_'+str(i)+'.png', cifar_blurred_train[i])
    cv2.imwrite('/home/alby/Downloads/test_b_'+str(i)+'.png', cifar_blurred_test[i])

cv2.imshow('train_0', cifar_train[0])
cv2.imshow('test_0', cifar_test[0])
cv2.imshow('train_b_0', cifar_blurred_train[0])
cv2.imshow('test_b_0', cifar_blurred_test[0])
cv2.waitKey(0)

cv2.imshow('train_0', cifar_train[1])
cv2.imshow('test_0', cifar_test[1])
cv2.imshow('train_b_0', cifar_blurred_train[1])
cv2.imshow('test_b_0', cifar_blurred_test[1])
cv2.waitKey(0)
'''

# Those are paths
reds = {'train_s': reds_train_sharp, 'train_b': reds_train_blur, 'val_s': reds_val_sharp, 'val_b': reds_val_blur,
        'test_b': reds_test_blur}

# reds = keras_folder(reds)

# for key in reds:
#     reds_merge(reds[k])

# train_datagen = ImageDataGenerator(rescale=rescale, validation_split=validation_split)
train_datagen = ImageDataGenerator(rescale=rescale)
test_datagen = ImageDataGenerator(rescale=rescale)

# Need a train_generator for both the sharp and blur images, which will be combined with a combined_generator
train_sharp_generator = train_datagen.flow_from_directory(
        reds['train_s'],
        target_size=target_size,
        batch_size=batch_size, class_mode=class_mode, seed=seed)
train_blur_generator = train_datagen.flow_from_directory(
        reds['train_b'],
        target_size=target_size,
        batch_size=batch_size, class_mode=class_mode, seed=seed)
# batch_size=batch_size, class_mode=class_mode, seed=seed, subset='training')


def combine_generators(sharp_generator, blur_generator):
    while True:
        sharp_batch = sharp_generator.next()
        blur_batch = blur_generator.next()

        yield [sharp_batch], [blur_batch]


train_generator = combine_generators(train_sharp_generator, train_blur_generator)

'''
validation_generator = train_datagen.flow_from_directory(
        '',
        target_size=target_size,
        batch_size=batch_size, class_mode=class_mode, seed=seed, subset='training')
test_generator = test_datagen.flow_from_directory(
        '',
        target_size=target_size,
        batch_size=batch_size, class_mode=class_mode, seed=seed, subset='training')
'''

'''
model = Sequential() # Sequential model

# Architecture of the CNN
model.add(Conv2D(32, KERNEL_SIZE, input_shape=INPUT_SHAPE3))
model.add(Activation(ACTIVATION_HIDDEN_LAYERS))
model.add(MaxPooling2D(pool_size=POOL_SIZE))

model.add(Conv2D(32, KERNEL_SIZE))
model.add(Activation(ACTIVATION_HIDDEN_LAYERS))
model.add(MaxPooling2D(pool_size=POOL_SIZE))

model.add(Conv2D(64, KERNEL_SIZE))
model.add(Activation(ACTIVATION_HIDDEN_LAYERS))
model.add(MaxPooling2D(pool_size=POOL_SIZE))

model.add(Flatten()) # Converts 3D feature maps to 1D feature vectors
model.add(Dense(64, activity_regularizer=REGULARIZER))
#model.add(Dense(64),
model.add(Activation(ACTIVATION_HIDDEN_LAYERS))
model.add(Dropout(DROPOUT))
model.add(Dense(OUTPUT_NEURONS))
model.add(Activation(ACTIVATION_OUTPUT_LAYER))

# Load or save the model
model = load_save(NAME, model)
'''

'''
data_size = train_sharp_generator.samples // batch_size
max_steps = int(epochs * data_size)

INITIAL_LR = 1e-4
LR = PolynomialDecay(initial_learning_rate=1e-4, decay_steps=max_steps, end_learning_rate=0.8, power=0.3)
OPTIMIZER = AdamOptimizer(learning_rate=LR)

LOSS = None
METRICS = []

input1 = Input(shape=input_shape, name='input1')
input2 = Input(shape=input_shape, name='input2')

output = deblur(input1, 'output')
model = Model(inputs=[input1, input2], outputs=output)

model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
'''

'''
model.compile()
history = model.fit_generator(multi_input_generator, epochs=epochs, steps_per_epoch=steps_per_epoch["train"],
callbacks=callbacks)
model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=validation_generator,
validation_steps=validation_steps)
print(model.summary())
val_score = model.evaluate_generator(val_generator, steps_per_epoch["val"])
test_score = model.evaluate_generator(test_generator, steps_per_epoch["test"])
predict = model.predict_generator(test_generator, steps=steps_per_epoch["test"])
'''
