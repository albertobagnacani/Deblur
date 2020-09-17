import datetime
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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, Activation, Conv2DTranspose, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

# policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

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
# validation_split = 0.2
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
batch_size = 8
class_mode = None

epochs = 50
# steps_per_epoch = 2000
# validation_steps = 800

'''
# Number of images in each folder
image_count = {"train": len(train_generator.filenames), "val": len(val_generator.filenames), 
               "test": len(test_generator.filenames)}
# Number of steps per epoch
steps_per_epoch = {"train": np.ceil(image_count["train"]/BATCH_SIZE), "val": np.ceil(image_count["val"]/BATCH_SIZE), 
                   "test": np.ceil(image_count["test"]/BATCH_SIZE)}
'''


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

train_datagen = ImageDataGenerator(rescale=rescale)  # validation_split=validation_split
test_datagen = ImageDataGenerator(rescale=rescale)

# Need a train_generator for both the sharp and blur images, which will be combined with a combined_generator
train_sharp_generator = train_datagen.flow_from_directory(
        reds['train_s'],
        target_size=target_size,
        batch_size=batch_size, class_mode=class_mode, seed=seed)
train_blur_generator = train_datagen.flow_from_directory(
        reds['train_b'],
        target_size=target_size,
        batch_size=batch_size, class_mode=class_mode, seed=seed)  # subset='training'

val_sharp_generator = train_datagen.flow_from_directory(
        reds['val_s'],
        target_size=target_size,
        batch_size=batch_size, class_mode=class_mode, seed=seed)
val_blur_generator = train_datagen.flow_from_directory(
        reds['val_b'],
        target_size=target_size,
        batch_size=batch_size, class_mode=class_mode, seed=seed)


def combine_generators(sharp_generator, blur_generator):
    while True:
        sharp_batch = sharp_generator.next()
        blur_batch = blur_generator.next()

        res = [sharp_batch, blur_batch]

        yield res


train_generator = combine_generators(train_sharp_generator, train_blur_generator)
validation_generator = combine_generators(val_sharp_generator, val_blur_generator)

'''
test_generator = test_datagen.flow_from_directory(
        '',
        target_size=target_size,
        batch_size=batch_size, class_mode=class_mode, seed=seed, subset='training')
'''


def res_net_block(x, filters, ksize):
    net = Conv2D(filters=filters, kernel_size=(ksize, ksize), padding='same', activation=Activation(tf.nn.relu))(x)
    net = Conv2D(filters=filters, kernel_size=(ksize, ksize), padding='same', activation=None)(net)

    return net


def generator(model, x_unwrap):
    channels = 3
    n_levels = 3
    starting_scale = 0.5

    inp_pred = model
    for i in range(n_levels):
        scale = starting_scale ** (n_levels - i - 1)
        hi = int(round((input_shape[0]*scale)))
        wi = int(round((input_shape[1]*scale)))

        inp_blur = tf.image.resize(model, [hi, wi])
        inp_pred = tf.image.resize(inp_pred, [hi, wi])
        inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')  # Use Keras layers?

        # Encoder
        conv1_1 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation=Activation(tf.nn.relu))(inp_all)
        conv1_2 = res_net_block(conv1_1, 32, 5)
        conv1_3 = res_net_block(conv1_2, 32, 5)
        conv1_4 = res_net_block(conv1_3, 32, 5)

        conv2_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=2, padding='same',
                         activation=Activation(tf.nn.relu))(conv1_4)
        conv2_2 = res_net_block(conv2_1, 64, 5)
        conv2_3 = res_net_block(conv2_2, 64, 5)
        conv2_4 = res_net_block(conv2_3, 64, 5)

        conv3_1 = Conv2D(filters=128, kernel_size=(5, 5), strides=2, padding='same',
                         activation=Activation(tf.nn.relu))(conv2_4)
        conv3_2 = res_net_block(conv3_1, 128, 5)
        conv3_3 = res_net_block(conv3_2, 128, 5)
        conv3_4 = res_net_block(conv3_3, 128, 5)

        deconv3_4 = conv3_4
        deconv3_3 = res_net_block(deconv3_4, 128, 5)
        deconv3_2 = res_net_block(deconv3_3, 128, 5)
        deconv3_1 = res_net_block(deconv3_2, 128, 5)

        # Decoder
        deconv2_4 = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=2, padding='same',
                                    activation=Activation(tf.nn.relu))(deconv3_1)
        # Skip connection
        cat2 = Add()([deconv2_4, conv2_4])  # merge([x, y], mode='sum')
        deconv2_3 = res_net_block(cat2, 64, 5)
        deconv2_2 = res_net_block(deconv2_3, 64, 5)
        deconv2_1 = res_net_block(deconv2_2, 64, 5)

        deconv1_4 = Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=2, padding='same',
                                    activation=Activation(tf.nn.relu))(deconv2_1)
        cat1 = Add()([deconv1_4, conv1_4])
        deconv1_3 = res_net_block(cat1, 32, 5)
        deconv1_2 = res_net_block(deconv1_3, 32, 5)
        deconv1_1 = res_net_block(deconv1_2, 64, 5)

        inp_pred = Conv2D(filters=channels, kernel_size=(5, 5), padding='same', activation=None)(deconv1_1)

        if i >= 0:
            x_unwrap.append(inp_pred)

    # return x_unwrap
    return inp_pred


def custom_loss(x_unwrap, img_gt): # Could be wrapped
    n_levels = 3

    loss_total = 0
    for i in range(n_levels):
        batch_s, hi, wi, channels = x_unwrap[i].get_shape().as_list()
        gt_i = tf.image.resize(img_gt, [hi, wi])
        loss = tf.reduce_mean((gt_i - x_unwrap[i]) ** 2)
        loss_total += loss

    return loss_total


data_size = train_sharp_generator.samples // batch_size
max_steps = int(epochs * data_size)

INITIAL_LR = 1e-4
END_LR = 0.0
POWER = 0.3
LR = PolynomialDecay(initial_learning_rate=INITIAL_LR, decay_steps=max_steps, end_learning_rate=END_LR, power=POWER)
OPTIMIZER = Adam(lr=INITIAL_LR)


def psnr(y_true, y_pred):
    return peak_signal_noise_ratio(y_true, y_pred)


def ssim(y_true, y_pred):
    return structural_similarity(y_true, y_pred, multichannel=True)


METRICS = ['mse', psnr, ssim]
METRICS = None

input_sharp = Input(shape=input_shape, name='input_sharp')
input_blur = Input(shape=input_shape, name='input_blur')

x_unwrap = []
output = generator(input_blur, x_unwrap)
model = Model(inputs=[input_sharp, input_blur], outputs=output)

model.add_loss(custom_loss(x_unwrap, input_sharp))
model.compile(optimizer=OPTIMIZER, metrics=METRICS)

print(model.summary())

log_dir = '../res/logs/reds' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)  # TODO1 histogram_freq=1

checkpoint_filepath = '../res/models/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.h5'

rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-3)
es = EarlyStopping(monitor='loss', patience=3)
mc = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', save_best_only=True)

callbacks = [tensorboard_callback, mc]

# print('Using GPU: {}'.format(tf.test.is_gpu_available()))
print(tf.config.list_physical_devices('GPU'))

train_steps = data_size
validation_steps = val_sharp_generator.samples // batch_size

if True:
    model.load_weights('../res/models/weights-2.h5')
    print('Loaded weights!')

history = model.fit(train_generator, epochs=epochs, steps_per_epoch=train_steps, callbacks=callbacks,
                    validation_data=validation_generator, validation_steps=validation_steps)

print('Saving model')
model.save('../res/models/final_model.h5')  # model = load_model('model.h5')
model.save_weights('../res/models/final_weights.h5')  # model.load_weights('weights.h5')

'''
val_score = model.evaluate_generator(val_generator, steps_per_epoch["val"])
test_score = model.evaluate_generator(test_generator, steps_per_epoch["test"])
predict = model.predict_generator(test_generator, steps=steps_per_epoch["test"])
'''
