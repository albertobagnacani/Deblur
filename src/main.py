import datetime
import json
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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, Activation, Conv2DTranspose, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import load_model

from TensorflowDatasetLoader import TensorflowDatasetLoader
from utils import load_cifar, blur_cifar, reshape_cifar, unpickle

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

json_path = 'params.json'

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
seed = 42
batch_size = 8
class_mode = None
epochs = 50
initial_lr = 1e-4
mc_period = 1

rescale = 1./255
# validation_split = 0.2
target_size = (720, 1280)
input_shape = (None, None, 3)
random_crop_size = (256, 256)

'''
# Number of images in each folder
image_count = {"train": len(train_generator.filenames), "val": len(val_generator.filenames), 
               "test": len(test_generator.filenames)}
# Number of steps per epoch
steps_per_epoch = {"train": np.ceil(image_count["train"]/BATCH_SIZE), "val": np.ceil(image_count["val"]/BATCH_SIZE), 
                   "test": np.ceil(image_count["test"]/BATCH_SIZE)}
'''

with open(json_path) as json_file:
    data = json.load(json_file)
    epochs = data['epochs']
    batch_size = data['batch_size']
    seed = data['seed']
    load_epoch = data['load_epoch']
    initial_lr = data['initial_lr']
    mc_period = data['mc_period']
    train = data['train']

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
test_generator = test_datagen.flow_from_directory(
        reds['test_b'],
        target_size=target_size,
        batch_size=batch_size, class_mode=class_mode, seed=seed)


def random_crop(sharp_batch, blur_batch):
    s = []
    b = []

    for image_s, image_b in zip(sharp_batch, blur_batch):
        height, width = image_s.shape[0], image_s.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        s.append(image_s[y:(y + dy), x:(x + dx), :])
        b.append(image_b[y:(y + dy), x:(x + dx), :])
    return np.array(s), np.array(b)


def combine_generators(sharp_generator, blur_generator):
    while True:
        sharp_batch = sharp_generator.next()
        blur_batch = blur_generator.next()

        sharp_batch, blur_batch = random_crop(sharp_batch, blur_batch)

        res = [sharp_batch, blur_batch]

        yield res


def combine_generators_no_random_crop(sharp_generator, blur_generator):
    while True:
        sharp_batch = sharp_generator.next()
        blur_batch = blur_generator.next()

        res = [sharp_batch, blur_batch]

        yield res


train_generator = combine_generators(train_sharp_generator, train_blur_generator)
validation_generator = combine_generators(val_sharp_generator, val_blur_generator)
test_val_generator = combine_generators_no_random_crop(val_sharp_generator, val_blur_generator)

'''
test_generator = test_datagen.flow_from_directory(
        '',
        target_size=target_size,
        batch_size=batch_size, class_mode=class_mode, seed=seed, subset='training')
'''


def res_net_block(x, filters, ksize):
    net = Conv2D(filters=filters, kernel_size=(ksize, ksize), padding='same', activation='relu')(x)
    net = Conv2D(filters=filters, kernel_size=(ksize, ksize), padding='same', activation=None)(net)

    return net


def generator(inp, x_unwrap=[]):
# def generator(inp):
    channels = 3
    n_levels = 3
    starting_scale = 0.5

    # b, h, w, c = inp.get_shape()

    if train:
        h, w = random_crop_size
    else:
        h, w = target_size[0], target_size[1]

    # x_unwrap = []
    inp_pred = inp
    for i in range(n_levels):
        scale = starting_scale ** (n_levels - i - 1)
        hi = int(round((h*scale)))
        wi = int(round((w*scale)))

        inp_blur = tf.image.resize(inp, [hi, wi])
        inp_pred = tf.image.resize(inp_pred, [hi, wi])
        inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')  # Use Keras layers? Why this?

        # Encoder
        conv1_1 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(inp_all)
        conv1_2 = res_net_block(conv1_1, 32, 5)
        conv1_3 = res_net_block(conv1_2, 32, 5)
        conv1_4 = res_net_block(conv1_3, 32, 5)

        conv2_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=2, padding='same',
                         activation='relu')(conv1_4)
        conv2_2 = res_net_block(conv2_1, 64, 5)
        conv2_3 = res_net_block(conv2_2, 64, 5)
        conv2_4 = res_net_block(conv2_3, 64, 5)

        conv3_1 = Conv2D(filters=128, kernel_size=(5, 5), strides=2, padding='same',
                         activation='relu')(conv2_4)
        conv3_2 = res_net_block(conv3_1, 128, 5)
        conv3_3 = res_net_block(conv3_2, 128, 5)
        conv3_4 = res_net_block(conv3_3, 128, 5)

        deconv3_4 = conv3_4
        deconv3_3 = res_net_block(deconv3_4, 128, 5)
        deconv3_2 = res_net_block(deconv3_3, 128, 5)
        deconv3_1 = res_net_block(deconv3_2, 128, 5)

        # Decoder
        deconv2_4 = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=2, padding='same',
                                    activation='relu')(deconv3_1)
        # Skip connection
        cat2 = Add()([deconv2_4, conv2_4])  # merge([x, y], mode='sum')
        deconv2_3 = res_net_block(cat2, 64, 5)
        deconv2_2 = res_net_block(deconv2_3, 64, 5)
        deconv2_1 = res_net_block(deconv2_2, 64, 5)

        deconv1_4 = Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=2, padding='same',
                                    activation='relu')(deconv2_1)
        cat1 = Add()([deconv1_4, conv1_4])
        deconv1_3 = res_net_block(cat1, 32, 5)
        deconv1_2 = res_net_block(deconv1_3, 32, 5)
        deconv1_1 = res_net_block(deconv1_2, 32, 5)

        inp_pred = Conv2D(filters=channels, kernel_size=(5, 5), padding='same', activation=None)(deconv1_1)

        if i >= 0:
            x_unwrap.append(inp_pred)

    # return x_unwrap
    return inp_pred


def custom_loss(x_unwrap, img_gt):  # Could be wrapped
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

'''
END_LR = 1e-5
POWER = 2
LR = PolynomialDecay(initial_learning_rate=initial_lr, decay_steps=max_steps, end_learning_rate=END_LR, power=POWER)
'''
OPTIMIZER = Adam(lr=initial_lr)


def custom_mse(x_unwrap, input_sharp):
    def c_mse(y_true, y_pred):
        n_levels = 3

        metric_total = 0
        for i in range(n_levels):
            batch_s, hi, wi, channels = x_unwrap[i].get_shape().as_list()
            gt_i = tf.image.resize(input_sharp, [hi, wi])
            metric = mean_squared_error(gt_i, x_unwrap[i])
            metric_total += metric

        return metric_total
    return c_mse


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def custom_psnr(x_unwrap, input_sharp):
    n_levels = 3

    metric_total = 0
    for i in range(n_levels):
        batch_s, hi, wi, channels = x_unwrap[i].get_shape().as_list()
        gt_i = tf.image.resize(input_sharp, [hi, wi])
        metric = 20*log10((1.0 ** 2) / tf.math.sqrt(tf.reduce_mean((gt_i - x_unwrap[i]) ** 2)))
        metric_total += metric

    metric_total /= 3

    return metric_total


input_sharp = Input(shape=input_shape, name='input_sharp')
input_blur = Input(shape=input_shape, name='input_blur')

x_unwrap = []
output = generator(input_blur, x_unwrap)
model = Model(inputs=[input_sharp, input_blur], outputs=output)

# x_unwrap = generator(input_blur)
# model = Model(inputs=[input_sharp, input_blur], outputs=x_unwrap)

METRICS = None
# METRICS = [custom_mse(x_unwrap, input_sharp)]

model.add_loss(custom_loss(x_unwrap, input_sharp))
# Since training happens on batch of images we will use the mean of SSIM values of all the images in the batch as the
# loss value -> Batch_mean(mean_scales_mse)
model.add_metric(custom_psnr(x_unwrap, input_sharp), name='mean_scales_psnr', aggregation='mean')
model.compile(optimizer=OPTIMIZER, metrics=METRICS)

# print(model.summary())

log_dir = '../res/logs/reds' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)  # TODO1 histogram_freq=1

save_weights_only = False

# checkpoint_filepath = '../res/models/checkpoints/'+ 'model' if not save_weights_only else 'weights'+'.{epoch:04d}-
# {val_loss:.4f}.h5'
checkpoint_filepath = '../res/models/checkpoints/model.{epoch:04d}-{val_loss:.4f}.h5'

rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-3)
es = EarlyStopping(monitor='loss', patience=3)
mc = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', save_best_only=False,
                     save_weights_only=save_weights_only, period=mc_period)

callbacks = [tensorboard_callback, mc, rlrop]

# print('Using GPU: {}'.format(tf.test.is_gpu_available()))
print(tf.config.list_physical_devices('GPU'))

train_steps = data_size
validation_steps = val_sharp_generator.samples // batch_size

if load_epoch != 0:
    model.load_weights('../res/models/model-'+str(load_epoch)+'.h5')
    # model = load_model('../res/models/model-'+str(load_epoch)+'.h5')
    print('Loaded model/weights!')

if train:
    history = model.fit(train_generator, epochs=epochs, steps_per_epoch=train_steps, callbacks=callbacks,
                        validation_data=validation_generator, validation_steps=validation_steps)
    # history = model.fit(TensorflowDatasetLoader('../res/datasets/REDS/train_b/', batch_size=batch_size).dataset,
    #                     epochs=epochs, steps_per_epoch=2, callbacks=callbacks)

    model.save('../res/models/final_model.h5')  # model = load_model('model.h5')
    model.save_weights('../res/models/final_weights.h5')  # model.load_weights('weights.h5')
    print('Saved model/weights!')
else:
    test_steps = test_generator.samples // batch_size

    pred = model.predict(test_val_generator, steps=validation_steps, batch_size=batch_size)

    count = 0
    for img in pred:
        imguint8 = img * 255
        cv2.imwrite('../res/datasets/REDS/out/test/'+str(count)+'.png', cv2.cvtColor(imguint8, cv2.COLOR_RGB2BGR))
        count += 1

    '''
    val_score = model.evaluate_generator(val_generator, steps_per_epoch["val"])
    test_score = model.evaluate_generator(test_generator, steps_per_epoch["test"])
    predict = model.predict_generator(test_generator, steps=steps_per_epoch["test"])
    '''
