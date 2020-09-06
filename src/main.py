import os
import pickle
import random
import shutil
from copy import deepcopy
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import tensorflow as tf


cifar_path = '../res/datasets/cifar-10/'
cifar_train_path = cifar_path+'modified/data_batch_unified'
cifar_test_path = cifar_path+'modified/test_batch'
cifar_blurred_train_path = cifar_path+'modified/data_batch_unified_blurred'
cifar_blurred_test_path = cifar_path+'modified/test_batch_blurred'
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


def reds_merge(input_path):
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
            # shutil.move(filename, os.path.join(input_path, str(count)+".png")) # TODO uncomment
            shutil.copyfile(filename, os.path.join(input_path, str(count)+".png")) # TODO comment

            count += 1

        # shutil.rmtree(root_n) # TODO uncomment


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

        with open(path_c, 'wb') as ff:
            pickle.dump(reshaped_ds, ff)
        with open(path_c_b, 'wb') as ff:
            pickle.dump(reshaped_ds_b, ff)

# Those are data
cifar = {'train': unpickle(cifar_train_path), 'test': unpickle(cifar_test_path),
         'train_b': unpickle(cifar_blurred_train_path), 'test_b': unpickle(cifar_blurred_test_path)}

'''
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

# for key in reds: # TODO uncomment
for k in ['val_s', 'val_b', 'test_b']: # TODO comment
    reds_merge(reds[k])
