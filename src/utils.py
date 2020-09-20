import os
import pickle
import random
import shutil
from copy import deepcopy
from os import listdir
from os.path import join, isfile
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

min_sigma = 0
max_sigma = 3
channel = 1024
image_size = 32


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
