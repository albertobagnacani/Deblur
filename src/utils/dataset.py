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
max_sigma = 3  # [0,3]
channel = 1024
image_size = 32


def unpickle(file):
    """
    Unpickle a pickled dict.

    :param file: Path to the pickled file
    :return: unpickled (dict): Unpickled dict
    """
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def load_cifar(path):
    """
    Loads the cifar dataset.

    :param path: Path to the cifar dataset
    :return: dict: for each set (train, val (not loaded here), test), the loaded images
    """
    # List all files
    files = [f for f in listdir(path) if isfile(join(path, f)) and 'batch' in f]
    files.sort()

    train = []
    test = []
    for file in files:
        filename = os.path.join(path, file)
        # Unpickle the batch
        batch = unpickle(filename)

        if 'data_batch' in filename:
            train.append(batch)
        elif 'test_batch' in filename:
            test.append(batch)

    return {'train': train, 'val': [], 'test': test}


def blur_cifar(ds):  # TODO1 do multiprocessing
    """
    Applies gaussian blurring with random stdev in [0, 3].

    The saved dataset was created with seed = 42.

    :param ds (np.array): Loaded dataset
    :return: blurred (np.array): Blurred dataset
    """
    print('Blurring')
    result = {key: [] for key in ds}

    for key in ds:
        for entry in ds[key]:
            tmp = []

            for image in entry[b'data']:
                blurred = []  # 0 = red, 1 = green, 2 = blue
                # Compute the random sigma
                sigma = random.randint(min_sigma, max_sigma)

                for i in range(3):
                    # For each channel, blur the image
                    img_c = np.reshape(image[channel * i:channel * (i + 1)], (image_size, image_size))

                    blurred_c = np.reshape(gaussian_filter(img_c, sigma), (channel,))
                    # blurred_c = np.reshape(cv2.GaussianBlur(src=img_c, sigmaX=sigma), (channel,)) # TODO1 check
                    blurred.extend(blurred_c)

                tmp.append(blurred)

            new_entry = deepcopy(entry)  # a bit heavy
            new_entry[b'data'] = np.array(tmp)
            result[key].append(new_entry)

    return result


def reshape_cifar(ds):  # TODO1 do multiprocessing
    """
    Reshape the cifar into the form (n_images, height, width, channels)

    :param ds (np.array): Loaded dataset
    :return: reshaped (np.array): Reshaped dataset
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
    Save the images of cifar dataset.

    :param ds (np.array): Loaded dataset
    :param path (string): Path where to save images
    :return: void
    """
    count = 0

    for image in ds:
        cv2.imwrite(path+str(count)+'.png', image)
        count += 1


def reds_merge(input_path):
    """
    Merge all the reds scene into a single folder (flow_from_directory format)
    :param input_path (string): Path containing the reds dataset
    :return: void
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

            # Copy to the new folder
            shutil.copyfile(filename, os.path.join(input_path, str(count)+".png"))

            count += 1

        # shutil.rmtree(root_n)


def keras_folder(paths):
    """
    Adds a dummy folder to be compatible with the flow_from_directory (e.g. train/train_blur/folder/<list of images>).
    :param paths (dict): dict containing for each set (train, val, test) the paths to the dataset
    :return: new_paths (dict): updated paths
    """
    print('Moving')

    res = {key: '' for key in paths}

    for key in paths:
        p = paths[key]
        # Add the 'folder/' directory
        new_p = p + 'folder/'

        # Create the folder if not existing
        Path(new_p).mkdir(parents=True, exist_ok=True)

        files = [f for f in listdir(p) if isfile(join(p, f))]
        for f in files:
            # Move to the new folder
            shutil.move(p + f, new_p)

        res[key] = new_p

    return res
