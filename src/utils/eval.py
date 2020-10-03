from os import listdir
from os.path import isfile, join

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error, structural_similarity


def avg_metric(sharp_path, deblurred_path):  # TODO1 do multiprocessing in those methods
    """
    Returns the metric evaluation (MSE, PSNR, SSIM) between the sharp_path and deblurred_path.

    :param sharp_path (string): Path to the sharp set
    :param deblurred_path (string): Path to the deblurred set
    :return: metrics (Tuple[float, float, float]): metric evaluation of MSE, PSNR, SSIM, respectively
    """
    sum_psnr = 0
    sum_mse = 0
    sum_ssim = 0

    # List all files
    files_orig = [f for f in listdir(sharp_path) if isfile(join(sharp_path, f))]
    files_deb = [f for f in listdir(deblurred_path) if isfile(join(deblurred_path, f))]

    count = 0
    for orig, deb in zip(files_orig, files_deb):
        orig_fn = join(sharp_path, orig)
        deb_fn = join(deblurred_path, deb)
        # Load images
        orig_img = cv2.imread(orig_fn)
        deb_img = cv2.imread(deb_fn)
        orig_img = np.divide(orig_img, 255)
        deb_img = np.divide(deb_img, 255)

        # Compute metrics
        sum_psnr += peak_signal_noise_ratio(orig_img, deb_img)
        sum_mse += mean_squared_error(orig_img, deb_img)
        sum_ssim += structural_similarity(orig_img, deb_img, multichannel=True)

        count += 1
        print('Analyzed: {}/{}'.format(count, len(files_orig)))

    # Average
    avg_psnr = sum_psnr/len(files_orig)
    avg_mse = sum_mse/len(files_orig)
    avg_ssim = sum_ssim/len(files_orig)

    return avg_mse, avg_psnr, avg_ssim


def avg_metric_loaded_array(sharp_arr, deblurred_arr):
    """
    Returns the metric evaluation (MSE, PSNR, SSIM) between the sharp array and deblurred array.

    :param sharp_arr (np.array): Loaded array of sharp set
    :param deblurred_arr (np.array): Loaded array of deblurred set
    :return: metrics (Tuple[float, float, float]): metric evaluation of MSE, PSNR, SSIM, respectively
    """
    sum_psnr = 0
    sum_mse = 0
    sum_ssim = 0

    count = 0
    for orig, deb in zip(sharp_arr, deblurred_arr):
        # Compute metrics
        orig /= 255.
        deb /= 255.
        sum_psnr += peak_signal_noise_ratio(orig, deb)
        sum_mse += mean_squared_error(orig, deb)
        sum_ssim += structural_similarity(orig, deb, multichannel=True)

        count += 1
        print('Analyzed: {}/{}'.format(count, len(sharp_arr)))

    # Average
    avg_psnr = sum_psnr/len(sharp_arr)
    avg_mse = sum_mse/len(sharp_arr)
    avg_ssim = sum_ssim/len(sharp_arr)

    return avg_mse, avg_psnr, avg_ssim
