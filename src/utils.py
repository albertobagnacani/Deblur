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