from os import listdir
from os.path import isfile, join

import tensorflow as tf
from numpy import float32


def select_patch(sharp, blur, patch_size_x, patch_size_y):
    """
    Select a patch on both sharp and blur images at the same localization.

    :param:
        sharp (tf.Tensor): Tensor for the sharp image
        blur (tf.Tensor): Tensor for the blur image
        patch_size_x (int): Size of patch along x axis
        patch_size_y (int): Size of patch along y axis
    :returns:
        patch (Tuple[tf.Tensor, tf.Tensor]): Tuple of tensors with shape (patch_size_x, patch_size_y, 3)
    """
    stack = tf.stack([sharp, blur], axis=0)
    patches = tf.image.random_crop(stack, size=[2, patch_size_x, patch_size_y, 3])
    return patches[0], patches[1]


class TensorflowDatasetLoader:
    """
    Class to load dataset using the TensorFlow Data API.
    """
    def __init__(self, dataset_path, batch_size=8, patch_size=(256, 256)):
        """
        Class constructor.

        :param dataset_path (string): Path to the dataset
        :param batch_size (int): batch size
        :param patch_size (Tuple[int, int]): dimension of the patch
        """
        p = dataset_path+'sharp/'
        # List all images paths
        # sharp_images_paths = [str(path) for path in Path(dataset_path).glob("*/sharp/*.png")]
        sharp_images_paths = [p+f for f in listdir(p) if isfile(join(p, f))]
        # if n_images is not None:
        #     sharp_images_paths = sharp_images_paths[0:n_images]

        # Generate corresponding blurred images paths
        blur_images_paths = [path.replace("sharp", "blur") for path in sharp_images_paths]

        # Load sharp and blurred images
        sharp_dataset = tf.data.Dataset.from_tensor_slices(sharp_images_paths).map(
            lambda path: self.load_image(path, float32),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        blur_dataset = tf.data.Dataset.from_tensor_slices(blur_images_paths).map(
            lambda path: self.load_image(path, float32),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        dataset = tf.data.Dataset.zip((sharp_dataset, blur_dataset))
        dataset = dataset.cache()

        # Select the same patch on the sharp image and its corresponding blurred
        dataset = dataset.map(
            lambda sharp_image, blur_image: select_patch(
                sharp_image, blur_image, patch_size[0], patch_size[1]
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        # Define dataset characteristics (batch_size, number_of_epochs, shuffling)
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=50)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        self.dataset = dataset

    @staticmethod
    def load_image(image_path, dtype):
        """
        Loads an image.

        :param image_path (string): Path to the image
        :param dtype (dtype): dtype of the loaded image
        :return:
        """
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, dtype)
        image = (image - 0.5) * 2

        return image
