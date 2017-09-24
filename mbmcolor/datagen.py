import os
import warnings
import random
import glob
import numpy as np

from mbmcolor.utils.image import read_image, resize, rgb2lab, lab2rgb


class ImagePreprocessor(object):
    # TODO: Write me a documentation

    def __init__(self, img_size, mean_shift=0.5, norm_factor=1.0, alexnet_resize=False):
        self.img_size = img_size
        self.mean_shift = mean_shift
        self.norm_factor = norm_factor
        self.alexnet_resize = alexnet_resize

        if alexnet_resize and img_size[0] != img_size[1]:
            self.img_size = (max(img_size), max(img_size))
            message = 'alexnet_resize was specified but img_size is not square. ' \
                      'img_size will be changed to (%d, %d)' % self.img_size
            warnings.warn(message, Warning)

    def resize_alexnet(self, img):
        """Normalize image size a la AlexNet.

        Args:
            img (numpy.array): Image.

        Returns:
            Resized and cropped image.
        """
        height, width, _ = img.shape
        new_size, _ = self.img_size  # img_size is square in this case

        # Resize so that the shorter size is equal to img_size
        # Then crop the central img_size patch
        if height < width:

            new_height, new_width = new_size, round(width * new_size / height)
            img = resize(img, (new_height, new_width))

            offset = (new_width - new_size) // 2
            img = img[:, offset:(offset + new_size)]

        else:

            new_height, new_width = round(height * new_size / width), new_size
            img = resize(img, (new_height, new_width))

            offset = (new_height - new_size) // 2
            img = img[offset:(offset + new_size), :]

        return img

    @staticmethod
    def image2array(img):

        assert img.dtype == 'float32', ValueError('Input image must be float32.')

        array = img

        if len(img.shape) == 2:
            array = np.expand_dims(array, axis=2)  # Add channel dimension
        elif len(img.shape) != 3:
            raise ValueError('Input image must be either a 2D or 3D array.')
        elif not 1 <= img.shape[2] <= 3:
            raise ValueError('Input image must contain at most 3 channels.')

        array = np.expand_dims(array, axis=0)  # Add samples dimension

        return array

    @staticmethod
    def array2image(array):

        assert array.dtype == 'float32', ValueError('Input array must be float32.')

        img = array

        if len(array.shape) != 4:
            raise ValueError('Input array must be 4-D (1, height, width, channels).')

        img = np.squeeze(img)  # Drop non-singleton dimensions

        return img

    def image2io(self, img):

        img_lab = rgb2lab(img)

        i = self.image2array(img_lab[:, :, 0])
        i -= self.mean_shift
        i /= self.norm_factor

        height, width, _ = img_lab[:, :, 1:].shape
        channels = 2  # TODO: Refactor channels
        o = np.reshape(img_lab[:, :, 1:], height * width * channels)

        return i, o

    def io2image(self, io):

        i, o = io

        i *= self.norm_factor
        i += self.mean_shift

        _, height, width, _ = i.shape
        channels = 2  # TODO: Refactor channels
        o = np.reshape(o, (1, height, width, channels))

        img_lab = np.concatenate((i, o), axis=3)
        img_lab = self.array2image(img_lab)

        return lab2rgb(img_lab)

    def build_image_generator(self, img_path, batch_size, n_imgs):

        img_paths = glob.glob(os.path.join(img_path, '*.JPEG'))
        paths = random.sample(img_paths, n_imgs)

        height, width = self.img_size
        x = np.zeros((batch_size, height, width, 1), dtype='float32')
        y = np.zeros((batch_size, height * width * 2), dtype='float32')

        while 1:

            n_samples = 0

            for path in paths:

                try:
                    img = read_image(path)

                except Exception as e:
                    print(e.message)

                else:
                    if len(img.shape) != 3:  # Skip if it is not a color image
                        continue

                    img = self.resize_alexnet(img)
                    x[n_samples % batch_size], y[n_samples % batch_size] = self.image2io(img)

                    n_samples += 1

                    if n_samples % batch_size == 0:
                        yield (x, y)
