import numpy as np


class ImagePreprocessor(object):

    def __init__(self, mean_shift=0.5, norm_factor=1.0):
        self.mean_shift = mean_shift
        self.norm_factor = norm_factor

    def image2array(self, img):

        assert img.dtype == 'float32', ValueError('Input image must be float32.')

        array = img

        if len(img.shape) == 2:
            array = np.expand_dims(array, axis=2)  # Add channel dimension
        elif len(img.shape) != 3:
            raise ValueError('Input image must be either a single- or 3-channel image.')

        array = np.expand_dims(array, axis=0)  # Add samples dimension

        array -= self.mean_shift
        array /= self.norm_factor

        return array

    def array2image(self, array):

        assert array.dtype == 'float32', ValueError('Input array must be float32.')

        img = array

        if len(array.shape) != 4:
            raise ValueError('Input array must be 4-D (1, height, width, channels).')

        img = np.squeeze(array)  # Drop non-singleton dimensions

        img *= self.norm_factor
        img += self.mean_shift

        return array

