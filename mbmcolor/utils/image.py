import numpy as np
import skimage.io
import skimage.color
import skimage.transform

# TODO: Write documentation of functions

# Constants lor Lab normalization
L_MIN = 0.0
L_MAX = 100.0
A_MIN = -86.1830297443950144042901229113340377807617187500
A_MAX = 98.2330538631131560123321833088994026184082031250
B_MIN = -107.8573002066948873789442586712539196014404296875
B_MAX = 94.4781222764782313561227056197822093963623046875


def read_image(path):
    """Read image from file.

    Args:
        path (str): Path to image file.

    Returns:
        RGB image.
    """
    return skimage.io.imread(path)


def resize(img, shape):
    """Resize image to specified shape.

    Args:
        img (numpy.array): Image.
        shape (tuple): Output shape.

    Returns:
        Resized image.
    """
    out = skimage.transform.resize(img, shape, preserve_range=True)

    if out.dtype != 'uint8':
        out = np.uint8(out)

    return out


def rgb2lab(img_rgb):
    """Transform image from RGB to Lab.
    This function rescales the Lab values to the [0, 1] interval.

    Args:
        img_rgb (numpy.array): RGB image of shape (height, width, 3).

    Returns:
        Lab image.
    """
    assert img_rgb.dtype == 'uint8'

    img_lab = skimage.color.rgb2lab(img_rgb)

    # Rescale to 0~1
    img_lab[:, :, 0] = (img_lab[:, :, 0] - L_MIN) / (L_MAX - L_MIN)
    img_lab[:, :, 1] = (img_lab[:, :, 1] - A_MIN) / (A_MAX - A_MIN)
    img_lab[:, :, 2] = (img_lab[:, :, 2] - B_MIN) / (B_MAX - B_MIN)

    return np.float32(img_lab)


def lab2rgb(img_lab):
    """Transform image from Lab to RGB.

    Args:
        img_lab (numpy.array): Lab image of shape (height, width, 3).

    Returns:
        RGB image.
    """
    assert img_lab.dtype == 'float32'

    temp = np.copy(img_lab)

    # Undo rescaling
    temp[:, :, 0] = (L_MAX - L_MIN) * temp[:, :, 0] + L_MIN
    temp[:, :, 1] = (A_MAX - A_MIN) * temp[:, :, 1] + A_MIN
    temp[:, :, 2] = (B_MAX - B_MIN) * temp[:, :, 2] + B_MIN

    img_rgb = skimage.color.lab2rgb(np.float64(temp))
    img_rgb = np.uint8(img_rgb * 255)

    return img_rgb
