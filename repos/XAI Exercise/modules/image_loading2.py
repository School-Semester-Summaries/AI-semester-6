from PIL import Image
from tensorflow import keras
import numpy as np


def get_img_array_via_path(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)

    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)

    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def get_img_array_via_image(img, size):

    # convert to pil image
    pil_img = Image.fromarray((img * 255).astype('uint8'))

    # resize to my size
    pil_img = pil_img.resize(size)

    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(pil_img)

    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array
