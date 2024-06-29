from os import makedirs
from os.path import exists, basename, join

from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from folder import Folder
from config import IMG_SIZE, SSIM_THRESHOLD, L2_RANGE
import numpy as np
from skimage.metrics import structural_similarity as ssim

import torch
from PIL import Image

# load the MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


def get_distance(v1, v2):
    return np.linalg.norm(v1/255 - v2/255)

def get_ssim(img_array, m_img_array, data_range=255):
    return ssim(img_array, m_img_array, data_range=data_range) * 100

def validate_mutation(img_array, m_img_array):
    img_l2 = np.linalg.norm(img_array)
    m_img_l2 = np.linalg.norm(m_img_array)
    distance = np.linalg.norm(img_array - m_img_array)
    ssi = ssim(img_array, m_img_array, data_range=255)
    # valid_mutation = img_l2 * (1 - L2_RANGE) <= m_img_l2 <= img_l2 * (1 + L2_RANGE) and ssi > SSIM_THRESHOLD
    valid_mutation = 0 < distance < img_l2 * (1 + L2_RANGE) and ssi >= SSIM_THRESHOLD
    return valid_mutation, ssi, distance, img_l2, m_img_l2

def make_image(inversed_img, img_scale_db = 0):
    inversed_img = inversed_img.squeeze()  # Remove singleton dimensions
    inversed_img = inversed_img / inversed_img.norm(float('inf'), dim=[0,1], keepdim=True).clip(1e-8, 1e8)
    inversed_img = inversed_img * (10 ** (img_scale_db / 20))
    inversed_img = (inversed_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    inversed_img = Image.fromarray(inversed_img.cpu().numpy())
    return inversed_img

def print_archive(archive):
    dst = Folder.DST_ARC + "_DJ"
    if not exists(dst):
        makedirs(dst)
    for i, ind in enumerate(archive):
        filename1 = join(dst, basename(
            'archived_' + str(i) +
            '_mem1_l_' + str(ind.m1.predicted_label) +
            '_seed_' + str(ind.seed)))
        plt.imsave(filename1, ind.m1.purified.reshape(28, 28),
                   cmap=cm.gray,
                   format='png')
        np.save(filename1, ind.m1.purified)
        assert (np.array_equal(ind.m1.purified,
                               np.load(filename1 + '.npy')))

        filename2 = join(dst, basename(
            'archived_' + str(i) +
            '_mem2_l_' + str(ind.m2.predicted_label) +
            '_seed_' + str(ind.seed)))
        plt.imsave(filename2, ind.m2.purified.reshape(28, 28),
                   cmap=cm.gray,
                   format='png')
        np.save(filename2, ind.m2.purified)
        assert (np.array_equal(ind.m2.purified,
                               np.load(filename2 + '.npy')))


def print_archive_experiment(archive):
    for i, ind in enumerate(archive):
        digit = ind.m1
        digit.export(ind.id)
        digit = ind.m2
        digit.export(ind.id)
        ind.export()


# Useful function that shapes the input in the format accepted by the ML model.
def reshape(v):
    v = (np.expand_dims(v, 0))
    # Shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        v = v.reshape(v.shape[0], 1, IMG_SIZE, IMG_SIZE)
    else:
        v = v.reshape(v.shape[0], IMG_SIZE, IMG_SIZE, 1)
    v = v.astype('float32')
    v = v / 255.0
    return v


def input_reshape(x):
    # shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        x_reshape = x.reshape(x.shape[0], 1, 28, 28)
    else:
        x_reshape = x.reshape(x.shape[0], 28, 28, 1)
    x_reshape = x_reshape.astype('float32')
    x_reshape /= 255.0
    return x_reshape

