import matplotlib.pyplot as plt
from predictor import Predictor
from utils import get_distance
from PIL import Image, ImageChops
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
from config import INIT_PKL , STYLEGAN_INIT, SEARCH_LIMIT
import os.path as osp
from stylegan.renderer_v2 import Renderer
import copy

def generate_seed(state=None):

    Renderer(disable_timing=True)._render_impl(
        res = state['generator_params'],  # res
        pkl = INIT_PKL,  # pkl
        w0_seeds= state['params']['w0_seeds'],  # w0_seed,
        class_idx = state['params']['class_idx'],  # class_idx,
        mixclass_idx = state['params']['mixclass_idx'],  # mix_idx,
        stylemix_idx = state['params']['stylemix_idx'],  # stylemix_idx,
        stylemix_seed = state['params']['stylemix_seed'],  # stylemix_seed,
        img_normalize = state['params']['img_normalize'],
        to_pil = state['params']['to_pil'],
    )

    info =  copy.deepcopy(state['params'])

    return state, info

layers = [[7], [6], [5], [4], [3], [5,6], [3,4], [3,4,5,6]]
layers_2 = [[0], [1], [2], [1,2], [0,1,2]]

digit_info = STYLEGAN_INIT
digit_info["params"]["w0_seeds"] = [[0, 1]]
digit_info["params"]["class_idx"] = 2
digit, _ = generate_seed(digit_info)
o_img = digit['generator_params'].image

fig, axs = plt.subplots(3, len(layers), figsize=(35, 15))
axs[0][0].imshow(o_img, cmap='gray')
axs[0][0].set_title('Original Image')
for i, layer in enumerate(layers):
    digit_info["params"]["stylemix_seed"] = 1
    digit_info["params"]["mixclass_idx"] = 5
    digit_info["params"]["stylemix_idx"] = layer
    m_digit, m_digit_info = generate_seed(digit_info)
    m_img = m_digit['generator_params'].image
    axs[1][i].imshow(m_img, cmap='gray')
    axs[1][i].set_title(f'Layer: {layer}, L2: {int(get_distance(np.array(o_img), np.array(m_img)))}')

for i, layer in enumerate(layers_2):
    digit_info["params"]["stylemix_seed"] = 1
    digit_info["params"]["mixclass_idx"] = 5
    digit_info["params"]["stylemix_idx"] = layer
    m_digit, m_digit_info = generate_seed(digit_info)
    m_img = m_digit['generator_params'].image
    axs[2][i].imshow(m_img, cmap='gray')
    axs[2][i].set_title(f'Layer: {layer}, L2: {int(get_distance(np.array(o_img), np.array(m_img)))}')

digit_info["params"]["w0_seeds"] = [[1, 1]]
digit_info["params"]["class_idx"] = 5
digit, _ = generate_seed(digit_info)
o_img = digit['generator_params'].image
axs[0][1].imshow(o_img, cmap='gray')
axs[0][1].set_title('StyleMix Image')
plt.savefig('fine-mix-2-5.png')
