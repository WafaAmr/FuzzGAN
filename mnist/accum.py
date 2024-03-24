import matplotlib.pyplot as plt
from predictor import Predictor
from utils import get_distance
from PIL import Image, ImageChops
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
from config import CACHE_DIR, STYLEGAN_INIT, SEARCH_LIMIT
import os.path as osp
from stylegan.renderer_v2 import Renderer
import copy

def generate_seed(state=None):
    valid_checkpoints_dict = {
    f.split('/')[-1].split('.')[0]: osp.join(CACHE_DIR, f)
    for f in os.listdir(CACHE_DIR)
    if (f.endswith('pkl') and osp.exists(osp.join(CACHE_DIR, f)))
}

    Renderer(disable_timing=True)._render_impl(
        res = state['generator_params'],  # res
        pkl = valid_checkpoints_dict[state['pretrained_weight']],  # pkl
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


fig, axs = plt.subplots(3, 10, figsize=(50, 15))
digit_info = STYLEGAN_INIT
start_class = 2
digit_info["params"]["w0_seeds"] = [[0, 1]]
digit_info["params"]["class_idx"] = start_class
digit, _ = generate_seed(digit_info)
img = digit['generator_params'].image
w = digit['generator_params'].w
axs[0,0].imshow(img, cmap='gray')
axs[0,0].set_title(f'ID: 0')


m_img = None
for i in range(1, 10):
    prev_img = m_img if m_img else img
    digit_info["params"]["stylemix_seed"] = i
    digit_info["params"]["mixclass_idx"] = 5
    digit_info["params"]["stylemix_idx"] = [6]
    digit_info["params"]["w_load"] = w
    m_digit, m_digit_info = generate_seed(digit_info)
    m_img = m_digit['generator_params'].image
    w = digit['generator_params'].w
    o_diff = ImageChops.difference(img, m_img)
    pre_diff = ImageChops.difference(prev_img, m_img)

    m_image = m_img.crop((2, 2, m_img.width - 2, m_img.height - 2))
    accepted, _confidence, predictions = Predictor().predict_datapoint(
    np.reshape(m_image, (-1, 28, 28, 1)),
    start_class
    )
    if accepted:
        digit_class = start_class
    else:
        digit_class = np.argsort(-predictions)[:1]
    axs[0, i].imshow(o_diff, cmap='jet', interpolation='nearest')
    axs[0, i].set_title(f'DIFF: {0} - {i}, L2: {int(get_distance(np.array(img), np.array(m_img)))}')
    axs[1, i].imshow(m_img, cmap='gray')
    axs[1, i].set_title(f'ID: {i}, Class: {digit_class}, Seed: {i}, Layer: 6')
    axs[2, i].imshow(pre_diff, cmap='jet', interpolation='nearest')
    axs[2, i].set_title(f'DIFF: {i-1} - {i}, L2: {int(get_distance(np.array(prev_img), np.array(m_img)))}')


plt.savefig('acc-2-5.png')

