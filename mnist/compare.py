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

fig, axs = plt.subplots(1, 11, figsize=(50, 10))

digit_info = STYLEGAN_INIT
digit_info["params"]["w0_seeds"] = [[0, 1], [1, 0]]
digit_info["params"]["class_idx"] = [2, 5]
digit, _ = generate_seed(digit_info)
o_img = digit['generator_params'].image
axs[0].imshow(o_img, cmap='gray')
axs[0].set_title(f'1:0')

digit_info = STYLEGAN_INIT
digit_info["params"]["w0_seeds"] = [[0, 0], [1, 1]]
digit_info["params"]["class_idx"] = [2, 5]
digit, _ = generate_seed(digit_info)
m_img = digit['generator_params'].image

step = 0.1
i = 1
while step < 1:
  digit_info = STYLEGAN_INIT
  digit_info["params"]["w0_seeds"] = [[0, 1-step], [1, step]]
  digit_info["params"]["class_idx"] = [2, 5]
  digit, _ = generate_seed(digit_info)
  img = digit['generator_params'].image
  m_image = img.crop((2, 2, img.width - 2, img.height - 2))
  _accepted, _confidence, predictions = Predictor().predict_datapoint(
  np.reshape(m_image, (-1, 28, 28, 1)),
  2
  )
  digit_class = np.argsort(-predictions)[:1]
  axs[i].imshow(img, cmap='gray')
  axs[i].set_title(f'Class: {digit_class},{int(get_distance(np.array(o_img), np.array(img)))} - {(round(1-step, 1))}:{(round(step, 1))} - {int(get_distance(np.array(m_img), np.array(img)))}')
  step += 0.1
  i += 1
axs[10].imshow(m_img, cmap='gray')
axs[10].set_title(f'0:1')
plt.savefig('latent-mixing-2-5.png')
plt.close()
digit_info["params"]["w0_seeds"] = [[0, 1]]
digit_info["params"]["class_idx"] = 2

fig, axs = plt.subplots(1, 9, figsize=(40, 10))
axs[0].imshow(o_img, cmap='gray')
axs[0].set_title('Layer: []')
for i in range(8):
    digit_info["params"]["stylemix_seed"] = 1
    digit_info["params"]["mixclass_idx"] = 5
    digit_info["params"]["stylemix_idx"] = [i]
    m_digit, m_digit_info = generate_seed(digit_info)
    m_img = m_digit['generator_params'].image
    axs[i+1].imshow(m_img, cmap='gray')
    axs[i+1].set_title(f'Layer: {i}, Distance: {int(get_distance(np.array(o_img), np.array(m_img)))}')
plt.savefig('latent-injecting-2-5.png')
