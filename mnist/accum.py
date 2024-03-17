import matplotlib.pyplot as plt
from predictor import Predictor
from utils import get_distance
from PIL import Image, ImageChops
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
from seed_utils_1 import Fuzzgan
from config import CACHE_DIR, STYLEGAN_INIT, SEARCH_LIMIT
import os.path as osp


root_path = 'mnist/search2/'
save_to_heatmap_folder = False
distance_percentages = []
mix_seeds = []

def generate_seed(state=None):
    valid_checkpoints_dict = {
    f.split('/')[-1].split('.')[0]: osp.join(CACHE_DIR, f)
    for f in os.listdir(CACHE_DIR)
    if (f.endswith('pkl') and osp.exists(osp.join(CACHE_DIR, f)))
}

    state['renderer']._render_impl(
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

    # print(state['generator_params'])
    init_image = state['generator_params'].image
    w = state['generator_params'].w
    # print(w.shape)
    state['images']['image_orig'] = init_image
    # layers = [[x] for x in range(state['generator_params'].num_ws)]


    return state, w


digit_info = STYLEGAN_INIT
img, w = generate_seed(digit_info)
img = img["images"]["image_orig"]
img.save('acc-0.png')
m_img = None
for i in range(1, 10):
    prev_img = m_img if m_img else img
    digit_info["params"]["stylemix_seed"] = i
    digit_info["params"]["w_load"] = w
    m_img, w = generate_seed(digit_info)
    m_img = m_img["images"]["image_orig"]
    print(get_distance(np.array(prev_img), np.array(m_img)))
    m_img.save(f'acc-{i}.png')

