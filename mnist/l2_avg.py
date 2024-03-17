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

    init_image = state['generator_params'].image
    state['images']['image_orig'] = init_image
    # layers = [[x] for x in range(state['generator_params'].num_ws)]


    return state


folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
for folder in folders:
  digit_path = os.path.join(root_path, folder)
  content = os.listdir(digit_path)
  files = [f for f in content if os.path.isfile(os.path.join(digit_path, f))]
  subfolders = [subfolder for subfolder in content if os.path.isdir(os.path.join(digit_path, subfolder))]
  for file in files:
    if file.endswith('.json') and not file.startswith('heatmap'):
        json_path = os.path.join(digit_path, file)
        print("###############################")
        with open(json_path, 'r') as f:
          params = json.load(f)
        digit_info = STYLEGAN_INIT
        digit_info['params']["w0_seeds"] = params["w0_seeds"]
        digit_info['params']["mixclass_idx"] = params["mixclass_idx"]
        digit_info['params']["stylemix_seed"] = params["stylemix_seed"]
        digit_info['params']["stylemix_idx"] = params["stylemix_idx"]

        img = generate_seed(digit_info)
        img = img["images"]["image_orig"]
        img.save('0.png')
        for subfolder in subfolders:
            m_path = os.path.join(digit_path, subfolder, 'optimal')
            if os.path.exists(m_path):
                m_pngs = [m_png for m_png in os.listdir(m_path) if os.path.isfile(os.path.join(m_path, m_png)) and m_png.endswith('.json')]
                for m_png in m_pngs:

                    if save_to_heatmap_folder:
                      heatmap_path = os.path.join('mnist/heatmaps', f'heatmap-{folder}-{subfolder}.png')
                    else:
                      heatmap_path = os.path.join(digit_path, f'heatmap-{subfolder}.png')

                    if not os.path.exists(heatmap_path):
                      mix_json_path = os.path.join(m_path, m_png)
                      with open(mix_json_path, 'r') as f:
                        m_params = json.load(f)

                      m_digit_raw = STYLEGAN_INIT
                      # m_digit_raw['params']["w0_seeds"] = [[m_params["stylemix_seed"], .5], [15, .5]]
                      m_digit_raw['params']["w0_seeds"] = [[m_params["stylemix_seed"], 1.0]]
                      m_digit_raw['params']["class_idx"] = m_params["mixclass_idx"]

                      m_img_raw = generate_seed(m_digit_raw)
                      m_img_raw = m_img_raw["images"]["image_orig"]
                      m_img_raw.save('1.png')

                      m_digit_mix = STYLEGAN_INIT
                      m_digit_mix['params']["w0_seeds"] = m_params["w0_seeds"]
                      m_digit_mix['params']["class_idx"] = m_params["class_idx"]
                      m_digit_mix['params']["mixclass_idx"] = m_params["mixclass_idx"]
                      m_digit_mix['params']["stylemix_seed"] = m_params["stylemix_seed"]
                      m_digit_mix['params']["stylemix_idx"] = m_params["stylemix_idx"]
                      m_img_mix = generate_seed(m_digit_mix)
                      m_img_mix = m_img_mix["images"]["image_orig"]
                      m_img_mix.save('2.png')

                      distance = get_distance(np.array(img), np.array(m_img_raw))
                      distance2 = get_distance(np.array(img), np.array(m_img_mix))
                      print(mix_json_path)
                      print(f'Distance: {round(distance, 3)}, Distance2: {round(distance2, 3)}, {round(distance2/distance, 3)}')
                      mag = np.linalg.norm(np.array(img))
                      distance_percentages.append([round(distance2, 3), round(mag * 0.75, 3), round(mag,3) ])
                      mix_seeds.append(m_params["stylemix_seed"])
                      m_digit_mix.save('2.png')


                    else:
                      print(f'Heatmap already exists at {heatmap_path}')


print(distance_percentages)
print(round(np.mean(distance_percentages[:][0])/np.mean(distance_percentages[:][2]), 3))
print(np.mean(mix_seeds))
