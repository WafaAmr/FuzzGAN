# import matplotlib.pyplot as plt
# from predictor import Predictor
from utils import get_distance
from PIL import Image, ImageChops
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse
import matplotlib.pyplot as plt
import json


# root_path = 'mnist/eval/HQ'
root_path = 'mnist/eval/HQ'
m_type = 'ssim'
save_heatmaps = False
# save_heatmaps = True
overwite_heatmaps = True
save_to_heatmap_folder = True

l2_range = []



model_folder = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
for class_folder in model_folder:
  class_path = os.path.join(root_path, class_folder)
  seed_class = [f for f in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, f))]
  for seed in seed_class:
    seed_path = os.path.join(class_path, seed)
    content = os.listdir(seed_path)
    files = [f for f in content if os.path.isfile(os.path.join(seed_path, f))]
    subfolders = [subfolder for subfolder in content if os.path.isdir(os.path.join(seed_path, subfolder))]

    for file in files:
      if file.endswith('.png') and not file.startswith('heatmap'):
          img_path = os.path.join(seed_path, file)
          for subfolder in subfolders:
              m_path = os.path.join(seed_path, subfolder, m_type)
              if os.path.exists(m_path):
                  m_pngs = sorted([m_png for m_png in os.listdir(m_path) if os.path.isfile(os.path.join(m_path, m_png)) and m_png.endswith('.png')])
                  for m_png in m_pngs:

                      img = Image.open(img_path)
                      img_array = np.array(img)

                      m_img_path = os.path.join(m_path, m_png)
                      m_img = Image.open(m_img_path)
                      m_img_array = np.array(m_img)

                      img_l2 = int(np.linalg.norm(img_array))
                      m_img_l2 = int(np.linalg.norm(m_img_array))
                      l2_range.append(m_img_l2/img_l2)

plt.figure(figsize=(20, 6))
plt.boxplot(l2_range, vert=False)
plt.title('L2 Range for all valid mutations with SSIM > 0.98')
plt.savefig(f'{root_path}/l2_range.png')
import numpy as np

print("Min:", np.min(l2_range))
print("Max:", np.max(l2_range))
print("Median:", np.median(l2_range))
print("Average:", np.average(l2_range))
print("First Quartile (25 percentile):", np.percentile(l2_range, 25))
print("Third Quartile (75 percentile):", np.percentile(l2_range, 75))