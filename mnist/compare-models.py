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
from predictor import Predictor

# root_path = 'mnist/eval/final/LQ-cross'
root_path = 'mnist/eval/final/HQ/'
# root_path = 'mnist/eval/final/HQ-100-10/'
# root_path = 'mnist/4000/eval/HQ/'
m_prefix = ''
diff_class = 0
mis = 0

stylemix_seeds = [[] for _ in range(11)]
seeds = [[] for _ in range(11)]
l2_comparison = []
con_classes = [[] for _ in range(10)]

model_folder = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
for class_folder in model_folder:
  if not class_folder in ['stats', 'heatmaps']:
    class_path = os.path.join(root_path, class_folder)
    seed_class = [f for f in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, f))]
    for seed in seed_class:
      seed_path = os.path.join(class_path, seed)
      content = os.listdir(seed_path)
      files = [f for f in content if os.path.isfile(os.path.join(seed_path, f))]
      subfolders = [subfolder for subfolder in content if os.path.isdir(os.path.join(seed_path, subfolder))]
      # class_seeds.append(int(seed))
      seeds[int(class_folder)].append(int(seed))
      seeds[10].append(int(seed))

      for file in files:
        if file.endswith('.png') and not file.startswith('heatmap'):
            img_path = os.path.join(seed_path, file)
            # for subfolder in subfolders:
                # m_path = os.path.join(seed_path, subfolder, m_prefix)
                # if os.path.exists(m_path):
                #     m_pngs = sorted([m_png for m_png in os.listdir(m_path) if os.path.isfile(os.path.join(m_path, m_png)) and m_png.endswith('.png')])
                #     for m_png in m_pngs[:1]:

            img = Image.open(img_path)
            img_array = np.array(img)

                        # m_img_path = os.path.join(m_path, m_png)
                        # m_img = Image.open(m_img_path)
                        # m_img_array = np.array(m_img)

                        # m_img_json_path = os.path.join(m_path, m_png.replace('.png', '.json'))

            m_accepted, confidence , m_predictions = Predictor().predict_datapoint(
              np.reshape(img_array, (-1, 28, 28, 1)),
              class_folder
            )
            m_class = np.argsort(-m_predictions)[:1]
            if not class_folder == str(m_class[0]):
              print(class_folder, m_accepted, m_class)
              mis += 1

            # m_class = np.argsort(-m_predictions)[:1]
            # pred = str(m_class[0])
            # if not class_folder == pred:
            #   print(class_folder, subfolder, m_accepted, m_class)
            #   mis += 1
            #   if subfolder == pred:
            #     diff_class += 1
print(mis)
print(diff_class)


