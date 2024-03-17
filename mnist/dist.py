# import matplotlib.pyplot as plt
# from predictor import Predictor
from utils import get_distance
from PIL import Image, ImageChops
import numpy as np
import os
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse

root_path = 'mnist/search/'
# root_path = 'mnist/search2/'
# m_type = 'non-optimal'
m_type = 'optimal'
save_to_heatmap_folder = False
i = 0
ssim_distances = []
mse_distances = []
psnr_distances = []
nrmse_distances = []
l2_distances = []
non_zero_pixels = []
averages = []




folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
for folder in folders:
  digit_path = os.path.join(root_path, folder)
  content = os.listdir(digit_path)
  files = [f for f in content if os.path.isfile(os.path.join(digit_path, f))]
  subfolders = [subfolder for subfolder in content if os.path.isdir(os.path.join(digit_path, subfolder))]
  for file in files:
    if file.endswith('.png') and not file.startswith('heatmap'):
        image_path = os.path.join(digit_path, file)
        for subfolder in subfolders:
            m_path = os.path.join(digit_path, subfolder, m_type)
            if os.path.exists(m_path):
                m_pngs = [m_png for m_png in os.listdir(m_path) if os.path.isfile(os.path.join(m_path, m_png)) and m_png.endswith('.png')]
                for m_png in m_pngs:
                    i += 1

                    if save_to_heatmap_folder:
                      heatmap_path = os.path.join('mnist/heatmaps', f'heatmap-{folder}-{subfolder}.png')
                    else:
                      heatmap_path = os.path.join(digit_path, f'heatmap-{subfolder}.png')

                    if not os.path.exists(heatmap_path):
                      m_image_path = os.path.join(m_path, m_png)
                      img = Image.open(image_path)
                      img_array = np.array(img)

                      m_img = Image.open(m_image_path)
                      m_img_array = np.array(m_img)

                      diff = ImageChops.difference(img, m_img)
                      non_zero_pixel = np.count_nonzero(diff)
                      non_zero_pixels.append(non_zero_pixel)
                      ssim_distance = ssim(img_array, m_img_array, data_range=255)
                      ssim_distances.append(ssim_distance)
                      mse_distance = mse(img_array, m_img_array)
                      mse_distances.append(mse_distance)
                      psnr_distance = psnr(img_array, m_img_array, data_range=255)
                      psnr_distances.append(psnr_distance)
                      nrmse_distance = nrmse(img_array, m_img_array, normalization='euclidean')
                      nrmse_distances.append(nrmse_distance)
                      l2_distance = np.linalg.norm(img_array - m_img_array)
                      l1_distance = np.linalg.norm(img_array - m_img_array, 1)
                      l2_distances.append(l2_distance)
                      average = np.mean(diff)
                      averages.append(average)
                      if ssim_distance > .8:
                        print(f'Comparing {image_path} with {m_image_path}')
                        print(f'{i}: SSIM: {round(ssim_distance * 100, 3)}, MSE: {round(mse_distance, 3)}, PSNR: {round(psnr_distance, 3)}, NRMSE: {round(nrmse_distance, 3)}, L1: {round(l1_distance, 3)}, L2: {round(l2_distance, 3)} Non-zero pixels: {non_zero_pixel}, Average: {round(average, 3)}')
print(f'Average SSIM: {round(np.mean(ssim_distances), 3)}')
print(f'Average MSE: {round(np.mean(mse_distances), 3)}')
print(f'Average PSNR: {round(np.mean(psnr_distances), 3)}')
print(f'Average NRMSE: {round(np.mean(nrmse_distances), 3)}')
print(f'Average L2: {round(np.mean(l2_distances), 3)}')
print(f'Average non-zero pixels: {round(np.mean(non_zero_pixels), 3)}')
print(f'Average average: {round(np.mean(averages), 3)}')
