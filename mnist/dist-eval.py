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
root_path = 'mnist/eval/LQ'
# m_type = 'optimal'
m_type = ''
# save_heatmaps = False
save_heatmaps = True
overwite_heatmaps = True
save_to_heatmap_folder = True


ssim_distances = []
mse_distances = []
psnr_distances = []
nrmse_distances = []
l2_distances = []
l1_distances = []
non_zero_pixels = []
pixel_difference = []
stylemix_layers = []
stylemix_seeds = []
l2_comparison = []



model_folder = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
for class_folder in model_folder:
  # class_folder = '0'
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
                  for m_png in m_pngs[:1]:

                      img = Image.open(img_path)
                      img_array = np.array(img)

                      m_img_path = os.path.join(m_path, m_png)
                      m_img = Image.open(m_img_path)
                      m_img_array = np.array(m_img)

                      m_img_json_path = os.path.join(m_path, m_png.replace('.png', '.json'))
                      with open(m_img_json_path, 'r') as f:
                        data = json.load(f)
                        stylemix_layer = data['stylemix_idx'][0]
                        stylemix_seed = data['stylemix_seed']
                        stylemix_layers.append(stylemix_layer)
                        stylemix_seeds.append(stylemix_seed)

                      diff = ImageChops.difference(img, m_img)
                      non_zero_elements = np.array(diff)[np.nonzero(diff)]
                      pixel_difference.extend(non_zero_elements)
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
                      l2_distances.append(l2_distance)
                      l1_distance = np.linalg.norm(img_array - m_img_array, 1)
                      l1_distances.append(l1_distance)

                      img_l2 = int(np.linalg.norm(img_array))
                      m_img_l2 = int(np.linalg.norm(m_img_array))
                      l2_comparison.append(img_l2 - m_img_l2)

                      if save_heatmaps:
                        if save_to_heatmap_folder:
                          os.makedirs(f'{root_path}/heatmaps/{class_folder}', exist_ok=True)
                          heatmap_path = os.path.join(f'{root_path}/heatmaps/{class_folder}', f'heatmap-{seed}-{subfolder}.png')
                          # print(heatmap_path)
                        else:
                          heatmap_path = os.path.join(seed_path, f'heatmap-{subfolder}.png')

                        if overwite_heatmaps or not os.path.exists(heatmap_path):
                          fig, axs = plt.subplots(1, 3, figsize=(18, 5))

                          # Display original image
                          axs[0].imshow(img, cmap='gray')

                          if img_l2 < m_img_l2:
                            color = 'red'
                            m_color = 'black'
                          else:
                            color = 'black'
                            m_color = 'red'

                          axs[0].set_title(f'Original Image - Class {class_folder} - L2: {img_l2}', color=color)

                          # Display modified image
                          axs[1].imshow(m_img, cmap='gray')
                          axs[1].set_title(f'Modified Image - Class {subfolder} - L2: {m_img_l2}', color=m_color)

                          # Display difference heatmap
                          divider = make_axes_locatable(axs[2])
                          cax = divider.append_axes("right", size="5%", pad=0.05)
                          im = axs[2].imshow(diff, cmap='jet', interpolation='nearest')
                          fig.colorbar(im, cax=cax, orientation="vertical")
                          axs[2].set_title(f'Difference Heatmap - L2: {int(l2_distance)}, SSIM: {round(ssim_distance *100, 1)}')
                          print(f'Heatmap saved to {heatmap_path}')
                          plt.savefig(heatmap_path)
                          plt.close()
                        else:
                          print(f'Heatmap already exists at {heatmap_path}')

print(f'Average SSIM: {round(np.mean(ssim_distances), 3)}')
print(f'Average MSE: {round(np.mean(mse_distances), 3)}')
print(f'Average PSNR: {round(np.mean(psnr_distances), 3)}')
print(f'Average NRMSE: {round(np.mean(nrmse_distances), 3)}')
print(f'Average L2: {round(np.mean(l2_distances), 3)}')
print(f'Average L1: {round(np.mean(l1_distances), 3)}')
print(f'Average non-zero pixels: {round(np.mean(non_zero_pixels), 3)}')

# plt.figure(figsize=(20, 6))
plt.hist(stylemix_layers, bins='auto')
plt.title('Histogram of stylemix_layer')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig(f'{root_path}/stylemix_layer.png')

plt.figure(figsize=(20, 6))
plt.boxplot(ssim_distances, vert=False, meanline=True, showmeans=True)
plt.title('Structural similarity index between original and modified images')
plt.savefig(f'{root_path}/SSIM.png')

plt.figure(figsize=(20, 6))
plt.boxplot(mse_distances, vert=False, meanline=True, showmeans=True)
plt.title('Mean squared error between original and modified images')
plt.savefig(f'{root_path}/MSE.png')

plt.figure(figsize=(20, 6))
plt.boxplot(psnr_distances, vert=False, meanline=True, showmeans=True)
plt.title('Peak signal-to-noise ratio between original and modified images')
plt.savefig(f'{root_path}/PSNR.png')

plt.figure(figsize=(20, 6))
plt.boxplot(nrmse_distances, vert=False, meanline=True, showmeans=True)
plt.title('Normalized root mean squared error between original and modified images')
plt.savefig(f'{root_path}/NRMSE.png')

plt.figure(figsize=(20, 6))
plt.boxplot(l2_distances, vert=False, meanline=True, showmeans=True)
plt.title('L2 distance of original and modified images difference')
plt.savefig(f'{root_path}/L2.png')

plt.figure(figsize=(20, 6))
plt.boxplot(l1_distances, vert=False, meanline=True, showmeans=True)
plt.title('L1 distance of original and modified images difference')
plt.savefig(f'{root_path}/L1.png')

plt.figure(figsize=(20, 6))
plt.boxplot(non_zero_pixels, vert=False, meanline=True, showmeans=True)
plt.title('Number of mutated pixels')
plt.savefig(f'{root_path}/non_zero_pixels.png')

plt.figure(figsize=(20, 6))
plt.boxplot(pixel_difference, vert=False, meanline=True, showmeans=True)
plt.title('Pixel value difference between original and modified images')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
plt.savefig(f'{root_path}/pixel_difference.png')
# print(f'min pixel difference: {min(pixel_difference)}')
# print(f'max pixel difference: {max(pixel_difference)}')
# print(f'average pixel difference: {round(np.mean(pixel_difference), 3)}')
# print(f'median pixel difference: {np.median(pixel_difference)}')
# print(f'mean pixel difference: {np.mean(pixel_difference)}')

plt.figure(figsize=(20, 6))
plt.boxplot(stylemix_seeds, vert=False, meanline=True, showmeans=True)
plt.title('Number of mutations')
plt.savefig(f'{root_path}/mutations.png')

plt.figure(figsize=(20, 6))
plt.boxplot(l2_comparison, vert=False, meanline=True, showmeans=True)
plt.title('L2 difference between original and modified images')
plt.savefig(f'{root_path}/l2_comparison.png')