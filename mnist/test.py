# import matplotlib.pyplot as plt
# from predictor import Predictor
# from utils import get_distance
# from PIL import Image, ImageChops
# import numpy as np
# import os
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# root_path = 'mnist/search2/'
# save_to_heatmap_folder = True

# folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
# for folder in folders:
#   digit_path = os.path.join(root_path, folder)
#   content = os.listdir(digit_path)
#   files = [f for f in content if os.path.isfile(os.path.join(digit_path, f))]
#   subfolders = [subfolder for subfolder in content if os.path.isdir(os.path.join(digit_path, subfolder))]
#   for file in files:
#     if file.endswith('.png') and not file.startswith('heatmap'):
#         image_path = os.path.join(digit_path, file)
#         for subfolder in subfolders:
#             m_path = os.path.join(digit_path, subfolder, 'optimal')
#             if os.path.exists(m_path):
#                 m_pngs = [m_png for m_png in os.listdir(m_path) if os.path.isfile(os.path.join(m_path, m_png)) and m_png.endswith('.png')]
#                 for m_png in m_pngs:

#                     if save_to_heatmap_folder:
#                       heatmap_path = os.path.join('mnist/heatmaps', f'heatmap-{folder}-{subfolder}.png')
#                     else:
#                       heatmap_path = os.path.join(digit_path, f'heatmap-{subfolder}.png')

#                     if not os.path.exists(heatmap_path):
#                       m_image_path = os.path.join(m_path, m_png)
#                       img = Image.open(image_path)
#                       image = np.reshape(np.array(img), (-1, 28, 28, 1))

#                       m_img = Image.open(m_image_path)
#                       m_image = np.reshape(np.array(m_img), (-1, 28, 28, 1))

#                       accepted, _, _, _   = Predictor().predict_generator(image, 5)
#                       m_accepted, _, not_class, _ = Predictor().predict_generator(m_image, 5)

#                       diff = ImageChops.difference(m_img, img)

#                       fig, axs = plt.subplots(1, 3, figsize=(18, 5))

#                       # Display original image
#                       axs[0].imshow(img, cmap='gray')
#                       axs[0].set_title('Original Image - Class 5')

#                       # Display modified image
#                       axs[1].imshow(m_img, cmap='gray')
#                       axs[1].set_title(f'Modified Image - Class {not_class}')

#                       # Display difference heatmap
#                       divider = make_axes_locatable(axs[2])
#                       cax = divider.append_axes("right", size="5%", pad=0.05)
#                       im = axs[2].imshow(diff, cmap='jet', interpolation='nearest')
#                       fig.colorbar(im, cax=cax, orientation="vertical")
#                       axs[2].set_title(f'Difference Heatmap - L2 Distance {int(get_distance(image, m_image))}')
#                       print(f'Heatmap saved to {heatmap_path}')
#                       plt.savefig(heatmap_path)
#                       plt.close()
#                     else:
#                       print(f'Heatmap already exists at {heatmap_path}')




# # image_path = 'mnist/search/518/0-0.png'  # Replace with the
# # m_image_path = 'mnist/search/518/0/optimal/1264-136-0-6-[0].png'  # Replace with the actual image file path

# # img = Image.open(image_path)
# # image = np.reshape(np.array(img), (-1, 28, 28, 1))

# # m_img = Image.open(m_image_path)
# # m_image = np.reshape(np.array(m_img), (-1, 28, 28, 1))

# # accepted, _, _, _   = Predictor().predict_generator(image, 5)
# # m_accepted, _, not_class, _ = Predictor().predict_generator(m_image, 5)

# # diff = ImageChops.difference(m_img, img)

# # fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# # # Display original image
# # axs[0].imshow(img, cmap='gray')
# # axs[0].set_title('Original Image - Class 5')

# # # Display modified image
# # axs[1].imshow(m_img, cmap='gray')
# # axs[1].set_title(f'Modified Image - Class {not_class}')

# # # Display difference heatmap
# # divider = make_axes_locatable(axs[2])
# # cax = divider.append_axes("right", size="5%", pad=0.05)
# # im = axs[2].imshow(diff, cmap='jet', interpolation='nearest')
# # fig.colorbar(im, cax=cax, orientation="vertical")
# # axs[2].set_title(f'Difference Heatmap - Distance {int(get_distance(image, m_image))}')

# # plt.savefig('heatmap_jet.png')

# print([[x] for x in range(8)])
# print([[x] for x in range(8-1, -1, -1)])
# process_count = 5
# process_list = zip(range(process_count), [process_count] * process_count)
# for i, threads in process_list:
#     print(i, threads)
# import nvidia_smi
# import pynvml

# # Initialize NVML
# pynvml.nvmlInit()

# # Get the number of GPUs
# device_count = pynvml.nvmlDeviceGetCount()

# # Iterate over each GPU
# for i in range(device_count):
#     # Get a handle to the GPU
#     handle = pynvml.nvmlDeviceGetHandleByIndex(i)

#     # Get the GPU's name
#     device_name = pynvml.nvmlDeviceGetName(handle)

#     # Get the GPU's memory info
#     meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

#     # Get the list of processes running on the GPU
#     procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

#     print(f"Device {i} - {device_name}")
#     print(f"Total memory: {meminfo.total}")
#     print(f"Free memory: {meminfo.free}")
#     print(f"Used memory: {meminfo.used}")
#     print(f"process count: {int(meminfo.free/1327497216)}")

#     # Print each process's PID and GPU memory usage
#     for proc in procs:
#         print(f"PID: {proc.pid}, Used GPU Memory: {proc.usedGpuMemory}")

# # Shut down NVML
# pynvml.nvmlShutdown()
# import nvidia_smi

# nvidia_smi.nvmlInit()

# handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

# info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

# print("Total memory:", info.total)
# print("Free memory:", info.free)
# print("Used memory:", info.used)

# nvidia_smi.nvmlShutdown()

#     model_memory_usage = 1327497216

#     nvidia_smi.nvmlInit()
#     handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
#     info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
#     process_count = int(info.free / model_memory_usage)
#     print(f"Process Count: {process_count}")