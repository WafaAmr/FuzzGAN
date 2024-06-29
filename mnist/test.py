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
# import numpy as np
# x = np.array([[1,2],[3,4]])
# print(x[:, 0])

# import numpy as np
# import torch

# # Suppose `array` is your NumPy array with shape (x, y)
# array = np.random.rand(5, 10)  # Replace with your actual array

# # Convert the NumPy array to a PyTorch tensor
# tensor = torch.from_numpy(array)

# print(tensor.shape)  #
# import itertools

# # Generate all combinations of numbers from 0 to 7
# combinations = list(itertools.product(range(8), repeat=8))

# # Print the combinations
# for combo in combinations:
#     print(combo)
# import itertools

# # Generate all possible combinations
# combinations = [list(comb) for r in range(9) for comb in itertools.combinations(range(8), r)]

# for comb in combinations:
#     if 0 not in comb and 1 not in comb and 2 not in comb and 3 in comb and 7 not in comb:
#         print(comb)


# def midpoint_and_precision(num1, num2):
#     """
#     Calculate the midpoint between two numbers and determine the precision based on the number of decimal places.

#     Parameters:
#     num1 (int or float): The first number.
#     num2 (int or float): The second number.

#     Returns:
#     tuple: A tuple containing the midpoint and the precision as an integer.
#     """
#     # Calculate midpoint
#     midpoint = (num1 + num2) / 2

#     # Convert midpoint to string to analyze decimal places
#     midpoint_str = f"{midpoint:.10f}"  # Convert to string with up to 10 decimal places

#     # Strip trailing zeros and split at the decimal point
#     midpoint_str = midpoint_str.rstrip('0')
#     if '.' in midpoint_str:
#         precision = len(midpoint_str.split('.')[1])
#     else:
#         precision = 0  # No decimal places if no decimal point is present

#     return midpoint, precision

# # Example usage
# result = midpoint_and_precision(3.145, 2.1)
# print("Midpoint:", result[0], "Precision:", result[1])
# import numpy as np
# from PIL import Image
# img = np.load('mnist/ref_digit/cinque_rp.npy')

# print(img.dtype)
# # Reshape the array to 2D and convert it to uint8
# img = img.reshape(28, 28).astype(np.uint8)

# # Convert the numpy array to an image
# img = Image.fromarray(img)

# # Save the image
# img.save('mnist/ref_digit/save.png')
import os, json
import numpy as np
import matplotlib.pyplot as plt

DATASET = 'mnist/original_dataset/5-HQ/'
content = os.listdir(DATASET)

x_test = []
for file in content:
    if file.endswith('.json'):
        with open(DATASET + file, 'r') as f:
            params = json.load(f)
            predictions = params["predictions"]
            _ , second_cls = np.argsort(-np.array(predictions))[:2]
            con = predictions[second_cls]
            # print(con)
            x_test.append(con)
boxprops = dict(facecolor='lightgray', color='black', linewidth=1)
plt.figure(figsize=(10, 2))
plt.grid(True, alpha=0.25)
plt.boxplot(x_test, vert=False, meanline=True, showmeans=True, patch_artist=True, boxprops=boxprops, zorder=3)
plt.xlabel('Confidence of the second most probable class for the digit 5')
plt.tight_layout()
plt.savefig('mnist/hq-second_class_confidence.png')