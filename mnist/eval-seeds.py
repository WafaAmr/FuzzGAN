import gzip
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from predictor import Predictor

# def compare_to_testset(seed_image, label):

# Specify the path to the dataset file
images_gz = '/home/upc/datasets/t10k-images-idx3-ubyte.gz'
labels_gz = '/home/upc/datasets/t10k-labels-idx1-ubyte.gz'
i = 0
total = 0

with gzip.open(images_gz, 'rb') as f:
    images = np.frombuffer(f.read(), np.uint8, offset=16)
with gzip.open(labels_gz, 'rb') as f:
    labels = np.frombuffer(f.read(), np.uint8, offset=8)

images = images.reshape(-1, 28, 28)
# labels = labels.reshape(-1, 1)

# Initialize a list of 10 empty lists, one for each class
classes = [[] for _ in range(10)]

# Iterate over the labels and images
for label, image in zip(labels, images):
    # Append the image to the appropriate class list
    classes[label].append(image)
# for i in range(10):
#     print(f'Class {i} has {len(classes[i])} samples')
#   l2 = np.linalg.norm(seed_image)
#   print(l2)
for idx in range(10):
  for test_image in classes[idx]:
    test_image = np.reshape(test_image, (-1, 28, 28, 1))
    not_class = None
    m_accepted = None
    nc = None
    conf = [[] for _ in range(10)]

    m_accepted, c, not_class, nc = Predictor().predict_generator(test_image, idx)
    if m_accepted and nc > 0:
      # print(f'Class {idx} - {c} - {not_class} - {nc}')
      i += 1
  print(f'Class {idx} has {i} samples')
  total += i
  i = 0
    #   ssi = ssim(seed_image, test_image, data_range=255)
    # #   l2_test = np.linalg.norm(test_image)
    #   # print(ssi)
    #   if ssi > 0.1:
    #     #   print(l2, l2_test)
    #       return test_image, ssi
    #   else:
    #     return None, None


# root_path = 'mnist/eval/HQ'


# model_folder = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
# for class_folder in model_folder:
#   class_path = os.path.join(root_path, class_folder)
#   seed_class = [f for f in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, f))]
#   for seed in seed_class:
#     seed_path = os.path.join(class_path, seed)
#     content = os.listdir(seed_path)
#     files = [f for f in content if os.path.isfile(os.path.join(seed_path, f))]
#     subfolders = [subfolder for subfolder in content if os.path.isdir(os.path.join(seed_path, subfolder))]

#     for file in files:
#     #   if file.startswith('testset'):
#     #         os.remove(os.path.join(seed_path, file))
#       if file.endswith('.png') and not file.startswith('testset') and not file.startswith('heatmap'):
#           img_path = os.path.join(seed_path, file)
#           img = Image.open(img_path)
#           img_array = np.array(img)
#           test_image, ssi = compare_to_testset(img_array, int(class_folder))
#         #   Image.fromarray(test_image).save(f'{seed_path}/testset-{ssi}.png')

#           if test_image is not None:
#               print(f'Found a similar image for {class_folder}, {ssi}, {img_path}')
#           else:
#               print(f'No similar image found for {class_folder}, {img_path}')
#           #     # break


  LQ = 618
  HQ = 6