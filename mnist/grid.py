import numpy as np
from PIL import Image
import os


DATASET = 'mnist/original_dataset/5/'
content = os.listdir(DATASET)

images = []
for file in content:
    if file.endswith(".png"):
        img = Image.open(DATASET + file)
        images.append(np.array(img))


image_grid = np.array([np.concatenate(images[i*33:(i+1)*33], axis=1) for i in range(3)])
final_image = np.concatenate(image_grid, axis=0)

# Convert the final numpy array back to an image
final_image = Image.fromarray(final_image)
final_image.save('mnist/original_dataset/5.png')