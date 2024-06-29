import numpy as np
from PIL import Image
import os
import json

# root = 'runs/HQ/DJ-HQ/'
root = 'runs/fix/R/LQ-G-10gen'

max_colunms = 15
max_rows = 40
pairs = os.listdir(f'{root}/inds')
images = []
for pair in pairs:
    if pair.startswith('ind'):
        with open(f'{root}/inds/{pair}', 'r') as f:
            ind = json.load(f)
            m1 = ind['m1']
            m2 = ind['m2']
            with open(f'{root}/archive/mbr{m1}.json', 'r') as f:
                m1_json = json.load(f)
                m1_misbehaviour = m1_json["misbehaviour"]
                with open(f'{root}/archive/mbr{m2}.json', 'r') as f:
                    m2_json = json.load(f)
                    m2_misbehaviour = m2_json["misbehaviour"]
                    if m1_misbehaviour or m2_misbehaviour:
                        img1 = Image.open(f'{root}/archive/mbr{m1}.png')
                        img2 = Image.open(f'{root}/archive/mbr{m2}.png')

                        img_pair = np.concatenate([np.array(img1), np.array(img2)], axis=1)
                        # img_pair = img_pair.reshape(-1, 56, 28)
                        img_pair = np.pad(img_pair, ((0,0), (2,2), (0,0)), 'constant', constant_values=255)
                        images.append(img_pair)
                        # img_pair = Image.fromarray(img_pair)
                        # img_pair.save(f'{root}/pairs/{pair}.png')




# Calculate the number of rows and columns
if len(images) < max_colunms:
    num_rows = 1
    num_cols = len(images)
else:
    num_rows = len(images) // max_colunms + (len(images) % max_colunms != 0)
    num_cols = max_colunms

# Create a list to store the rows
rows = []

for i in range(num_rows):
    # Get the images for this row
    row_images = images[i*max_colunms:(i+1)*max_colunms]

    # If this is the last row and it has less than 15 images, add white images
    while len(row_images) < num_cols:
        # Create a white image of the same size and type as the other images
        white_image = np.ones_like(row_images[0]) * 255
        row_images.append(white_image)

    # Concatenate the images in this row and add the result to the list of rows
    rows.append(np.concatenate(row_images, axis=1))

# Concatenate the rows to get the final image
if len(rows) > max_rows:
    nuber_of_images = len(rows)//max_rows + (len(rows) % max_rows != 0)
    for i in range(nuber_of_images):
        final_image = np.concatenate(rows[i*max_rows:(i+1)*max_rows], axis=0)
        final_image = Image.fromarray(final_image.astype(np.uint8))
        final_image.save(f'{root}/{root.split("/")[-1]}_{i}.png')
else:
    final_image = np.concatenate(rows, axis=0)
    # Convert the final numpy array back to an image
    final_image = Image.fromarray(final_image.astype(np.uint8))
    final_image.save(f'{root}/{root.split("/")[-1]}.png')