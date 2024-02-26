from config import STYLEGAN_INIT
from seed_utils import init_images
from utils import get_distance
import numpy as np
import pickle

with open('dataset-pci-0.7.pkl', 'rb') as f:
    dataset = pickle.load(f)

image = dataset[0]["images"]["image_orig"]
seed = dataset[0]["params"]["seed"]
confidence = dataset[0]["predictor"]["confidence"]
not_class = dataset[0]["predictor"]["not_class"]
not_class_confidence = dataset[0]["predictor"]["not_class_confidence"]
print(seed, confidence, not_class, not_class_confidence)

# init_state_1 = STYLEGAN_INIT
# state_1 = init_images(STYLEGAN_INIT)
# image_1 = state_1['images']['image_show']
# image_1.save('1.png')

# init_state_2 = STYLEGAN_INIT
# init_state_2["params"]["seed"] = 423064
# init_state_2["params"]["class_idx"] = 5
# init_state_2["params"]["trunc_psi"] = 1
# state_2 = init_images(STYLEGAN_INIT)
# image_2 = state_2['images']['image_orig']
# image_2 = image_2.crop((2, 2, image_2.width - 2, image_2.height - 2))

# image_2.save('2.png')



# print(get_distance(np.array(image_1), np.array(image_2)))


