from predictor import Predictor
from utils import get_distance
from PIL import Image
import numpy as np
import os

# Read image from file
folder_path = 'mnist/try/'
folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]


image_path = 'mnist/search/52922/0-9.png'  # Replace with the
image = np.array(Image.open(image_path))
image = np.reshape(image, (-1, 28, 28, 1))


# Read image from file
m_image_path = 'mnist/search3/52922/0-9.png'  # Replace with the actual image file path
m_image = np.array(Image.open(m_image_path))
m_image = np.reshape(m_image, (-1, 28, 28, 1))

accepted, _, _, _   = Predictor().predict_generator(image, 5)
m_accepted, _, _, _ = Predictor().predict_generator(m_image, 5)

print(f"Original: {accepted}, Mutated: {m_accepted}, Distance: {get_distance(image, m_image)}")