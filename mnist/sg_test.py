from config import STYLEGAN_INIT
from seed_utils import init_images

state = init_images(STYLEGAN_INIT)

image = state['images']['image_show']

image.save('1.png')