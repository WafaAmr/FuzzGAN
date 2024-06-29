from PIL import Image
from stylegan.renderer_v2 import Renderer
from mnist.config import INIT_PKL, STYLEGAN_INIT
import dnnlib
from PIL import ImageOps
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from PIL import ImageChops
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch


def make_image(inversed_img, img_scale_db = 0):
    inversed_img = inversed_img.squeeze()  # Remove singleton dimensions
    inversed_img = inversed_img / inversed_img.norm(float('inf'), dim=[0,1], keepdim=True).clip(1e-8, 1e8)
    inversed_img = inversed_img * (10 ** (img_scale_db / 20))
    inversed_img = (inversed_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    inversed_img = Image.fromarray(inversed_img.cpu().numpy())
    return inversed_img

renderer = Renderer(disable_timing=True)
state = STYLEGAN_INIT
w_path = "mnist/inv_2/5/1"
w_load = np.load(f'{w_path}-inv.npy')
inv_image = Image.open(f'{w_path}-inv.png')
og = Image.open(f'{w_path}-og.png')
res = dnnlib.EasyDict()
renderer._render_impl(
            res = res,  # res
            pkl = INIT_PKL,
            w0_seeds= [[0, 1]],
            # w_load = w_load,
            # noise_mode='none',
            # # class_idx = 5,
            # # mixclass_idx = state['params']['mixclass_idx'],
            # # stylemix_idx = state['params']['stylemix_idx'],
            # # stylemix_seed = state['params']['stylemix_seed'],
            # force_fp32 = True,
            # img_normalize = state['params']['img_normalize'],
            # to_pil = state['params']['to_pil'],
        )
# image = res.image
w_load = torch.from_numpy(w_load).to(torch.device('cuda'), dtype=torch.float32)
print(w_load.shape)
G = renderer.G
latent_path = []

img_gen = G.synthesis(w_load, noise_mode='const')
image = make_image(img_gen)



l2_saved = np.linalg.norm(np.array(og) - np.array(inv_image))
l2_gen = np.linalg.norm(np.array(og) - np.array(image))
l2_gen_saved = np.linalg.norm(np.array(image) - np.array(inv_image))

diff = ImageChops.difference(inv_image, image)
fig, axs = plt.subplots(1, 4, figsize=(18, 5))
axs[0].imshow(og, cmap='gray')
axs[0].set_title('Original Image')

axs[1].imshow(inv_image, cmap='gray')
axs[1].set_title(f'Inverted Image Saved L2: {int(l2_saved)}')

axs[2].imshow(image, cmap='gray')
axs[2].set_title(f'Inverted Image Generated L2: {int(l2_gen)}')

# Display difference heatmap
divider = make_axes_locatable(axs[3])
cax = divider.append_axes("right", size="5%", pad=0.05)
im = axs[3].imshow(diff, cmap='jet', interpolation='nearest')
fig.colorbar(im, cax=cax, orientation="vertical")
axs[3].set_title(f'Difference Heatmap L2: {int(l2_gen_saved)}')
plt.savefig('render.png')
plt.close()

