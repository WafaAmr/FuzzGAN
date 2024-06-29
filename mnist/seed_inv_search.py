from PIL import Image, ImageChops
from mnist.config import INIT_PKL, TEST_IMAGES, TEST_LABELS, STYLEGAN_INIT, STYLEMIX_SEED_LIMIT
import dnnlib
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import gzip
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
from mnist.predictor import Predictor
from mnist.utils import validate_mutation
import random
from stylegan.renderer_v2 import Renderer
from gan_inv.inversion import PTI
from gan_inv.lpips import util



root = "mnist/eval/inverted2/5"

with gzip.open(TEST_IMAGES, 'rb') as f:
    images = np.frombuffer(f.read(), np.uint8, offset=16)
with gzip.open(TEST_LABELS, 'rb') as f:
    labels = np.frombuffer(f.read(), np.uint8, offset=8)


images = images.reshape(-1, 28, 28)
images = np.pad(images, [(0,0), (2,2), (2,2)], 'constant', constant_values=0)

classes = [[] for _ in range(10)]

for label, image in zip(labels, images):
    classes[label].append(image)

seed_class = 5
layers = [[x] for x in range(7, -1, -1)]
for id, image in enumerate(classes[seed_class][0:100]):

    renderer = Renderer(disable_timing=True)
    res = dnnlib.EasyDict()
    renderer._render_impl(
        res = res,
        pkl = INIT_PKL,
        w0_seeds = [[0, 1]],
    )
    percept = util.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=True
    )
    pti = PTI(renderer.G,percept)

    test_image = Image.fromarray(image)

    inversed_img, ws = pti.train(test_image, True)
    del percept
    del pti

    state = STYLEGAN_INIT
    renderer._render_impl(
                res = res,  # res
                pkl = INIT_PKL,
                w0_seeds= state['params']['w0_seeds'],
                w_load = ws.detach().cpu().numpy(),
                class_idx = seed_class,
                mixclass_idx = state['params']['mixclass_idx'],
                stylemix_idx = state['params']['stylemix_idx'],
                stylemix_seed = state['params']['stylemix_seed'],
                img_normalize = state['params']['img_normalize'],
                to_pil = state['params']['to_pil'],
            )
    inversed_img = res.image
    image = inversed_img.crop((2, 2, inversed_img.width - 2, inversed_img.height - 2))
    image_array = np.array(image)
    w = res.w
    path = f"{root}/{id}"
    img_path = f"{path}/{id}.png"
    if not os.path.exists(img_path):
        os.makedirs(path, exist_ok=True)
        image.save(img_path)

    m_classes = [{} for _ in range(10)]
    for stylemix_class in range(10):

        state["params"]["mixclass_idx"] = stylemix_class
        stylemix_seed = 1

        while stylemix_seed < STYLEMIX_SEED_LIMIT:

            # require unique seed for each stylemix
            # r_seed = random.randint(0, 350000)
            r_seed = stylemix_seed
            state["params"]["stylemix_seed"] = r_seed

            for idx, layer in enumerate(layers):
                state["params"]["stylemix_idx"] = layer

                renderer._render_impl(
                            res = res,  # res
                            pkl = INIT_PKL,
                            w0_seeds= state['params']['w0_seeds'],
                            w_load = ws.detach().cpu().numpy(),
                            class_idx = seed_class,
                            mixclass_idx = state['params']['mixclass_idx'],
                            stylemix_idx = state['params']['stylemix_idx'],
                            stylemix_seed = state['params']['stylemix_seed'],
                            img_normalize = state['params']['img_normalize'],
                            to_pil = state['params']['to_pil'],
                        )


                m_image = res.image
                m_image = m_image.crop((2, 2, m_image.width - 2, m_image.height - 2))
                m_image_array = np.array(m_image)

                m_accepted, confidence , m_predictions = Predictor().predict_datapoint(
                    np.reshape(m_image_array, (-1, 28, 28, 1)),
                    seed_class
                )
                m_class = np.argsort(-m_predictions)[:1]

                valid_mutation, ssi, l2_distance, img_l2, m_img_l2 = validate_mutation(image_array, m_image_array)
                if m_class != seed_class:
                    print(f'{id}:{stylemix_class}_{stylemix_seed}_{r_seed}:{layer[0]}, {int(l2_distance)}, {int(ssi*100)} {m_class}')

                if m_classes[m_class[0]] == {} or m_classes[m_class[0]]['ssi'] < ssi:
                    m_classes[m_class[0]]['image'] = m_image
                    m_classes[m_class[0]]['ssi'] = ssi
                    m_classes[m_class[0]]['l2'] = l2_distance
                    m_classes[m_class[0]]['stylemix_seed'] = stylemix_seed
                    m_classes[m_class[0]]['stylemix_idx'] = layer[0]



            stylemix_seed += 1
    for m_class, data in enumerate(m_classes):
        if data != {} and m_class != seed_class:
            ssi = data['ssi']
            if ssi > .6:
                m_image = data['image']
                l2_distance = data['l2']
                stylemix_seed = data['stylemix_seed']
                layer = data['stylemix_idx']
                m_path = f"{path}/{m_class}"
                m_name = f"/{int(l2_distance)}-{int(ssi * 100)}-{stylemix_seed}-{layer}-{m_class}"
                os.makedirs(m_path, exist_ok=True)
                m_image.save(f"{m_path}/{m_name}.png")
    del renderer
    del res