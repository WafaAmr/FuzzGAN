import os, sys
import os.path as osp
import numpy as np
from PIL import Image
from stylegan.renderer_v2 import Renderer
from config import STYLEGAN_INIT, SEARCH_LIMIT, STYLEMIX_SEED_LIMIT, SSIM_THRESHOLD, INIT_PKL
from predictor import Predictor
from utils import validate_mutation
import dnnlib
from stylegan.renderer_v2 import Renderer
import random

def dlmimic(w_path):

    seed_class = int(os.path.basename(os.path.dirname(w_path)))
    root = f"mnist/eval/final/inv-LQ/{seed_class}/"
    layers = [[3,4,5,6,7]]
    con_class = {0:[6,8], 1:[4,8], 2:[8,4], 3:[5,2], 4:[9,6], 5:[3,5,6,8], 6:[0,4,8], 7:[2,4], 8:[6,9], 9:[4,8]}

    state = STYLEGAN_INIT
    res = dnnlib.EasyDict()
    w_load = np.load(w_path)
    renderer = Renderer()
    renderer._render_impl(
                res = res,  # res
                pkl = INIT_PKL,
                w0_seeds= state['params']['w0_seeds'],
                w_load = w_load,
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

    file_id = osp.basename(w_path).split('-')[0]
    img_root = osp.join(root, file_id)
    img_path = osp.join(img_root, 'inv.png')

    print(f"Saving {img_path}")
    if not osp.exists(img_path):
        os.makedirs(img_root, exist_ok=True)
        image.save(img_path)

        m_classes = [{} for _ in range(10)]
        # for stylemix_class in con_class[seed_class]:
        found_mutation = False
        tried_all_layers = False

        # state["params"]["mixclass_idx"] = stylemix_class
        stylemix_seed = 1

        while not found_mutation and not tried_all_layers and stylemix_seed < STYLEMIX_SEED_LIMIT:
            stylemix_class = random.choice(con_class[seed_class])
            state["params"]["mixclass_idx"] = stylemix_class
            # require unique seed for each stylemix
            r_seed = random.randint(0, 350000)
            # r_seed = stylemix_seed
            state["params"]["stylemix_seed"] = r_seed

            for idx, layer in enumerate(layers):
                state["params"]["stylemix_idx"] = layer

                renderer._render_impl(
                            res = res,  # res
                            pkl = INIT_PKL,
                            w0_seeds= state['params']['w0_seeds'],
                            w_load = w,
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

                if not m_accepted:

                    valid_mutation, ssi, l2_distance, img_l2, m_img_l2 = validate_mutation(image_array, m_image_array)
                    print(f"SSI: {round(ssi*100, 2)}, L2: {round(l2_distance/img_l2, 2)}, Valid: {valid_mutation}")

                    if valid_mutation:
                        found_mutation = True

                        m_path = f"{img_root}/{stylemix_class}"
                        m_name = f"/{int(l2_distance)}-{int(ssi * 100)}-{stylemix_seed}-{stylemix_class}-{layer[0]}-{m_class}"
                        os.makedirs(m_path, exist_ok=True)
                        # with open(f"{m_path}/{m_name}.json", 'w') as f:
                        #     (json.dump(m_digit_info, f, sort_keys=True, indent=4))
                        m_image.save(f"{m_path}/{m_name}.png")
                if idx == len(layers) and found_mutation:
                    tried_all_layers = True
                    break
            print(f"Stylemix Seed: {stylemix_seed}")
            stylemix_seed += 1
        del renderer
        del res

if __name__ == "__main__":

  input_path = sys.argv[1]

  print(f"Reading from {input_path}")
  dlmimic(input_path)
