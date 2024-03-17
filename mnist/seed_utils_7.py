import os
import os.path as osp
import copy
import json

import gradio as gr
import numpy as np
from PIL import Image

import dnnlib
from stylegan.renderer_v2 import Renderer
from config import CACHE_DIR, STYLEGAN_INIT, SEARCH_LIMIT
from predictor import Predictor
import pickle
from utils import get_distance, get_ssim

import threading
class Fuzzgan:

    def __init__(self, w0_seed=0, stylemix_seed=0, stylemix_seed_limit=5000, optimal_distance=1500, threads=None):
        self.state = STYLEGAN_INIT
        self.mix_state = None
        self.dataset = []
        self.search_limit = SEARCH_LIMIT
        # self.w0_seed = w0_seed
        # self.stylemix_seed = stylemix_seed
        self.stylemix_seed_limit = stylemix_seed_limit
        # self.optimal_distance = optimal_distance
        self.distance_limit = None
        self.layers = None
        self.threads = threads

    def generate_seed(self, state=None):
        if state is None:
            state = self.state

        state['renderer']._render_impl(
            res = state['generator_params'],  # res
            pkl = valid_checkpoints_dict[state['pretrained_weight']],  # pkl
            w0_seeds= state['params']['w0_seeds'],  # w0_seed,
            class_idx = state['params']['class_idx'],  # class_idx,
            mixclass_idx = state['params']['mixclass_idx'],  # mix_idx,
            stylemix_idx = state['params']['stylemix_idx'],  # stylemix_idx,
            stylemix_seed = state['params']['stylemix_seed'],  # stylemix_seed,
            img_normalize = state['params']['img_normalize'],
            to_pil = state['params']['to_pil'],
        )

        init_image = state['generator_params'].image
        state['images']['image_orig'] = init_image
        self.layers = [[x] for x in range(state['generator_params'].num_ws)]

        info =  copy.deepcopy(state['params'])

        return state, info

    def generate_dataset(self):
        # self.w0_seed = 205915 # 52895 # 52900
        self.w0_seed = 0
        self.stylemix_seed = 0
        digit_class = 2
        root = f"mnist/eval/LQ/{digit_class}/"

        data_point = 0
        if self.threads:
            step_size =  self.threads
        else:
            step_size = 1

        while data_point < self.search_limit:
        # while data_point < 1:
            state = self.state

            state["params"]["class_idx"] = digit_class
            state["params"]["w0_seeds"] = [[self.w0_seed, 1.0]]
            state["params"]["stylemix_idx"] = []
            state["params"]["stylemix_seed"] = 0

            digit, digit_info = self.generate_seed()


            label = digit["params"]["class_idx"]
            image = digit["images"]["image_orig"]
            image = image.crop((2, 2, image.width - 2, image.height - 2))
            image_array = np.array(image)

            accepted, confidence, predictions = Predictor().predict_datapoint(
                np.reshape(image_array, (-1, 28, 28, 1)),
                label
            )

            digit_info["accepted"] = accepted.tolist()
            digit_info["exp-confidence"] = float(confidence)
            digit_info["predictions"] = predictions.tolist()

            if accepted:
                _ , second_cls = np.argsort(-predictions)[:2]
                second_cls_confidence = predictions[second_cls]
                if second_cls_confidence:
                    data_point += 1
                    for stylemix_cls, cls_confidence in enumerate(predictions):
                        if stylemix_cls != label and cls_confidence:
                            # found mutation below threshold
                            found_mutation = False
                            found_L2 = False
                            found_SSIM = False
                            tried_all_layers = False

                            state["params"]["mixclass_idx"] = stylemix_cls
                            self.stylemix_seed = 0
                            while not found_mutation and not tried_all_layers and self.stylemix_seed < self.stylemix_seed_limit:

                                # require unique seed for each stylemix
                                if self.stylemix_seed == self.w0_seed:
                                    self.stylemix_seed += 1
                                state["params"]["stylemix_seed"] = self.stylemix_seed

                                for idx, layer in enumerate(self.layers):
                                    if idx == len(self.layers) - 1 and found_mutation:
                                        tried_all_layers = True
                                        break
                                    state["params"]["stylemix_idx"] = layer

                                    m_digit, m_digit_info = self.generate_seed()
                                    m_image = m_digit["images"]["image_orig"]
                                    m_image = m_image.crop((2, 2, m_image.width - 2, m_image.height - 2))
                                    m_image_array = np.array(m_image)


                                    m_accepted, confidence , m_predictions = Predictor().predict_datapoint(
                                        np.reshape(m_image_array, (-1, 28, 28, 1)),
                                        label
                                    )
                                    m_class = np.argsort(-m_predictions)[:1]
                                    m_digit_info["accepted"] = m_accepted.tolist()
                                    m_digit_info["predicted-class"] = m_class.tolist()
                                    m_digit_info["exp-confidence"] = float(confidence)
                                    m_digit_info["predictions"] = m_predictions.tolist()

                                    self.distance_limit = np.linalg.norm(image_array)


                                    distance = get_distance(image_array, m_image_array)
                                    ssim = get_ssim(image_array, m_image_array)


                                    if not m_accepted and stylemix_cls == m_class and (0 < distance < self.distance_limit or 90 < ssim):
                                        path = f"{root}{self.w0_seed}/"
                                        seed_name = f"0-{second_cls}"
                                        img_path = f"{path}/{seed_name}.png"
                                        if not os.path.exists(img_path):
                                            os.makedirs(path, exist_ok=True)
                                            image.save(img_path)

                                            digit_info["l2_norm"] = self.distance_limit
                                            with open(f"{path}/{seed_name}.json", 'w') as f:
                                                (json.dump(digit_info, f, sort_keys=True, indent=4))

                                        if 0 < distance < self.distance_limit and 95 < ssim:
                                            found_mutation = True
                                            optimal_path = f"{path}/{stylemix_cls}/optimal"
                                            optimal_name = f"/{int(distance)}-{int(ssim)}-{self.stylemix_seed}-{stylemix_cls}-{idx}-{m_class}"
                                            os.makedirs(optimal_path, exist_ok=True)
                                            with open(f"{optimal_path}/{optimal_name}.json", 'w') as f:
                                                (json.dump(m_digit_info, f, sort_keys=True, indent=4))
                                            m_image.save(f"{optimal_path}/{optimal_name}.png")
                                            print("Found optimal")
                                        elif distance < self.distance_limit * 0.5:
                                            found_L2 = True
                                            l2_path = f"{path}/{stylemix_cls}/l2"
                                            l2_name = f"/{int(distance)}-{int(ssim)}-{self.stylemix_seed}-{stylemix_cls}-{idx}-{m_class}"
                                            os.makedirs(l2_path, exist_ok=True)
                                            with open(f"{l2_path}/{l2_name}.json", 'w') as f:
                                                (json.dump(m_digit_info, f, sort_keys=True, indent=4))
                                            m_image.save(f"{l2_path}/{l2_name}.png")
                                            print("Found l2")
                                        elif 98 < ssim:
                                            found_SSIM = True
                                            ssim_path = f"{path}/{stylemix_cls}/ssim"
                                            ssim_name = f"/{int(distance)}-{int(ssim)}-{self.stylemix_seed}-{stylemix_cls}-{stylemix_cls}-{idx}-{m_class}"
                                            os.makedirs(ssim_path, exist_ok=True)
                                            with open(f"{ssim_path}/{ssim_name}.json", 'w') as f:
                                                (json.dump(m_digit_info, f, sort_keys=True, indent=4))
                                            m_image.save(f"{ssim_path}/{ssim_name}.png")
                                            print("Found SSIM")

                                        if not found_mutation:
                                            found_mutation = found_L2 and found_SSIM
                                self.stylemix_seed += 1
            self.w0_seed += step_size




valid_checkpoints_dict = {
    f.split('/')[-1].split('.')[0]: osp.join(CACHE_DIR, f)
    for f in os.listdir(CACHE_DIR)
    if (f.endswith('pkl') and osp.exists(osp.join(CACHE_DIR, f)))
}
print(f'\nFile under CACHE_DIR ({CACHE_DIR}):')
print(os.listdir(CACHE_DIR))
print('\nValid checkpoint file:')
print(valid_checkpoints_dict)

if __name__ == "__main__":
    fuzzgan = Fuzzgan(threads=1)
    fuzzgan.generate_dataset()
    # fuzzgan1 = Fuzzgan(w0_seed=0, threads=3)
    # fuzzgan1.generate_dataset()
    # fuzzgan2 = Fuzzgan(w0_seed=1, threads=3)
    # fuzzgan2.generate_dataset()
    # fuzzgan3 = Fuzzgan(w0_seed=3, threads=3)
    # fuzzgan3.generate_dataset()