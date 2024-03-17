import os
import os.path as osp
import copy
import json
import shutil


import gradio as gr
import numpy as np
from PIL import Image

import dnnlib
from stylegan.renderer_v2 import Renderer
from config import CACHE_DIR, STYLEGAN_INIT, SEARCH_LIMIT, STYLEMIX_SEED_LIMIT, SSIM_THRESHOLD
from predictor import Predictor
import pickle
from utils import get_distance, validate_mutation

import threading
class Fuzzgan:

    def __init__(self, w0_seed=0, stylemix_seed=0, threads=None):
        self.state = STYLEGAN_INIT
        self.mix_state = None
        self.dataset = []
        self.search_limit = SEARCH_LIMIT
        # self.w0_seed = w0_seed
        # self.stylemix_seed = stylemix_seed
        self.stylemix_seed_limit = STYLEMIX_SEED_LIMIT
        self.layers = None
        self.threads = threads

    def init_network(self, state=None):
        if state is None:
            state = self.state

        state['renderer']._render_impl(
            res = state['generator_params'],  # res
            pkl = valid_checkpoints_dict[state['pretrained_weight']],  # pkl
            w0_seeds= state['params']['w0_seeds'],  # w0_seed,
            class_idx = state['params']['class_idx'],  # class_idx,
            mixclass_idx = state['params']['mixclass_idx'],  # mix_idx,
            stylemix_idx = state['params']['stylemix_idx'],  # stylemix_idx,
            patch_idxs = state['params']['patch_idxs'],  # patch_idxs,
            stylemix_seed = state['params']['stylemix_seed'],  # stylemix_seed,
            img_normalize = state['params']['img_normalize'],
            to_pil = state['params']['to_pil'],
        )

        self.layers = [[x] for x in range(state['generator_params'].num_ws)]

        info =  copy.deepcopy(state['params'])
        return state, info

    def generate_dataset(self):
        self.w0_seed = 500
        digit_class = 5
        root = f"mnist/eval/test3/{digit_class}/"

        data_point = 0
        if self.threads:
            step_size =  self.threads
        else:
            step_size = 1

        # while data_point < self.search_limit:
        while data_point < 10:
            state = self.state

            state["params"]["class_idx"] = digit_class
            state["params"]["w0_seeds"] = [[self.w0_seed, 1.0]]
            state["params"]["stylemix_idx"] = []
            state["params"]["mixclass_idx"] = None
            state["params"]["stylemix_seed"] = None
            state["params"]["patch_idxs"] = None

            digit, digit_info = self.init_network()


            label = digit["params"]["class_idx"]
            image = digit["generator_params"].image
            image = image.crop((2, 2, image.width - 2, image.height - 2))
            # os.makedirs(f'{root}seed/', exist_ok=True)
            # image.save(f"{root}seed/{self.w0_seed}.png")
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
                        # if cls_confidence:
                            # found mutation below threshold
                            found_mutation = False
                            tried_all_layers = False
                            best_found = {}

                            state["params"]["mixclass_idx"] = stylemix_cls
                            self.stylemix_seed = 0
                            while not found_mutation and not tried_all_layers and self.stylemix_seed < self.stylemix_seed_limit:

                                # require unique seed for each stylemix
                                if self.stylemix_seed == self.w0_seed:
                                    self.stylemix_seed += 1
                                state["params"]["stylemix_seed"] = self.stylemix_seed
                                state["params"]["patch_idxs"] = self.layers

                                m_digit, m_digit_info = self.init_network()

                                generator_params = m_digit["generator_params"]
                                patch_stylemix = generator_params.patch_idxs

                                for key, layer in patch_stylemix.items():

                                    m_image = generator_params[key].image
                                    m_image = m_image.crop((2, 2, m_image.width - 2, m_image.height - 2))
                                    m_image_array = np.array(m_image)


                                    m_accepted, confidence , m_predictions = Predictor().predict_datapoint(
                                        np.reshape(m_image_array, (-1, 28, 28, 1)),
                                        label
                                    )
                                    m_class = np.argsort(-m_predictions)[:1]
                                    print(layer,f"m_accepted: {m_accepted}, m_predictions: {m_predictions}")
                                    if not m_accepted:

                                        valid_mutation, ssi, l2_distance, img_l2, m_img_l2 = validate_mutation(image_array, m_image_array)
                                        print(layer, f"valid_mutation: {valid_mutation}, layer: {layer} ssi: {round(ssi * 100, 1)}, l2_distance: {int(l2_distance)}, img_l2: {int(img_l2)}, m_img_l2: {int(m_img_l2)}, ratio: {round(img_l2/m_img_l2,3)}")

                                        path = f"{root}{self.w0_seed}/"
                                        seed_name = f"0-{second_cls}"
                                        img_path = f"{path}/{seed_name}.png"
                                        if not os.path.exists(img_path):
                                            os.makedirs(path, exist_ok=True)
                                            image.save(img_path)

                                            digit_info["l2_norm"] = img_l2
                                            with open(f"{path}/{seed_name}.json", 'w') as f:
                                                (json.dump(digit_info, f, sort_keys=True, indent=4))

                                        if valid_mutation:
                                            found_mutation = True

                                            m_digit_info["accepted"] = m_accepted.tolist()
                                            m_digit_info["predicted-class"] = m_class.tolist()
                                            m_digit_info["exp-confidence"] = float(confidence)
                                            m_digit_info["predictions"] = m_predictions.tolist()
                                            m_digit_info["ssi"] = float(ssi)
                                            m_digit_info["l2_norm"] = m_img_l2
                                            m_digit_info["l2_distance"] = l2_distance
                                            m_digit_info["stylemix_idx"] = layer


                                            m_path = f"{path}/{stylemix_cls}"
                                            m_name = f"/{int(l2_distance)}-{int(ssi * 100)}-{self.stylemix_seed}-{stylemix_cls}-{layer[0]}-{m_class}"
                                            os.makedirs(m_path, exist_ok=True)
                                            with open(f"{m_path}/{m_name}.json", 'w') as f:
                                                (json.dump(m_digit_info, f, sort_keys=True, indent=4))
                                            m_image.save(f"{m_path}/{m_name}.png")
                                        else:
                                            if not best_found or l2_distance < best_found["l2_distance"]:
                                                # print(f"valid_mutation: {valid_mutation}, layer: {layer} ssi: {round(ssi * 100, 1)}, l2_distance: {int(l2_distance)}, img_l2: {int(img_l2)}, m_img_l2: {int(m_img_l2)}, ratio: {round(img_l2/m_img_l2,3)}")
                                                best_found =  copy.deepcopy(m_digit_info)
                                                best_found["accepted"] = m_accepted.tolist()
                                                best_found["predicted-class"] = m_class.tolist()
                                                best_found["exp-confidence"] = float(confidence)
                                                best_found["predictions"] = m_predictions.tolist()
                                                best_found["ssi"] = float(ssi)
                                                best_found["l2_norm"] = m_img_l2
                                                best_found["l2_distance"] = l2_distance
                                                best_found["stylemix_idx"] = layer
                                print("---------------------------------------")
                                self.stylemix_seed += 1
                            if not found_mutation:
                                l2_distance = best_found["l2_distance"]
                                ssi = best_found["ssi"]
                                stylemix_seed = best_found["stylemix_seed"]
                                stylemix_cls = best_found["mixclass_idx"]
                                layer = best_found["stylemix_idx"]
                                m_class = best_found["predicted-class"]

                                m_path = f"{path}/{stylemix_cls}/bf/"
                                m_name = f"/{int(l2_distance)}-{int(ssi * 100)}-{stylemix_seed}-{stylemix_cls}-{layer[0]}-{m_class}"
                                os.makedirs(m_path, exist_ok=True)
                                with open(f"{m_path}/{m_name}.json", 'w') as f:
                                    (json.dump(best_found, f, sort_keys=True, indent=4))
                                m_image.save(f"{m_path}/{m_name}.png")
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