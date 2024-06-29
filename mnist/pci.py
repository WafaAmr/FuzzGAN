import os
import os.path as osp
import copy
import json

import gradio as gr
import numpy as np
from PIL import Image

import dnnlib
from stylegan.renderer_v2 import Renderer
from config import STYLEGAN_INIT, SEARCH_LIMIT, INIT_PKL
from predictor import Predictor
import pickle
from utils import get_distance

import threading
class Fuzzgan:

    def __init__(self, w0_seed=0, stylemix_seed=0, stylemix_seed_limit=5000, optimal_distance=1500, distance_limit=3000, threads=None):
        self.state = STYLEGAN_INIT
        self.mix_state = None
        self.dataset = []
        self.search_limit = SEARCH_LIMIT
        # self.w0_seed = w0_seed
        # self.stylemix_seed = stylemix_seed
        self.stylemix_seed_limit = stylemix_seed_limit
        self.optimal_distance = optimal_distance
        self.distance_limit = distance_limit
        self.layers = None
        self.threads = threads
        self.state['renderer'] = Renderer()

    def generate_seed(self, state=None):
        if state is None:
            state = self.state

        state['renderer']._render_impl(
            res = state['generator_params'],  # res
            pkl = INIT_PKL,  # pkl
            w0_seeds= state['params']['w0_seeds'],  # w0_seed,
            class_idx = state['params']['class_idx'],  # class_idx,
            mixclass_idx = state['params']['mixclass_idx'],  # mix_idx,
            stylemix_idx = state['params']['stylemix_idx'],  # stylemix_idx,
            stylemix_seed = state['params']['stylemix_seed'],  # stylemix_seed,
            trunc_psi = state['params']['trunc_psi'],  # trunc_psi,
            trunc_cutoff = 8,
            img_normalize = state['params']['img_normalize'],
            to_pil = state['params']['to_pil'],
        )

        # init_image = state['generator_params'].image
        # state['images']['image_orig'] = init_image
        # self.layers = [[x] for x in range(state['generator_params'].num_ws)]

        info =  copy.deepcopy(state['params'])

        return state, info

    def generate_dataset(self):
        self.w0_seed = 0 # 53029 # 53034
        self.stylemix_seed = 0
        trunc_psi = 1

        data_point = 0
        if self.threads:
            step_size =  self.threads
        else:
            step_size = 1
        image = None
        while data_point < self.search_limit:
        # while data_point < 1:
            state = self.state
            if image is not None:
                old_image = image
            state["params"]["trunc_psi"] = trunc_psi
            state["params"]["w0_seeds"] = [[self.w0_seed, 1.0]]
            state["params"]["class_idx"] = 5
            state["params"]["stylemix_idx"] = []
            state["params"]["stylemix_seed"] = 0


            digit, digit_info = self.generate_seed()


            label = digit["params"]["class_idx"]
            image = digit['generator_params'].image
            image = image.crop((2, 2, image.width - 2, image.height - 2))
            image_array = np.reshape(np.array(image), (-1, 28, 28, 1))

            accepted, confidence, predictions = Predictor().predict_datapoint(image_array, label)
            print(predictions)

            digit_info["accepted"] = accepted.tolist()
            digit_info["exp-confidence"] = float(confidence)
            digit_info["predictions"] = predictions.tolist()

            if accepted:
                trunc_psi  -= 0.01
                # if trunc_psi <= 2:
                #     trunc_psi  += 0.1
                # else:
                #     trunc_psi = 1
                #     self.w0_seed += 1
            else:
                image.save(f"mnist/pci-2/{self.w0_seed}-{round(trunc_psi, 2)}.png")
                old_image.save(f"mnist/pci-2/{self.w0_seed}-{round(trunc_psi + .01, 2)}.png")
                # break
                trunc_psi = 1
                self.w0_seed += 10
                self.state['renderer'] = Renderer()






if __name__ == "__main__":
    fuzzgan = Fuzzgan(threads=5)
    fuzzgan.generate_dataset()
    # fuzzgan1 = Fuzzgan(w0_seed=0, threads=3)
    # fuzzgan1.generate_dataset()
    # fuzzgan2 = Fuzzgan(w0_seed=1, threads=3)
    # fuzzgan2.generate_dataset()
    # fuzzgan3 = Fuzzgan(w0_seed=3, threads=3)
    # fuzzgan3.generate_dataset()