import os
import os.path as osp
import copy
import json
import numpy as np
from PIL import Image
from stylegan.renderer_v2 import Renderer
from config import STYLEGAN_INIT, SEARCH_LIMIT, STYLEMIX_SEED_LIMIT, INIT_PKL
from predictor import Predictor
from utils import validate_mutation
import pickle
import dnnlib
from multiprocessing import Process, set_start_method

class Fuzzgan:

    def __init__(self, w0_seed=0, stylemix_seed=0, search_limit=SEARCH_LIMIT , process_count=None):
        self.state = STYLEGAN_INIT
        self.mix_state = None
        self.dataset = []
        self.search_limit = search_limit
        self.w0_seed = w0_seed
        # self.stylemix_seed = stylemix_seed
        self.stylemix_seed_limit = STYLEMIX_SEED_LIMIT
        self.layers = None
        self.process_count = process_count
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
            img_normalize = state['params']['img_normalize'],
            to_pil = state['params']['to_pil'],
        )

        self.layers = [[7], [6], [5], [4], [4,5], [4,6], [5,6], [4,5,6], [3,4,5,6]]

        info =  copy.deepcopy(state['params'])

        return state, info

    def generate_dataset(self):
        digit_class = 5
        root = f"mnist/eval/rejected/{digit_class}/"

        data_point = 0
        if self.process_count:
            step_size =  self.process_count
        else:
            step_size = 1

        images = []
        for i in range(0, 10):
            data_point = 0
            self.w0_seed = 0
            while data_point < 90:
                state = self.state

                state["params"]["class_idx"] = i
                state["params"]["w0_seeds"] = [[self.w0_seed, 1.0]]
                state["params"]["stylemix_idx"] = []
                state["params"]["mixclass_idx"] = None
                state["params"]["stylemix_seed"] = None

                digit, digit_info = self.generate_seed()


                label = digit["params"]["class_idx"]
                image = digit['generator_params'].image
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

                if not accepted:
                    data_point += 1
                    if not os.path.exists(root):
                        os.makedirs(root, exist_ok=True)
                    images.append(image)
                    # image.save(f"{root}{self.w0_seed}.png")


                self.w0_seed += step_size
        image_grid = np.array([np.concatenate(images[i*30:(i+1)*30], axis=1) for i in range(30)])
        final_image = np.concatenate(image_grid, axis=0)

        # Convert the final numpy array back to an image
        final_image = Image.fromarray(final_image)

        # Save the final image
        final_image.save(f'{root}/final_image.jpg')

if __name__ == "__main__":
    fuzzgan = Fuzzgan(process_count=1)
    fuzzgan.generate_dataset()

    # # Set the start method to 'spawn'
    # set_start_method('spawn')
    # # Create processes
    # process_count = 5
    # process_list = zip(range(process_count), [process_count] * process_count)
    # processes = [Process(target=run_fuzzgan, args=(w0_seed, p_count)) for w0_seed, p_count in process_list]

    # # Start processes
    # for process in processes:
    #     process.start()

    # # Wait for all processes to complete
    # for process in processes:
    #     process.join()