import os
import os.path as osp
import copy
import json
import numpy as np
from PIL import Image
from stylegan.renderer_v2 import Renderer
from config import STYLEGAN_INIT, SEARCH_LIMIT, STYLEMIX_SEED_LIMIT, SSIM_THRESHOLD, TEST_IMAGES, TEST_LABELS, INIT_PKL
from predictor import Predictor
from utils import validate_mutation, make_image
import gzip
import dnnlib
from multiprocessing import Process, Pool, set_start_method
from gan_inv.inversion import PTI
from gan_inv.lpips import util

class Fuzzgan:

    def __init__(self, seed_class=None, w0_seed=0, stylemix_seed=0, search_limit=SEARCH_LIMIT , process_count=None):
        self.state = STYLEGAN_INIT
        self.mix_state = None
        self.dataset = []
        self.search_limit = search_limit
        self.seed_class = seed_class
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
            w_load = state['params']['w_load'],  # w_load,
            w_load_seed = state['params']['w_load_seed'],  # w_load_seed,
            class_idx = state['params']['class_idx'],  # class_idx,
            mixclass_idx = state['params']['mixclass_idx'],  # mix_idx,
            stylemix_idx = state['params']['stylemix_idx'],  # stylemix_idx,
            stylemix_seed = state['params']['stylemix_seed'],  # stylemix_seed,
            img_normalize = state['params']['img_normalize'],
            to_pil = state['params']['to_pil'],
        )

        self.layers = [[x] for x in range(state['generator_params'].num_ws-1, -1, -1)]

        info =  copy.deepcopy(state['params'])

        return state, info

    def generate_dataset(self):
        seed_class = self.seed_class
        root = f"mnist/eval/testset/{seed_class}/"

        with gzip.open(TEST_IMAGES, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16)
        with gzip.open(TEST_LABELS, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)

        images = images.reshape(-1, 28, 28)
        images = np.pad(images, [(0,0), (2,2), (2,2)], 'constant', constant_values=0)

        labels = labels.reshape(-1, 1) # Convert images to int data type
        classes = [[] for _ in range(10)]
        state = self.state
        renderer = state['renderer']

        for label, image in zip(labels, images):
            # Append the image to the appropriate class list
            classes[int(label)].append(image)

        data_point = 0
        if self.process_count:
            step_size =  self.process_count
        else:
            step_size = 1

        renderer._render_impl(
            res = dnnlib.EasyDict(),
            pkl = INIT_PKL,
            w0_seeds = [[0, 1]]
        )

        percept = util.PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=True
        )


        for id, test_image in enumerate(classes[seed_class][:self.search_limit]):
            pti = PTI(renderer.G,percept)
            test_image = Image.fromarray(test_image)
            inversed_img, ws = pti.train(test_image, True)

            inversed_img = make_image(inversed_img)
            state["params"]["w_load"] = ws.detach().cpu().numpy(),
            state["params"]["w_load_seed"] = 0

            # inversed_img = inversed_img.crop((2, 2, inversed_img.width - 2, inversed_img.height - 2))

            for mixclass in range(10):
                found_mutation = False
                tried_all_layers = False

                state["params"]["mixclass_idx"] = mixclass
                self.stylemix_seed = 0
                while not found_mutation and not tried_all_layers and self.stylemix_seed < self.stylemix_seed_limit:

                    # require unique seed for each stylemix
                    if self.stylemix_seed == self.w0_seed:
                        self.stylemix_seed += 1
                    state["params"]["stylemix_seed"] = self.stylemix_seed

                    for idx, layer in enumerate(self.layers):
                        state["params"]["stylemix_idx"] = layer
                                state["params"]["w0_seeds"] = [[0, 1]]

                        m_digit, m_digit_info = self.generate_seed()
                        m_image = m_digit['generator_params'].image
                        m_image = m_image.crop((2, 2, m_image.width - 2, m_image.height - 2))
                        m_image_array = np.array(m_image)

                        m_accepted, confidence , m_predictions = Predictor().predict_datapoint(
                            np.reshape(m_image_array, (-1, 28, 28, 1)),
                            label
                        )

                        m_class = np.argsort(-m_predictions)[:1]
                        if not m_accepted and stylemix_cls == m_class:

                            valid_mutation, ssi, l2_distance, img_l2, m_img_l2 = validate_mutation(image_array, m_image_array)

                            if valid_mutation:
                                if not found_at_least_one:
                                    data_point += 1
                                    found_at_least_one = True

                                path = f"{root}{self.w0_seed}/"
                                seed_name = f"0-{second_cls}"
                                img_path = f"{path}/{seed_name}.png"
                                if not os.path.exists(img_path):
                                    os.makedirs(path, exist_ok=True)
                                    image.save(img_path)

                                    digit_info["l2_norm"] = img_l2
                                    with open(f"{path}/{seed_name}.json", 'w') as f:
                                        (json.dump(digit_info, f, sort_keys=True, indent=4))

                                found_mutation = True

                                m_digit_info["accepted"] = m_accepted.tolist()
                                m_digit_info["predicted-class"] = m_class.tolist()
                                m_digit_info["exp-confidence"] = float(confidence)
                                m_digit_info["predictions"] = m_predictions.tolist()
                                m_digit_info["ssi"] = float(ssi)
                                m_digit_info["l2_norm"] = m_img_l2
                                m_digit_info["l2_distance"] = l2_distance


                                m_path = f"{path}/{stylemix_cls}"
                                m_name = f"/{int(l2_distance)}-{int(ssi * 100)}-{self.stylemix_seed}-{stylemix_cls}-{layer[0]}-{m_class}"
                                os.makedirs(m_path, exist_ok=True)
                                with open(f"{m_path}/{m_name}.json", 'w') as f:
                                    (json.dump(m_digit_info, f, sort_keys=True, indent=4))
                                m_image.save(f"{m_path}/{m_name}.png")
                        if idx == len(self.layers) and found_mutation:
                            tried_all_layers = True
                            break
                    self.stylemix_seed += 1
                self.w0_seed += step_size




def run_fuzzgan(args):
    seed_class, w0_seed, process_count = args
    print('----------------------------------------------------')
    print(f"Process for class:{seed_class}:{w0_seed} Started")
    print('----------------------------------------------------')
    search_limit = int(SEARCH_LIMIT / process_count)
    Fuzzgan(seed_class=seed_class, w0_seed=w0_seed, search_limit=search_limit, process_count=process_count).generate_dataset()
    print('###################################################')
    print(f"Process for class:{seed_class}:{w0_seed} finished")
    print('###################################################')

if __name__ == "__main__":
    fuzzgan = Fuzzgan(seed_class=5)
    fuzzgan.generate_dataset()

    # args_list = []
    # process_count = 5

    # for seed_class in range(10):
    #     for w0_seed in range(process_count):
    #         args_list.append((seed_class, w0_seed, process_count))

    # print(f"Args List: {args_list}")
    # set_start_method('spawn')
    # with Pool(processes=process_count) as pool:
    #     pool.map(run_fuzzgan, args_list, chunksize=1)