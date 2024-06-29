import os
import os.path as osp
import copy
import json
import numpy as np
import dnnlib
from stylegan.renderer_v2 import Renderer
from config import STYLEGAN_INIT, SEARCH_LIMIT, STYLEMIX_SEED_LIMIT, INIT_PKL
from predictor import Predictor
from utils import validate_mutation
from multiprocessing import Pool, set_start_method

class mimicry:

    def __init__(self, class_idx=None, w0_seed=0, stylemix_seed=0, search_limit=SEARCH_LIMIT , step_size=1):
        self.state = STYLEGAN_INIT
        self.class_idx = class_idx
        self.w0_seed = w0_seed
        self.stylemix_seed = stylemix_seed
        self.search_limit = search_limit
        self.stylemix_seed_limit = STYLEMIX_SEED_LIMIT
        # self.layers = [[7], [6], [5], [4], [3], [5,6], [3,4], [3,4,5,6]]
        self.layers = [[3,4,5,6]]
        # self.con_class = {0:[6,8], 1:[4,8], 2:[8,4], 3:[5,2], 4:[9,6], 5:[6,8], 6:[0,4], 7:[2,4], 8:[6,9], 9:[4,8]}
        self.con_class = {0:[5], 1:[2], 2:[6], 3:[8], 5:[3], 6:[2], 7:[1,9], 8:[1,3], 9:[4,7]}

        self.step_size = step_size
        self.sg = Renderer()
        self.res = dnnlib.EasyDict()

    def render(self, state=None):
        if state is None:
            state = self.state

        self.sg._render_impl(
            res = self.res,  # res
            pkl = INIT_PKL,  # pkl
            w0_seeds= state['params']['w0_seeds'],  # w0_seed,
            class_idx = state['params']['class_idx'],  # class_idx,
            mixclass_idx = state['params']['mixclass_idx'],  # mix_idx,
            stylemix_idx = state['params']['stylemix_idx'],  # stylemix_idx,
            stylemix_seed = state['params']['stylemix_seed'],  # stylemix_seed,
            img_normalize = state['params']['img_normalize'],
            to_pil = state['params']['to_pil'],
        )

        info =  copy.deepcopy(state['params'])

        return state, info

    def search(self):
        root = f"mnist/eval/final/LQ-100/{self.class_idx}/"
        frontier_seed_count = 0
        while frontier_seed_count < self.search_limit:
            state = self.state

            state["params"]["class_idx"] = self.class_idx
            state["params"]["w0_seeds"] = [[self.w0_seed, 1.0]]
            state["params"]["stylemix_idx"] = []
            state["params"]["mixclass_idx"] = None
            state["params"]["stylemix_seed"] = None

            digit, digit_info = self.render()


            label = digit["params"]["class_idx"]
            image = self.res.image
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
                found_at_least_one = False
                _ , second_cls = np.argsort(-predictions)[:2]
                second_cls_confidence = predictions[second_cls]
                if not second_cls_confidence:
                    for stylemix_cls in self.con_class[self.class_idx]:
                        # found mutation below threshold
                        found_mutation = False
                        # tried_all_layers = False

                        state["params"]["mixclass_idx"] = stylemix_cls
                        self.stylemix_seed = 0
                        # while not found_mutation and not tried_all_layers and self.stylemix_seed < self.stylemix_seed_limit:
                        while not found_mutation  and self.stylemix_seed < self.stylemix_seed_limit:

                            # require unique seed for each stylemix
                            if self.stylemix_seed == self.w0_seed:
                                self.stylemix_seed += 1
                            state["params"]["stylemix_seed"] = self.stylemix_seed

                            for idx, layer in enumerate(self.layers):
                                state["params"]["stylemix_idx"] = layer

                                m_digit, m_digit_info = self.render()
                                m_image = self.res.image
                                m_image = m_image.crop((2, 2, m_image.width - 2, m_image.height - 2))
                                m_image_array = np.array(m_image)


                                m_accepted, confidence , m_predictions = Predictor().predict_datapoint(
                                    np.reshape(m_image_array, (-1, 28, 28, 1)),
                                    label
                                )

                                m_class = np.argsort(-m_predictions)[:1]

                                if not m_accepted:

                                    valid_mutation, ssi, l2_distance, img_l2, m_img_l2 = validate_mutation(image_array, m_image_array)
                                    print(f"SSI: {round(ssi*100, 2)}, L2: {round(l2_distance/img_l2, 2)}, Valid: {valid_mutation}")



                                    if valid_mutation:
                                        if not found_at_least_one:
                                            frontier_seed_count += 1
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
                                #     else:
                                #         if save_BF and (not best_found or ssi > best_found["ssi"]):
                                #             best_found =  copy.deepcopy(m_digit_info)
                                #             best_found["accepted"] = m_accepted.tolist()
                                #             best_found["predicted-class"] = m_class.tolist()
                                #             best_found["exp-confidence"] = float(confidence)
                                #             best_found["predictions"] = m_predictions.tolist()
                                #             best_found["ssi"] = float(ssi)
                                #             best_found["l2_norm"] = m_img_l2
                                #             best_found["l2_distance"] = l2_distance
                                # if idx == len(self.layers) and found_mutation:
                                #     tried_all_layers = True
                                #     break
                                # print(con_class)
                            self.stylemix_seed += 1
                        # if save_BF and not found_mutation and best_found:
                        #     frontier_seeds.add(self.w0_seed)
                        #     if not os.path.exists(img_path):
                        #         os.makedirs(path, exist_ok=True)
                        #         image.save(img_path)

                            #     with open(f"{path}/{seed_name}.json", 'w') as f:
                            #         (json.dump(digit_info, f, sort_keys=True, indent=4))

                            # l2_distance = best_found["l2_distance"]
                            # ssi = best_found["ssi"]
                            # stylemix_seed = best_found["stylemix_seed"]
                            # stylemix_cls = best_found["mixclass_idx"]
                            # layer = best_found["stylemix_idx"]
                            # m_class = best_found["predicted-class"]

                            # m_path = f"{path}/{stylemix_cls}/bf/"
                            # m_name = f"/{int(l2_distance)}-{int(ssi * 100)}-{stylemix_seed}-{stylemix_cls}-{layer[0]}-{m_class}"
                            # os.makedirs(m_path, exist_ok=True)
                            # with open(f"{m_path}/{m_name}.json", 'w') as f:
                            #     (json.dump(best_found, f, sort_keys=True, indent=4))
                            # m_image.save(f"{m_path}/{m_name}.png")
            self.w0_seed += self.step_size







def run_mimicry(class_idx, w0_seed=0, step_size=1):
    mimicry(class_idx=class_idx, w0_seed=w0_seed, step_size=step_size).search()

if __name__ == "__main__":
    for i in range(8,10):
        run_mimicry(i, 0)
    # run_mimicry(7, 30)


    # for seed_class in range(5):
    #     for w0_seed in range(process_count):
    #         args_list.append((seed_class, w0_seed, process_count))

    # args_list = [i for i in range(10)]
    # args_list = [6,7,8,9]
    # process_count = 3
    # # for seed_class in range(3):
    # #     args_list.append((seed_class))

    # print(f"Args List: {args_list}")
    # set_start_method('spawn')
    # with Pool(processes=process_count) as pool:
    #     pool.map(run_mimicry, args_list, chunksize=1)