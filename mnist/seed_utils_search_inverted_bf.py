import os
import os.path as osp
import numpy as np
from PIL import Image
from stylegan.renderer_v2 import Renderer
from config import STYLEGAN_INIT, SEARCH_LIMIT, STYLEMIX_SEED_LIMIT, SSIM_THRESHOLD, INIT_PKL
from predictor import Predictor
from utils import validate_mutation
import dnnlib
from multiprocessing import Process, Pool, set_start_method
from stylegan.renderer_v2 import Renderer


class Dlmimic:
    def __init__(self):
        self.state = STYLEGAN_INIT
        self.state['renderer'] = Renderer()
        self.layers = [[7], [6], [5], [4], [3], [5,6], [3,4], [3,4,5,6], [3,4,5,6,7]]
        self.con_class = {0:[6,8], 1:[4,8], 2:[8,4], 3:[5,2], 4:[9,6], 5:[6,8], 6:[0,4,8], 7:[2,4], 8:[6,9], 9:[4,8]}

    def search(self, seed_class, w_path):
        seed_class = int(seed_class)
        state = self.state
        root = f"mnist/eval/final/inv/{seed_class}/"

        res = dnnlib.EasyDict()
        w_load = np.load(w_path)
        print(f"Loading {seed_class}")
        state['renderer']._render_impl(
                    res = res,  # res
                    pkl = INIT_PKL,
                    w0_seeds= state['params']['w0_seeds'],
                    w_load = w_load,
                    class_idx = seed_class,
                    mixclass_idx = seed_class,
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
        img_path = osp.join(root, f'{file_id}.png')

        print(f"Saving {img_path}")
        if not osp.exists(img_path):
            os.makedirs(root, exist_ok=True)
            image.save(img_path)

            m_classes = [{} for _ in range(10)]
            for stylemix_class in range(10):

                state["params"]["mixclass_idx"] = stylemix_class
                stylemix_seed = 1

                while stylemix_seed < 100:

                    # require unique seed for each stylemix
                    # r_seed = random.randint(0, 350000)
                    r_seed = stylemix_seed
                    state["params"]["stylemix_seed"] = r_seed

                    for idx, layer in enumerate(self.layers):
                        state["params"]["stylemix_idx"] = layer

                        state['renderer']._render_impl(
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

                        valid_mutation, ssi, l2_distance, img_l2, m_img_l2 = validate_mutation(image_array, m_image_array)
                        if m_class != seed_class:
                            print(f'{file_id}:{stylemix_class}_{stylemix_seed}_{r_seed}:{layer[0]}, {int(l2_distance)}, {int(ssi*100)} {m_class}')

                        if m_classes[m_class[0]] == {} or m_classes[m_class[0]]['l2'] > l2_distance:
                            m_classes[m_class[0]]['image'] = m_image
                            m_classes[m_class[0]]['ssi'] = ssi
                            m_classes[m_class[0]]['l2'] = l2_distance
                            m_classes[m_class[0]]['stylemix_seed'] = stylemix_seed
                            m_classes[m_class[0]]['stylemix_idx'] = layer[0]



                    stylemix_seed += 1
            for m_class, data in enumerate(m_classes):
                if data != {} and m_class != seed_class:
                    ssi = data['ssi']
                    if ssi > .1:
                        m_image = data['image']
                        l2_distance = data['l2']
                        stylemix_seed = data['stylemix_seed']
                        layer = data['stylemix_idx']
                        m_path = osp.join(root, file_id, str(m_class))
                        m_name = f"/{int(l2_distance)}-{int(ssi * 100)}-{stylemix_seed}-{layer}-{m_class}"
                        os.makedirs(m_path, exist_ok=True)
                        m_image.save(f"{m_path}/{m_name}.png")
            del renderer
            del res



def run_dlmimic(args):
    print(f"Processing {args}")
    seed_class, w_path = args
    dl = Dlmimic()
    dl.search(seed_class, w_path)

if __name__ == "__main__":
    run_dlmimic((0, "/home/upc/Desktop/FuzzGAN/mnist/inv_2/0/0-inv.npy"))
    # dlmimic((0, "/home/upc/Desktop/FuzzGAN/mnist/inv_2/0/1-inv.npy"))
    # args_list = []
    # process_count = 2

    # root_path = "/home/upc/Desktop/FuzzGAN/mnist/inv_2/"
    # digit_classes = [f for f in os.listdir(root_path) if osp.isdir(osp.join(root_path, f))]
    # for digit_class in digit_classes[:1]:
    #     digit_class_path = osp.join(root_path, digit_class)
    #     for file in os.listdir(digit_class_path):
    #         if file.endswith(".npy"):
    #             w_path = osp.join(digit_class_path, file)
    #             args_list.append((digit_class, w_path))



    # # print(f"Args List: {args_list}")
    # set_start_method('spawn')
    # with Pool(processes=process_count) as pool:
    #     result = pool.map(run_dlmimic, args_list, chunksize=1)