import os
import os.path as osp

import gradio as gr
import numpy as np
from PIL import Image

import dnnlib
from stylegan.renderer_v2 import Renderer
from config import CACHE_DIR, STYLEGAN_INIT, POPSIZE, SEARCH_FACTOR
from predictor import Predictor
import pickle
from utils import get_distance

import threading

def init_images(global_state):
    """This function is called only ones with Gradio App is started.
    0. pre-process global_state, unpack value from global_state of need
    1. Re-init renderer
    2. run `renderer._render_drag_impl` with `is_drag=False` to generate
       new image
    3. Assign images to global state and re-generate mask
    """

    if isinstance(global_state, gr.State):
        state = global_state.value
    else:
        state = global_state
    # print(state['params']['stylemix_seed'])
    state['renderer']._render_impl(
        state['generator_params'],  # res
        valid_checkpoints_dict[state['pretrained_weight']],  # pkl
        state['params']['w0_seeds'],  # w0_seed,
        state['params']['class_idx'],  # class_idx,
        state['params']['mixclass_idx'],  # mix_idx,
        state['params']['stylemix_idx'],  # stylemix_idx,
        state['params']['stylemix_seed'],  # stylemix_seed,
        img_normalize=True,
        to_pil = True
    )
    init_image = state['generator_params'].image
    state['images']['image_orig'] = init_image
    return global_state

def generate_dataset(start_seed):
    dataset = []
    search_population = POPSIZE * SEARCH_FACTOR
    w0_seed = start_seed
    j = 0
    stylemix_seed = 0
    found = False
    stylemix_seed_limit = 5000
    distance_limit = 1500

    lst = [1, 2, 3, 4, 5, 6, 7]
    all_combinations = [[x] for x in lst]

    # import itertools
    # all_combinations = []
    # for r in range(1, len(lst) + 1):
    #     combinations_object = itertools.combinations(lst, r)
    #     combinations = [c for c in combinations_object]
    #     all_combinations.extend(combinations)


    while j < 10000:

        state_init = STYLEGAN_INIT
        state_init["params"]["w0_seeds"] = [[w0_seed, 1.0]]
        state_init["params"]["stylemix_idx"] = []
        state_init["params"]["stylemix_seed"] = 0
        digit =init_images(state_init)
        print(f"Generated {w0_seed} digits")
        w0_seed += 2
        expected_label = digit["params"]["class_idx"]
        image = digit["images"]["image_orig"]
        image = image.crop((2, 2, image.width - 2, image.height - 2))
        image_array = np.array(image)
        image_array = np.reshape(image_array, (-1, 28, 28, 1))
        accepted, confidence, not_class, not_class_confidence = Predictor().predict_generator(image_array, expected_label)
        if accepted and not_class_confidence != 0.0:
            found = False
            j += 1
            print(accepted, confidence, not_class, not_class_confidence)
            for cls in range(10):
                state_init["params"]["mixclass_idx"] = cls
                while not found and stylemix_seed < stylemix_seed_limit:
                    if stylemix_seed == w0_seed:
                        stylemix_seed += 1
                    state_init["params"]["stylemix_seed"] = stylemix_seed
                    for idx, layer in enumerate(all_combinations):
                        # print(f"mixing: {idx}/{len(all_combinations)}")
                        state_init["params"]["stylemix_idx"] = list(layer)
                        m_digit =init_images(state_init)
                        m_image = digit["images"]["image_orig"]
                        # print(m_digit["params"]["stylemix_seed"])
                        m_image = m_image.crop((2, 2, m_image.width - 2, m_image.height - 2))
                        m_image_array = np.array(m_image)
                        m_image_array = np.reshape(m_image_array, (-1, 28, 28, 1))
                        m_accepted, _, m_not_class, _ = Predictor().predict_generator(m_image_array, expected_label)
                        distance = get_distance(np.array(image), np.array(m_image))
                        if not m_accepted and not found and 0 < distance < distance_limit:
                            os.makedirs(f"mnist/try/{w0_seed}/", exist_ok=True)
                            image.save(f"mnist/try/{w0_seed}/0-{confidence}-{not_class}-{not_class_confidence}.png")
                            m_image.save(f"mnist/try/{w0_seed}/{m_accepted}_{int(distance)}_{idx}_{m_not_class}_{stylemix_seed}.png")
                            print(f"Found: {not m_accepted}, class: {m_not_class}, stylemix_seed: {stylemix_seed}, distance: {distance}")
                            found = True
                        elif found:
                            break
                        # print(w0_seed)
                    stylemix_seed += 1
                    # distance.append(get_distance(image, m_image))
                # smallest_index = distance.index(min(distance))
                # print(smallest_index)


    #         image.save(f"dataset/pci-0.7/true/{i}-{confidence}-{not_class}-{not_class_confidence}.png")
    #         if 'predictor' not in digit:
    #             digit['predictor'] = {}
    #         digit["predictor"]["confidence"] = confidence
    #         digit["predictor"]["not_class"] = not_class
    #         digit["predictor"]["not_class_confidence"] = not_class_confidence
    #         dataset.append(digit)
    #     elif not accepted:
    #         image.save(f"dataset/pci-0.7/false/{i}-{not_class}-{not_class_confidence}-{confidence}.png")
    #     i += 1
    #     print(f"Generated {i} digits")
    # dataset.sort(key=lambda digit: digit["predictor"]["confidence"], reverse=True)
    # with open('dataset-pci-0.7.pkl', 'wb') as f:
    #     pickle.dump(dataset, f)
    # dataset = dataset[:POPSIZE]
    # print(f"Selected {len(dataset)} digits")
    # for digit in dataset:
    #     print(digit["predictor"]["not_class_confidence"])
    #     i =digit['params']['seed']
    #     image = digit["images"]["image_orig"]
    #     image.save(f"dataset/{i}.png")
    return dataset










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
    generate_dataset(0)