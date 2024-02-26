import os
import os.path as osp

import gradio as gr
import numpy as np
from PIL import Image

import dnnlib
from stylegan.renderer import Renderer
from config import CACHE_DIR, STYLEGAN_INIT, POPSIZE, SEARCH_FACTOR
from predictor import Predictor
import pickle

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

    state['renderer'].init_network(
        state['generator_params'],  # res
        valid_checkpoints_dict[state['pretrained_weight']],  # pkl
        state['params']['seed'],  # w0_seed,
        state['params']['class_idx'],  # class_idx,
        None,  # w_load
        state['params']['latent_space'] == 'w',  # w or w+
        'const',  # noise_mode
        state['params']['trunc_psi'],  # trunc_psi,
        state['params']['trunc_cutoff'],  # trunc_cutoff,
        None,  # input_transform
        state['params']['lr']  # lr,
    )

    # state['renderer2']._render_drag_impl(state['generator_params'],
    #                                     class_idx=state['params']['class_idx'],
    #                                     is_drag=False,
    #                                     to_pil=True)

    init_image = state['generator_params'].image
    state['images']['image_orig'] = init_image
    # state['images']['image_raw'] = init_image
    # state['images']['image_show'] = Image.fromarray(
    #     np.array(init_image))
    # state['mask'] = np.ones((init_image.size[1], init_image.size[0]),
    #                         dtype=np.uint8)
    return global_state

def generate_dataset():
    dataset = []
    search_population = POPSIZE * SEARCH_FACTOR
    i = 0
    j = 0

    while j < 1000:
        state_init = STYLEGAN_INIT
        state_init["params"]["seed"] = i
        digit =init_images(state_init)
        expected_label = digit["params"]["class_idx"]
        image = digit["images"]["image_orig"]
        image = image.crop((2, 2, image.width - 2, image.height - 2))
        image_array = np.array(image)
        image_array = np.reshape(image_array, (-1, 28, 28, 1))
        accepted, confidence, not_class, not_class_confidence = Predictor().predict_generator(image_array, expected_label)
        if accepted and not_class_confidence != 0.0:
            j += 1
            image.save(f"dataset/pci-0.7/true/{i}-{confidence}-{not_class}-{not_class_confidence}.png")
            if 'predictor' not in digit:
                digit['predictor'] = {}
            digit["predictor"]["confidence"] = confidence
            digit["predictor"]["not_class"] = not_class
            digit["predictor"]["not_class_confidence"] = not_class_confidence
            dataset.append(digit)
        elif not accepted:
            image.save(f"dataset/pci-0.7/false/{i}-{not_class}-{not_class_confidence}-{confidence}.png")
        i += 1
        print(f"Generated {i} digits")
    dataset.sort(key=lambda digit: digit["predictor"]["confidence"], reverse=True)
    with open('dataset-pci-0.7.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    dataset = dataset[:POPSIZE]
    print(f"Selected {len(dataset)} digits")
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
    generate_dataset()