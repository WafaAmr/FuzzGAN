import os
import os.path as osp

import gradio as gr
import numpy as np
from PIL import Image

import dnnlib
from stylegan.renderer import Renderer, add_watermark_np
from config import CACHE_DIR


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
        state['params']['latent_space'] == 'w+',  # w_plus
        'const',  # noise_mode
        state['params']['trunc_psi'],  # trunc_psi,
        state['params']['trunc_cutoff'],  # trunc_cutoff,
        None,  # input_transform
        state['params']['lr']  # lr,
    )

    state['renderer']._render_drag_impl(state['generator_params'],
                                        class_idx=state['params']['class_idx'],
                                        is_drag=False,
                                        to_pil=True)

    init_image = state['generator_params'].image
    state['images']['image_orig'] = init_image
    state['images']['image_raw'] = init_image
    state['images']['image_show'] = Image.fromarray(
        np.array(init_image))
    state['mask'] = np.ones((init_image.size[1], init_image.size[0]),
                            dtype=np.uint8)
    return global_state

valid_checkpoints_dict = {
    f.split('/')[-1].split('.')[0]: osp.join(CACHE_DIR, f)
    for f in os.listdir(CACHE_DIR)
    if (f.endswith('pkl') and osp.exists(osp.join(CACHE_DIR, f)))
}
print(f'\nFile under CACHE_DIR ({CACHE_DIR}):')
print(os.listdir(CACHE_DIR))
print('\nValid checkpoint file:')
print(valid_checkpoints_dict)