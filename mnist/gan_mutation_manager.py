import os
from config import CACHE_DIR
import os.path as osp
from stylegan.renderer_v2 import Renderer
import copy

def render_seed(state=None):

    Renderer(disable_timing=True)._render_impl(
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

    info =  copy.deepcopy(state['params'])

    return state, info

def apply_mutoperator1(state, extent):
    state['params']['stylemix_idx'] = extent['stylemix_idx']

    return state


def apply_mutoperator2(state, extent):
    stylemix_seed = state['params']['stylemix_seed']
    mixclass_idx = state['params']['mixclass_idx']
    state['params']['stylemix_idx'] = None
    state['params']['stylemix_seed'] = None
    state['params']['mixclass_idx'] = None


    return state


def mutate(state, operator_name, mutation_extent):
    if operator_name == 1:
        state = apply_mutoperator1(state, mutation_extent)
    elif operator_name == 2:
        state = apply_mutoperator2(state, mutation_extent)
    return state

valid_checkpoints_dict = {
    f.split('/')[-1].split('.')[0]: osp.join(CACHE_DIR, f)
    for f in os.listdir(CACHE_DIR)
    if (f.endswith('pkl') and osp.exists(osp.join(CACHE_DIR, f)))
}
print(f'\nFile under CACHE_DIR ({CACHE_DIR}):')
print(os.listdir(CACHE_DIR))
print('\nValid checkpoint file:')
print(valid_checkpoints_dict)