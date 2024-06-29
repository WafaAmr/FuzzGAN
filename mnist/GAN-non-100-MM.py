from config import INIT_PKL
import dnnlib
from utils import get_distance
import numpy as np
import random
from stylegan.renderer_v2 import Renderer

renderer = Renderer()


def apply_mutoperator1(state, extent=None):
    least_distance = None
    stylemix_seed = random.randint(0, 350000)
    if "w" in state["res"]:
        w_load = state["res"].w
    else:
        w_load = None
    for stylemix_idx in range(state["res"].num_ws):
        current_res = dnnlib.EasyDict()
        state['stylemix_idx'] = [stylemix_idx]
        renderer._render_impl(
            res=current_res,  # res
            pkl=INIT_PKL,  # pkl
            w0_seeds=state['w0_seeds'],
            w_load = w_load,
            class_idx=state['class_idx'],
            mixclass_idx=state['second_cls'],
            stylemix_idx= state['stylemix_idx'],
            stylemix_seed=stylemix_seed,
            trunc_psi=state['trunc_psi'],
            img_normalize=state['img_normalize'],
            to_pil=state['to_pil'],
        )
        m_image = current_res.image
        m_image_array = np.array(m_image.crop((2, 2, m_image.width - 2, m_image.height - 2)))
        current_res.image_array = m_image_array
        l2_distance = get_distance(state['res'].image_array, m_image_array)

        if least_distance is None or l2_distance < least_distance:
            state["m_res"] = current_res
            least_distance = l2_distance
    return state

def apply_mutoperator2(state):

    if "w" in state["res"]:
        w_load = state["res"].w
    else:
        w_load = None

    if "second_cls" in state["res"]:
        second_cls = state["second_cls"]
    else:
        second_cls = random.randint(0, 9)

    current_res = dnnlib.EasyDict()
    renderer._render_impl(
        res=current_res,  # res
        pkl=INIT_PKL,  # pkl
        w0_seeds=state['w0_seeds'],
        w_load = w_load,
        class_idx=state['class_idx'],
        mixclass_idx=second_cls,
        stylemix_idx= state['stylemix_idx'],
        stylemix_seed=state['stylemix_seed'],
        trunc_psi=state['trunc_psi'],
        img_normalize=state['img_normalize'],
        to_pil=state['to_pil'],
    )
    m_image = current_res.image
    m_image_array = np.array(m_image.crop((2, 2, m_image.width - 2, m_image.height - 2)))
    current_res.image_array = m_image_array
    state["m_res"] = current_res

    return state


def mutate(state, operator_name):
    if operator_name == 1:
        state = apply_mutoperator2(state)
    elif operator_name == 2:
        state = apply_mutoperator2(state)
    return state

def generate(state, operator_name):
    if operator_name == 1:
        state = apply_mutoperator1(state)
    elif operator_name == 2:
        state = apply_mutoperator1(state)
    return state

