import os
import json
import numpy as np
from config import INIT_PKL
from stylegan.renderer_v2 import Renderer
import dnnlib
from predictor import Predictor


renderer = Renderer()
m = False

def render_seed(state):
    renderer._render_impl(
        res = state['res'],  # res
        pkl = INIT_PKL,  # pkl
        w0_seeds= state['w0_seeds'],
        w_load= state['w_load'],  # w_load,
        w_load_seed = state['w_load_seed'],
        class_idx = state['class_idx'],
        mixclass_idx = state['mixclass_idx'],
        stylemix_idx = state['stylemix_idx'],
        stylemix_seed = state['stylemix_seed'],
        trunc_psi = state['trunc_psi'],
        img_normalize = state['img_normalize'],
        to_pil = state['to_pil'],
    )
    return state



DATASET = 'mnist/eval/HQ/5'
subfolders = [os.path.join(DATASET, subfolder) for subfolder in os.listdir(DATASET) if os.path.isdir(os.path.join(DATASET, subfolder))]

x_test = []
for subfolder in subfolders:
    content = os.listdir(subfolder)
    if m:
        for folder in content:
            if os.path.isdir(os.path.join(subfolder,folder)):
                scd_class = os.path.join(subfolder,folder)
                for file in os.listdir(scd_class):
                    if file.endswith('.json'):
                        with open(os.path.join(subfolder, folder, file), 'r') as f:
                            params = json.load(f)
                            x_test.append(params)
    else:
        for file in content:
            if file.endswith('.json'):
                with open(os.path.join(subfolder, file), 'r') as f:
                    params = json.load(f)
                    x_test.append(params)

DATASET = 'mnist/inv_2/5'
content = os.listdir(DATASET)

ws_load = []
for file in content:
    if file.endswith('.npy'):
        w_load = np.load(os.path.join(DATASET, file))
        ws_load.append(w_load)

for w_load, state in zip(ws_load, x_test):
    seed = state['w0_seeds'][0][0]
    state['w0_seeds'] = [[0, 1]]
    state['w_load'] = w_load
    state['w_load_seed'] = 0
    state["res"] = dnnlib.EasyDict()
    state = render_seed(state)
    image = state['res'].image
    image_array = np.array(image.crop((2, 2, image.width - 2, image.height - 2)))
    state["res"].image_array = image_array
    label = state["class_idx"]
    accepted, confidence, predictions = Predictor().predict_datapoint(
    np.reshape(image_array, (-1, 28, 28, 1)),
    label
    )
    # if not accepted:
    image.save(f"mnist/eval/5/{seed}.png")
    print(predictions)




