import os
import os.path as osp
import copy
import numpy as np
from stylegan.renderer_v2 import Renderer
from config import STYLEGAN_INIT, MODEL, POPSIZE, INIT_PKL
from predictor import Predictor
from utils import validate_mutation
import json


seed = 400
class_idx = 5
path = f"mnist/original_dataset/{class_idx}"

os.makedirs(path, exist_ok=True)
renderer = Renderer()
state = STYLEGAN_INIT
state['params']['class_idx'] = class_idx
dataset_size = 0

while dataset_size < POPSIZE:
  state['params']['w0_seeds'] = [[seed, 1]]
  renderer._render_impl(
    res = state['generator_params'],  # res
    pkl = INIT_PKL,  # pkl
    w0_seeds= state['params']['w0_seeds'],
    class_idx = state['params']['class_idx'],
    mixclass_idx = state['params']['mixclass_idx'],
    stylemix_idx = state['params']['stylemix_idx'],
    stylemix_seed = state['params']['stylemix_seed'],
    trunc_psi = state['params']['trunc_psi'],
    img_normalize = state['params']['img_normalize'],
    to_pil = state['params']['to_pil'],
  )
  image = state['generator_params'].image
  label = state["params"]["class_idx"]
  image = image.crop((2, 2, image.width - 2, image.height - 2))
  image_array = np.array(image)
  accepted, confidence, predictions = Predictor().predict_datapoint(
    np.reshape(image_array, (-1, 28, 28, 1)),
    label
  )

  if accepted:
    _ , second_cls = np.argsort(-predictions)[:2]
    second_cls_confidence = predictions[second_cls]
    if second_cls_confidence:
      dataset_size += 1
      info = copy.deepcopy(state["params"])
      info["predictions"] = predictions.tolist()
      info["second_cls"] = int(second_cls)
      info["model"] = MODEL

      with open(f"{path}/{seed}.json", 'w') as f:
          (json.dump(info, f, indent=4))
      image.save(f"{path}/{seed}.png")
  seed += 1