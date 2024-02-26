import dnnlib
# from stylegan.renderer import Renderer
from stylegan.renderer_v2 import Renderer

DEVICE = 'cuda'

INIT_PKL = 'stylegan2_mnist_32x32-con'

CACHE_DIR = 'checkpoints'

SEARCH_LIMIT = 10000

# DRAGGAN_INIT = {
#     "images": {
#         # image_orig: the original image, change with seed/model is changed
#         # image_raw: image with mask and points, change durning optimization
#         # image_show: image showed on screen
#     },
#     "temporal_params": {
#         # stop
#     },
#     'mask':
#     None,  # mask for visualization, 1 for editing and 0 for unchange
#     'last_mask': None,  # last edited mask
#     'show_mask': True,  # add button
#     "generator_params": dnnlib.EasyDict(),
#     "params": {
#         "seed": 0,
#         "class_idx": 5,
#         "motion_lambda": 20,
#         "r1_in_pixels": 1,
#         "r2_in_pixels": 3,
#         "magnitude_direction_in_pixels": 1.0,
#         "latent_space": "w",
#         "trunc_psi": 0.7,
#         "trunc_cutoff": None,
#         "lr": 0.001,
#     },
#     "device": DEVICE,
#     "draw_interval": 1,
#     "renderer": Renderer(disable_timing=True),
#     "points": {},
#     "curr_point": None,
#     "curr_type_point": "start",
#     'editing_state': 'add_points',
#     'pretrained_weight': INIT_PKL
# }
STYLEGAN_INIT = {
    "images": {
        # image_orig: the original image, change with seed/model is changed
        # image_raw: image with mask and points, change durning optimization
        # image_show: image showed on screen
    },
    "temporal_params": {
        # stop
    },
    'mask':
    None,  # mask for visualization, 1 for editing and 0 for unchange
    'last_mask': None,  # last edited mask
    'show_mask': True,  # add button
    "generator_params": dnnlib.EasyDict(),
    "params": {
        "w0_seeds": [[0, 1.0]],
        "class_idx": 5,
        "mixclass_idx": 0,
        "stylemix_idx": [], # [1, 2, 3, 4, 5, 6, 7]
        "stylemix_seed": 0,
        "trunc_psi": 1.0,
        "trunc_cutoff": 8,
        "random_seed": 0,
        "noise_mode": 'const',
        "force_fp32": False,
        "layer_name": None,
        "sel_channels": 3,
        "base_channel": 0,
        "img_scale_db": 0.0,
        "img_normalize": True,
        "to_pil": True,
        "fft_show": False,
        "fft_all": True,
        "fft_range_db": 50,
        "fft_beta": 8,
        "untransform": False,
    },
    "device": DEVICE,
    "draw_interval": 1,
    "renderer": Renderer(),
    "points": {},
    "curr_point": None,
    "curr_type_point": "start",
    'editing_state': 'add_points',
    'pretrained_weight': INIT_PKL
}

#################################################

DJ_DEBUG = 1

# GA Setup
POPSIZE = 100

STOP_CONDITION = "iter"
#STOP_CONDITION = "time"

NGEN = 100
RUNTIME = 3600
STEPSIZE = 10
# Mutation Hyperparameters
# range of the mutation
MUTLOWERBOUND = 0.01
MUTUPPERBOUND = 0.6

# Reseeding Hyperparameters
# extent of the reseeding operator
RESEEDUPPERBOUND = 10

K_SD = 0.1

# K-nearest
K = 1

# Archive configuration
ARCHIVE_THRESHOLD = 4.0

#------- NOT TUNING ----------

# mutation operator probability
MUTOPPROB = 0.5
MUTOFPROB = 0.5

IMG_SIZE = 28
num_classes = 10


# INITIALPOP = 'seeded'
INITIALPOP = 'random'

GENERATE_ONE_ONLY = False

MODEL2 = 'mnist/models/cnnClassifier_lowLR.h5'
MODEL = 'mnist/models/cnnClassifier.h5'
#MODEL = "models/regular3"
#MODEL = 'models/cnnClassifier_001.h5'
#MODEL = 'models/cnnClassifier_op.h5'

RESULTS_PATH = 'results'
REPORT_NAME = 'stats.csv'
DATASET = 'original_dataset/janus_dataset_comparison.h5'
EXPLABEL = 5

#TODO: set interpreter
INTERPRETER = '/home/vin/yes/envs/tf_gpu/bin/python'