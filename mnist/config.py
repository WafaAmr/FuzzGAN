import dnnlib
from stylegan.renderer import Renderer

DEVICE = 'cuda'

INIT_PKL = 'stylegan2_mnist_32x32-con'

CACHE_DIR = '../checkpoints'

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
        "seed": 0,
        "class_idx": 5,
        "motion_lambda": 20,
        "r1_in_pixels": 1,
        "r2_in_pixels": 3,
        "magnitude_direction_in_pixels": 1.0,
        "latent_space": "w+",
        "trunc_psi": 0.7,
        "trunc_cutoff": None,
        "lr": 0.001,
    },
    "device": DEVICE,
    "draw_interval": 1,
    "renderer": Renderer(disable_timing=True),
    "points": {},
    "curr_point": None,
    "curr_type_point": "start",
    'editing_state': 'add_points',
    'pretrained_weight': INIT_PKL
}

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

MODEL2 = 'models/cnnClassifier_lowLR.h5'
MODEL = 'models/cnnClassifier.h5'
#MODEL = "models/regular3"
#MODEL = 'models/cnnClassifier_001.h5'
#MODEL = 'models/cnnClassifier_op.h5'

RESULTS_PATH = 'results'
REPORT_NAME = 'stats.csv'
DATASET = 'original_dataset/janus_dataset_comparison.h5'
EXPLABEL = 5

#TODO: set interpreter
INTERPRETER = '/home/vin/yes/envs/tf_gpu/bin/python'