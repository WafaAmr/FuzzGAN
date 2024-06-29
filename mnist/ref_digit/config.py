import dnnlib
# from stylegan.renderer import Renderer
from stylegan.renderer_v2 import Renderer
import os.path as osp
import os

DEVICE = 'cuda'

INIT_PKL = 'checkpoints/stylegan2_mnist_32x32-con.pkl'
# INIT_PKL = 'checkpoints/stylegan2_mnist_004800.pkl'
GAN_NET_SIZE = 8

TEST_IMAGES = '/home/upc/datasets/t10k-images-idx3-ubyte.gz'
TEST_LABELS = '/home/upc/datasets/t10k-labels-idx1-ubyte.gz'

# SEARCH_LIMIT = 100
# SSIM_THRESHOLD = 0.95
# L2_RANGE = 0.2
# STYLEMIX_SEED_LIMIT = 100
SSIM_THRESHOLD = 0.75
# L2_RANGE = 0.25
SEARCH_LIMIT = 10
# SSIM_THRESHOLD = 0.65
L2_RANGE = 1
STYLEMIX_SEED_LIMIT = 500

STYLEGAN_INIT = {
    "generator_params": dnnlib.EasyDict(),
    "params": {
        "w0_seeds": [[0, 1]],
        "w_load": None,
        "class_idx": None,
        "mixclass_idx": None,
        "stylemix_idx": [],
        "patch_idxs": None,
        "stylemix_seed": None,
        "trunc_psi": 1,
        "trunc_cutoff": 0,
        "random_seed": 0,
        "noise_mode": 'const',
        "force_fp32": False,
        "layer_name": None,
        "sel_channels": 3,
        "base_channel": 0,
        "img_scale_db": 0,
        "img_normalize": True,
        "to_pil": True,
        "input_transform" : None,
        "untransform": False,
    },
    "device": DEVICE,
    "renderer": None,
    'pretrained_weight': INIT_PKL
}

#################################################

DJ_DEBUG = 1

# GA Setup
POPSIZE = 100

STOP_CONDITION = "iter"
#STOP_CONDITION = "time"

NGEN = 1000
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

GENERATE_ONE_ONLY = True

MODEL = 'mnist/models/cnnClassifier_lowLR.h5'
# MODEL = 'mnist/models/cnnClassifier.h5'
#MODEL = "models/regular3"
#MODEL = 'models/cnnClassifier_001.h5'
#MODEL = 'models/cnnClassifier_op.h5'

RESULTS_PATH = 'results'
REPORT_NAME = 'stats.csv'
DATASET = 'mnist/original_dataset/janus_dataset_comparison.h5'
EXPLABEL = 5

#TODO: set interpreter
INTERPRETER = '/home/vin/yes/envs/tf_gpu/bin/python'

# inversion is costly
# easier/faster to label the data
#  less acurate and more auto
