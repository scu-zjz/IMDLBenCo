from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4

# Cudnn parameters
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# Model parameters
_C.MODEL = CN()
_C.MODEL.NAME = 'detconfcmx'
_C.MODEL.PRETRAINED = ''
_C.MODEL.MODS = ('RGB','NP++')
_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.EXTRA.DETECTION = None
_C.MODEL.EXTRA.CONF = False

# Dataset parameters
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.TRAIN = []
_C.DATASET.VALID = []
_C.DATASET.NUM_CLASSES = 2

# Testing parameters
_C.TEST = CN()
_C.TEST.MODEL_FILE = ''


def update_config(cfg, args, config_path):
    cfg.defrost()
    cfg.merge_from_file(config_path)
    # if args.opts:
    #     cfg.merge_from_list(args.opts)
    cfg.freeze()