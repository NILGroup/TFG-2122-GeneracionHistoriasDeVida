from socketserver import DatagramRequestHandler


DIR             = 'D2T'

KELM            = "https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/quadruples-train.tsv"
DART            = "https://raw.githubusercontent.com/Yale-LILY/dart/master/data/v1.1.1/dart-v1.1.1-full-train.json"
WEBNLG          = "https://gitlab.com/shimorina/webnlg-dataset/-/archive/master/webnlg-dataset-master.zip?path=release_v3.0/en/train"

SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",                    
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}

EPOCHS          = 4
LR              = 1e-3
EPS             = (1e-30, 1e-3),
CLIP_THRESHOLD  = 1.0,
DECAY_RATE      = -0.8,
BETA1           = None,
WEIGHT_DECAY    = 0.0,
RELATIVE_STEP   = False,
SCALE_PARAMETER = False,
WARMUP_INIT     = False

BATCH_SIZE      = 4
NUM_OF_EPOCHS   = 2