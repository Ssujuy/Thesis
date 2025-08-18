from enum import Enum
import torch

### DNABERT6 Defaults
DEFAULT_DNABERT6_MODEL_ID               = "zhihan1996/DNA_bert_6"
DEFAULT_DNABER6_DATASET_PATH            = "train.csv"
DEFAULT_DNABER6_DATASET_PERCENTAGE      = 100
DEFAULT_DNABERT6_PROJECTION_DIMENSION   = 768
DEFAULT_DNABERT6_WINDOW_SIZE            = 512
DEFAULT_DNABERT6_LEARNING_RATE          = 2e-5
DEFAULT_DNABERT6_EPOCHS                 = 4
DEFAULT_DNABERT6_BATCH_SIZE             = 16 
DEFAULT_DNABERT6_WEIGHT_DECAY           = 0.01 
DEFAULT_DNABERT6_WARMUP_RATIO           = 0.2
DEFAULT_DNABERT6_DEVICE                 = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DNABER6_SAVE_DIRECTORY          = "dnabert6_smorfs_ft" 
### end

class KmerAmbiguousState(Enum):

    MASK = 0,
    UNK = 1

class ProjectionState(Enum):

    NO_PROJECTION = 0
    NOT_TRAINABLE = 1
    TRAINABLE = 2

class HiddenState(Enum):

    CLS     = 0,
    MEAN    = 1,
    BOTH    = 2