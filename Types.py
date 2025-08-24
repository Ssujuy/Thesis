from enum import Enum
import torch
import torch.nn as nn

### DNABERT6 Defaults
DEFAULT_DNABERT6_MODEL_ID:                  str         = "zhihan1996/DNA_bert_6"
DEFAULT_DNABER6_DATASET_PATH:               str         = "train.csv"
DEFAULT_DNABERT6_KMER_SIZE:                 int         = 6
DEFAULT_DNABER6_DATASET_PERCENTAGE:         int         = 100
DEFAULT_DNABERT6_PROJECTION_DIMENSION:      int         = 768
DEFAULT_DNABERT6_WINDOW_SIZE:               int         = 512
DEFAULT_DNABERT6_LEARNING_RATE:             float       = 2e-5
DEFAULT_DNABERT6_EPOCHS:                    int         = 4
DEFAULT_DNABERT6_BATCH_SIZE:                int         = 16
DEFAULT_DNABERT6_WEIGHT_DECAY:              float       = 0.01
DEFAULT_DNABERT6_WARMUP_RATIO:              float       = 0.2
DEFAULT_DNABERT6_DEVICE:                    str         = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DNABER6_SAVE_DIRECTORY:             str         = "dnabert6_smorfs_ft"
DEFAULT_DNABERT6_STRATEGY:                  str         = "epoch"
DEFAULT_DNABER6_METRIC:                     str         = "f1"
### end

### Pytorch file printing defaults
DEFAULT_PT_ROWS_PRINT:          int         = 6
DEFAULT_PT_LENGTH_PRINT:        int         = 10
### end

### Convolution Block Defaults
DEFAULT_CONVOLUTION_PADDING                 = None
DEFAULT_CONVOLUTION_DILATION:   int         = 1
DEFAULT_CONVOLUTION_STRIDE:     int         = 1
DEFAULT_CONVOLUTION_GROUPS:     int         = 1
DEFAULT_CONVOLUTION_ACTIVATION: str         = "gelu"
DEFAULT_CONVOLUTION_DROPOUT:    float       = 0.0
### end

activationFunctionMapping = {
    "gelu":     nn.GELU(),
    "relu":     nn.ReLU(),
    "lrelu":    nn.LeakyReLU(),
    "silu":     nn.SiLU(),
    "tahn":     nn.Tanh(),
}

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