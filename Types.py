from enum import Enum
import torch
import torch.nn as nn

PRINT_BLUE = "\033[34m"
PRINT_GREEN = "\033[32m"
PRINT_RED = "\033[31m"
PRINT_RESET = "\033[0m"

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
DEFAULT_PT_ROWS_PRINT:                      int         = 6
DEFAULT_PT_LENGTH_PRINT:                    int         = 10
### end

### Convolution Block Defaults
DEFAULT_CONVOLUTION_PADDING                             = None
DEFAULT_CONVOLUTION_DILATION:               int         = 1
DEFAULT_CONVOLUTION_STRIDE:                 int         = 1
DEFAULT_CONVOLUTION_GROUPS:                 int         = 1
DEFAULT_CONVOLUTION_ACTIVATION:             str         = "gelu"
DEFAULT_CONVOLUTION_DROPOUT:                float       = 0.0
### end

### Multi Kernel convolution Defaults
DEFAULT_MULTI_KERNEL_KERNEL_LIST:           list        = [3,5,7,11,15]
DEFAULT_MULTI_KERNEL_PER_KERNEL_OUTPUTCH:   int         = 64
### end

### Temporal Head Defaults
DEFAULT_TEMPORAL_HIDDEN_CHANNELS:            int         = 128
DEFAULT_TEMPORAL_KERNEL_REDUCTION:           int         = 1
DEFAULT_TEMPORAL_KERNEL_RESIDUAL:            int         = 3
DEFAULT_TEMPORAL_DROPOUT:                    float       = 0.1
DEFAULT_TEMPORAL_MULTI_DILATION:             bool        = True
DEFAULT_RESIDUAL_BLOCKS_NMB:                 int         = 2
### end

### smORFs CNN Classifier Defaults
DEFAULT_SMORFCNN_TEMPORAL_HEAD:              bool        = True
DEFAULT_SMORFCNN_MULTI_KERNEL:               bool        = True
DEFAULT_SMORFCNN_ONEHOT_KERNEL_LIST:         list        = [3,5,7,11,15]
DEFAULT_SMORFCNN_EMBEDDINGS_KERNEL_LIST:     list        = [3,11]
DEFAULT_SMORFCNN_OUTPUT_CHANNELS_KERNEL:     int         = 64
DEFAULT_SMORFCNN_OUTPUT_CHANNELS_TEMPORAL:   int         = 128
DEFAULT_SMORFCNN_RESIDUAL_BLOCKS:            int         = 2
DEFAULT_SMORFCNN_DROPOUT:                    float       = 0.1
DEFAULT_SMORFCNN_CLASSES:                    int         = 1
DEFAULT_SMORFCNN_CLASSIFIER_OUTPUT:          int         = 256
DEFAULT_SMORFCNN_SEED:                       int         = 42
DEFAULT_SMORFCNN_DETERMINISTIC:              bool        = True 
DEFAULT_SMORFCNN_DEVICE:                     str         = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SMORFCNN_TRAIN_SPLIT:                float       = 0.8
DEFAULT_SMORFCNN_VALIDATION_SPLIT:           float       = 0.1
DEFAULT_SMORFCNN_TEST_SPLIT:                 float       = 0.1
DEFAULT_SMORFCNN_THRESHOLD:                  float       = 0.5
DEFAULT_SMORFCNN_MAX_GRAD_NORM:              float       = 1.0
DEFAULT_SMORFCNN_LEARNING_RATE:              float       = 1e-3
DEFAULT_SMORFCNN_WEIGHT_DECAY:               float       = 1e-4
DEFAULT_SMORFCNN_TRAIN_BATCH_SIZE:           int         = 16
DEFAULT_SMORFCNN_VALIDATION_BATCH_SIZE:      int         = 16
DEFAULT_SMORFCNN_TEST_BATCH_SIZE:            int         = 16
DEFAULT_SMORFCNN_KFOLD:                      int         = 10
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