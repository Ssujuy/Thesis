from enum import Enum
import torch
import torch.nn as nn

########## ----------- Generic Defaults --------- ##########
DEFAULT_DEBUG_MODE:                         bool        = False
########## ----------- End --------- ##########

########## ----------- DNABERT6 Defaults --------- ##########
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
DEFAULT_DNABER6_TEST_SPLIT:                 float       = 0.2
DEFAULT_DNABERT6_DEVICE:                    str         = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DNABER6_SAVE_DIRECTORY:             str         = "dnabert6_smorfs_ft"
DEFAULT_DNABERT6_STRATEGY:                  str         = "epoch"
DEFAULT_DNABER6_METRIC:                     str         = "f1"
########## ----------- End --------- ##########


########## ----------- PyTorch File Print Defaults --------- ##########
DEFAULT_PT_ROWS_PRINT:                      int         = 6
DEFAULT_PT_LENGTH_PRINT:                    int         = 10
########## ----------- End --------- ##########

########## ----------- Convolution Block Defaults --------- ##########
DEFAULT_CONVOLUTION_PADDING:                str         = "valid"
DEFAULT_CONVOLUTION_DILATION:               int         = 1
DEFAULT_CONVOLUTION_STRIDE:                 int         = 1
DEFAULT_CONVOLUTION_GROUPS:                 int         = 1
DEFAULT_CONVOLUTION_ACTIVATION:             str         = "gelu"
DEFAULT_CONVOLUTION_DROPOUT:                float       = 0.0
DEFAULT_CONVOLUTION_BIAS:                   bool        = False
########## ----------- End --------- ##########

########## ----------- Multikernel Convolution Defaults --------- ##########
DEFAULT_MULTI_KERNEL_KERNEL_LIST:           list        = [3,5,7,11,15]
DEFAULT_MULTI_KERNEL_PER_KERNEL_OUTPUT:     int         = 64
########## ----------- End --------- ##########

########## ----------- Multi Gap Kernel Convolution Defaults --------- ##########
DEFAULT_MULTI_GAP_KERNEL_KERNEL_LIST:        list        = [3,5,7,11,15]
DEFAULT_MULTI_GAP_KERNEL_GAP_LIST:           list        = [1,2,3]
DEFAULT_MULTI_GAP_KERNEL_OUTPUT:             int         = 64
########## ----------- End --------- ##########

########## ----------- Temporal Head Defaults --------- ##########
DEFAULT_TEMPORAL_HIDDEN_CHANNELS:            int         = 512
DEFAULT_TEMPORAL_KERNEL_REDUCTION:           int         = 1
DEFAULT_TEMPORAL_KERNEL_RESIDUAL:            int         = 3
DEFAULT_TEMPORAL_MULTI_DILATION:             bool        = True
DEFAULT_RESIDUAL_BLOCKS_NMB:                 int         = 2
########## ----------- Temporal Head Defaults --------- ##########

########## ----------- SMORF CNN Classifier Defaults --------- ##########
DEFAULT_SMORFCNN_SAVE_DIR_PATH:              str         = "smorfCNN"
DEFAULT_SMORFCNN_MULTI_KERNEL:               bool        = True
DEFAULT_SMORFCNN_MULTI_GAP_KERNEL:           bool        = True
DEFAULT_SMORFCNN_DNABERT:                    bool        = True
DEFAULT_SMORFCNN_MULTI_KERNEL_LIST:          list        = [3,4,5,6,7,11]
DEFAULT_SMORFCNN_MULTI_GAP_KERNEL_LIST:      list        = [3,4,5,6,7,11]
DEFAULT_SMORFCNN_MULTI_GAP_KERNEL_GAP_LIST:  list        = [1,2,3]      
DEFAULT_SMORFCNN_OUTPUT_CHANNELS_KERNEL:     int         = 256
DEFAULT_SMORFCNN_OUTPUT_CHANNELS_TEMPORAL:   int         = 512
DEFAULT_SMORFCNN_RESIDUAL_BLOCKS:            int         = 2
DEFAULT_SMORFCNN_CLASSES:                    int         = 1
DEFAULT_SMORFCNN_CLASSIFIER_L1_OUTPUT:       int         = 1024
DEFAULT_SMORFCNN_CLASSIFIER_L2_OUTPUT:       int         = 256
DEFAULT_SMORFCNN_CLASSIFIER_DROPOUT:         float       = 0.2
DEFAULT_SMORFCNN_LEARNING_RATE:              float       = 1e-3
DEFAULT_SMORFCNN_WEIGHT_DECAY:               float       = 2e-3
DEFAULT_SMORFCNN_EPS:                        float       = 1e-8
DEFAULT_SMORFCNN_BETAS:                      tuple       = (0.9, 0.999)
DEFAULT_SMORFCNN_MINIMUM_LEARNING_RATE:      float       = 1e-6
DEFAULT_SMORFCNN_SCHEDULER_FACTOR:           float       = 0.01
DEFAULT_SMORFCNN_SCHEDULER_WARMUP:           float       = 0.1
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
DEFAULT_SMORFCNN_TRAIN_BATCH_SIZE:           int         = 64
DEFAULT_SMORFCNN_VALIDATION_BATCH_SIZE:      int         = 64
DEFAULT_SMORFCNN_TEST_BATCH_SIZE:            int         = 64
DEFAULT_SMORFCNN_EPOCHS:                     int         = 10
DEFAULT_SMORFCNN_KFOLD:                      int         = 10
########## ----------- End --------- ##########

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

class HiddenState(Enum):

    CLS     = 0,
    MEAN    = 1,
    BOTH    = 2

class Colours(Enum):

    WHITE   = 0,
    BLUE    = 1,
    GREEN   = 2,
    RED     = 3,
    PURPLE  = 4