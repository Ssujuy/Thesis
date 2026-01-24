import torch,os,random,json
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim.lr_scheduler as lrScheduler
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import pandas as pd

import Types, Helpers
from dataRead import fastaToList
from MultiKernelConvolution import MultiKernelConvolution
from MultiGapKernelConvolution import MultiGapKernelConvolution
from MultiStridedKernelConvolution import MultiStridedKernelConvolution
from dnabert6 import DNABERT6
from ComputationalFeatures import ComputationalFeatures

# ---------------------------------------------
# The CNNClassifier for smORFs
# ---------------------------------------------

class SmORFCNN(nn.Module):
    """
    Multi branch Classifier that predicts coding and non-coding smORF DNA sequences.
    DNABERT-6 branch that produces embeddings features, depending on hidden state (CLS, MEAN or BOTH).
    Uses onehot encoded sequences of size [B, 4, 512] for the Convolution feature extraction branches.    
    Multiple Kernel Convolution branch uses a list of kernel widths to extract features from the 1hot encoded DNA sequences.
    Multiple Gapped Kernel Convolution uses a list of kernel widths along with a list of gaps to extract features from the 1hot encoded DNA sequences.
    Computational Features branch uses Hexamer Score, Fickett Score, Codon Bias and Nucleotide Bias to extract features.

    In order to create a balance between the sizes of features, all the features were reduced to 256.
    Finally, the features are passed to a classifer to determine coding potential.

    Attributes
    ----------
    forwardDebugLimit : int
        Limit for times debug logs are printed in forward.
    
    forwardDebugCounter : int
        Counter for debug logs in forward.

    debugMode : bool
        Turns debug mode on when true (more information).

    onehotInputChannels : int
        Size of onehot encoded sequences.

    embeddingsInputChannels : int
        Size of dnabert6 embeddings.

    trainPath: str
        Path to pyTorch file storing onehot encoded sequences as Tensors [B,4,Length], masked Tensors.
        for valid=1, padded=0 positions and dnabert6 embeddings

    saveModelPathDir : str
        Directory to save the trained model and configuration.
    
    multikernel : bool
        Activates Multiple Kernel Convolution branch for feature extraction.

    multiGapKernel : bool
        Activates Multiple Gap Kernel Convolution branch for feature extraction.

    multiStrideKernel : bool
        Activates Multiple Strided Kernel Convolution branch for feature extraction.

    dnabertEmbeddings : bool
        Activates DNABERT-6 branch for embeddings feature extraction.

    computationalFeatures : bool
        Activates computational branch for feature extraction.

    multiKernelList : list
        List of different kernel sizes for Multi Kernel Convolution.

    multiGapKernelList : list
        List of different kernel sizes Multi Gap Kernel Convolution.

    multiStrideKernelList : list
        List of different kernel sizes Multi Strided Kernel Convolution.

    multiGapKernelGapList : list
        List of gaps for Multi Gap Kernel Convolution.            

    multiStrideKernelStrideList : list       
        List of strides for Multi Strided Kernel Convolution.                    
    
    outputChannelsPerKernel : int
        Size of C_out, output channels per kernel convolution block.
    
    outputChannelsPerGapKernel : int
        Size of C_out, output channels per gap kernel convolution block.

    outputChannelsPerStrideKernel : int
        Size of C_out, output channels per stride kernel convolution block.

    dnabertReductionSize : int
        Size of dnabert6 embeddings after reduction.

    mkcReductionSize : int
        Size of Multiple Kernel Convolution features after reduction.

    mgkcReductionSize : int
        Size of Multiple Gap Kernel Convolution features after reduction.

    mskcReductionSize : int
        Size of Multiple Stride Kernel Convolution features after reduction.

    compFeaturesIncreaseSize : int
        Size of computational features after increase.
    
    classes : int
        Number of classes produced at the end of classifier.

    layer1Output : int
        Output size of layer1 in classifier.

    layer2Output : int
        Output size of layer2 in classifier.

    classifierDropout : float
        Dropout between each layer in classifer.

    learningRate : float
        Initial Learning Rate passed in AdamW optimizer.

    weightDecay : float
        Weight decay parameter passed in optimizer.

    minLearningRate : float
        Minimum Learning Rate parameter passed in scheduler.

    schedulerFactor : float
        Factor parameter passed in scheduler.
    
    schedulerPatience : int
        Patience parameter passed in scheduler.
    
    threshold : float
        threshold used to differentiate negative from positive smORFs, from sigmoid probs.

    maxGradNorm : float
        Maximum grad norm used to clip grad norm.

    seed : int
        Random see of the model.

    deterministic : bool
        Enables model to use only deterministic algorithms, helps reproducibillity.

    device : str
        Device used by model.

    trainBatchSize : int
        Batch size used for training DataLoader.

    valBatchSize : int
        Batch size used for validation DataLoader.

    testBatchSize : int
        Batch size used for testing DataLoader.

    trainSplit : float
        Split of the total Dataset used for training.

    valSplit : float
        plit of the total Dataset used for validation.

    testSplit : float
        plit of the total Dataset used for testing.

    epochs : int
        Number of epochs for fit.
    
    Methods
    ----------
    _initializeEnvironment() -> None:
        Sets environment config for deterministic algorithm (helps with reproducibillity).
        Sets seed and random seed with model's seed.
    
    _calculateFeaturesDim(self) -> int:
        Calculates total features dim, which is the classifier's input channel, based on active branches.
    
    optimizerInit(self) -> torch.optim.Optimizer:
        Initializes AdamW optimizer, adds weight decay to weight kernels and removes it from norms and biases.

    _schedulerInit(self) -> torch.optim.lr_scheduler:
        Initializes sequential Learning rate schedulers, with warmpup LinearLR and cosine CosineAnnealingLR.

    initializeDataset() -> None:
        Uses Helpers function loadFeaturesFromPt in order to load all tensors from pyTorch file in a TensorDataset.
        Then, uses Helpers toDataLoaders function, in order to split the Dataset in training, validation and testing DataLoaders.
        Finally prints some minor information for its split and a label distribution for each DataLoader and sum.
    
    forward(xScores: torch.Tensor, xEmbeddings: torch.Tensor, xOnehot: torch.Tensor, maskOnehot: torch.Tensor) -> torch.Tensor
        Takes 4 Tensor input arguments: xScore is the computational features, xEmbeddings is the dnabert6 embedddings, xOnehot is the onehot encoded dna sequence and maskOnehot is the mask. 
        Passes onehot encoded sequences ([B,4,512]) along with their mask through ([B,512,1]) through Multiple Kernel Convolution and Multiple Gapped Kernel Convolution.
        Finally, creates a features tensor with all features drawn from all branches and passes them to the classifer.
        Logits are returned.


    trainEpoch(epochIndex: int) -> dict:
        Sets the model to training mode with self.train(), initializes loss function BCEWithLogits.
        Iterates the trainingDataLoader and for each batch, moves inputs to device and uses model's forward function.
        Then, computes probabillities and loss, calls back propagation, clips grad norm and uses optimizer and scheduler step.
        Finally, calculates runningLoss and computes all epoch metrics and returns them as a dict. ("loss", "acc", "precision", "recall", "f1", etc.).

    validateEpoch(epochIndex: int) -> dict:
        Uses torch.no_grad(), sets the model to evaluation mode using self.eval(), initializes loss function BCEWithLogits.
        Iterates the validationDataLoader and for each batch, moves inputs to device and uses model's forward function.
        Then, computes probabillities and loss. Finally, calculates runningLoss and computes all epoch metrics and returns them as a dict. ("loss", "acc", "precision", "recall", "f1", etc.), along with epoch ROC AUC.


    test() -> dict:
        Uses torch.no_grad(), sets the model to evaluation mode using self.eval(), initializes loss function BCEWithLogits.
        Iterates the validationDataLoader and for each batch, moves inputs to device and uses model's forward function.
        Then, computes probabillities and loss. Finally, calculates runningLoss and computes all epoch metrics and returns them as a dict. ("loss", "acc", "precision", "recall", "f1", etc.), along with epoch ROC AUC.

    fit(epochs: int):
        Moves model to device, initializes scheduler and keeps dictionaries for training and validation metrics.
        For each epoch trainEpoch and validateEpoch functions are called, their metrics are saved, along with lr and best state.
        Finally after the epochs are finished test function is called to test the model and curves for acc,loss,f1 and auc are printed.

    kFoldCrossValidation(k: int = Types.DEFAULT_SMORFCNN_KFOLD):
        Performs a Stratified k-fold Cross Validation.
        Creates a new Dataset, extracts labels and initializes StratifiedKFold.
        Seed, dataset, optimizer and schedulers change per fold.
        First the kFold CV trains, then validates the best model produced from fit function.
        Finally, the best model is saved.

    predict()
        Reads sequences from FASTA file and converts them to a list.
        For each sequence create 1hot encoded sequence, mask and dnabert6 embeddings.
        Pass the tensor through forward and find the label based on the probabillity from sigmoid.
        Finally create a csv file with all the sequences, labels and probibillities.

        Csv file saved to class atribute path: `predictOutputPath`.

        Finally, prints a count of coding vs non-coding labels and a percentage, based on the total sequences.

    print():
        Prints member variables of the class and number of model parameters and trainable model parameters.

    _debugInit():
        Prints initial member variables of the model and total parameters.

    _debugIn(xOnehot : Tensor, xEmbeddings : Tensor, maskOnehot : Tensor):
        Prints model's forward input shapes and dtypes for xOnehot, maskOnehot and xEmbeddings.
        Prints will occur until limit is reached and debugMode is True.

    debugLogits(self, logits : Tensor):
        Prints logits shape and a small sample.
        Prints will occur until limit is reached and debugMode is True.

    def _debugFinal(probabilities : Tensor, targets : Tensor, runningLoss : float, n : int, function : str, epochIndex : int)
        Prints final stats at the end of epoch. Prints shape of probabillities and targets.
        Prints epoch's loss. Prints will occur only for first epoch.
    """
    def __init__(
        self,
        onehotInputChannels: int,
        embeddingsInputChannels: int,
        trainPath: str,
        predictOutputPath: str,
        hiddenState: str                        = Types.HiddenState.BOTH,
        dnabertDirectory: str                   = Types.DEFAULT_DNABERT6_SAVE_DIRECTORY,
        saveModelPathDir: str                   = Types.DEFAULT_SMORFCNN_SAVE_DIR_PATH,
        multiKernel: bool                       = Types.DEFAULT_SMORFCNN_MULTI_KERNEL,
        multiGapKernel: bool                    = Types.DEFAULT_SMORFCNN_MULTI_GAP_KERNEL,
        multiStrideKernel: bool                 = Types.DEFAULT_SMORFCNN_MULTI_STRIDE_KERNEL,
        dnabertEmbeddings: bool                 = Types.DEFAULT_SMORFCNN_DNABERT,
        computationalFeatures: bool             = Types.DEFAULT_SMORFCNN_COMPUTATIONAL_FEATURES,
        multiKernelList: list                   = Types.DEFAULT_SMORFCNN_MULTI_KERNEL_LIST,
        multiGapKernelList: list                = Types.DEFAULT_SMORFCNN_MULTI_GAP_KERNEL_LIST,
        multiStrideKernelList: list             = Types.DEFAULT_SMORFCNN_MULTI_S_KERNEL_K_LIST,
        multiGapKernelGapList: list             = Types.DEFAULT_SMORFCNN_MULTI_GAP_KERNEL_GAP_LIST,
        multiStrideKernelStrideList: list       = Types.DEFAULT_SMORFCNN_MULTI_S_KERNEL_S_LIST,
        outputChannelsPerKernel: int            = Types.DEFAULT_SMORFCNN_OUTPUT_CHANNELS_KERNEL,
        outputChannelsPerGapKernel: int         = Types.DEFAULT_SMORFCNN_OUTPUT_CHANNELS_G_KERNEL,
        outputChannelsPerStrideKernel: int      = Types.DEFAULT_SMORFCNN_OUTPUT_CHANNELS_S_KERNEL,
        dnabertReductionSize: int               = Types.DEFAULT_SMORFCNN_DNABERT_REDUCTION_SIZE,
        mkcReductionSize: int                   = Types.DEFAULT_SMORFCNN_MKC_REDUCTION_SIZE,
        mgkcReductionSize: int                  = Types.DEFAULT_SMORFCNN_MGKC_REDUCTION_SIZE,
        mskcReductionSize: int                  = Types.DEFAULT_SMORFCNN_MSKC_REDUCTION_SIZE,
        compFeaturesIncreaseSize: int           = Types.DEFAULT_SMORFCNN_SCALAR_INCREASE_SIZE,
        classes: int                            = Types.DEFAULT_SMORFCNN_CLASSES,
        layer1Output: int                       = Types.DEFAULT_SMORFCNN_CLASSIFIER_L1_OUTPUT,
        layer2Output: int                       = Types.DEFAULT_SMORFCNN_CLASSIFIER_L2_OUTPUT,
        classifierDropout: float                = Types.DEFAULT_SMORFCNN_CLASSIFIER_DROPOUT,
        learningRate: float                     = Types.DEFAULT_SMORFCNN_LEARNING_RATE,
        weightDecay: float                      = Types.DEFAULT_SMORFCNN_WEIGHT_DECAY,
        minLearningRate: float                  = Types.DEFAULT_SMORFCNN_MINIMUM_LEARNING_RATE,
        schedulerFactor: float                  = Types.DEFAULT_SMORFCNN_SCHEDULER_FACTOR,
        schedulerPatience: int                  = Types.DEFAULT_SMORFCNN_SCHEDULER_PATIENCE,
        threshold: float                        = Types.DEFAULT_SMORFCNN_THRESHOLD,
        maxGradNorm: float                      = Types.DEFAULT_SMORFCNN_MAX_GRAD_NORM,
        seed: int                               = Types.DEFAULT_SMORFCNN_SEED,
        deterministic: bool                     = Types.DEFAULT_SMORFCNN_DETERMINISTIC,
        device: str                             = Types.DEFAULT_SMORFCNN_DEVICE,
        trainBatchSize: int                     = Types.DEFAULT_SMORFCNN_TRAIN_BATCH_SIZE,
        valBatchSize: int                       = Types.DEFAULT_SMORFCNN_VALIDATION_BATCH_SIZE,
        testBatchSize: int                      = Types.DEFAULT_SMORFCNN_TEST_BATCH_SIZE,
        trainSplit: float                       = Types.DEFAULT_SMORFCNN_TRAIN_SPLIT,
        valSplit: float                         = Types.DEFAULT_SMORFCNN_VALIDATION_SPLIT,
        testSplit: float                        = Types.DEFAULT_SMORFCNN_TEST_SPLIT,
        epochs: int                             = Types.DEFAULT_SMORFCNN_EPOCHS,
        debug: bool                             = Types.DEFAULT_DEBUG_MODE,
        forwardDebugLimit: int                  = Types.DEFAULT_FORWARD_DEBUG_LIMIT
    ):
        """
        Constructs the complete smORF Classifier, using multi branched Multiple Kernel Convolution and DNABERT6 embeddings.
        Also, initializes model's parameters and the final 2-layer classifier.

        Parameters
        ----------
        onehotInputChannels : int
            Size of onehot encoded sequences.

        embeddingsInputChannels : int
            Size of dnabert6 embeddings.

        trainPath: str
            Path to pyTorch file storing onehot encoded sequences as Tensors [B,4,Length], masked Tensors.
            For valid=1, padded=0 positions and dnabert6 embeddings

        saveModelPathDir : str
            Directory to save the trained model and configuration.

        multikernel : bool
            Activates Multiple Kernel Convolution branch for feature extraction.

        multiGapKernel : bool
            Activates Multiple Gap Kernel Convolution branch for feature extraction.

        multiStrideKernel : bool
            Activates Multiple Strided Kernel Convolution branch for feature extraction.

        dnabertEmbeddings : bool
            Activates DNABERT-6 branch for embeddings feature extraction.

        computationalFeatures : bool
            Activates computational branch for feature extraction.

        multiKernelList : list
            List of different kernel sizes for Multi Kernel Convolution.

        multiGapKernelList : list
            List of different kernel sizes Multi Gap Kernel Convolution.
        
        multiStrideKernelList : list
            List of different kernel sizes Multi Strided Kernel Convolution.

        multiGapKernelGapList : list
            List of gaps for Multi Gap Kernel Convolution.

        multiStrideKernelStrideList : list
            List of strides for Multi Strided Kernel Convolution.

        outputChannelsPerKernel : int
            Size of C_out, output channels per kernel convolution block.
        
        outputChannelsPerGapKernel : int
            Size of C_out, output channels per gap kernel convolution block.

        outputChannelsPerStrideKernel : int
            Size of C_out, output channels per stride kernel convolution block.

        dnabertReductionSize : int
            Size of dnabert6 embeddings after reduction.

        mkcReductionSize : int
            Size of Multiple Kernel Convolution features after reduction.

        mgkcReductionSize : int
            Size of Multiple Gap Kernel Convolution features after reduction.

        mskcReductionSize : int
            Size of Multiple Stride Kernel Convolution features after reduction.            

        compFeaturesIncreaseSize : int
            Size of computational features after increase.

        classes : int
            Number of classes produced at the end of classifier.

        layer1Output : int
            Output size of layer1 in classifier.

        layer2Output : int
            Output size of layer2 in classifier.

        classifierDropout : float
            Dropout between each layer in classifer.

        learningRate : float
            Initial Learning Rate passed in AdamW optimizer.

        weightDecay : float
            Weight decay parameter passed in optimizer.

        minLearningRate : float
            Minimum Learning Rate parameter passed in scheduler.

        schedulerFactor : float
            Factor parameter passed in scheduler.
        
        schedulerPatience : int
            Patience parameter passed in scheduler.
        
        threshold : float
            threshold used to differentiate negative from positive smORFs, from sigmoid probs.

        maxGradNorm : float
            Maximum grad norm used to clip grad norm.

        seed : int
            Random see of the model.

        deterministic : bool
            Enables model to use only deterministic algorithms, helps reproducibillity.

        device : str
            Device used by model.

        trainBatchSize : int
            Batch size used for training DataLoader.

        valBatchSize : int
            Batch size used for validation DataLoader.

        testBatchSize : int
            Batch size used for testing DataLoader.

        trainSplit : float
            Split of the total Dataset used for training.

        valSplit : float
            plit of the total Dataset used for validation.

        testSplit : float
            plit of the total Dataset used for testing.

        epochs : int
            Number of epochs for fit.

        debug : bool
            Turns debug mode on when true (more information).

        forwardDebugLimit : int
            Limit for times debug logs are printed in forward.
        """
        super().__init__()

        self.debugMode = debug
        self.forwardDebugCounter = 0
        self.forwardDebugLimit = forwardDebugLimit
        self.seed = seed
        self.deterministic = deterministic
        self.device = device
        self.trainPath = trainPath

        self._initializeEnvironment()

        self.saveModelPathDir = saveModelPathDir
        self.saveModelConfigPath = f"{self.saveModelPathDir}/{self.saveModelPathDir}.json"
        self.saveModelWeightsPath = f"{self.saveModelPathDir}/{self.saveModelPathDir}.pt"

        self.onehotInputChannels = onehotInputChannels
        self.embeddingsInputChannels =  embeddingsInputChannels

        self.dnabertDirectory = dnabertDirectory
        self.dnabertHiddenState = hiddenState

        self.predictOutputPath = predictOutputPath

        self.multiKernel = multiKernel
        self.multiGapKernel = multiGapKernel
        self.multiStrideKernel = multiStrideKernel
        self.dnabertEmbeddings = dnabertEmbeddings
        self.computationalFeatures = computationalFeatures
        self.multiKernelList = multiKernelList
        self.multiGapKernelList = multiGapKernelList
        self.multiStrideKernelList = multiStrideKernelList
        self.multiGapKernelGapList = multiGapKernelGapList
        self.multiStrideKernelStrideList = multiStrideKernelStrideList
        self.outputChannelsPerKernel = outputChannelsPerKernel
        self.outputChannelsPerGapKernel = outputChannelsPerGapKernel
        self.outputChannelsPerStrideKernel = outputChannelsPerStrideKernel
        self.dnabertReductionSize = dnabertReductionSize
        self.mkcReductionSize = mkcReductionSize
        self.mgkcReductionSize = mgkcReductionSize
        self.mskcReductionSize = mskcReductionSize
        self.compFeaturesIncreaseSize = compFeaturesIncreaseSize
        self.layer1Output = layer1Output
        self.layer2Output = layer2Output
        self.classifierDropout = classifierDropout
        self.classes = classes
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.eps = Types.DEFAULT_SMORFCNN_EPS
        self.betas = Types.DEFAULT_SMORFCNN_BETAS
        self.minLearningRate = minLearningRate
        self.schedulerFactor = schedulerFactor
        self.schedulerPatience = schedulerPatience
        self.threshold = threshold
        self.maxGradNorm = maxGradNorm
        self.trainBatchSize = trainBatchSize
        self.validationBatchSize = valBatchSize
        self.testBatchSize = testBatchSize
        self.trainSplit = trainSplit
        self.validationSplit = valSplit
        self.testSplit = testSplit
        self.epochs = epochs

        self.dnabert6Class = None
        self.dnabert6Reduction = None

        self.multiKernelClass = None
        self.multipleKernelReduction = None

        self.multiGapKernelClass = None
        self.multiGapKernelReduction = None

        self.multiStidedKernelClass = None
        self.multiStridedKernelReduction = None

        self.compFeaturesClass = None
        self.compFeaturesIncrease = None

        if self.dnabertEmbeddings:
            self.dnabert6Class = DNABERT6(hiddenState=self.dnabertHiddenState)
            self.dnabert6Class.load(self.dnabertDirectory)

            self.dnabert6Reduction = nn.Sequential(
                nn.Linear(self.embeddingsInputChannels, self.dnabertReductionSize),
                nn.BatchNorm1d(self.dnabertReductionSize),
                nn.GELU(),
                nn.Dropout(self.classifierDropout)
            )

        if self.multiKernel:
            self.multiKernelClass = MultiKernelConvolution(
                inputChannels=self.onehotInputChannels,
                outputChannelsKernel=self.outputChannelsPerKernel,
                kernelList=self.multiKernelList,
                debug=self.debugMode,
                forwardDebugLimit=self.forwardDebugLimit
            )

            self.multipleKernelReduction = nn.Sequential(
                nn.Linear(self.multiKernelClass.outputChannels * 2, self.mkcReductionSize),
                nn.BatchNorm1d(self.mkcReductionSize),
                nn.GELU(),
                nn.Dropout(self.classifierDropout)
            )

        if self.multiStrideKernel:
            self.multiStidedKernelClass = MultiStridedKernelConvolution(
                inputChannels=self.onehotInputChannels,
                outputChannelsSKernel=self.outputChannelsPerKernel,
                kernelList=self.multiStrideKernelList,
                strideList=self.multiStrideKernelStrideList,
                debug=self.debugMode,
                forwardDebugLimit=self.forwardDebugLimit
            )

            self.multiStridedKernelReduction = nn.Sequential(
                nn.Linear(self.multiStidedKernelClass.outputChannels * 2, self.mskcReductionSize),
                nn.BatchNorm1d(self.mskcReductionSize),
                nn.GELU(),
                nn.Dropout(self.classifierDropout)
            )

        if self.multiGapKernel:
            self.multiGapKernelClass = MultiGapKernelConvolution(
                inputChannels=self.onehotInputChannels,
                outputChannelsGKernel=self.outputChannelsPerGapKernel,
                kernelList=self.multiGapKernelList,
                gapList=self.multiGapKernelGapList,
                debug=self.debugMode,
                forwardDebugLimit=self.forwardDebugLimit
            )

            self.multiGapKernelReduction = nn.Sequential(
                nn.Linear(self.multiGapKernelClass.outputChannels * 2, self.mgkcReductionSize),
                nn.BatchNorm1d(self.mgkcReductionSize),
                nn.GELU(),
                nn.Dropout(self.classifierDropout)
            )
        
        if self.computationalFeatures:
            self.compFeaturesClass = ComputationalFeatures(self.trainPath)
            self.compFeaturesIncrease = nn.Sequential(
                    nn.Linear(self.compFeaturesClass.input, self.compFeaturesIncreaseSize),
                    nn.BatchNorm1d(self.compFeaturesIncreaseSize),
                    nn.GELU(),
                    nn.Dropout(self.classifierDropout),
                )

        self.featuresDim = self._calculateFeaturesDim()

        self.classifier = nn.Sequential(
            nn.Linear(self.featuresDim, self.layer1Output),
            nn.GELU(),
            nn.Dropout(self.classifierDropout),

            nn.Linear(self.layer1Output, self.layer2Output),
            nn.GELU(),
            nn.Dropout(self.classifierDropout),

            nn.Linear(self.layer2Output, self.classes)
        )

        self.optimizer = self._optimizerInit()
        self.scheduler = self._schedulerInit()

        self.modelParams = sum(p.numel() for p in self.parameters())
        self.modelTrainableParams = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self._debugInit()

        self.trainDataLoader = None
        self.validationDataLoader = None
        self.testDataLoader = None

        self.to(self.device)

    @classmethod
    def fromJson(cls, path: str) -> "SmORFCNN":
        """
        Read JSON config file from configPath file path, initialize and the SmORFCNN model.
        Using @classmethod, cls can be used as self for SmORFCNN class.

        Parameters
        ----------
        cls : Type[Self@SmORFCNN]
            Same as self for SmORFCNN class.
        
        path : str 
            Path to JSON file to load configuration.
        
        Return
        ----------
        SmORFCNN
            class model SmORFCNN.
        """

        if not Path(path).exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        config = json.loads(Path(path).read_text(encoding="utf-8"))

        if config["onehotInputChannels"] is None:
            raise KeyError("Key onehotInputChannels does not exist in the configuration file.")
        
        if config["embeddingsInputChannels"] is None:
            raise KeyError("Key embeddingsInputChannels does not exist in the configuration file.")

        if config["trainPath"] is None:
            raise KeyError("Key trainPath does not exist in the configuration file.")       

        hiddenState = config.get("hiddenState", Types.HiddenState.BOTH)

        if hiddenState == "cls":
            hiddenState = Types.HiddenState.CLS

        elif hiddenState == "mean":
            hiddenState = Types.HiddenState.MEAN

        elif hiddenState == "both":
            hiddenState = Types.HiddenState.BOTH

        return cls(
            onehotInputChannels         = config["onehotInputChannels"],
            embeddingsInputChannels     = config["embeddingsInputChannels"],
            trainPath                   = config["trainPath"],
            predictOutputPath           = config["predictOutputPath"],
            hiddenState                 = hiddenState,
            dnabertDirectory            = config.get("dnabertDirectory",              Types.DEFAULT_DNABERT6_SAVE_DIRECTORY),
            saveModelPathDir            = config.get("saveModelPathDir",              Types.DEFAULT_SMORFCNN_SAVE_DIR_PATH),
            multiKernel                 = config.get("multiKernel",                   Types.DEFAULT_SMORFCNN_MULTI_KERNEL),
            multiGapKernel              = config.get("multiGapKernel",                Types.DEFAULT_SMORFCNN_MULTI_GAP_KERNEL),
            dnabertEmbeddings           = config.get("dnabertEmbeddings",             Types.DEFAULT_SMORFCNN_DNABERT),
            computationalFeatures       = config.get("computationalFeatures",         Types.DEFAULT_SMORFCNN_COMPUTATIONAL_FEATURES),
            multiKernelList             = config.get("multiKernelList",               Types.DEFAULT_SMORFCNN_MULTI_KERNEL_LIST),
            multiGapKernelList          = config.get("multiGapKernelList",            Types.DEFAULT_SMORFCNN_MULTI_GAP_KERNEL_LIST),
            multiGapKernelGapList       = config.get("multiGapKernelGapList",         Types.DEFAULT_SMORFCNN_MULTI_GAP_KERNEL_GAP_LIST),
            outputChannelsPerKernel     = config.get("outputChannelsPerKernel",       Types.DEFAULT_SMORFCNN_OUTPUT_CHANNELS_KERNEL),
            outputChannelsPerGapKernel  = config.get("outputChannelsPerGapKernel",    Types.DEFAULT_MULTI_GAP_KERNEL_OUTPUT),
            dnabertReductionSize        = config.get("dnabertReductionSize",          Types.DEFAULT_SMORFCNN_DNABERT_REDUCTION_SIZE),
            mkcReductionSize            = config.get("mkcReductionSize",              Types.DEFAULT_SMORFCNN_MKC_REDUCTION_SIZE),
            mgkcReductionSize           = config.get("mgkcReductionSize",             Types.DEFAULT_SMORFCNN_MGKC_REDUCTION_SIZE),
            compFeaturesIncreaseSize    = config.get("compFeaturesIncreaseSize",      Types.DEFAULT_SMORFCNN_SCALAR_INCREASE_SIZE),
            classes                     = config.get("classes",                       Types.DEFAULT_SMORFCNN_CLASSES),
            layer1Output                = config.get("layer1Output",                  Types.DEFAULT_SMORFCNN_CLASSIFIER_L1_OUTPUT),
            layer2Output                = config.get("layer2Output",                  Types.DEFAULT_SMORFCNN_CLASSIFIER_L2_OUTPUT),
            classifierDropout           = config.get("classifierDropout",             Types.DEFAULT_SMORFCNN_CLASSIFIER_DROPOUT),
            learningRate                = config.get("learningRate",                  Types.DEFAULT_SMORFCNN_LEARNING_RATE),
            weightDecay                 = config.get("weightDecay",                   Types.DEFAULT_SMORFCNN_WEIGHT_DECAY),
            minLearningRate             = config.get("minLearningRate",               Types.DEFAULT_SMORFCNN_MINIMUM_LEARNING_RATE),
            schedulerFactor             = config.get("schedulerFactor",               Types.DEFAULT_SMORFCNN_SCHEDULER_FACTOR),
            schedulerPatience           = config.get("schedulerPatience",             Types.DEFAULT_SMORFCNN_SCHEDULER_PATIENCE),
            threshold                   = config.get("threshold",                     Types.DEFAULT_SMORFCNN_THRESHOLD),
            maxGradNorm                 = config.get("maxGradNorm",                   Types.DEFAULT_SMORFCNN_MAX_GRAD_NORM),
            seed                        = config.get("seed",                          Types.DEFAULT_SMORFCNN_SEED),
            deterministic               = config.get("deterministic",                 Types.DEFAULT_SMORFCNN_DETERMINISTIC),
            device                      = config.get("device",                        Types.DEFAULT_SMORFCNN_DEVICE),
            trainBatchSize              = config.get("trainBatchSize",                Types.DEFAULT_SMORFCNN_TRAIN_BATCH_SIZE),
            valBatchSize                = config.get("valBatchSize",                  Types.DEFAULT_SMORFCNN_VALIDATION_BATCH_SIZE),
            testBatchSize               = config.get("testBatchSize",                 Types.DEFAULT_SMORFCNN_TEST_BATCH_SIZE),
            trainSplit                  = config.get("trainSplit",                    Types.DEFAULT_SMORFCNN_TRAIN_SPLIT),
            valSplit                    = config.get("valSplit",                      Types.DEFAULT_SMORFCNN_VALIDATION_SPLIT),
            testSplit                   = config.get("testSplit",                     Types.DEFAULT_SMORFCNN_TEST_SPLIT),
            epochs                      = config.get("epochs",                        Types.DEFAULT_SMORFCNN_EPOCHS),
            debug                       = config.get("debug",                         Types.DEFAULT_DEBUG_MODE),
        )

    def toJson(self) -> dict:
        """
        Saves all attributes of the model inside a dictionary and returns it.

        Return
        ----------
        config
            model's attributes in dictionary.
        """

        hiddenState = ""

        if self.dnabertHiddenState == Types.HiddenState.CLS:
            hiddenState = "cls"

        elif self.dnabertHiddenState == Types.HiddenState.MEAN:
            hiddenState = "mean"

        elif self.dnabertHiddenState == Types.HiddenState.BOTH:
            hiddenState = "both"

        config = {
            "saveModelPathDir":           self.saveModelPathDir,
            "onehotInputChannels":        self.onehotInputChannels,
            "embeddingsInputChannels":    self.embeddingsInputChannels,
            "trainPath":                  self.trainPath,
            "predictOutputPath":          self.predictOutputPath,
            "hiddenState":                hiddenState,
            "dnabertDirectory":           self.dnabertDirectory,
            "multiKernel":                self.multiKernel,
            "multiGapKernel":             self.multiGapKernel,
            "dnabertEmbeddings":          self.dnabertEmbeddings,
            "computationalFeatures":      self.computationalFeatures,
            "multiKernelList":            self.multiKernelList,
            "multiGapKernelList":         self.multiGapKernelList,
            "multiGapKernelGapList":      self.multiGapKernelGapList,
            "outputChannelsPerKernel":    self.outputChannelsPerKernel,
            "outputChannelsPerGapKernel": self.outputChannelsPerGapKernel,
            "dnabertReductionSize":       self.dnabertReductionSize,
            "mkcReductionSize":           self.mkcReductionSize,
            "mgkcReductionSize":          self.mgkcReductionSize,
            "compFeaturesIncreaseSize":   self.compFeaturesIncreaseSize,
            "classes":                    self.classes,
            "layer1Output":               self.layer1Output,
            "layer2Output":               self.layer2Output,
            "classifierDropout":          self.classifierDropout,
            "learningRate":               self.learningRate,
            "weightDecay":                self.weightDecay,
            "minLearningRate":            self.minLearningRate,
            "schedulerFactor":            self.schedulerFactor,
            "schedulerPatience":          self.schedulerPatience,
            "threshold":                  self.threshold,
            "maxGradNorm":                self.maxGradNorm,
            "seed":                       self.seed,
            "deterministic":              self.deterministic,
            "device":                     self.device,
            "trainBatchSize":             self.trainBatchSize,
            "valBatchSize":               self.validationBatchSize,
            "testBatchSize":              self.testBatchSize,
            "trainSplit":                 self.trainSplit,
            "valSplit":                   self.validationSplit,
            "testSplit":                  self.testSplit,
            "epochs":                     self.epochs,
            "debug":                      self.debugMode,
        }

        return config

    def saveConfig(self) -> None:
        """
        Saves config file to saveModelConfigPath file path.
        """

        path = Path(self.saveModelConfigPath)
        path.parent.mkdir(parents=True, exist_ok=True)

        config = self.toJson()

        with open(path, "w", encoding="utf-8") as file:
            json.dump(config, file, indent=2, sort_keys=True)

        Helpers.colourPrint(Types.Colours.GREEN, f"Configuration saved to: {path.resolve()}")

    def _initializeEnvironment(self) -> None:
        """
        Sets environment config for deterministic algorithm (helps with reproducibillity).
        Sets seed and random seed with model's seed.
        """
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        os.environ["PYTHONHASHSEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        torch.manual_seed(self.seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        # if self.deterministic:
        #     try:
        #         torch.use_deterministic_algorithms(True)
        #     except Exception:
        #         pass
        #     torch.backends.cudnn.deterministic = True
        #     torch.backends.cudnn.benchmark = False

    def initializeDataset(self) -> None:
        """
        Uses Helpers function loadFeaturesFromPt in order to load all tensors from pyTorch file in a TensorDataset.
        Then, uses Helpers toDataLoaders function, in order to split the Dataset in training, validation and testing DataLoaders.
        Finally prints some minor information for its split and a label distribution for each DataLoader and sum.
        """

        Helpers.colourPrint(Types.Colours.GREEN, f"Initiallizing dataset for model training from csv in path: {self.trainPath}")

        self.csv_path = Path(self.trainPath)
        dataFrame = pd.read_csv(self.csv_path)

        initialLen = len(dataFrame)
        dataFrame = dataFrame.drop_duplicates(subset=['sequence'])
        cleanLen = len(dataFrame)

        Helpers.colourPrint(Types.Colours.WHITE,f"Dropped {initialLen - cleanLen} duplicate sequences.")
        Helpers.colourPrint(Types.Colours.WHITE,f"Initial Dataset length {initialLen}")
        Helpers.colourPrint(Types.Colours.WHITE,f"Clean Dataset length {cleanLen}")

        self.trainDataLoader, self.validationDataLoader, self.testDataLoader = Helpers.toDataloaders(
            dataFrame,
            self.validationSplit,
            self.testSplit,
            self.trainBatchSize,
            self.validationBatchSize,
            self.testBatchSize,
            self.seed
        )

        Helpers.printDataloader("Training", self.trainDataLoader)
        Helpers.printDataloader("Validation", self.validationDataLoader)
        Helpers.printDataloader("Testing", self.testDataLoader)

        Helpers.plotLabelDistribution(self.trainDataLoader, self.validationDataLoader, self.testDataLoader)

    def _calculateFeaturesDim(self) -> int:
        """
        Calculates total features dim, which is the classifier's input channel, based on active features extraction branches.
        DNABERT branch output + MKC branch output + MGKC branch output + computational branch output.
        
        Return
        ----------
        int
            Calculated features dimension.
        """
        fusedDim = 0

        if self.dnabertEmbeddings:
            fusedDim += self.dnabertReductionSize
        
        if self.multiKernel:
            fusedDim += self.mkcReductionSize
        
        if self.multiGapKernel:
            fusedDim += self.mgkcReductionSize

        if self.multiStrideKernel:
            fusedDim += self.mskcReductionSize

        if self.computationalFeatures:
            fusedDim += self.compFeaturesIncreaseSize

        return fusedDim

    def _optimizerInit(self) -> torch.optim.Optimizer:
        """
        Initializes AdamW optimizer, adds weight decay to weight kernels and removes it from norms and biases.

        Return
        ----------
        torch.optim.Optimizer
            AdamW optimizer.
        """
        decay, noDecay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or name.endswith(".bias"):
                noDecay.append(param)
            else:
                decay.append(param)

        parameterGroups = [
            {"params": decay,    "weight_decay": self.weightDecay},
            {"params": noDecay, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(parameterGroups, lr=self.learningRate, betas=self.betas, eps=self.eps)

    def _schedulerInit(self) -> torch.optim.lr_scheduler:
        """
        Initializes Learning Rate scheduler Reduce LR on plateau.

        Return
        ----------
        torch.optim.lr_scheduler
            Learning Rate scheduler Reduce LR on plateau.
        """

        return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=self.schedulerFactor, 
                patience=self.schedulerPatience, 
            )

    def forward(self, xScores: torch.Tensor, xEmbeddings: torch.Tensor, xOnehot: torch.Tensor, maskOnehot: torch.Tensor) -> torch.Tensor:
        """
        Takes 4 Tensor input arguments: xScore is the computational features, xEmbeddings is the dnabert6 embedddings, xOnehot is the onehot encoded dna sequence and maskOnehot is the mask. 
        Passes onehot encoded sequences ([B,4,512]) along with their mask through ([B,512,1]) through Multiple Kernel Convolution and Multiple Gapped Kernel Convolution.
        Finally, creates a features tensor with all features drawn from all branches and passes them to the classifer.
        Logits are returned.

        Parameters
        ----------
        xScores : Tensor
            Computational Features Tensor [B,4]
        
        xEmbeddings : Tensor
            DNABERT6 embeddings of shape [B,1536,1].

        xOnehot : Tensor
            Onehot encoded sequences of shape [B,4,512].

        maskOnehot : Tensor
            Mask used to differentiate valid from padded positions, shape [B,512,1].

        Return
        ----------
        Tensor
            Logits Tensor after the classifier's output.
        """

        self._debugIn(xOnehot, xEmbeddings, maskOnehot)

        features = []
        inputMKCOneHot = xOnehot
        inputMGKCOneHot = xOnehot
        inputMSKCOneHot = xOnehot
        inputEmbeddings = xEmbeddings
        inputScores = xScores

        if xOnehot is None:
            raise AttributeError("Did not receive onehot encoded input!")
        
        if xEmbeddings is None:
            raise AttributeError("Did not receive dnabert6 embeddings input!")

        if self.multiKernel:

            inputMKCOneHot, maskMKC = self.multiKernelClass(inputMKCOneHot, maskOnehot)

            gap = Helpers.globalAveragePooling(inputMKCOneHot, maskMKC)
            gmp = Helpers.globalMaxPooling(inputMKCOneHot, maskMKC)

            reduction = self.multipleKernelReduction(torch.cat([gap, gmp], dim=1))

            features.append(reduction)

        if self.multiStrideKernel:

            inputMSKCOneHot, maskMSKC = self.multiStidedKernelClass(inputMSKCOneHot, maskOnehot)

            gap = Helpers.globalAveragePooling(inputMSKCOneHot, maskMSKC)
            gmp = Helpers.globalMaxPooling(inputMSKCOneHot, maskMSKC)

            reduction = self.multiStridedKernelReduction(torch.cat([gap, gmp], dim=1))

            features.append(reduction)

        if self.multiGapKernel:

            inputMGKCOneHot, maskMGKC = self.multiGapKernelClass(inputMGKCOneHot, maskOnehot)

            gap = Helpers.globalAveragePooling(inputMGKCOneHot, maskMGKC)
            gmp = Helpers.globalMaxPooling(inputMGKCOneHot, maskMGKC)

            reduction = self.multiGapKernelReduction(torch.cat([gap, gmp], dim=1))

            features.append(reduction)

        if self.dnabertEmbeddings:

            reduction = self.dnabert6Reduction(inputEmbeddings.squeeze(-1))

            features.append(reduction)

        if self.computationalFeatures:

            increase = self.compFeaturesIncrease(inputScores)
            features.append(increase)

        if not features:
            raise RuntimeError("Failed to produce any features!")

        fused = features[0] if len(features) == 1 else torch.cat(features, dim=1)

        if fused.size(1) != self.featuresDim:
            raise ValueError(f"Expected fused dimension {self.featuresDim} , got {fused.size(1)}")

        fused = fused.squeeze(-1)

        logits = self.classifier(fused)

        self._debugLogits(logits)

        if self.debugMode and self.forwardDebugLimit > self.forwardDebugCounter:
            self.forwardDebugCounter += 1

        return logits.squeeze(-1)

    def trainEpoch(self, epochIndex: int) -> dict:
        """
        Sets the model to training mode with self.train(), initializes loss function BCEWithLogits.
        Iterates the trainingDataLoader and for each batch, moves inputs to device and uses model's forward function.
        Then, computes probabillities and loss, calls back propagation, clips grad norm and uses optimizer and scheduler step.
        Finally, calculates runningLoss and computes all epoch metrics and returns them as a dict. ("loss", "acc", "precision", "recall", "f1", etc.).

        Parameters
        ----------
        epochIndex : int
            Current epoch number.

        Return
        ----------
        dict
            Dictionary of all epoch metrics during training ("acc", "loss", precision, etc.).
        """
        self.train()

        n = 0
        runningLoss = 0.0
        probabilities = []
        targets = []

        lossFunction = torch.nn.BCEWithLogitsLoss()

        Helpers.colourPrint(Types.Colours.WHITE, f"Starting training for epoch {epochIndex}")

        try:
            iterator = tqdm(
                    enumerate(self.trainDataLoader),
                    total=len(self.trainDataLoader),
                    desc=f"Epoch {epochIndex}",
                    leave=False
                )
        except Exception:
            iterator = enumerate(self.trainDataLoader)

        for _, batch in iterator:

            xOnehot, maskOnehot, sequences, y = batch

            xEmbed = self.dnabert6Class.embeddings(sequences)

            xScores = self.compFeaturesClass.score(sequences)

            xOnehot = xOnehot.to(self.device)
            maskOnehot = maskOnehot.to(self.device)
            xEmbed = xEmbed.to(self.device)
            xScores = xScores.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            outputs = self(xScores, xEmbed, xOnehot, maskOnehot)

            loss = lossFunction(outputs, y.float())
            probs = torch.sigmoid(outputs)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), self.maxGradNorm)

            self.optimizer.step()
            batchSize = y.size(0)
            n += batchSize
            runningLoss += loss.item() * batchSize
            probabilities.append(probs.detach().view(-1).cpu())
            targets.append(y.detach().long().view(-1).cpu())

            if 'tqdm' in locals():
                iterator.set_postfix(loss=runningLoss / n)

        probabilities = torch.cat(probabilities, dim=0)
        targets = torch.cat(targets, dim=0)

        self._debugFinal(probabilities, targets, runningLoss, n, "Training", epochIndex)

        return Helpers.computeEpochMetrics(probabilities, targets, runningLoss, n, self.threshold, epochIndex)

    @torch.no_grad()
    def validateEpoch(self, epochIndex: int) -> dict:
        """
        Uses torch.no_grad(), sets the model to evaluation mode using self.eval(), initializes loss function BCEWithLogits.
        Iterates the validationDataLoader and for each batch, moves inputs to device and uses model's forward function.
        Then, computes probabillities and loss. Finally, calculates runningLoss and computes all epoch metrics and returns them as a dict. ("loss", "acc", "precision", "recall", "f1", etc.).

        Parameters
        ----------
        epochIndex : int
            Current epoch number.

        Return
        ----------
        dict
            Dictionary of all epoch metrics during validation ("acc", "loss", precision, etc.), along with ROC AUC.
        """
        self.eval()

        n = 0
        runningLoss = 0.0
        probabilities = []
        targets = []

        lossFunction = torch.nn.BCEWithLogitsLoss()

        Helpers.colourPrint(Types.Colours.WHITE, f"Starting validation for epoch {epochIndex}")

        iterator = tqdm(
                enumerate(self.validationDataLoader),
                total=len(self.validationDataLoader),
                desc=f"Epoch {epochIndex}",
                leave=False
            )

        for _, batch in iterator:

            xOnehot, maskOnehot, sequences, y = batch

            xEmbed = self.dnabert6Class.embeddings(sequences)

            xScores = self.compFeaturesClass.score(sequences)

            xOnehot = xOnehot.to(self.device)
            maskOnehot = maskOnehot.to(self.device)
            xEmbed = xEmbed.to(self.device)
            xScores = xScores.to(self.device)
            y = y.to(self.device)

            outputs = self(xScores, xEmbed, xOnehot, maskOnehot)

            loss = lossFunction(outputs, y.float())
            probs = torch.sigmoid(outputs)

            batchSize = y.size(0)
            n += batchSize
            runningLoss += loss.item() * batchSize
            probabilities.append(probs.detach().cpu())
            targets.append(y.detach().cpu())

            if 'tqdm' in locals():
                iterator.set_postfix(loss=runningLoss / n)

        probabilities = torch.cat(probabilities, dim=0)
        targets = torch.cat(targets, dim=0)

        self._debugFinal(probabilities, targets, runningLoss, n, "Validation", epochIndex)

        metrics = Helpers.computeEpochMetrics(probabilities, targets, runningLoss, n, self.threshold, epochIndex)

        metrics = metrics | Helpers.computeEpochROC(probabilities, targets, epochIndex)

        metrics = metrics | Helpers.computeEpochMCC(metrics, epochIndex)

        return metrics

    @torch.no_grad()
    def test(self) -> dict:
        """
        Uses torch.no_grad(), sets the model to evaluation mode using self.eval(), initializes loss function BCEWithLogits.
        Iterates the validationDataLoader and for each batch, moves inputs to device and uses model's forward function.
        Then, computes probabillities and loss. Finally, calculates runningLoss and computes all epoch metrics and returns them as a dict. ("loss", "acc", "precision", "recall", "f1", etc.), along with epoch ROC AUC.

        Return
        ----------
        dict
            Dictionary of all epoch metrics during validation ("acc", "loss", precision, etc.), along with ROC AUC.
        """
        self.eval()

        n = 0
        runningLoss = 0.0
        probabilities = []
        targets = []

        lossFunction = torch.nn.BCEWithLogitsLoss()

        Helpers.colourPrint(Types.Colours.WHITE, f"Starting testing")

        iterator = tqdm(
                enumerate(self.testDataLoader),
                total=len(self.testDataLoader),
                desc=f"{self.testDataLoader}",
                leave=False
            )

        for _, batch in iterator:

            xOnehot, maskOnehot, sequences, y = batch

            xEmbed = self.dnabert6Class.embeddings(sequences)

            xScores = self.compFeaturesClass.score(sequences)

            xOnehot = xOnehot.to(self.device)
            maskOnehot = maskOnehot.to(self.device)
            xEmbed = xEmbed.to(self.device)
            xScores = xScores.to(self.device)
            y = y.to(self.device)

            outputs = self(xScores, xEmbed, xOnehot, maskOnehot)

            loss = lossFunction(outputs, y.float())
            probs = torch.sigmoid(outputs)

            batchSize = y.size(0)
            n += batchSize
            runningLoss += loss.item() * batchSize
            probabilities.append(probs.detach().cpu())
            targets.append(y.detach().cpu()) 

            if 'tqdm' in locals():
                iterator.set_postfix(loss=runningLoss / n)

        probabilities = torch.cat(probabilities, dim=0)
        targets = torch.cat(targets, dim=0)

        self._debugFinal(probabilities, targets, runningLoss, n, "Testing", 1)

        metrics = Helpers.computeEpochMetrics(probabilities, targets, runningLoss, n, self.threshold, 0)

        metrics = metrics | Helpers.computeEpochROC(probabilities, targets, 0)

        metrics = metrics | Helpers.computeEpochMCC(metrics, 0)

        return metrics

    def fit(self, epochs: int):
        """
        Moves model to device, initializes scheduler and keeps dictionaries for training and validation metrics.
        For each epoch trainEpoch and validateEpoch functions are called, their metrics are saved, along with lr and best state.
        Finally after the epochs are finished test function is called to test the model and curves for acc,loss,f1 and auc are printed.

        Parameters
        ----------
        epochIndex : int
            Current epoch number.
        """
        self.to(self.device)

        bestF1 = -1.0
        bestEpoch = 0
        bestState: dict[str, torch.Tensor] | None = None

        trainingMetrics = {
            "loss": [],
            "acc": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "TP": [],
            "TN": [],
            "FP": [],
            "FN": [],
        }

        validationMetrics = {
            "loss": [],
            "acc": [],
            "precision": [],
            "recall": [],
            "learningRate": [],
            "f1": [],
            "TP": [],
            "TN": [],
            "FP": [],
            "FN": [],
            "auc": [],
            "fpr": [],
            "tpr": [],
            "mcc": []
        }

        epochIter = tqdm(
            enumerate(range(1, epochs + 1), start=1),
            total=epochs,
            desc="Epochs",
            leave=True
        )

        for epoch,_ in epochIter:

            epochTrainMetrics = self.trainEpoch(epochIndex=epoch)

            epochValMetrics = self.validateEpoch(epochIndex=epoch)

            for key in epochTrainMetrics:
                trainingMetrics[key].append(epochTrainMetrics[key])

            for key in epochValMetrics:
                validationMetrics[key].append(epochValMetrics[key])

            epochLR = self.optimizer.param_groups[0]["lr"]
            validationMetrics["learningRate"].append(epochLR)

            Helpers.colourPrint(Types.Colours.BLUE, f"[Scheduler] epoch={epoch} val_loss={epochValMetrics["loss"]:.6f} lr={epochLR:.3e}")

            self.scheduler.step(epochValMetrics["loss"])

            if validationMetrics["f1"][epoch - 1] > bestF1:
                bestF1 = validationMetrics["f1"][epoch - 1]
                bestEpoch = epoch
                bestState = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

            try:
                epochIter.set_postfix_str(
                    f"F1 {validationMetrics['f1']:.4f}, AUC {validationMetrics.get('roc_auc', float('nan')):.4f}, LR {epochLR:.2e}"
                )
            except Exception:
                pass

        if bestState is not None:
            self.load_state_dict(bestState)
            self.to(self.device)
        
        testMetrics = self.test()

        Helpers.printFitSummary(trainingMetrics, validationMetrics)

        Helpers.plotFitCurves(trainingMetrics, validationMetrics)

        Helpers.plotROCCurve(validationMetrics, bestEpoch)

        Helpers.plotConfusionPie(trainingMetrics, validationMetrics, testMetrics, epochs)

        self.saveModel()

        return trainingMetrics, validationMetrics

    def kFoldCrossValidation(self, k: int = Types.DEFAULT_SMORFCNN_KFOLD):
        """
        Performs a Stratified k-fold Cross Validation.
        Creates a new Dataset, extracts labels and initializes StratifiedKFold.
        Seed, dataset, optimizer and schedulers change per fold.
        First the kFold CV trains, then validates the best model produced from fit function.
        Finally, the best model is saved.

        Parameters
        ----------
        k : int
            K is the number of folds the Cross Validation will have.
        """

        originalTrainDataLoader = self.trainDataLoader
        originalValidationDataLoader = self.validationDataLoader

        fullDataset = ConcatDataset([originalTrainDataLoader.dataset, originalValidationDataLoader.dataset])

        labels = np.array([item[-1].item() for item in fullDataset])

        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.seed)
        splits = list(splitter.split(np.arange(len(labels)), labels))

        foldMetrics = []
        bestFoldF1 = -1.0
        bestFoldState = None
        bestFold = None

        initialState = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

        iterator = tqdm(enumerate(splits, start=1),total=k,desc=f"{k}-Fold Cross Validation",leave=True)

        for foldIndex, (trainIndex, valIndex) in iterator:

            Helpers.colourPrint(Types.Colours.WHITE, f"\n=== Fold {foldIndex}/{k}: train={len(trainIndex)}  val={len(valIndex)} ===")

            torch.manual_seed(self.seed + foldIndex)
            np.random.seed(self.seed + foldIndex)
            random.seed(self.seed + foldIndex)

            self.load_state_dict(initialState)
            self.to(self.device)

            trainDataset = Subset(fullDataset, indices=trainIndex)
            valDataset = Subset(fullDataset, indices=valIndex)

            self.trainDataLoader = DataLoader(
                trainDataset,
                batch_size=self.trainBatchSize,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                drop_last=False,
                worker_init_fn=lambda wid: np.random.seed(self.seed + foldIndex + wid),
                generator=torch.Generator().manual_seed(self.seed + foldIndex)
            )
            self.validationDataLoader = DataLoader(
                valDataset,
                batch_size=self.validationBatchSize,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )

            self.optimizer  = self._optimizerInit()
            self.scheduler  = self._schedulerInit()

            _ = self.fit(epochs=self.epochs)
            metrics = self.validateEpoch(epochIndex=foldIndex)
            foldMetrics.append(metrics)

            if metrics["f1"] > bestFoldF1:
                bestFoldF1 = metrics["f1"]
                bestFold = foldIndex
                bestFoldState = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

            iterator.set_postfix_str(
                f"F1 {metrics['f1']:.4f}  AUC {metrics.get('roc_auc', float('nan')):.4f}"
            )

        if bestFoldState is not None:
            self.load_state_dict(bestFoldState)
            self.to(self.device)
            Helpers.colourPrint(Types.Colours.RED, f"\nRestored best fold #{bestFold} (val F1={bestFoldF1:.4f})")
            self.saveModel()

        self.trainDataLoader = originalTrainDataLoader
        self.validationDataLoader = originalValidationDataLoader

        summary = Helpers.kFoldSummary(foldMetrics)

        Helpers.printKFoldMetrics(foldMetrics, summary)

        Helpers.plotMeanROC(foldMetrics, summary)

        return foldMetrics, summary

    @torch.no_grad()
    def predict(self, codingFastaPath: str, nCodingFastaPath: str):
        """
        Reads sequences from two separate FASTA files (Coding and Non-Coding).
        Predicts labels and calculates accuracy against the known ground truth.
        """
        self.eval()
        results = []

        try:
            coding = fastaToList(codingFastaPath)
            nonCoding = fastaToList(nCodingFastaPath)
        except Exception as e:
            Helpers.colourPrint(Types.Colours.RED, f"Error reading FASTA files: {e}")
            return

        Helpers.colourPrint(Types.Colours.GREEN, f"Loaded {len(coding)} Coding sequences.")
        Helpers.colourPrint(Types.Colours.GREEN, f"Loaded {len(nonCoding)} Non-Coding sequences.")


        allSequences = [(s, 1) for s in coding] + [(s, 0) for s in nonCoding]

        Helpers.colourPrint(Types.Colours.GREEN, "Starting Prediction...")

        iterator = tqdm(allSequences, desc="Predicting") if 'tqdm' in globals() else allSequences

        for sequence, label in iterator:

            xOnehot = torch.stack([Helpers.sequenceTo1Hot(sequence)], dim=0).to(torch.float32)
            xOnehot = xOnehot.permute(0, 2, 1).contiguous()
            maskOnehot = (xOnehot.sum(dim=1) > 0).to(torch.float32)

            xEmbed = self.dnabert6Class.embeddings([sequence])
            xScores = self.compFeaturesClass.score([sequence])

            xOnehot = xOnehot.to(self.device)
            maskOnehot = maskOnehot.to(self.device)
            xEmbed = xEmbed.to(self.device)
            xScores = xScores.to(self.device)

            outputs = self(xScores, xEmbed, xOnehot, maskOnehot)
            prob = torch.sigmoid(outputs).item()
            pred = int(prob >= self.threshold)

            results.append({
                "sequence": sequence,
                "probability": float(prob),
                "predictedLabel": pred,
                "trueLabel": label
            })

        Helpers.colourPrint(Types.Colours.GREEN, f"Saving predictions to {self.predictOutputPath}")
        df = pd.DataFrame(results)
        df.to_csv(self.predictOutputPath, index=False)

        total = len(df)

        if total > 0:
            predCoding = int(df["predictedLabel"].sum())
            prednCoding = total - predCoding

            correctPredictions = (df["predictedLabel"] == df["trueLabel"]).sum()
            accuracy = (correctPredictions / total) * 100

            codingDf = df[df["trueLabel"] == 1]
            if len(codingDf) > 0:
                codingAcc = (codingDf["predictedLabel"] == 1).mean() * 100
            else: 
                codingAcc = 0.0

            nCodingDf = df[df["trueLabel"] == 0]
            if len(nCodingDf) > 0:
                nCodingAcc = (nCodingDf["predictedLabel"] == 0).mean() * 100
            else:
                nCodingAcc = 0.0

            print("-" * 30)
            Helpers.colourPrint(Types.Colours.GREEN, f"Total Sequences: {total}")
            print("-" * 30)
            Helpers.colourPrint(Types.Colours.BLUE, f"Overall Accuracy: {accuracy:.2f}%")
            print("-" * 30)
            Helpers.colourPrint(Types.Colours.GREEN, f"Correct predictions {correctPredictions}")
            print("-" * 30)            
            Helpers.colourPrint(Types.Colours.GREEN, f"Predicted Coding: {predCoding} (True Coding Accuracy: {codingAcc:.2f}%)")
            Helpers.colourPrint(Types.Colours.GREEN, f"Predicted Non-Coding: {prednCoding} (True Non-Coding Accuracy: {nCodingAcc:.2f}%)")
            print("-" * 30)
        else:
            Helpers.colourPrint(Types.Colours.RED, "No predictions made.")

    def saveModel(self) -> None:
        """
        Save a single checkpoint file with config + weights (+ optional opt/sched) to saveModelWeightsPath file path.
        """

        self.saveConfig()

        path = Path(self.saveModelWeightsPath)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "format_version": 1,
            "model_class": self.__class__.__name__,
            "configPath": self.saveModelConfigPath,
            "state_dict": self.state_dict(),
            "threshold": self.threshold,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
        }

        torch.save(checkpoint, str(path))
        Helpers.colourPrint(Types.Colours.GREEN, f"Trained CNN smORF Classifier successfully saved to: {path.resolve()}")

    @classmethod
    def load(cls, path: str) -> "SmORFCNN":
        """
        Load model (architecture + weights) from directory path, then return the initialized model.

        Parameters
        ----------
        cls : Type[Self@SmORFCNN]
            Same as self for SmORFCNN class.
        
        path : str 
            Path to pyTorch file that the model's checkpoint was saved.
        
        Return
        ----------
        model
            SmORFCNN class model, loaded from pyTorch file and JSOn configuration file.
        """

        path = Path(path)

        savedModel = torch.load(path, map_location="cpu")
        model = cls.fromJson(savedModel["configPath"])
        model.load_state_dict(savedModel["state_dict"], strict=True)
        model.threshold = savedModel["threshold"]
        model.to(model.device)
        model.eval()

        Helpers.colourPrint(Types.Colours.GREEN, f"Trained CNN smORF Classifier successfully loaded from: {path.resolve()}")
        return model

    def _debugInit(self):
        """
        Prints initial member variables of the model and total parameters.
        """

        if self.debugMode:
            Helpers.colourPrint(Types.Colours.PURPLE, f"[SmORFCNN][INIT] total params={self.modelParams} trainable={self.modelTrainableParams}")
            Helpers.colourPrint(Types.Colours.PURPLE, f"[SmORFCNN][INIT] device={self.device} multiKernel={self.multiKernel} multiGapKernel={self.multiGapKernel}")
            Helpers.colourPrint(Types.Colours.PURPLE, f"[SmORFCNN][INIT] fusedDim={self.featuresDim} classifier={self.classifier}")

    def _debugIn(
            self,
            xOnehot: torch.Tensor,
            xEmbeddings: torch.Tensor,
            maskOnehot: torch.Tensor
        ):
        """
        Prints model's forward input shapes and dtypes for xOnehot, maskOnehot and xEmbeddings.
        Prints will occur until limit is reached and debugMode is True.

        Parameters
        ----------
        xOnehot : Tensor
            Input Tensor of onehot encoded sequences.

        maskOnehot : Tensor
            Input Tensor for mask of onehot encoded sequences.

        xEmbeddings : Tensor
            Embeddings Tensor for dnabert6 embeddings.
        """

        if self.debugMode and self.forwardDebugLimit > self.forwardDebugCounter:
            Helpers.colourPrint(Types.Colours.PURPLE, f"[SmORFCNN.forward] Input xOnehot shape={tuple(xOnehot.shape)}-dtype={xOnehot.dtype}")
            Helpers.colourPrint(Types.Colours.PURPLE, f"[SmORFCNN.forward] Input xEmbeddings shape={tuple(xEmbeddings.shape)}-dtype={xEmbeddings.dtype}")
            Helpers.colourPrint(Types.Colours.PURPLE, f"[SmORFCNN.forward] Input maskOnehot shape={tuple(maskOnehot.shape)}-dtype={maskOnehot.dtype}")
            Helpers.colourPrint(Types.Colours.PURPLE, f"[SmORFCNN.forward] Input maskOnehot sum per-batch={maskOnehot.sum(dim=1).detach().cpu().tolist()[:8]}")
    
    def _debugLogits(self, logits: torch.Tensor):
        """
        Prints logits shape and a small sample.
        Prints will occur until limit is reached and debugMode is True.

        Parameters
        ----------
        logits : Tensor
            Logits produced from classifier.
        """

        if self.debugMode and self.forwardDebugLimit > self.forwardDebugCounter:
            Helpers.colourPrint(Types.Colours.PURPLE, f"[SmORFCNN.forward] Logits shape={tuple(logits.shape)}")
            Helpers.colourPrint(Types.Colours.PURPLE, f"[SmORFCNN.forward] Logits sample={logits.detach().view(-1)[:8].cpu().tolist()}")
    
    def _debugFinal(
            self,
            probabilities: torch.Tensor,
            targets: torch.Tensor,
            runningLoss: float,
            n: int,
            function: str,
            epochIndex: int
        ):
        """
        Prints final stats at the end of epoch. Prints shape of probabillities and targets.
        Prints epoch's loss. Prints will occur only for first epoch.

        Parameters
        ----------
        probabilities : Tensor
            Probabillities Tensor.

        targets : Tensor
            Targets Tensor.

        runningLoss : float
            Epoch's current loss.

        n : int
            Number of batches.
        
        function : str
            Function name calling debugFinal.

        epchIndex: int
            Current epoch.   
        """

        if epochIndex == 1 and self.debugMode:
            Helpers.colourPrint(Types.Colours.PURPLE, f"[SmORFCNN][{function} Epoch-{epochIndex}] End of epoch probs shape={tuple(probabilities.shape)} targets shape={tuple(targets.shape)}")
            Helpers.colourPrint(Types.Colours.PURPLE, f"[SmORFCNN][{function} Epoch-{epochIndex}] End of epoch loss average={runningLoss/max(1,n):.6f}")


mymodel = SmORFCNN(4,1536,"train.csv","feljfow",debug=False,multiStrideKernel=False)
#mymodel = SmORFCNN.load("smorfCNN/smorfCNN.pt")
mymodel.initializeDataset()
mymodel.fit(10)
#mymodel.predict("CPPred_test/Human.small_coding_RNA_test.fa", "CPPred_test/Homo38.small_ncrna_test.fa")
#mymodel.predict("csORF-finder_test_data/H.sapiens_Ribo-csORFs_testp.txt", "csORF-finder_test_data/H.sapiens_Ribo-ncsORFs_testn.txt")
#mymodel.predict("DeepCPP_raw_data/human_mrnasorf.fa", "DeepCPP_raw_data/human_lncsorf.fa")