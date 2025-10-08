import torch,os,random
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim.lr_scheduler as lrScheduler
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

import Types, Helpers
from MultiKernelConvolution import MultiKernelConvolution
from MultiGapKernelConvolution import MultiGapKernelConvolution
from TemporalHead import TemporalHead

# ---------------------------------------------
# The CNNClassifier for smORFs
# ---------------------------------------------

class SmORFCNN(nn.Module):
    """
    Multi branch Classifier that takes dnabert 6 embeddings, CLS embeddings and mean embeddings of size [B,1536,1].\n
    Uses onehot encoded sequences of size [B, 4, 512] that are passed through a Multi Kernel Convolution and then are\n
    concatenated creating [B, C_out * len(kernelList), L_min]. The features from Multi Kernel are passed to Temporal Head,\n
    which reduces the feature's size [B, reduces, 498] then passes them through residual mapping 2 residual blocks and computes\n
    Global Average Pooling and Global Max Pooling [B, C_out * 2].

    
    Attributes
    ----------
    saveModelPath : str
        Path to save the trained model.

    onehotInputChannels : int
        Size of onehot encoded sequences.

    embeddingsInputChannels : int
        Size of dnabert6 embeddings.

    featuresPath: str
        Path to pyTorch file storing onehot encoded sequences as Tensors [B,4,Length], masked Tensors\n
        for valid=1, padded=0 positions and dnabert6 embeddings

    temporalHead : bool
        Activates Temporal Head Block.
    
    multikernel : bool
        Activates Multi Kernel Block.

    multiKernelList : list
        List of different kernel sizes for Multi Kernel Convolution.

    multiGapKernelList : list
        List of different kernel sizes Multi Gap Kernel Convolution.
    
    multiGapKernelGapList : list
        List of gaps for Multi Gap Kernel Convolution.            

    outputChannelsPerKernel : int
        Size of C_out, output channels per kernel convolution block.
    
    outputChannelsPerGapKernel : int
        Size of C_out, output channels per gap kernel convolution block.

    temporalHeadOutputChannels : int
        Size of C_out, output channels of temporal head block (* 2).

    residualBlocks : int
        Number of residual blocks used in Temporal head.

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
    
    schedulerWarmup : float
        Warmup parameter passed in scheduler.
    
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
        Sets environment config for deterministic algorithm (helps with reproducibillity),\n
        Sets seed and random seed with model's seed.
    
    _calculateFeaturesDim(self) -> int:
        Calculates total features dim, which is the classifier's input channel, based on active branches.
    
    optimizerInit(self) -> torch.optim.Optimizer:
        Initializes AdamW optimizer, adds weight decay to weight kernels\n
        and removes it from norms and biases.

    _schedulerInit(self) -> torch.optim.lr_scheduler:
        Initializes sequential Learning rate schedulers, with warmpup LinearLR and cosine CosineAnnealingLR.

    initializeDataset() -> None:
        Uses Helpers function loadFeaturesFromPt in order to load all tensors from pyTorch file in a TensorDataset.\n
        Then, uses Helpers toDataLoaders function, in order to split the Dataset in training, validation and testing DataLoaders.\n
        Finally prints some minor information for its split and a label distribution for each DataLoader and sum.
    
    forward(xOnehot: Tensor, xEmbeddings: Tensor, maskOnehot: Tensor) -> Tensor:
        Takes dnabert6 embeddings and appends them to features list.\n
        Passes onehot encoded sequences ([B,4,512]) along with their mask through ([B,512,1]) through Multiple Kernel Convolution,\n
        then through Temporal Head and appends them to the features list as well.\n
        Finally, creates a features tensor with all features drawn from all branches and passes them to the classifer.\n
        Logits are returned.


    trainEpoch(epochIndex: int) -> dict:
        Sets the model to training mode with self.train(), initializes loss function BCEWithLogits.\n
        Iterates the trainingDataLoader and for each batch, moves inputs to device and uses model's forward function.\n
        Then, computes probabillities and loss, calls back propagation, clips grad norm and uses optimizer and scheduler step.\n
        Finally, calculates runningLoss and computes all epoch metrics and returns them as a dict. ("loss", "acc", "precision", "recall", "f1", etc.).

    validateEpoch(epochIndex: int) -> dict:
        Uses torch.no_grad(), sets the model to evaluation mode using self.eval(), initializes loss function BCEWithLogits.\n
        Iterates the validationDataLoader and for each batch, moves inputs to device and uses model's forward function.\n
        Then, computes probabillities and loss. Finally, calculates runningLoss and computes all epoch metrics\n
        and returns them as a dict. ("loss", "acc", "precision", "recall", "f1", etc.), along with epoch ROC AUC.


    test() -> dict:
        Uses torch.no_grad(), sets the model to evaluation mode using self.eval(), initializes loss function BCEWithLogits.\n
        Iterates the validationDataLoader and for each batch, moves inputs to device and uses model's forward function.\n
        Then, computes probabillities and loss. Finally, calculates runningLoss and computes all epoch metrics\n 
        and returns them as a dict. ("loss", "acc", "precision", "recall", "f1", etc.), along with epoch ROC AUC.

    fit(epochs: int):
        Moves model to device, initializes scheduler and keeps dictionaries for training and validation metrics.\n
        For each epoch trainEpoch and validateEpoch functions are called, their metrics are saved, along with lr and best state.\n
        Finally after the epochs are finished test function is called to test the model and curves for acc,loss,f1 and auc are printed.

    kFoldCrossValidation()

    print():
        Prints member variables of the class and number of model parameters and trainable model parameters.
    """
    def __init__(
        self,
        saveModelPath: str,
        onehotInputChannels: int,
        embeddingsInputChannels: int,
        featuresPath: str,
        multiKernel: bool                       = Types.DEFAULT_SMORFCNN_MULTI_KERNEL,
        multiGapKernel: bool                    = Types.DEFAULT_SMORFCNN_MULTI_GAP_KERNEL,
        dnabertEmbeddings: bool                 = Types.DEFAULT_SMORFCNN_DNABERT,
        multiKernelList: list                   = Types.DEFAULT_SMORFCNN_MULTI_KERNEL_LIST,
        multiGapKernelList: list                = Types.DEFAULT_SMORFCNN_MULTI_GAP_KERNEL_LIST,
        multiGapKernelGapList: list             = Types.DEFAULT_SMORFCNN_MULTI_GAP_KERNEL_GAP_LIST,
        outputChannelsPerKernel: int            = Types.DEFAULT_SMORFCNN_OUTPUT_CHANNELS_KERNEL,
        outputChannelsPerGapKernel: int         = Types.DEFAULT_MULTI_GAP_KERNEL_OUTPUT,
        temporalHeadOutputChannels: int         = Types.DEFAULT_SMORFCNN_OUTPUT_CHANNELS_TEMPORAL,
        residualBlocks: int                     = Types.DEFAULT_SMORFCNN_RESIDUAL_BLOCKS,
        classes: int                            = Types.DEFAULT_SMORFCNN_CLASSES,
        layer1Output: int                       = Types.DEFAULT_SMORFCNN_CLASSIFIER_L1_OUTPUT,
        layer2Output: int                       = Types.DEFAULT_SMORFCNN_CLASSIFIER_L2_OUTPUT,
        classifierDropout: float                = Types.DEFAULT_SMORFCNN_CLASSIFIER_DROPOUT,
        learningRate: float                     = Types.DEFAULT_SMORFCNN_LEARNING_RATE,
        weightDecay: float                      = Types.DEFAULT_SMORFCNN_WEIGHT_DECAY,
        minLearningRate: float                  = Types.DEFAULT_SMORFCNN_MINIMUM_LEARNING_RATE,
        schedulerFactor: float                  = Types.DEFAULT_SMORFCNN_SCHEDULER_FACTOR,
        schedulerWarmup: float                  = Types.DEFAULT_SMORFCNN_SCHEDULER_WARMUP,
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
        debug: bool                             = Types.DEFAULT_DEBUG_MODE
    ):
        """
        Constructs the complete smORF Classifier, using multi branched Multiple Kernel Convolution and DNABERT6 embeddings.\n
        Also, initializes model's parameters and the final 2-layer classifier.

        Parameters
        ----------
        saveModelPath : str
            Path to save the trained model.

        onehotInputChannels : int
            Size of onehot encoded sequences.

        embeddingsInputChannels : int
            Size of dnabert6 embeddings.

        featuresPath: str
            Path to pyTorch file storing onehot encoded sequences as Tensors [B,4,Length], masked Tensors\n
            for valid=1, padded=0 positions and dnabert6 embeddings

        temporalHead : bool
            Activates Temporal Head Block.
        
        multikernel : bool
            Activates Multi Kernel Block.

        multiKernelList : list
            List of different kernel sizes for Multi Kernel Convolution.

        multiGapKernelList : list
            List of different kernel sizes Multi Gap Kernel Convolution.
        
        multiGapKernelGapList : list
            List of gaps for Multi Gap Kernel Convolution.            

        outputChannelsPerKernel : int
            Size of C_out, output channels per kernel convolution block.
        
        outputChannelsPerGapKernel : int
            Size of C_out, output channels per gap kernel convolution block.

        temporalHeadOutputChannels : int
            Size of C_out, output channels of temporal head block (* 2).

        residualBlocks : int
            Number of residual blocks used in Temporal head.

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
        
        schedulerWarmup : float
            Warmup parameter passed in scheduler.
        
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
        """
        super().__init__()

        self.debugMode = debug
        self.forwardDebugOnce = debug
        self.seed = seed
        self.deterministic = deterministic
        self.device = device
        self.featuresPath = featuresPath

        self._initializeEnvironment()

        self.saveModelPath = saveModelPath
        self.onehotInputChannels = onehotInputChannels
        self.embeddingsInputChannels =  embeddingsInputChannels

        self.multiKernel = multiKernel
        self.multiGapKernel = multiGapKernel
        self.dnabertEmbeddings = dnabertEmbeddings
        self.multiKernelList = multiKernelList
        self.multiGapKernelList = multiGapKernelList
        self.multiGapKernelGapList = multiGapKernelGapList
        self.outputChannelsPerKernel = outputChannelsPerKernel
        self.outputChannelsPerGapKernel = outputChannelsPerGapKernel
        self.temporalHeadOutputChannels = temporalHeadOutputChannels
        self.residualBlocks = residualBlocks
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
        self.schedulerWarmpup = schedulerWarmup
        self.threshold = threshold
        self.maxGradNorm = maxGradNorm
        self.trainBatchSize = trainBatchSize
        self.validationBatchSize = valBatchSize
        self.testBatchSize = testBatchSize
        self.trainSplit = trainSplit
        self.validationSplit = valSplit
        self.testSplit = testSplit
        self.epochs = epochs

        self.multiKernelClass = None
        self.multiKernelTemporalClass = None
        self.multiGapKernelClass = None
        self.multiGapKernelTemporalClass = None

        if self.multiKernel:
            self.multiKernelClass = MultiKernelConvolution(
                inputChannels=self.onehotInputChannels,
                outputChannelsKernel=self.outputChannelsPerKernel,
                kernelList=self.multiKernelList
            )

            self.multiKernelTemporalClass = TemporalHead(
                self.multiKernelClass.outputChannels,
                hiddenChannels=self.temporalHeadOutputChannels,
                residualBlocks=self.residualBlocks
            )

        if self.multiGapKernel:
            self.multiGapKernelClass = MultiGapKernelConvolution(
                inputChannels=self.onehotInputChannels,
                outputChannelsGKernel=self.outputChannelsPerGapKernel,
                kernelList=self.multiGapKernelList,
                gapList=self.multiGapKernelGapList
            )

            self.multiGapKernelTemporalClass = TemporalHead(
                self.multiGapKernelClass.outputChannels,
                hiddenChannels=self.temporalHeadOutputChannels,
                residualBlocks=self.residualBlocks
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

        self.modelParams = sum(p.numel() for p in self.parameters())
        self.modelTrainableParams = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self._debugInit()

        self.trainDataLoader = None
        self.validationDataLoader = None
        self.testDataLoader = None

        self.to(self.device)

    def _initializeEnvironment(self) -> None:
        """
        Sets environment config for deterministic algorithm (helps with reproducibillity),\n
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

        if self.deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def initializeDataset(self) -> None:
        """
        Uses Helpers function loadFeaturesFromPt in order to load all tensors from pyTorch file in a TensorDataset.\n
        Then, uses Helpers toDataLoaders function, in order to split the Dataset in training, validation and testing DataLoaders.\n
        Finally prints some minor information for its split and a label distribution for each DataLoader and sum.
        """
        tensorDataset = Helpers.loadFeaturesFromPt(self.featuresPath)

        self.trainDataLoader, self.validationDataLoader, self.testDataLoader = Helpers.toDataloaders(
            tensorDataset,
            self.trainSplit,
            self.validationSplit,
            self.testSplit,
            self.trainBatchSize,
            self.validationBatchSize,
            self.testBatchSize,
            self.seed
        )

        self._debugDataLoader()

        Helpers.printDataloader("Training", self.trainDataLoader)
        Helpers.printDataloader("Validation", self.validationDataLoader)
        Helpers.printDataloader("Testing", self.testDataLoader)

        Helpers.plotLabelDistribution(self.trainDataLoader, self.validationDataLoader, self.testDataLoader)

    def _calculateFeaturesDim(self) -> int:
        """
        Calculates total features dim, which is the classifier's input channel, based on active branches.
        
        Return
        ----------
        int
            Calculated features dimension.
        """
        fusedDim = 0

        if self.dnabertEmbeddings:
            fusedDim += self.embeddingsInputChannels
        
        if self.multiKernel:
            fusedDim += 2 * self.temporalHeadOutputChannels
        
        if self.multiGapKernel:
            fusedDim += 2 * self.temporalHeadOutputChannels

        return fusedDim

    def _optimizerInit(self) -> torch.optim.Optimizer:
        """
        Initializes AdamW optimizer, adds weight decay to weight kernels\n
        and removes it from norms and biases.

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
        Initializes sequential Learning rate schedulers, with warmpup LinearLR and cosine CosineAnnealingLR.
        """
        if self.trainDataLoader is None:
            AttributeError("Traindataloader must be initialized to calculate size")

        stepsPerEpoch = len(self.trainDataLoader)
        totalSteps = stepsPerEpoch * self.epochs
        warmupSteps = int(self.schedulerWarmpup * totalSteps)
        cosineSteps = totalSteps - warmupSteps

        warmup = lrScheduler.LinearLR(self.optimizer, start_factor=self.schedulerFactor, total_iters=warmupSteps)

        cosine = lrScheduler.CosineAnnealingLR(self.optimizer, T_max=cosineSteps, eta_min=self.minLearningRate)

        return lrScheduler.SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[warmupSteps])    

    def forward(self, xOnehot: torch.Tensor, xEmbeddings:  torch.Tensor, maskOnehot: torch.Tensor) -> torch.Tensor:
        """
        Takes dnabert6 embeddings and appends them to features list.\n
        Passes onehot encoded sequences ([B,4,512]) along with their mask through ([B,512,1]) through Multiple Kernel Convolution,\n
        then through Temporal Head and appends them to the features list as well.\n
        Finally, creates a features tensor with all features drawn from all branches and passes them to the classifer.\n
        Logits are returned.

        Parameters
        ----------
        xOnehot : Tensor
            Onehot encoded sequences of shape [B,4,512].

        xEmbeddings : Tensor
            DNABERT6 embeddings of shape [B,1536,1].

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
        inputEmbeddings = xEmbeddings

        if xOnehot is None:
            raise AttributeError("Did not receive onehot encoded input!")
        
        if xEmbeddings is None:
            raise AttributeError("Did not receive dnabert6 embeddings input!")

        if self.multiKernel:

            inputMKCOneHot, maskMKC = self.multiKernelClass(inputMKCOneHot, maskOnehot)
            inputMKCOneHot = self.multiKernelTemporalClass(inputMKCOneHot, maskMKC)
            features.append(inputMKCOneHot)

        if self.multiGapKernel:

            inputMGKCOneHot, maskMGKC = self.multiGapKernelClass(inputMGKCOneHot, maskOnehot)
            inputMGKCOneHot = self.multiGapKernelTemporalClass(inputMGKCOneHot, maskMGKC)
            features.append(inputMGKCOneHot)

        if self.dnabertEmbeddings:

            features.append(inputEmbeddings.squeeze(-1))

        if not features:
            raise RuntimeError("Failed to produce any features!")

        fused = features[0] if len(features) == 1 else torch.cat(features, dim=1)

        if fused.size(1) != self.featuresDim:
            raise ValueError(f"Expected fused dimension {self.featuresDim} , got {fused.size(1)}")

        fused = fused.squeeze(-1)

        logits = self.classifier(fused)

        self._debugLogits(logits)

        return logits.squeeze(-1) if self.classes == 1 else logits

    def trainEpoch(self, epochIndex: int) -> dict:
        """
        Sets the model to training mode with self.train(), initializes loss function BCEWithLogits.\n
        Iterates the trainingDataLoader and for each batch, moves inputs to device and uses model's forward function.\n
        Then, computes probabillities and loss, calls back propagation, clips grad norm and uses optimizer and scheduler step.\n
        Finally, calculates runningLoss and computes all epoch metrics and returns them as a dict. ("loss", "acc", "precision", "recall", "f1", etc.).

        Parameters
        ----------
        epochIndex : int
            Current epoch number.

        Return
        ----------
        dict
            Dictionary of all epoch metrics during triaing ("acc", "loss", precision, etc.).
        """
        self.train()

        n = 0
        runningLoss = 0.0
        probabilities = []
        targets = []

        lossFunction = torch.nn.BCEWithLogitsLoss()

        print(f"Starting training for epoch {epochIndex}")

        try:
            iterator = tqdm(
                    enumerate(self.trainDataLoader),
                    total=len(self.trainDataLoader),
                    desc=f"Epoch {epochIndex}",
                    leave=False
                )
        except Exception:
            iterator = enumerate(self.trainDataLoader)

        for i, batch in iterator:
            xOnehot, maskOnehot, xEmbed, y = batch

            self._debugInEpoch(xOnehot, maskOnehot, xEmbed, y, "Training", i, epochIndex)

            xOnehot = xOnehot.to(self.device)
            maskOnehot = maskOnehot.to(self.device)
            xEmbed = xEmbed.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            outputs = self(xOnehot, xEmbed, maskOnehot)

            self._debugOutEpoch(outputs, "Training", i, epochIndex)

            loss = lossFunction(outputs, y.float())
            probs = torch.sigmoid(outputs)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), self.maxGradNorm)

            self.optimizer.step()
            self.scheduler.step()
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
        Uses torch.no_grad(), sets the model to evaluation mode using self.eval(), initializes loss function BCEWithLogits.\n
        Iterates the validationDataLoader and for each batch, moves inputs to device and uses model's forward function.\n
        Then, computes probabillities and loss. Finally, calculates runningLoss and computes all epoch metrics\n
        and returns them as a dict. ("loss", "acc", "precision", "recall", "f1", etc.).

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

        print(f"Starting validation for epoch {epochIndex}")

        iterator = tqdm(
                enumerate(self.validationDataLoader),
                total=len(self.validationDataLoader),
                desc=f"Epoch {epochIndex}",
                leave=False
            )

        for j, batch in iterator:

            xOnehot, maskOnehot, xEmbed, y = batch

            self._debugInEpoch(xOnehot, maskOnehot, xEmbed, y, "Validation", j, epochIndex)

            xOnehot = xOnehot.to(self.device)
            maskOnehot = maskOnehot.to(self.device)
            xEmbed = xEmbed.to(self.device)
            y = y.to(self.device)

            outputs = self(xOnehot, xEmbed, maskOnehot)

            self._debugOutEpoch(outputs, "Validation", j, epochIndex)

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

        return metrics

    @torch.no_grad()
    def test(self) -> dict:
        """
        Uses torch.no_grad(), sets the model to evaluation mode using self.eval(), initializes loss function BCEWithLogits.\n
        Iterates the validationDataLoader and for each batch, moves inputs to device and uses model's forward function.\n
        Then, computes probabillities and loss. Finally, calculates runningLoss and computes all epoch metrics\n 
        and returns them as a dict. ("loss", "acc", "precision", "recall", "f1", etc.), along with epoch ROC AUC.

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

        print(f"Starting testing")

        iterator = tqdm(
                enumerate(self.testDataLoader),
                total=len(self.testDataLoader),
                desc=f"{self.testDataLoader}",
                leave=False
            )

        for k, batch in iterator:

            xOnehot, maskOnehot, xEmbed, y = batch

            self._debugInEpoch(xOnehot, maskOnehot, xEmbed, y, "Testing", k, 1)

            xOnehot = xOnehot.to(self.device)
            maskOnehot = maskOnehot.to(self.device)
            xEmbed = xEmbed.to(self.device)
            y = y.to(self.device)

            outputs = self(xOnehot, xEmbed, maskOnehot)

            self._debugOutEpoch(outputs, "Testing", k, 1)

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

        return metrics

    def fit(self, epochs: int):
        """
        Moves model to device, initializes scheduler and keeps dictionaries for training and validation metrics.\n
        For each epoch trainEpoch and validateEpoch functions are called, their metrics are saved, along with lr and best state.\n
        Finally after the epochs are finished test function is called to test the model and curves for acc,loss,f1 and auc are printed.

        Parameters
        ----------
        epochIndex : int
            Current epoch number.
        """
        self.to(self.device)

        self.scheduler = self._schedulerInit()

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
            "tpr": []
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
            print(f"[Scheduler] epoch={epoch} val_loss={epochValMetrics["loss"]:.6f} lr={epochLR:.3e}")


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
        """

        originalTrainDataLoader = self.trainDataLoader
        originalValidationDataLoader = self.validationDataLoader

        fullDataset = ConcatDataset([originalTrainDataLoader.dataset, originalValidationDataLoader.dataset])

        tSubset, vSubset = originalTrainDataLoader.dataset, originalValidationDataLoader.dataset

        tBase, vBase = tSubset.dataset, vSubset.dataset

        tIdx = torch.as_tensor(tSubset.indices, dtype=torch.long)
        vIdx = torch.as_tensor(vSubset.indices, dtype=torch.long)

        tLabels = tBase.tensors[-1].index_select(0, tIdx)
        vLabels = vBase.tensors[-1].index_select(0, vIdx)
        labels = torch.cat([tLabels, vLabels], dim=0).detach().cpu().long().numpy()

        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.seed)
        splits = list(splitter.split(np.arange(len(labels)), labels))

        foldMetrics = []
        bestFoldF1 = -1.0
        bestFoldState = None
        bestFold = None

        initialState = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

        iterator = tqdm(enumerate(splits, start=1),total=k,desc=f"{k}-Fold CV",leave=True)

        for foldIndex, (trainIndex, valIndex) in iterator:

            print(f"\n=== Fold {foldIndex}/{k}: train={len(trainIndex)}  val={len(valIndex)} ===")

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
            print(f"\nRestored best fold #{bestFold} (val F1={bestFoldF1:.4f})")

        self.trainDataLoader = originalTrainDataLoader
        self.validationDataLoader = originalValidationDataLoader

        summary = Helpers.kFoldSummary(foldMetrics)

        Helpers.printKFoldMetrics(foldMetrics, summary)

        Helpers.plotMeanROC(foldMetrics, summary)

        return foldMetrics, summary

    def saveModel(self) -> None:
        """
        Save a single checkpoint file with config + weights (+ optional opt/sched).
        """
        path = Path(self.saveModelPath)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "format_version": 1,
            "model_class": self.__class__.__name__,
            # "config": self.to_config(),
            "state_dict": self.state_dict(),
            "threshold": getattr(self, "threshold", None),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
        }

        torch.save(checkpoint, str(path))
        print(f"Trained CNN smORF Classifier successfully saved to: {path.resolve()}")

    @classmethod
    def load(path: str, strict: bool = True) -> "SmORFCNN":
        """
        Load model (architecture + weights) for inference or training.
        """
        ckpt = torch.load(str(path), map_location="cpu")
        config = ckpt["config"]
        model = cls.from_config(config)
        model.load_state_dict(ckpt["state_dict"], strict=strict)
        model.threshold = ckpt["threshold"]
        model.to(config.get("device", "cpu"))
        model.eval()
        return model

    def _debugInit(self):

        if self.debugMode:
            print(f"[INIT] total params={self.modelParams} trainable={self.modelTrainableParams}")
            print(f"[INIT] optimizer param_groups sizes={[len(g['params']) for g in self.optimizer.param_groups]}")
            print(f"[INIT] device={self.device} multiKernel={self.multiKernel} temporalHead={self.temporalHead}")
            print(f"[INIT] onehot kernels={self.onehotKernelList}")
            print(f"[INIT] fusedDim={self.featuresDim} classifier={self.classifier}")
    
    def _debugDataLoader(self):

        if self.debugMode:
            tIdx  = set(self.trainDataLoader.dataset.indices)
            vIdx  = set(self.validationDataLoader.dataset.indices)
            teIdx = set(self.testDataLoader.dataset.indices)
            print(f"[SPLIT] |train|={len(tIdx)} |val|={len(vIdx)} |test|={len(teIdx)} "
                f"overlap train∩val={len(tIdx & vIdx)} train∩test={len(tIdx & teIdx)} val∩test={len(vIdx & teIdx)}")

    def _debugIn(self, xOnehot, xEmbeddings, maskOnehot):
        if self.forwardDebugOnce and self.debugMode:
            print(f"[SmORFCNN.forward] xOnehot shape={tuple(xOnehot.shape)} "
                    f"min/max/mean=({xOnehot.detach().min().item():.3f}/{xOnehot.detach().max().item():.3f}/{xOnehot.detach().float().mean().item():.3f})")
            print(f"[SmORFCNN.forward] xEmbeddings shape={tuple(xEmbeddings.shape)} "
                    f"min/max/mean=({xEmbeddings.detach().min().item():.3f}/{xEmbeddings.detach().max().item():.3f}/{xEmbeddings.detach().float().mean().item():.3f})")
            print(f"[SmORFCNN.forward] maskOnehot shape={tuple(maskOnehot.shape)} sum per-batch={maskOnehot.sum(dim=1).detach().cpu().tolist()[:8]}")

    def _debugOnehot(self, inputOneHot):
        if self.forwardDebugOnce and self.debugMode:
            print(f"[SmORFCNN.forward] onehot features shape={tuple(inputOneHot.shape)}")

    def _debugEmbeddings(self, inputEmbeddings):
        if self.forwardDebugOnce and self.debugMode:
            print(f"[SmORFCNN.forward] embeddings features shape={tuple(inputEmbeddings.shape)}")
    
    def _debugLogits(self, logits):
        if self.forwardDebugOnce and self.debugMode:
            print(f"[SmORFCNN.forward] logits shape={tuple(logits.shape)} "
                f"min/max/mean=({logits.detach().min().item():.3f}/{logits.detach().max().item():.3f}/{logits.detach().float().mean().item():.3f}) "
                f"sample={logits.detach().view(-1)[:8].cpu().tolist()}")
            self.forwardDebugOnce = False

    def _debugInEpoch(self, xOnehot, maskOnehot, xEmbed, y, func, index, epochIndex):
        if index == 0 and epochIndex == 1 and self.debugMode:
            print(f"[{func} Epoch-{epochIndex}] batch0 shapes: xOnehot={tuple(xOnehot.shape)} maskOnehot={tuple(maskOnehot.shape)} "
                f"y pos rate={(y.sum().item()/max(1,y.numel())):.3f}")
            print(f"[{func} Epoch-{epochIndex}] xEmbed stats: min={xEmbed.min().item():.3f} max={xEmbed.max().item():.3f} mean={xEmbed.float().mean().item():.3f} "
                f"len(T)={xEmbed.shape[-1]}")
    
    def _debugOutEpoch(self, outputs, func, index, epochIndex):
        if index == 0 and epochIndex == 1 and self.debugMode:
            print(f"[{func} Epoch-{epochIndex}] outputs shape={tuple(outputs.shape)} "
                f"min/max/mean=({outputs.detach().min().item():.3f}/{outputs.detach().max().item():.3f}/{outputs.detach().float().mean().item():.3f})")
    
    def _debugFinal(self, probabilities, targets, runningLoss, n, func, epochIndex):
        if epochIndex == 0 and self.debugMode:
            print(f"[{func} Epoch{epochIndex}] epoch probs shape={tuple(probabilities.shape)} targets shape={tuple(targets.shape)} "
            f"loss_avg={runningLoss/max(1,n):.6f}")

mymodel = SmORFCNN("smorfCNN.pt",4,1536,"features.pt",debug=False)
mymodel.initializeDataset()
mymodel.fit(1)