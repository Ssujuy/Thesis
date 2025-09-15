import Types, Helpers
import torch,os,random
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrScheduler
from torch.utils.data import DataLoader, ConcatDataset, Subset
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.model_selection import StratifiedKFold


class ConvolutionBlock(nn.Module):
    """
    Conv1d -> BatchNorm -> GELU -> (Dropout)
    - BatchNorm stabilizes training
    - GELU is a strong modern activation

    Where:
    1. kernel: number of adjacent positiona single filter looks at.
    2. stride: how far the kernel moves during each step.
    3. padding: ensures edge positions get full windows and layer shapes stay compatible.
    4. dilation: spacing between kernel taps.
    5. groups: splits the inputChannels in groups and applies independent filters.
    6. activation: activation function to use, configurable for testing.
    7. dropout: zeros activations to reduce co-adaptation and overfitting
    """

    def __init__(
            self,
            inputChannels: int,
            outputChannels: int,
            kernel: int,
            stride: int                 = Types.DEFAULT_CONVOLUTION_STRIDE,
            padding                     = Types.DEFAULT_CONVOLUTION_PADDING,
            dilation: int               = Types.DEFAULT_CONVOLUTION_DILATION,
            groups: int                 = Types.DEFAULT_CONVOLUTION_GROUPS,
            activation: str             = Types.DEFAULT_CONVOLUTION_ACTIVATION,
            dropout: float              = Types.DEFAULT_CONVOLUTION_DROPOUT,
            debug: bool                 = Types.DEFAULT_DEBUG_MODE
        ):
        super().__init__()

        self.forwardDebugOnce = debug
        self.debugMode = debug
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.activation = activation
        self.dropout = dropout

        if self.padding is None:
            # "Same" padding for odd kernels under given dilation.
            self.padding = (dilation * (kernel - 1)) // 2

        self.conv1d = nn.Conv1d(
            self.inputChannels,
            self.outputChannels,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=False
        )

        self.batchNormalization = nn.BatchNorm1d(outputChannels)
        self.activation = Types.activationFunctionMapping.get(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        self._debugIn(x)
        x = self.conv1d(x)
        x = self.batchNormalization(x)
        x = self.activation(x)
        x = self.dropout(x)
        self._debugOut(x)
        return x

    def print(self) -> None:

        print("Convolution Block Parameters:")
        print(f" - Input Channels: {self.inputChannels}")
        print(f" - Output Channels: {self.outputChannels}")
        print(f" - Kernel: {self.kernel}")
        print(f" - Stride: {self.stride}")
        print(f" - Padding: {self.padding}")
        print(f" - Groups: {self.groups}")
        print(f" - Activation: {self.activation}")
        print(f" - Dropout: {self.dropout}")
        print(f" - Model Parameters: {self.dropout}")

    def _debugIn(self,x):
        if self.forwardDebugOnce and self.debugMode:
            print(f"[ConvolutionBlock] in shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
            f"min={x.detach().min().item():.4f} max={x.detach().max().item():.4f} mean={x.detach().float().mean().item():.4f}")

    def _debugOut(self,x):
        if self.forwardDebugOnce and self.debugMode:
            print(f"[ConvolutionBlock] out shape={tuple(x.shape)}")
            self.forwardDebugOnce = False
    
class ResidualBlock(nn.Module):
    """
    Defining 2 Convolution Block classes. This class computes a residual mapping
    y = x + F(x) introduced by ResNets. inputChannels length is equal to output channels length.
    """
    def __init__(
            self,
            inputChannels: int,
            outputChannels: int,
            kernel: int,
            stride: int                 = Types.DEFAULT_CONVOLUTION_STRIDE,
            padding                     = Types.DEFAULT_CONVOLUTION_PADDING,
            dilation: int               = Types.DEFAULT_CONVOLUTION_DILATION,
            groups: int                 = Types.DEFAULT_CONVOLUTION_GROUPS,
            activation: str             = Types.DEFAULT_CONVOLUTION_ACTIVATION,
            dropout: float              = Types.DEFAULT_CONVOLUTION_DROPOUT,
            debug: bool                 = Types.DEFAULT_DEBUG_MODE
        ):
        super().__init__()

        self.debugMode = debug
        self.forwardDebugOnce = debug
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.activation = activation
        self.dropout = dropout

        self.convolution1 = ConvolutionBlock(
            self.inputChannels,
            self.outputChannels,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            activation=self.activation,
            dropout=self.dropout
        )

        self.convolution2 = ConvolutionBlock(
            self.inputChannels,
            self.outputChannels,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            activation=self.activation,
            dropout=self.dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._debugIn(x)
        out = x + self.convolution2(self.convolution1(x))
        self._debugOut(out)
        return out

    def print(self) -> None:

        print("Residual Block Parameters:")
        print(f" - Input Channels: {self.inputChannels}")
        print(f" - Output Channels: {self.outputChannels}")        
        print(f" - Kernel: {self.kernel}")
        print(f" - Stride: {self.stride}")
        print(f" - Padding: {self.padding}")
        print(f" - Groups: {self.groups}")
        print(f" - Activation: {self.activation}")
        print(f" - Dropout: {self.dropout}")
        print(f" - Model Parameters: {self.activation}")

    def _debugIn(self,x):
        if self.forwardDebugOnce and self.debugMode:
            print(f"[ResidualBlock] in shape={tuple(x.shape)}")

    def _debugOut(self,out):
        if self.forwardDebugOnce and self.debugMode:
            print(f"[ResidualBlock] out shape={tuple(out.shape)}")
            self.forwardDebugOnce = False

class MultiKernelConvolution(nn.Module):
    """
    Parallel convs with different kernel widths (e.g., 3/7/11/15)
    Captures short motifs (codons, start/stop) and longer context.
    """
    def __init__(
            self,
            inputChannels: int,
            outputChannelsKernel: int   = Types.DEFAULT_MULTI_KERNEL_PER_KERNEL_OUTPUTCH,
            kernelList: list            = Types.DEFAULT_MULTI_KERNEL_KERNEL_LIST,
            stride: int                 = Types.DEFAULT_CONVOLUTION_STRIDE,
            padding                     = Types.DEFAULT_CONVOLUTION_PADDING,
            dilation: int               = Types.DEFAULT_CONVOLUTION_DILATION,
            groups: int                 = Types.DEFAULT_CONVOLUTION_GROUPS,
            activation: str             = Types.DEFAULT_CONVOLUTION_ACTIVATION,
            dropout: float              = Types.DEFAULT_CONVOLUTION_DROPOUT,
            debug: bool                 = Types.DEFAULT_DEBUG_MODE
        ):
        super().__init__()

        self.debugMode = debug
        self.forwardDebugOnce = debug
        self.inputChannels = inputChannels
        self.outputChannelsKernel = outputChannelsKernel
        self.kernelList = kernelList
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.activation = activation
        self.dropout = dropout

        self.branches = nn.ModuleList()

        for kernel in kernelList:
            conv = ConvolutionBlock(
                self.inputChannels,
                self.outputChannelsKernel,
                kernel,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                activation=self.activation,
                dropout=self.dropout
            )
            self.branches.append(conv)
        
        self.outputChannels = outputChannelsKernel * len(kernelList)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = []

        for branch in self.branches:
            b = branch(x)
            output.append(b)

        out = torch.cat(output, dim=1)
        return out

    def print(self) -> None:

        print("Multiple Kernel Convolution Parameters:")
        print(f" - Input Channels: {self.inputChannels}")
        print(f" - Output Channels per Kernel: {self.outputChannelsKernel}")
        print(f" - Output Channels total: {self.outputChannels}")
        print(f" - Kernel List: {self.kernelList}")
        print(f" - Stride: {self.stride}")
        print(f" - Padding: {self.padding}")
        print(f" - Groups: {self.groups}")
        print(f" - Activation: {self.activation}")
        print(f" - Dropout: {self.dropout}")
        print(f" - Model Parameters: {self.dropout}")
    
    def _debugIn(self,x):
        if self.forwardDebugOnce and self.debugMode:
            print(f"[MultiKernelConv] in shape={tuple(x.shape)} kernels={self.kernelList}")

    def _debugBranch(self,b):
        if self.forwardDebugOnce and self.debugMode:
            print(f"[MultiKernelConv] out shape={tuple(b.shape)}")

    def _debugOut(self,out):
        if self.forwardDebugOnce and self.debugMode:
            print(f"[MultiKernelConv] concat out shape={tuple(out.shape)}")

class TemporalHead(nn.Module):
    """
    Reduce channels -> residual stack -> masked global pooling (max + avg).
    Assumes sequences are right-padded and `mask` marks valid positions: [B, L] with 1=valid, 0=pad.
    """
    def __init__(
        self,
        inputChannels,
        hiddenChannels: int         = Types.DEFAULT_TEMPORAL_HIDDEN_CHANNELS,
        kernelResidual: int         = Types.DEFAULT_TEMPORAL_KERNEL_RESIDUAL,
        kernerReduction: int        = Types.DEFAULT_TEMPORAL_KERNEL_REDUCTION,
        stride: int                 = Types.DEFAULT_CONVOLUTION_STRIDE,
        padding                     = Types.DEFAULT_CONVOLUTION_PADDING,
        groups: int                 = Types.DEFAULT_CONVOLUTION_GROUPS,
        activation: str             = Types.DEFAULT_CONVOLUTION_ACTIVATION,
        dropout: float              = Types.DEFAULT_TEMPORAL_DROPOUT,
        multipleDilation: bool      = Types.DEFAULT_TEMPORAL_MULTI_DILATION,
        residualBlocks: int         = Types.DEFAULT_RESIDUAL_BLOCKS_NMB
        ):
        super().__init__()
        self.firsttime = True
        self.inputChannels = inputChannels
        self.hiddenChannels = hiddenChannels
        self.kernelResidual = kernelResidual
        self.kernelReduction = kernerReduction
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.activation = activation
        self.dropout = dropout
        self.multipleDilation = multipleDilation
        self.residualBlocks = residualBlocks

        self.reduce = ConvolutionBlock(
            self.inputChannels,
            self.hiddenChannels,
            self.kernelReduction,
            stride=self.stride,
            padding=self.padding,
            dilation=Types.DEFAULT_CONVOLUTION_DILATION,
            groups=self.groups,
            activation=self.activation,
            dropout=self.dropout
        )

        blocks = []

        for i in range(self.residualBlocks):
            dilation = 2**i if self.multipleDilation else Types.DEFAULT_CONVOLUTION_DILATION
            
            blocks.append(
                ResidualBlock(
                    self.hiddenChannels,
                    self.hiddenChannels,
                    self.kernelResidual,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=dilation,
                    groups=self.groups,
                    activation=self.activation,
                    dropout=self.dropout
                )
            )       
        
        self.residualBlocks = nn.Sequential(*blocks)

    def print(self) -> None:

        print("Temporal Head Parameters:")
        print(f" - Input Channels: {self.inputChannels}")
        print(f" - Hidden Channels: {self.hiddenChannels}")
        print(f" - Kernel for Residual Block: {self.kernelResidual}")
        print(f" - Kernel for size Reduction: {self.kernelReduction}")
        print(f" - Stride: {self.stride}")
        print(f" - Padding: {self.padding}")
        print(f" - Groups: {self.groups}")
        print(f" - Activation: {self.activation}")
        print(f" - Dropout: {self.dropout}")
        print(f" - Change Dilation for Residual Block: {self.multipleDilation}")
        print(f" - Number of Residual blocks: {self.residualBlocks}")


    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, C, L]   feature map
        mask: [B, L]      1 for valid tokens, 0 for right-padded tail.

        1. Project to 128 size
        2. Calculate residualBlocks number of Residuals sequentially.
        3. Calculate global max pooling and global average pooling, then concatenate.

        returns: [B, 2*C] (global-max ⊕ masked-global-avg)
        """
        # refine features (length preserved)
        if self.firsttime:
            print(f"[TemporalHead] in x shape={tuple(x.shape)} mask shape={tuple(mask.shape)} "
            f"mask sum per-batch={mask.sum(dim=1).detach().cpu().tolist()[:4]}")
        x = self.reduce(x)        # [B, C, L]
        if self.firsttime:
            print(f"[TemporalHead] after reduce shape={tuple(x.shape)}")
        x = self.residualBlocks(x)        # [B, C, L]
        if self.firsttime:    
            print(f"[TemporalHead] after residualBlocks shape={tuple(x.shape)}")

        gmp = Helpers.globalMaxPooling(x, mask)
        gap = Helpers.globalAveragePooling(x, mask)
        if self.firsttime:
            print(f"[TemporalHead] GMP type={type(gmp)} shape={tuple(gmp.shape)} "
            f"GAP type={type(gap)} shape={tuple(gap.shape)}")
        out = torch.cat([gmp, gap], dim=1)
        if self.firsttime:
            print(f"[TemporalHead] out shape={tuple(out.shape)}")
        self.firsttime = False
        return out

    

# ---------------------------------------------
# The CNNClassifier for smORFs
# ---------------------------------------------

class SmORFCNN(nn.Module):
    """
    Two-branch CNN for coding vs non-coding prediction.

    Inputs
    ------
    x_onehot: [B,4,L]            # optional one-hot DNA
    x_embed:  [B,E,T]            # optional DNABERT6 embeddings (channels-first)
    mask_onehot: [B,L] optional  # 1=valid
    mask_embed:  [B,T] optional  # 1=valid

    Design notes (theory links):
      - Local convs + weight sharing (translation equivariance)
      - Multi-scale kernels capture motifs & context
      - 1x1 conv to squeeze large E (NIN)
      - Dilations increase receptive field efficiently
      - Residual skips ease optimization
      - Global pooling yields fixed-size representation
    """
    def __init__(
        self,
        onehotInputChannels: int,
        embeddingsInputChannels: int,
        featuresPath: str,
        temporalHead: bool                      = Types.DEFAULT_SMORFCNN_TEMPORAL_HEAD,
        multiKernel: bool                       = Types.DEFAULT_SMORFCNN_MULTI_KERNEL,
        onehotKernelList: list                  = Types.DEFAULT_SMORFCNN_ONEHOT_KERNEL_LIST,
        embeddingsKernelList: list              = Types.DEFAULT_SMORFCNN_EMBEDDINGS_KERNEL_LIST,
        outputChannelsKernel: int               = Types.DEFAULT_SMORFCNN_OUTPUT_CHANNELS_KERNEL,
        temporalHeadOutputChannels: int         = Types.DEFAULT_SMORFCNN_OUTPUT_CHANNELS_TEMPORAL,
        residualBlocks: int                     = Types.DEFAULT_SMORFCNN_RESIDUAL_BLOCKS,
        dropout: float                          = Types.DEFAULT_SMORFCNN_DROPOUT,
        classes: int                            = Types.DEFAULT_SMORFCNN_CLASSES,
        classifierOutput: int                   = Types.DEFAULT_SMORFCNN_CLASSIFIER_OUTPUT,
        learningRate: float                     = Types.DEFAULT_SMORFCNN_LEARNING_RATE,
        weightDecay: float                      = Types.DEFAULT_SMORFCNN_WEIGHT_DECAY,
        eps: float                              = Types.DEFAULT_SMORFCNN_EPS,
        betas: tuple                            = Types.DEFAULT_SMORFCNN_BETAS,
        factor: float                           = Types.DEFAULT_SMORFCNN_FACTOR,
        patience: int                           = Types.DEFAULT_SMORFCNN_PATIENCE,
        minLearningRate: float                  = Types.DEFAULT_SMORFCNN_MINIMUM_LEARNING_RATE,
        threshold: float                        = Types.DEFAULT_SMORFCNN_THRESHOLD,
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

    ):
        super().__init__()
        self.firsttime = True
        self.seed = seed
        self.deterministic = deterministic
        self.device = device
        self.featuresPath = featuresPath
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        os.environ["PYTHONHASHSEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        torch.manual_seed(self.seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        if self.deterministic:
            # Make ops deterministic where possible
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.onehotInputChannels = onehotInputChannels
        self.embeddingsInputChannels =  embeddingsInputChannels

        self.temporalHead = temporalHead
        self.multiKernel = multiKernel
        self.onehotKernelList = onehotKernelList
        self.embeddingsKernelList = embeddingsKernelList
        self.outputChannelsKernel = outputChannelsKernel
        self.temporalHeadOutputChannels = temporalHeadOutputChannels
        self.residualBlocks = residualBlocks
        self.dropout = dropout
        self.classes = classes
        self.classifierOutput = classifierOutput
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.eps = eps
        self.betas = betas
        self.factor = factor
        self.patience = patience
        self.minLearningRate = minLearningRate
        self.threshold = threshold
        self.trainBatchSize = trainBatchSize
        self.validationBatchSize = valBatchSize
        self.testBatchSize = testBatchSize
        self.trainSplit = trainSplit
        self.validationSplit = valSplit
        self.testSplit = testSplit
        self.epochs = epochs

        self.onehotMultiKernelClass = None
        self.onehotTemporalClass = None
        self.embeddingsMultiKernelClass = None
        self.embeddingsTemporalClass = None

        if self.multiKernel:
            self.onehotMultiKernelClass = MultiKernelConvolution(
                inputChannels=self.onehotInputChannels,
                outputChannelsKernel=self.outputChannelsKernel,
                kernelList=self.onehotKernelList,
                dropout=self.dropout
            )
            
        if self.temporalHead:
            self.onehotTemporalClass = TemporalHead(
                self.onehotMultiKernelClass.outputChannels,
                hiddenChannels=self.temporalHeadOutputChannels,
                residualBlocks=self.residualBlocks,
                dropout=self.dropout
            )

        if self.multiKernel:
            self.embeddingsMultiKernelClass = MultiKernelConvolution(
                inputChannels=self.embeddingsInputChannels,
                outputChannelsKernel=self.outputChannelsKernel,
                kernelList=self.embeddingsKernelList,
                dropout=self.dropout
            )

        if self.temporalHead:
            self.embeddingsTemporalClass = TemporalHead(
                inputChannels=self.embeddingsMultiKernelClass.outputChannels,
                hiddenChannels=self.temporalHeadOutputChannels,
                residualBlocks=self.residualBlocks,
                dropout=self.dropout
            )

        self.fusedDim = self._calculateFusedDim()

        self.classifier = nn.Sequential(
            nn.Linear(self.fusedDim, self.classifierOutput),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.classifierOutput, self.classes)
        )

        self.optimizer = self._optimizerInit()
        self.scheduler = self._schedulerInit()
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[INIT] total params={total_params:,} trainable={trainable_params:,}")
        print(f"[INIT] optimizer param_groups sizes={[len(g['params']) for g in self.optimizer.param_groups]}")
        print(f"[INIT] device={self.device} multiKernel={self.multiKernel} temporalHead={self.temporalHead}")
        print(f"[INIT] onehot kernels={self.onehotKernelList} embed kernels={self.embeddingsKernelList}")
        print(f"[INIT] fusedDim={self.fusedDim} classifier={self.classifier}")

        self.trainDataLoader = None
        self.validationDataLoader = None
        self.testDataLoader = None

    def initializeDataset(self) -> None:

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

        # Helpers.printDataloader("Training", self.trainDataLoader)
        # Helpers.printDataloader("Validation", self.validationDataLoader)
        # Helpers.printDataloader("Testing", self.testDataLoader)

        # Helpers.plotLabelDistribution(self.trainDataLoader, self.validationDataLoader, self.testDataLoader)
        try:
            t_idx  = set(self.trainDataLoader.dataset.indices)
            v_idx  = set(self.validationDataLoader.dataset.indices)
            te_idx = set(self.testDataLoader.dataset.indices)
            print(f"[SPLIT] |train|={len(t_idx)} |val|={len(v_idx)} |test|={len(te_idx)} "
                f"overlap train∩val={len(t_idx & v_idx)} train∩test={len(t_idx & te_idx)} val∩test={len(v_idx & te_idx)}")
        except Exception as e:
            print(f"[SPLIT] couldn't inspect indices: {e}")

    def _calculateFusedDim(self) -> int:
        # If TemporalHead is ON, each branch outputs 2*hidden → two branches = 4*hidden.
        temporalHeadDim = (4 * self.temporalHeadOutputChannels) * int(self.temporalHead)

        # When TemporalHead is OFF, we pool directly. Each branch contributes 2*C_branch,
        # where C_branch is MultiKernel out_ch if multiKernel=True, else the raw input channels.
        if self.temporalHead:
            poolingNoTemporalDim = 0
        else:
            onehot_ch = (self.onehotMultiKernelClass.outputChannels
                            if self.multiKernel else self.onehotInputChannels)
            embed_ch  = (self.embeddingsMultiKernelClass.outputChannels
                            if self.multiKernel else self.embeddingsInputChannels)
            poolingNoTemporalDim = 2 * (onehot_ch + embed_ch)

        # Do NOT add MultiKernel channels separately when TemporalHead is ON;
        # they’re already reduced inside the head. (Avoid double counting.)
        multiKernelDim = 0

        print(f"[FUSED DIM] temporalHeadDim={temporalHeadDim} poolingNoTemporalDim={poolingNoTemporalDim} -> fusedDim={temporalHeadDim + multiKernelDim + poolingNoTemporalDim}")

        return temporalHeadDim + multiKernelDim + poolingNoTemporalDim

    def _optimizerInit(self) -> torch.optim.Optimizer:
        """
        AdamW with a common param-group trick:
        - decay on weights of conv/linear
        - NO decay on biases and norm parameters (1D parameters)
        """

        decay, noDecay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or name.endswith(".bias"):
                noDecay.append(param)     # norms & biases
            else:
                decay.append(param)        # weight matrices/kernels

        parameterGroups = [
            {"params": decay,    "weight_decay": self.weightDecay},
            {"params": noDecay, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(parameterGroups, lr=self.learningRate, betas=self.betas, eps=self.eps)
    
    def _schedulerInit(self) -> torch.optim.lr_scheduler:

        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.factor,
            patience=self.patience,
            min_lr=self.minLearningRate,
        )        

    def forward(
        self,
        xOnehot: torch.Tensor,
        xEmbeddings:  torch.Tensor,
        maskOnehot: torch.Tensor,
        maskEmbeddings:  torch.Tensor,
        # rc_augment: bool = False
    ) -> torch.Tensor:
        """
        forward function that passes onehot encoded sequences/embeddings through
        MultiKernel CNN and Temporal Head CNN (if branches are active!!).

        xOnehot --> [B, C1, L] \n
        xEmbeddings --> [B, C2, T] \n
        maskOnehot --> [B, L] (1=valid, 0=pad) \n
        maskEmbeddings --> [B, T] (1=valid, 0=pad) \n

        Returns:
        logits: [B] if self.classes==1 else [B, self.classes]
        """
        if self.firsttime:
            print(f"[SmORFCNN.forward] xOnehot shape={tuple(xOnehot.shape)} "
                    f"min/max/mean=({xOnehot.detach().min().item():.3f}/{xOnehot.detach().max().item():.3f}/{xOnehot.detach().float().mean().item():.3f})")
            print(f"[SmORFCNN.forward] xEmbeddings shape={tuple(xEmbeddings.shape)} "
                    f"min/max/mean=({xEmbeddings.detach().min().item():.3f}/{xEmbeddings.detach().max().item():.3f}/{xEmbeddings.detach().float().mean().item():.3f})")
            print(f"[SmORFCNN.forward] maskOnehot shape={tuple(maskOnehot.shape)} sum per-batch={maskOnehot.sum(dim=1).detach().cpu().tolist()[:8]}")
            print(f"[SmORFCNN.forward] maskEmbeddings shape={tuple(maskEmbeddings.shape)} sum per-batch={maskEmbeddings.sum(dim=1).detach().cpu().tolist()[:8]}")
        features = []
        inputOneHot = xOnehot
        inputEmbeddings = xEmbeddings

        if xOnehot is None:
            raise AttributeError("Did not receive onehot encoded input!")
        
        if xEmbeddings is None:
            raise AttributeError("Did not receive dnabert6 embeddings input!")

        if self.multiKernel:
            if self.firsttime:
                print("[SmORFCNN.forward] Passing onehot through MultiKernelConvolution")
            inputOneHot = self.onehotMultiKernelClass(inputOneHot)

        if self.temporalHead:
            if self.firsttime:
                print("[SmORFCNN.forward] Passing onehot through TemporalHead")
            inputOneHot = self.onehotTemporalClass(inputOneHot, maskOnehot)
            
        else:
            if self.firsttime:
                print("[SmORFCNN.forward] Onehot branch: applying masked GMP+GAP without TemporalHead")
            inputOneHot = torch.cat(
                [
                    Helpers.globalMaxPooling(inputOneHot, maskOnehot),
                    Helpers.globalAveragePooling(inputOneHot, maskOnehot)
                ],
                    dim=1
            )  

        features.append(inputOneHot)

        if self.firsttime:
            print(f"[SmORFCNN.forward] onehot features shape={tuple(inputOneHot.shape)}")

        if self.multiKernel:
            if self.firsttime:
                print("[SmORFCNN.forward] Passing embeddings through MultiKernelConvolution")
            inputEmbeddings = self.embeddingsMultiKernelClass(inputEmbeddings)

        if self.temporalHead:
            if self.firsttime:
                print("[SmORFCNN.forward] Passing embeddings through TemporalHead")
            inputEmbeddings = self.embeddingsTemporalClass(inputEmbeddings, maskEmbeddings)

        else:
            if self.firsttime:
                print("[SmORFCNN.forward] Embeddings branch: applying masked GMP+GAP without TemporalHead")
            inputEmbeddings = torch.cat([
                    Helpers.globalMaxPooling(inputEmbeddings, maskEmbeddings),
                    Helpers.globalAveragePooling(inputEmbeddings, maskEmbeddings)],
                dim=1
            )
        if self.firsttime:
            print(f"[SmORFCNN.forward] embeddings features shape={tuple(inputEmbeddings.shape)}")
        features.append(inputEmbeddings)

        if not features:
            raise RuntimeError("Failed to produce any features!")

        fused = features[0] if len(features) == 1 else torch.cat(features, dim=1)
        if self.firsttime:
            print(f"[SmORFCNN.forward] fused features shape={tuple(fused.shape)} (expected last dim={self.fusedDim})")

        if fused.size(1) != self.fusedDim:
            raise ValueError(f"Expected fused dimension {self.fusedDim} , got {fused.size(1)}")

        logits = self.classifier(fused)
        if self.firsttime:
            print(f"[SmORFCNN.forward] logits shape={tuple(logits.shape)} "
                f"min/max/mean=({logits.detach().min().item():.3f}/{logits.detach().max().item():.3f}/{logits.detach().float().mean().item():.3f}) "
                f"sample={logits.detach().view(-1)[:8].cpu().tolist()}")
        self.firsttime = False
        return logits.squeeze(-1) if self.classes == 1 else logits

    def trainEpoch(
            self,
            trainingData: DataLoader,
            epochIndex: int,
            maxGradNorm: float      = Types.DEFAULT_SMORFCNN_MAX_GRAD_NORM,
            threshold: float        = Types.DEFAULT_SMORFCNN_THRESHOLD
        ) -> dict:
        
        """
        Trainer function
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
                    enumerate(trainingData),
                    total=len(trainingData),
                    desc=f"Epoch {epochIndex}",
                    leave=False
                )
        except Exception:
            iterator = enumerate(trainingData)

        for i, batch in iterator:
            xOnehot, maskOnehot, xEmbed, maskEmbed, y = batch
            if i == 0:
                print(f"[trainEpoch e{epochIndex}] batch0 shapes: xOnehot={tuple(xOnehot.shape)} maskOnehot={tuple(maskOnehot.shape)} "
                      f"xEmbed={tuple(xEmbed.shape)} maskEmbed={tuple(maskEmbed.shape)} y={tuple(y.shape)} "
                      f"y pos rate={(y.sum().item()/max(1,y.numel())):.3f}")
                print(f"[trainEpoch e{epochIndex}] xEmbed stats: min={xEmbed.min().item():.3f} max={xEmbed.max().item():.3f} mean={xEmbed.float().mean().item():.3f} "
                      f"len(T)={xEmbed.shape[-1]}")
            xOnehot = xOnehot.to(self.device)
            maskOnehot = maskOnehot.to(self.device)
            xEmbed = xEmbed.to(self.device)
            maskEmbed = maskEmbed.to(self.device)
            y = y.to(self.device)
            if i == 0:
                len_oh = maskOnehot.sum(-1).float()
                pos = (y == 1)
                neg = ~pos
                def _m(v): return (v.mean().item() if v.numel() > 0 else float('nan'))
                def _s(v): return (v.std().item()  if v.numel() > 1 else float('nan'))
                print(f"[trainEpoch e{epochIndex}] lenOH pos/neg mean={_m(len_oh[pos]):.1f}/{_m(len_oh[neg]):.1f} "
                    f"std={_s(len_oh[pos]):.1f}/{_s(len_oh[neg]):.1f}")
            self.optimizer.zero_grad()

            outputs = self(xOnehot, xEmbed, maskOnehot, maskEmbed)
            if i == 0:
                print(f"[trainEpoch e{epochIndex}] outputs shape={tuple(outputs.shape)} "
                      f"min/max/mean=({outputs.detach().min().item():.3f}/{outputs.detach().max().item():.3f}/{outputs.detach().float().mean().item():.3f})")
            loss = lossFunction(outputs, y.float())
            probs = torch.sigmoid(outputs)

            loss.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), maxGradNorm)
            if i == 0:
                print(f"[trainEpoch e{epochIndex}] total_grad_norm(before clip)={total_norm:.4f} "
                    f"clipped={bool(total_norm > maxGradNorm)}")
                print(f"[trainEpoch e{epochIndex}] classifier[0].weight grad |mean|="
                    f"{self.classifier[0].weight.grad.abs().mean().item():.6f}")

            if i == 0:
                total_norm_sq = 0.0
                for p in self.parameters():
                    if p.grad is not None:
                        gn = p.grad.data.norm(2).item()
                        total_norm_sq += gn*gn
                print(f"[trainEpoch e{epochIndex}] grad_norm(L2)={total_norm_sq**0.5:.4f} loss={loss.item():.6f} "
                      f"probs sample={probs.detach().view(-1)[:8].cpu().tolist()}")
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
        print(f"[trainEpoch e{epochIndex}] epoch probs shape={tuple(probabilities.shape)} targets shape={tuple(targets.shape)} "
              f"loss_avg={runningLoss/max(1,n):.6f}")
        return Helpers.computeEpochMetrics(
            probabilities,
            targets,
            runningLoss,
            n,
            threshold,
            epochIndex
        )
    @torch.no_grad()
    def validateEpoch(
            self,
            validationData: DataLoader,
            epochIndex: int,
            threshold: float = Types.DEFAULT_SMORFCNN_THRESHOLD,
        ) -> dict:

        self.eval()

        n = 0
        runningLoss = 0.0
        probabilities = []
        targets = []

        lossFunction = torch.nn.BCEWithLogitsLoss()

        print(f"Starting training for epoch {epochIndex}")

        iterator = tqdm(
                enumerate(validationData),
                total=len(validationData),
                desc=f"Epoch {epochIndex}",
                leave=False
            )

        for j, batch in iterator:
            xOnehot, maskOnehot, xEmbed, maskEmbed, y = batch
            if j == 0:
                print(f"[validateEpoch e{epochIndex}] batch0 shapes: xOnehot={tuple(xOnehot.shape)} maskOnehot={tuple(maskOnehot.shape)} "
                      f"xEmbed={tuple(xEmbed.shape)} maskEmbed={tuple(maskEmbed.shape)} y={tuple(y.shape)} "
                      f"y pos rate={(y.sum().item()/max(1,y.numel())):.3f}")
            xOnehot = xOnehot.to(self.device)
            maskOnehot = maskOnehot.to(self.device)
            xEmbed = xEmbed.to(self.device)
            maskEmbed = maskEmbed.to(self.device)
            y = y.to(self.device)
            if j == 0:
                len_oh = maskOnehot.sum(-1).float()
                pos = (y == 1)
                neg = ~pos
                def _m(v): return (v.mean().item() if v.numel() > 0 else float('nan'))
                def _s(v): return (v.std().item()  if v.numel() > 1 else float('nan'))
                print(f"[valEpoch e{epochIndex}] lenOH pos/neg mean={_m(len_oh[pos]):.1f}/{_m(len_oh[neg]):.1f} "
                    f"std={_s(len_oh[pos]):.1f}/{_s(len_oh[neg]):.1f}")

            outputs = self(xOnehot, xEmbed, maskOnehot, maskEmbed)
            if j == 0:
                print(f"[validateEpoch e{epochIndex}] outputs shape={tuple(outputs.shape)} "
                      f"min/max/mean=({outputs.detach().min().item():.3f}/{outputs.detach().max().item():.3f}/{outputs.detach().float().mean().item():.3f})")

            loss = lossFunction(outputs, y.float())
            probs = torch.sigmoid(outputs)

            batchSize = y.size(0)
            n += batchSize
            runningLoss += loss.item() * batchSize
            probabilities.append(probs.detach())
            targets.append(y.detach())

            if 'tqdm' in locals():
                iterator.set_postfix(loss=runningLoss / n)

        probabilities = torch.cat(probabilities, dim=0)
        targets = torch.cat(targets, dim=0)
        print(f"[validateEpoch e{epochIndex}] epoch probs shape={tuple(probabilities.shape)} targets shape={tuple(targets.shape)} "
              f"loss_avg={runningLoss/max(1,n):.6f}")
        metrics = Helpers.computeEpochMetrics(
            probabilities,
            targets,
            runningLoss,
            n,
            threshold,
            epochIndex
        )

        metrics = metrics | Helpers.computeEpochROC(
            probabilities,
            targets,
            epochIndex
        )

        return metrics

    @torch.no_grad()
    def test(
            self,
            testData: DataLoader,
            threshold: float = Types.DEFAULT_SMORFCNN_THRESHOLD,
        ) -> dict:

        self.eval()

        n = 0
        runningLoss = 0.0
        probabilities = []
        targets = []

        lossFunction = torch.nn.BCEWithLogitsLoss()

        print(f"Starting testing")

        iterator = tqdm(
                enumerate(testData),
                total=len(testData),
                desc=f"{testData}",
                leave=False
            )

        for k, batch in iterator:
            xOnehot, maskOnehot, xEmbed, maskEmbed, y = batch
            if k == 0:
                print(f"[test] batch0 shapes: xOnehot={tuple(xOnehot.shape)} maskOnehot={tuple(maskOnehot.shape)} "
                      f"xEmbed={tuple(xEmbed.shape)} maskEmbed={tuple(maskEmbed.shape)} y={tuple(y.shape)} "
                      f"y pos rate={(y.sum().item()/max(1,y.numel())):.3f}")
            xOnehot = xOnehot.to(self.device)
            maskOnehot = maskOnehot.to(self.device)
            xEmbed = xEmbed.to(self.device)
            maskEmbed = maskEmbed.to(self.device)
            y = y.to(self.device)
            if k == 0:
                len_oh = maskOnehot.sum(-1).float()
                pos = (y == 1)
                neg = ~pos
                def _m(v): return (v.mean().item() if v.numel() > 0 else float('nan'))
                def _s(v): return (v.std().item()  if v.numel() > 1 else float('nan'))
                print(f"[test] lenOH pos/neg mean={_m(len_oh[pos]):.1f}/{_m(len_oh[neg]):.1f} "
                    f"std={_s(len_oh[pos]):.1f}/{_s(len_oh[neg]):.1f}")

            outputs = self(xOnehot, xEmbed, maskOnehot, maskEmbed)
            if k == 0:
                print(f"[test] outputs shape={tuple(outputs.shape)} "
                      f"min/max/mean=({outputs.detach().min().item():.3f}/{outputs.detach().max().item():.3f}/{outputs.detach().float().mean().item():.3f})")
            loss = lossFunction(outputs, y.float())
            probs = torch.sigmoid(outputs)

            batchSize = y.size(0)
            n += batchSize
            runningLoss += loss.item() * batchSize
            probabilities.append(probs.detach())
            targets.append(y.detach()) 

            if 'tqdm' in locals():
                iterator.set_postfix(loss=runningLoss / n)

        probabilities = torch.cat(probabilities, dim=0)
        targets = torch.cat(targets, dim=0)
        print(f"[test] epoch probs shape={tuple(probabilities.shape)} targets shape={tuple(targets.shape)} "
              f"loss_avg={runningLoss/max(1,n):.6f}")
        metrics = Helpers.computeEpochMetrics(
            probabilities,
            targets,
            runningLoss,
            n,
            threshold,
            0
        )

        metrics = metrics | Helpers.computeEpochROC(
            probabilities,
            targets,
            0
        )

        return metrics

    def fit(
        self,
        trainingDataloader:     DataLoader,
        validationDataloader:   DataLoader,
        epochs:                 int,
    ):
        """
        Train the model for `epochs` using train/val loaders and return a training history dict.
        Keeps the best (val F1) checkpoint loaded at the end.
        History keys:
        - train_loss, val_loss
        - train_acc,  val_acc
        - train_precision, val_precision
        - train_recall,    val_recall
        - train_f1,        val_f1
        - val_roc_auc
        - lr  (the learning rate each epoch, if scheduler/optimizer exposes it)
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
            "tpr": []
        }

        epochIter = tqdm(
            enumerate(range(1, epochs + 1), start=1),
            total=epochs,
            desc="Epochs",
            leave=True
        )

        for epoch,_ in epochIter:

            epochTrainMetrics = self.trainEpoch(
                trainingData=trainingDataloader,
                epochIndex=epoch,
                maxGradNorm=Types.DEFAULT_SMORFCNN_MAX_GRAD_NORM,
                threshold=self.threshold,
            )

            epochValMetrics = self.validateEpoch(
                validationData=validationDataloader,
                epochIndex=epoch,
                threshold=self.threshold,
            )

            # ---- 3) (Optional) scheduler step — typically once per epoch ----
                # Many schedulers expect .step() AFTER val metrics (e.g., ReduceLROnPlateau uses val loss)
                # If you use ReduceLROnPlateau, call as: scheduler.step(val_metrics["loss"])

            if isinstance(self.scheduler, lrScheduler.ReduceLROnPlateau):
                self.scheduler.step(epochValMetrics["loss"])
            else:
                self.scheduler.step()

            for key in epochTrainMetrics:
                trainingMetrics[key].append(epochTrainMetrics[key])

            for key in epochValMetrics:
                validationMetrics[key].append(epochValMetrics[key])

            epochLR = self.optimizer.param_groups[0]["lr"]
            validationMetrics["learningRate"].append(epochLR)

            # ---- 5) Save best-by-val-F1 weights ----
            if validationMetrics["f1"][epoch - 1] > bestF1:
                bestF1 = validationMetrics["f1"][epoch - 1]
                bestEpoch = epoch
                # Keep a CPU copy; avoids GPU memory bloat and is device-agnostic to reload
                bestState = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

            try:
                epochIter.set_postfix_str(
                    f"F1 {validationMetrics['f1']:.4f}, AUC {validationMetrics.get('roc_auc', float('nan')):.4f}, LR {epochLR:.2e}"
                )
            except Exception:
                pass

        # ---- 7) Restore the best checkpoint (by val F1) ----
        if bestState is not None:
            self.load_state_dict(bestState)
            self.to(self.device)
        
        testMetrics = self.test(self.testDataLoader)

        # Helpers.printFitSummary(trainingMetrics, validationMetrics)

        # Helpers.plotFitCurves(trainingMetrics, validationMetrics)

        # Helpers.plotROCCurve(validationMetrics, bestEpoch)

        # Helpers.plotConfusionPie(trainingMetrics, validationMetrics, testMetrics, epochs)

        return trainingMetrics, validationMetrics

    def kFoldCrossValidation(
        self,
        k: int = Types.DEFAULT_SMORFCNN_KFOLD,
    ):
        """
        """
    # ---------- A) Build CV pool from your existing loaders ----------
        # Concat order is [train subset, val subset]
        fullDataset = ConcatDataset([self.trainDataLoader.dataset, self.validationDataLoader.dataset])

        # ---------- B) Extract labels directly (no type branching) ----------
        # Assumptions:
        #   trainLoader.dataset           -> Subset
        #   trainLoader.dataset.dataset   -> TensorDataset
        #   trainLoader.dataset.indices   -> index list/array into the base TensorDataset
        #   base TensorDataset.tensors[-1] is labels [N]
        tSubset = self.trainDataLoader.dataset
        valSubset = self.validationDataLoader.dataset

        tBase = tSubset.dataset          # TensorDataset
        valBase = valSubset.dataset          # TensorDataset

        tIdx = torch.as_tensor(tSubset.indices, dtype=torch.long)
        valIdx = torch.as_tensor(valSubset.indices, dtype=torch.long)

        tLabels = tBase.tensors[-1].index_select(0, tIdx)  # [N_tr]
        valLabels = valBase.tensors[-1].index_select(0, valIdx)  # [N_va]

        labels = torch.cat([tLabels, valLabels], dim=0).detach().cpu().long().numpy()  # [N_tr+N_va]

        # ---------- C) Stratified KFold splits ----------
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.seed)
        splits = list(splitter.split(np.arange(len(labels)), labels))

        # ---------- D) Bookkeeping ----------
        foldMetrics = []
        bestFoldF1 = -1.0
        bestFoldState = None
        bestFold = None

        iterator = tqdm(enumerate(splits, start=1),total=k,desc=f"{k}-Fold CV",leave=True)

        for foldIndex, (trainIndex, valIndex) in iterator:

            print(f"\n=== Fold {foldIndex}/{k}: train={len(trainIndex)}  val={len(valIndex)} ===")

            # ---------- E) Build fold-specific loaders ----------
            trainDataset = Subset(fullDataset, indices=trainIndex)
            valDataset = Subset(fullDataset, indices=valIndex)

            trainDataloader = DataLoader(
                trainDataset,
                batch_size=self.trainBatchSize,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                drop_last=False
            )
            validationDataloader = DataLoader(
                valDataset,
                batch_size=self.validationBatchSize,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )

            # # ---------- F) Reinitialize model for an independent fold run ----------
            # def _init(m):
            #     if hasattr(m, "reset_parameters"):
            #         m.reset_parameters()
            # self.apply(_init)
            # self.to(self.device)

            _ = self.fit(
                trainLoader=trainDataloader,
                valLoader=validationDataloader,
                epochs=self.epochs
            )

            # Final metrics on this fold's validation split
            metrics = self.validateEpoch(
                validationData=validationDataloader,
                epochIndex=foldIndex,
            )

            foldMetrics.append(metrics)

            # Keep best-by-F1 weights across folds
            if metrics["f1"] > bestFoldF1:
                bestFoldF1 = metrics["f1"]
                bestFold = foldIndex
                bestFoldState = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

            # Update fold-level tqdm postfix
            if 'tqdm' in locals():
                iterator.set_postfix_str(f"F1 {metrics['f1']:.4f}  AUC {metrics.get('roc_auc', float('nan')):.4f}")

        # ---------- G) Restore the best fold checkpoint ----------
        if bestFoldState is not None:
            self.load_state_dict(bestFoldState)
            self.to(self.device)
            print(f"\nRestored best fold #{bestFold} (val F1={bestFoldF1:.4f})")

        summary = Helpers.kFoldSummary(foldMetrics)

        # Helpers.printKFoldMetrics(foldMetrics, summary)

        # Helpers.plotMeanROC(foldMetrics, summary)

        return foldMetrics, summary

    def saveModel() -> None:
        return
    

mymodel = SmORFCNN(4,768,"features.pt")
mymodel.initializeDataset()
mymodel.fit(
    mymodel.trainDataLoader,
    mymodel.validationDataLoader,
    5
)