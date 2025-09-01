import Types, Helpers
import torch,os,random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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
            dropout: float              = Types.DEFAULT_CONVOLUTION_DROPOUT
        ):
        super().__init__()

        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.activation = activation
        self.dropout = dropout

        if padding is None:
            # "Same" padding for odd kernels under given dilation.
            padding = (dilation * (kernel - 1)) // 2

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(x)
        x = self.batchNormalization(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    
class ResidualBlock(nn.Module):
    """
    Defining 2 Convolution Block classes. This class computes a residual mapping
    y = x + F(x) introduced by ResNets. inputChannels length is equal to output channels length.
    """
    def __init__(
            self,
            inputChannels: int,
            kernel: int,
            stride: int                 = Types.DEFAULT_CONVOLUTION_STRIDE,
            padding                     = Types.DEFAULT_CONVOLUTION_PADDING,
            dilation: int               = Types.DEFAULT_CONVOLUTION_DILATION,
            groups: int                 = Types.DEFAULT_CONVOLUTION_GROUPS,
            activation: str             = Types.DEFAULT_CONVOLUTION_ACTIVATION,
            dropout: float              = Types.DEFAULT_CONVOLUTION_DROPOUT
        ):
        super().__init__()

        self.inputChannels = inputChannels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.activation = activation
        self.dropout = dropout

        self.convolution1 = ConvolutionBlock(
            self.inputChannels,
            self.inputChannels,
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
            self.inputChannels,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            activation=self.activation,
            dropout=self.dropout
        )

    def print(self) -> None:

        print("Residual Block Parameters:")
        print(f" - Input Channels: {self.inputChannels}")
        print(f" - Kernel: {self.kernel}")
        print(f" - Stride: {self.stride}")
        print(f" - Padding: {self.padding}")
        print(f" - Groups: {self.groups}")
        print(f" - Activation: {self.activation}")
        print(f" - Dropout: {self.dropout}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.convolution2(self.convolution1(x))

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
            dropout: float              = Types.DEFAULT_CONVOLUTION_DROPOUT
        ):
        super().__init__()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = []

        for branch in self.branches:
            output.append(branch(x))

        return torch.cat(output, dim=1)


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
                    self.inputChannels,
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
        x = self.reduce(x)        # [B, C, L]
        x = self.residualBlocks(x)        # [B, C, L]

        return torch.cat([
                Helpers.globalMaxPooling(x, mask),
                Helpers.globalAveragePooling(x, mask)],
                dim=1
        )         # [B, 2*C]

    

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
        seed: int                               = Types.DEFAULT_SMORFCNN_SEED,
        deterministic: bool                     = Types.DEFAULT_SMORFCNN_DETERMINISTIC,
        device: str                             = Types.DEFAULT_SMORFCNN_DEVICE
    ):
        super().__init__()

        self.seed = seed
        self.deterministic = deterministic
        self.device = device
        self.featuresPath = featuresPath

        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

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

        dataset = Helpers.Dataset(self.featuresPath)
        self.trainDataset, self.validationDataset, self.testDataset = Helpers.shuffleSPlitTDataset(
            dataset,
            Types.DEFAULT_SMORFCNN_TRAIN_SPLIT,
            Types.DEFAULT_SMORFCNN_VALIDATION_SPLIT,
            Types.DEFAULT_SMORFCNN_TEST_SPLIT,
            Types.DEFAULT_SMORFCNN_SEED
        )

    def _calculateFusedDim(self) -> int:

        temporalHeadDim = 4 * int(self.temporalHead) * self.temporalHeadOutputChannels
        multiKernelDim = int(self.multiKernel) * (len(self.onehotKernelList) + len(self.embeddingsKernelList)) * self.outputChannelsKernel
        poolingNoTemporalDim = (1 - self.temporalHead) * 2 * (self.embeddingsInputChannels + self.onehotInputChannels)

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

        features = []
        inputOneHot = xOnehot
        inputEmbeddings = xEmbeddings

        if xOnehot is None:
            raise AttributeError("Did not receive onehot encoded input!")
        
        if xEmbeddings is None:
            raise AttributeError("Did not receive dnabert6 embeddings input!")

        if self.multiKernel:
            inputOneHot = self.onehotMultiKernelClass(inputOneHot)

        if self.temporalHead:
            inputOneHot = self.onehotTemporalClass(inputOneHot, maskOnehot)
            
        else:

            inputOneHot = torch.cat(
                [
                    Helpers.globalMaxPooling(inputOneHot, maskOnehot),
                    Helpers.globalAveragePooling(inputOneHot, maskOnehot)
                ],
                    dim=1
            )  

        features.append(inputOneHot)


        if self.multiKernel:
            inputEmbeddings = self.embeddingsMultiKernelClass(inputEmbeddings)

        if self.temporalHead:
            inputEmbeddings = self.embeddingsTemporalClass(inputEmbeddings, maskEmbeddings)
            
        else:
            
            inputEmbeddings = torch.cat([
                    Helpers.globalMaxPooling(inputEmbeddings, maskEmbeddings),
                    Helpers.globalAveragePooling(inputEmbeddings, maskEmbeddings)],
                dim=1
            )
                
        features.append(inputEmbeddings)
        
        if not features:
            raise RuntimeError("Failed to produce any features!")
        
        fused = features[0] if len(features) == 1 else torch.cat(features, dim=1)

        if fused.size(1) != self.fusedDim:
            raise ValueError(f"Expected fused dimension {self.fusedDim} , got {fused.size(1)}")

        logits = self.classifier(fused)

        return logits.squeeze(-1) if self.classes == 1 else logits


    def trainEpoch(
            self,
            data,
            optimizer: torch.optim.Optimizer,
            maxGradNorm: float      = Types.DEFAULT_SMORFCNN_MAX_GRAD_NORM,
            threshold: float        = Types.DEFAULT_SMORFCNN_THRESHOLD
        ) -> dict:
        
        """
        Trainer function
        """

        self.train()

        loss = 0.0
        n = 0
        all_probs = []
        all_targets = []

        lossFunction = torch.nn.BCEWithLogitsLoss()

        return

    def validateEpoch():
        return
    def fit():
        return
    def kFoldCrossValidation():
        return
    def saveModel() -> None:
        return