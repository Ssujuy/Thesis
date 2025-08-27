import Types, Helpers
import torch
import torch.nn as nn

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

        if padding is None:
            # "Same" padding for odd kernels under given dilation.
            padding = (dilation * (kernel - 1)) // 2

        self.conv1d = nn.Conv1d(
            inputChannels,
            outputChannels,
            kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )

        self.batchNormalization = nn.BatchNorm1d(outputChannels)
        self.activation = Types.activationFunctionMapping.get(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

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
        
        self.convolution1 = ConvolutionBlock(
            inputChannels,
            inputChannels,
            kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            activation=activation,
            dropout=dropout
        )

        self.convolution2 = ConvolutionBlock(
            inputChannels,
            inputChannels,
            kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            activation=activation,
            dropout=dropout
        )

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

        self.branches = []

        for kernel in kernelList:
            conv = ConvolutionBlock(
                inputChannels,
                outputChannelsKernel,
                kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                activation=activation,
                dropout=dropout
            )
            self.branches.append(conv)
        
        self.outputChannels = outputChannelsKernel * len(kernelList)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = []

        for branch in self.branches:
            output.append(branch(x))

        return torch.cat(output, dim=1)