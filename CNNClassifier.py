import Types, Helpers
import torch
import torch.nn as nn

class ConvolutionBlock(nn.Module):
    """
    Conv1d -> BatchNorm -> GELU -> (Dropout)
    - BatchNorm stabilizes training
    - GELU is a strong modern activation

    Where:
    1. kernelWidth: number of adjacent positiona single filter looks at.
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
            kernelWidth: int,
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
            padding = (dilation * (kernelWidth - 1)) // 2

        self.conv1d = nn.Conv1d(
            inputChannels,
            outputChannels,
            kernelWidth,
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