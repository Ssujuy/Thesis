import torch
import torch.nn as nn
from Convolution import ConvolutionBlock

import Types

# ---------------------------------------------
# Residual Block
# ---------------------------------------------

class ResidualBlock(nn.Module):
    """
    Defining 2 Convolution Block classes. This class computes a residual mapping\n
    y = x + F(x) introduced by ResNets. inputChannels length is equal to output channels length.

    Attributes
    ----------
    forwardDebugOnce : bool
        Ensures forward will only print logs once to avoid overflowing output

    debugMode : bool
        Turns debug mode on when true (more information).

    inputChannels: int
        Size of input

    outputChannels: int
        Size of output.

    kernel : int
        Number of adjacent positions a single filter looks at.

    stride : int
        How far the kernel moves during each step.

    padding : int
        Ensures edge positions get full windows and layer shapes stay compatible.

    dilation : int
        Spacing between kernel taps.

    groups : int
        Splits the inputChannels in groups and applies independent filters.

    activation : str
        Activation function to use, configurable for testing.

    dropout : float
        Zeros activations to reduce co-adaptation and overfitting.

    bias : bool
        If True, adds a learnable bias to the output.

    convolution1 : ConvolutionBlock
        First Convolution block to compute convolution1(x).

    convolution2 : ConvolutionBlock
        Second Convolution Block to compute convolution2(convolution1(x)).

    modelParams : int
        Total of model's parameters.

    modelTrainableParams : int
        Total of model's trainable parameters, requires gradient.

    Methods
    ----------
    forward(x : Tensor, mask : Tensor) -> tuple[Tensor, Tensor]:
        Computes Residual mapping. First computes convolution1(x, mask),\n
        then convolution2(y1, m1) taken from convolution1 output,\n
        finally x + y2, which is x + convolution2(convolution1(x)) and multiply by the final mask (m2).\n
        Then returns the output Tensor and output mask Tensor.

    print():
        Prints member variables of the class and number of model parameters and trainable model parameters.
    """

    def __init__(
            self,
            inputChannels: int,
            outputChannels: int,
            kernel: int,
            stride: int                 = Types.DEFAULT_CONVOLUTION_STRIDE,
            padding: str                = Types.DEFAULT_CONVOLUTION_PADDING,
            dilation: int               = Types.DEFAULT_CONVOLUTION_DILATION,
            groups: int                 = Types.DEFAULT_CONVOLUTION_GROUPS,
            activation: str             = Types.DEFAULT_CONVOLUTION_ACTIVATION,
            dropout: float              = Types.DEFAULT_CONVOLUTION_DROPOUT,
            bias: bool                  = Types.DEFAULT_CONVOLUTION_BIAS,
            debug: bool                 = Types.DEFAULT_DEBUG_MODE
        ):
        """
        Constructs 2 Convolution Block classes for Residual Mapping and initializes member variables.

        Parameters
        ----------
        inputChannels: int
            Size of input.

        outputChannels: int
            Size of output.

        kernel : int
            Number of adjacent positions a single filter looks at.

        stride : int
            How far the kernel moves during each step.

        padding : int
            Ensures edge positions get full windows and layer shapes stay compatible.

        dilation : int
            Spacing between kernel taps.

        groups : int
            Splits the inputChannels in groups and applies independent filters.

        activation : str
            Activation function to use, configurable for testing.

        dropout : float
            Zeros activations to reduce co-adaptation and overfitting.
        
        bias : bool
            If True, adds a learnable bias to the output.

        debug : bool
            Turns debug mode on when true (more information).
        """
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
        self.bias = bias

        self.convolution1 = ConvolutionBlock(
            self.inputChannels,
            self.outputChannels,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            activation=self.activation,
            dropout=self.dropout,
            bias=self.bias,
            debug=self.debugMode
        )

        self.convolution2 = ConvolutionBlock(
            self.outputChannels,
            self.outputChannels,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            activation=self.activation,
            dropout=self.dropout,
            bias=self.bias,
            debug=self.debugMode
        )

        self.modelParams = sum(p.numel() for p in self.parameters())
        self.modelTrainableParams = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Residual mapping. First computes convolution1(x, mask),\n
        then convolution2(y1, m1) taken from convolution1 output,\n
        finally x + y2, which is x + convolution2(convolution1(x)) and multiply by the final mask (m2).\n
        Then returns the output Tensor and output mask Tensor.

        Parameters
        ----------
        x : Tensor
            Input Tensor.
        
        mask : Tensor
            Mask Tensor.

        Return
        ----------
        tuple[Tensor, Tensor]
            Output Tensor and mask Tensor after Residual Mapping.
        """

        if not isinstance(mask, torch.Tensor) or not isinstance(x, torch.Tensor):
            raise TypeError("Input x or mask arguments given is not a Tensor")

        if x is None or mask is None:
            raise ValueError("Input x or mask tensor is None")

        y1, m1 = self.convolution1(x, mask)
        y2, m2 = self.convolution2(y1, m1)

        x = x[..., :y2.size(-1)]
        x = x * m2.unsqueeze(1)

        out = (x + y2) * m2.unsqueeze(1)

        return out, m2

    def print(self) -> None:
        """
        Prints member variables of the class and number of model parameters and trainable model parameters.
        """

        print("Residual Block Parameters:")
        print(f" - Input Channels: {self.inputChannels}")
        print(f" - Output Channels: {self.outputChannels}")        
        print(f" - Kernel: {self.kernel}")
        print(f" - Stride: {self.stride}")
        print(f" - Padding: {self.padding}")
        print(f" - Groups: {self.groups}")
        print(f" - Activation: {self.activation}")
        print(f" - Dropout: {self.dropout}")
        print(f" - Bias: {self.bias}")
        print(f" - Debug mode: {self.debugMode}")
        print(f" - Model Parameters: {self.modelParams}")
        print(f" - Model Trainable Parameters: {self.modelTrainableParams}")

    def _debugIn(self, x: torch.Tensor) -> None:
        """
        Prints shape of input Tensor in forward. Only works once for forward and when debugMode is True.

        Parameters
        ----------
        x : Tensor
            Input Tensor.        
        """

        if self.forwardDebugOnce and self.debugMode:
            print(f"[ResidualBlock] in shape={tuple(x.shape)}")

    def _debugOut(self, out: torch.Tensor) -> None:
        """
        Prints shape of output Tensor in forward. Only works once for forward and when debugMode is True.

        Parameters
        ----------
        x : Tensor
            Input Tensor.        
        """

        if self.forwardDebugOnce and self.debugMode:
            print(f"[ResidualBlock] out shape={tuple(out.shape)}")
            self.forwardDebugOnce = False