import torch
import torch.nn as nn
from Convolution import ConvolutionBlock

import Types, Helpers

# ---------------------------------------------
# Residual Block
# ---------------------------------------------

class ResidualBlock(nn.Module):
    """
    Defining 2 Convolution Block classes. This class computes a residual mapping\n
    y = x + F(x) introduced by ResNets. inputChannels length is equal to output channels length.

    Attributes
    ----------
    forwardDebugLimit : int
        Limit for times debug logs are printed in forward.
    
    forwardDebugCounter : int
        Counter for debug logs in forward.

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
            debug: bool                 = Types.DEFAULT_DEBUG_MODE,
            forwardDebugLimit: int      = Types.DEFAULT_FORWARD_DEBUG_LIMIT
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

        forwardDebugLimit : int
            Limit for times debug logs are printed in forward.
        """
        super().__init__()

        self.forwardDebugCounter = 0
        self.forwardDebugLimit = forwardDebugLimit
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

        self._debugIn(x, mask)
        y1, m1 = self.convolution1(x, mask)
        y2, m2 = self.convolution2(y1, m1)

        x = x[..., :y2.size(-1)]
        x = x * m2.unsqueeze(1)

        self._debugResidual(x, mask, y2, m2)

        out = (x + y2) * m2.unsqueeze(1)

        self._debugOut(out, m2)

        if self.debugMode and self.forwardDebugLimit > self.forwardDebugCounter:
            self.forwardDebugCounter += 1

        return out, m2

    def print(self) -> None:
        """
        Prints member variables of the class and number of model parameters and trainable model parameters.
        """

        Helpers.colourPrint(Types.Colours.BLUE, "Residual Block Parameters:")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Input Channels: {self.inputChannels}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Output Channels: {self.outputChannels}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Kernel: {self.kernel}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Stride: {self.stride}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Padding: {self.padding}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Groups: {self.groups}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Activation: {self.activation}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Dropout: {self.dropout}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Debug mode: {self.debugMode}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Model Paramters: {self.modelParams}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Model Trainable Parameters: {self.modelTrainableParams}")

    def _debugIn(self, x: torch.Tensor, mask: torch.Tensor) -> None:
        """
        Prints shape of input Tensor in forward.\n
        Prints will occur until limit is reached and debugMode is True.

        Parameters
        ----------
        x : Tensor
            Input Tensor.

        mask : Tensor
            Mask Tensor of input.
        """

        if self.debugMode and self.forwardDebugLimit > self.forwardDebugCounter:
            Helpers.colourPrint(
                Types.Colours.PURPLE,
                f"[ResidualBlock] Input x.shape={tuple(x.shape)}-dtype={x.dtype}, mask.shape={tuple(mask.shape)}-dtype={mask.dtype}\n"
                f"[ResidualBlock] mask sum per-batch={mask.sum(dim=1).detach().cpu().tolist()[:4]}"
            )

    def _debugResidual(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            y2: torch.Tensor,
            mask2: torch.Tensor
        ) -> None:
        """
        Prints shape of Tensors in forward before residual.\n
        Prints will occur until limit is reached and debugMode is True.

        Parameters
        ----------
        x : Tensor
            Input Tensor.

        mask : Tensor
            Mask Tensor of input.

        y2 : Tensor
            Input Tensor.

        mask2 : Tensor
            Mask Tensor of input.
        """

        if self.debugMode and self.forwardDebugLimit > self.forwardDebugCounter:
            Helpers.colourPrint(
                Types.Colours.PURPLE,
                f"[ResidualBlock] Input x.shape={tuple(x.shape)}-dtype={x.dtype}, mask.shape={tuple(mask.shape)}-dtype={mask.dtype}\n"
                f"[ResidualBlock] mask sum per-batch={mask.sum(dim=1).detach().cpu().tolist()[:4]}"
            )
            Helpers.colourPrint(
                Types.Colours.PURPLE,
                f"[ResidualBlock] Input y2.shape={tuple(y2.shape)}-dtype={y2.dtype}, mask.shape={tuple(mask2.shape)}-dtype={mask2.dtype}\n"
                f"[ResidualBlock] mask sum per-batch={mask2.sum(dim=1).detach().cpu().tolist()[:4]}"
            )

    def _debugOut(self, out: torch.Tensor, mask: torch.Tensor) -> None:
        """
        Prints shape of output Tensor in forward.\n
        Prints will occur until limit is reached and debugMode is True.

        Parameters
        ----------
        out : Tensor
            Output Tensor.

        mask : Tensor
            Mask Tensor of input.
        """

        if self.debugMode and self.forwardDebugLimit > self.forwardDebugCounter:
            Helpers.colourPrint(
                Types.Colours.PURPLE,
                f"[ResidualBlock] Output x.shape={tuple(out.shape)}-dtype={out.dtype}, mask.shape={tuple(mask.shape)}-dtype={mask.dtype}\n"
                f"[ResidualBlock] mask sum per-batch={mask.sum(dim=1).detach().cpu().tolist()[:4]}"
            )