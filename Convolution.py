import torch
import torch.nn as nn
import torch.nn.functional as F

import Types, Helpers

# ---------------------------------------------
# Masked Convolution
# ---------------------------------------------

class ConvolutionBlock(nn.Module):
    """
    Constructs a basic Convolution Network with configurable parameters:\n
    kernel, stride, padding, dilation, groups, activation and dropout

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
    
    conv1d : nn.Conv1d
        Convolution Conv1d from pyTorch.

    batchNormalization : nn.BatchNorm1d
        Batch Normalization.

    activation : nn
        Activation Function.

    modelParams : int
        Total of model's parameters.

    modelTrainableParams : int
        Total of model's trainable parameters, requires gradient.

    Methods
    ----------
    forward(x : Tensor, mask : Tensor) -> tuple[Tensor, Tensor]:
        Passes input x through convolution, batch normalization, activation and dropout.\n
        Then passes mask across the same convolution block and updates the mask, then\n
        x is multiplied by the mask to mark padded positions so they dont produce features.\n
        Returns tuple of y, mask.

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
        Constructs Conv1d, BatchNormalization, acvtivation function and dropout.

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
        self.bias = bias

        self.conv1d = nn.Conv1d(
            self.inputChannels,
            self.outputChannels,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias
        )

        self.register_buffer("mask_kernel", torch.ones(1, 1, kernel, dtype=torch.float32))

        self.activation = Types.activationFunctionMapping.get(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.modelParams = sum(p.numel() for p in self.parameters())
        self.modelTrainableParams = sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def _updateMask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Passes mask tensor through convolution with kernel=maskKernel (1,1,1).\n
        Then uses convolution.eq to compare every element of convolution to the scalar self.kernel.

        Parameters
        ----------
        mask : Tensor
            Mask Tensor.

        Return
        ----------
        Tensor
            Updated mask after convolution.eq.
        """
        if mask is None:
            raise ValueError("mask tensor is None")

        if not isinstance(mask, torch.Tensor):
            raise TypeError("Mask argument given is not a Tensor")

        mask = mask.unsqueeze(1)
        mask = mask.to(dtype=torch.float32)

        convolution = F.conv1d(
            mask,
            self.mask_kernel,
            stride=self.stride,
            padding=0,
            dilation=self.dilation
        )

        mask = convolution.eq(float(self.kernel)).squeeze(1)
        return mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Passes input x through convolution, batch normalization, activation and dropout.\n
        Updates the mask of input x then multiplies the output after convolution with mask\n
        and multiply again the output after activation and dropout.\n
        Returns a Tensor with the output of size outputChannels and mask with 1 as positions without pad.

        Parameters
        ----------
        x : Tensor
            Input Tensor.

        Return
        ----------
        tuple[Tensor, Tensor]
            Output Tensor and mask Tensor after convolution.  
        """
        
        if not isinstance(mask, torch.Tensor) or not isinstance(x, torch.Tensor):
            raise TypeError("Input x or mask arguments given is not a Tensor")

        if x is None or mask is None:
            raise ValueError("Input x or mask tensor is None")

        self._debugIn(x)
        y = self.conv1d(x)
        updatedMask = self._updateMask(mask)
        y = y * updatedMask.unsqueeze(1)
        y = self.activation(y)
        y = self.dropout(y)
        y = y * updatedMask.unsqueeze(1)
        self._debugOut(y)
        return y, updatedMask

    def print(self) -> None:
        """
        Prints member variables of the class and number of model parameters and trainable model parameters.
        """

        Helpers.colourPrint(Types.Colours.BLUE, "Convolution Block Parameters:")
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

    def _debugIn(self, x: torch.Tensor) -> None:
        """
        Prints debug information for input shape and type in forward. Only works once for forward and when debugMode is True.

        Parameters
        ----------
        x : Tensor
            Input Tensor.        
        """

        if self.forwardDebugOnce and self.debugMode:
            print(f"[ConvolutionBlock] in shape={tuple(x.shape)} dtype={x.dtype} device={x.device}"
            f"min={x.detach().min().item():.4f} max={x.detach().max().item():.4f} mean={x.detach().float().mean().item():.4f}")

    def _debugOut(self, x: torch.Tensor) -> None:
        """
        Prints debug information for output shape in forward. Only works once for forward and when debugMode is True.\n
        Toggles forwardDebugOnce attribute, to only run once.

        Parameters
        ----------
        x : Tensor
            Input Tensor.        
        """

        if self.forwardDebugOnce and self.debugMode:
            print(f"[ConvolutionBlock] out shape={tuple(x.shape)}")
            self.forwardDebugOnce = False