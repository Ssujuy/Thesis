import torch
import torch.nn as nn
from Convolution import ConvolutionBlock

import Types

# ---------------------------------------------
# Multiple Kernel Convolution
# ---------------------------------------------

class MultiKernelConvolution(nn.Module):
    """
    Parallel Convolution Blocks with different kernel widths, to capture short motifs (codons, start/stop) and longer context.

    Attributes
    ----------
    forwardDebugOnce : bool
        Ensures forward will only print logs once to avoid overflowing output

    debugMode : bool
        Turns debug mode on when true (more information).

    inputChannels: int
        Size of input.

    outputChannelsKernel: int
        Size of output channels per kernel.

    kernelList : list
        A list of different sized kernels.

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

    branches : ModuleList
        List of Convolution Blocks with different kernel value, with length equal to kernelList length.

    modelParams : int
        Total of model's parameters.

    modelTrainableParams : int
        Total of model's trainable parameters, requires gradient.

    Methods
    ----------
    forward(x : Tensor, mask : Tensor) -> tuple[Tensor, Tensor]:
        Compute convolution with different kernel widths. Output of each branch is L_out = L_in - k + 1 and output shape is (B,C_out,L_out).\n
        Because different kernel widths produce different L_out we crop to minimum, which is L_min = 498.\n
        Mask first is updated from each convolution, cropped to L_min and combined with or statement for each mask.\n
        Finally, the features of shape (B, C_out, L_min) are concatenated and create (B, C_out x 5, L_min)

    print():
        Prints member variables of the class and number of model parameters and trainable model parameters.
    """

    def __init__(
            self,
            inputChannels: int,
            outputChannelsKernel: int   = Types.DEFAULT_MULTI_KERNEL_PER_KERNEL_OUTPUT,
            kernelList: list            = Types.DEFAULT_MULTI_KERNEL_KERNEL_LIST,
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
        Constructs List of Convolution Blocks to draw features with different kernel sizes and initializes member variables.

        Parameters
        ----------
        inputChannels: int
            Size of input.

        outputChannelsKernel: int
            Size of output for each kernel.

        kernelList : list
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
        self.outputChannelsKernel = outputChannelsKernel
        self.kernelList = kernelList
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.activation = activation
        self.dropout = dropout
        self.bias = bias

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
                dropout=self.dropout,
                bias=self.bias,
                debug=self.debugMode
            )
            self.branches.append(conv)
        
        self.outputChannels = outputChannelsKernel * len(kernelList)

        self.modelParams = sum(p.numel() for p in self.parameters())
        self.modelTrainableParams = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute convolution with different kernel widths. Output of each branch is L_out = L_in - k + 1 and output shape is (B,C_out,L_out).\n
        Because different kernel widths produce different L_out we crop to minimum, which is L_min = 498.\n
        Mask first is updated from each convolution, cropped to L_min and combined with or statement for each mask.\n
        Finally, the features of shape (B, C_out, L_min) are concatenated and create (B, C_out x 5, L_min).

        Parameters
        ----------
        x : Tensor
            Input Tensor.

        Return
        ----------
        tuple[Tensor, Tensor]
            Output Tensor and mask Tensor after multiple convolution with different kernel size.
        """

        outputs = []
        masks = []

        self._debugIn(x)

        for branch in self.branches:
            y, m = branch(x, mask)
            outputs.append(y)
            masks.append(m)
            self._debugBranch(y)

        Lmin = min(y.size(-1) for y in outputs)
        outputs = [y[..., :Lmin] for y in outputs]
        masks   = [m[..., :Lmin] for m in masks]

        out = torch.cat(outputs, dim=1)
        combined_mask = masks[0]
        for m in masks[1:]:
            combined_mask = combined_mask | m

        self._debugOut(out)
        return out, combined_mask

    def print(self) -> None:
        """
        Prints member variables of the class and number of model parameters and trainable model parameters.
        """

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
        print(f" - Bias: {self.bias}")
        print(f" - Debug mode: {self.debugMode}")
        print(f" - Model Parameters: {self.modelParams}")
        print(f" - Model Trainable Parameters: {self.modelTrainableParams}")
    
    def _debugIn(self,x):
        """
        Prints shape of input Tensor in forward and kernel list of model. Only works once for forward and when debugMode is True.

        Parameters
        ----------
        x : Tensor
            Input Tensor.        
        """

        if self.forwardDebugOnce and self.debugMode:
            print(f"[MultiKernelConv] in shape={tuple(x.shape)} kernels={self.kernelList}")

    def _debugBranch(self,b):
        """
        Prints shape of each branche's Tensor in forward. Only works once for forward and when debugMode is True.

        Parameters
        ----------
        b : Tensor
            Input Tensor.        
        """

        if self.forwardDebugOnce and self.debugMode:
            print(f"[MultiKernelConv] out shape={tuple(b.shape)}")

    def _debugOut(self, out: torch.Tensor) -> None:
        """
        Prints shape of output Tensor in forward. Only works once for forward and when debugMode is True.

        Parameters
        ----------
        out : Tensor
            Input Tensor.        
        """

        if self.forwardDebugOnce and self.debugMode:
            print(f"[MultiKernelConv] concat out shape={tuple(out.shape)}")
            self.forwardDebugOnce = False