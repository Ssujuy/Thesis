import torch
import torch.nn as nn
from Convolution import ConvolutionBlock

import Types, Helpers

# ---------------------------------------------
# Multiple Kernel Convolution
# ---------------------------------------------

class MultiKernelConvolution(nn.Module):
    """
    Parallel Convolution Blocks with different kernel widths, to capture short motifs (codons, start/stop) and longer context.

    Attributes
    ----------
    forwardDebugLimit : int
        Limit for times debug logs are printed in forward.
    
    forwardDebugCounter : int
        Counter for debug logs in forward.

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
        Compute convolution with different kernel widths. Output of each branch is L_out = L_in - k + 1 and output shape is (B,C_out,L_out).
        Because different kernel widths produce different L_out we crop to minimum, which is L_min = 498.
        Mask first is updated from each convolution, cropped to L_min and combined with or statement for each mask.s
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
            debug: bool                 = Types.DEFAULT_DEBUG_MODE,
            forwardDebugLimit: int      = Types.DEFAULT_FORWARD_DEBUG_LIMIT
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

        forwardDebugLimit : int
            Limit for times debug logs are printed in forward.
        """
        super().__init__()

        self.debugMode = debug
        self.forwardDebugCounter = 0
        self.forwardDebugLimit = forwardDebugLimit
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
        Compute convolution with different kernel widths. Output of each branch is L_out = L_in - k + 1 and output shape is (B,C_out,L_out).
        Because different kernel widths produce different L_out we crop to minimum, which is L_min = 498.
        Mask first is updated from each convolution, cropped to L_min and combined with or statement for each mask.
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

        self._debugIn(x, mask)

        for branch in self.branches:
            y, m = branch(x, mask)
            outputs.append(y)
            masks.append(m)
            self._debugBranch(y, m)

        Lmin = min(y.size(-1) for y in outputs)
        outputs = [y[..., :Lmin] for y in outputs]
        masks = [m[..., :Lmin] for m in masks]

        out = torch.cat(outputs, dim=1)
        combinedMask = masks[0]
        for m in masks[1:]:
            combinedMask = (combinedMask | m)

        self._debugOut(out, combinedMask)

        if self.debugMode and self.forwardDebugLimit > self.forwardDebugCounter:
            self.forwardDebugCounter += 1

        return out, combinedMask

    def print(self) -> None:
        """
        Prints member variables of the class and number of model parameters and trainable model parameters.
        """

        Helpers.colourPrint(Types.Colours.BLUE, "Multiple Kernel Convolution Parameters:")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Input Channels: {self.inputChannels}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Output Channels per Kernel: {self.outputChannelsKernel}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Output Channels: {self.outputChannels}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Kernel List: {self.kernelList}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Stride: {self.stride}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Padding: {self.padding}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Groups: {self.groups}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Activation: {self.activation}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Dropout: {self.dropout}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Debug mode: {self.debugMode}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Model Paramters: {self.modelParams}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Model Trainable Parameters: {self.modelTrainableParams}")
    
    def _debugIn(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Prints shape of input Tensor in forward and mask.
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
                f"[MultiKernelConvolution] Input x.shape={tuple(x.shape)}-dtype={x.dtype}, mask.shape={tuple(mask.shape)}-dtype={mask.dtype}\n"
                f"[MultiKernelConvolution] mask sum per-batch={mask.sum(dim=1).detach().cpu().tolist()[:4]}"
            )

    def _debugBranch(self, b: torch.Tensor, m: torch.Tensor):
        """
        Prints shape of each branche's Tensor in forward.
        Prints will occur until limit is reached and debugMode is True.
        
        Parameters
        ----------
        b : Tensor
            Input Tensor.

        m : Tensor
            Mask Tensor.
        """

        if self.debugMode and self.forwardDebugLimit > self.forwardDebugCounter:
            Helpers.colourPrint(
                Types.Colours.PURPLE,
                f"[MultiKernelConvolution] Branch x.shape={tuple(b.shape)}-dtype={b.dtype}, mask.shape={tuple(m.shape)}-dtype={m.dtype}\n"
                f"[MultiKernelConvolution] mask sum per-batch={m.sum(dim=1).detach().cpu().tolist()[:4]}"
            )

    def _debugOut(self, out: torch.Tensor, m: torch.Tensor) -> None:
        """
        Prints shape of output Tensor in forward.
        Prints will occur until limit is reached and debugMode is True.

        Parameters
        ----------
        out : Tensor
            Input Tensor.

        m : Tensor
            Mask Tensor.
        """

        if self.debugMode and self.forwardDebugLimit > self.forwardDebugCounter:
            Helpers.colourPrint(
                Types.Colours.PURPLE,
                f"[MultiKernelConvolution] Output out.shape={tuple(out.shape)}-dtype={out.dtype}, mask.shape={tuple(m.shape)}-dtype={m.dtype}\n"
                f"[MultiKernelConvolution] mask sum per-batch={m.sum(dim=1).detach().cpu().tolist()[:4]}"
            )