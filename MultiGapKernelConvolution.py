import torch
import torch.nn as nn
from Convolution import ConvolutionBlock

import Types, Helpers

# ---------------------------------------------
# Multiple Gapped Kernel Convultion
# ---------------------------------------------

class MultiGapKernelConvolution(nn.Module):
    """
    Parallel Convolution Blocks with different kernel widths and different gap sizes (dilation).
    To capture short motifs (codons, start/stop) and longer context resembles g-gap for DNA sequences.

    Attributes
    ----------
    forwardDebugLimit : int
        Limit for times debug logs are printed in forward.
    
    forwardDebugCounter : int
        Counter for debug logs in forward.

    debugMode : bool
        Turns debug mode on when true (more information)

    inputChannels : int
        Size of input.
    
    outputChannelsBranch : int
        Size of output channels per gap kernel.
    
    kernelList : list
        A list of different sized kernels.
    
    gapList : list
        A list of different sized gaps(dillations).
    
    stride : int
        How far the kernel moves during each step.

    padding : int
        Ensures edge positions get full windows and layer shapes stay compatible.

    groups : int
        Splits the inputChannels in groups and applies independent filters.

    activation : str
        Activation function to use, configurable for testing.

    dropout : float
        Zeros activations to reduce co-adaptation and overfitting.

    bias : bool
        If True, adds a learnable bias to the output.

    modelParams : int
        Total of model's parameters.

    modelTrainableParams : int
        Total of model's trainable parameters, requires gradient.

    Methods
    ----------
    forward(x : Tensor, mask : Tensor) -> tuple[Tensor, Tensor]:
        Compute convolution with different kernel widths and different gap sizes (dillation).
        Output of each branch is L_out = L_in - k + 1 and output shape is (B,C_out,L_out).
        Because different kernel widths produce different L_out we crop to minimum, which is L_min = 498.
        Mask first is updated from each convolution, cropped to L_min and combined with or statement for each mask.
        Finally, the features of shape (B, C_out, L_min) are concatenated and create (B, C_out x 5, L_min)
    """

    def __init__(
        self,
        inputChannels,
        outputChannelsGKernel: int,
        kernelList: list            = Types.DEFAULT_MULTI_GAP_KERNEL_KERNEL_LIST,
        gapList: list               = Types.DEFAULT_MULTI_GAP_KERNEL_GAP_LIST,
        stride: int                 = Types.DEFAULT_CONVOLUTION_STRIDE,
        padding: str                = Types.DEFAULT_CONVOLUTION_PADDING,
        groups: int                 = Types.DEFAULT_CONVOLUTION_GROUPS,
        activation: str             = Types.DEFAULT_CONVOLUTION_ACTIVATION,
        dropout: float              = Types.DEFAULT_CONVOLUTION_DROPOUT,
        bias: bool                  = Types.DEFAULT_CONVOLUTION_BIAS,
        debug: bool                 = Types.DEFAULT_DEBUG_MODE,
        forwardDebugLimit: int      = Types.DEFAULT_FORWARD_DEBUG_LIMIT
        ):
        """
        Constructs List of Convolution Blocks to draw features with different kernel sizes and gap sizes, initialize member variables.

        Parameters
        ----------
        inputChannels: int
            Size of input.

        outputChannelsGKernel: int
            Size of output for each gap kernel.

        kernelList : list
            Number of adjacent positions a single filter looks at.
        
        gapList : list
            Number of dillation.

        stride : int
            How far the kernel moves during each step.

        padding : str
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

        self.inputChannels = inputChannels
        self.outputChannelsBranch = outputChannelsGKernel
        self.kernelList = kernelList
        self.gapList = gapList
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.activation = activation
        self.dropout = dropout
        self.bias = bias
        self.debugMode = debug
        self.forwardDebugCounter = 0
        self.forwardDebugLimit = forwardDebugLimit

        self.branches = nn.ModuleList()
        self.pairs = []

        for k in kernelList:
            for g in gapList:
                dil = g + 1
                self.branches.append(
                    ConvolutionBlock(
                        inputChannels=self.inputChannels,
                        outputChannels=self.outputChannelsBranch,
                        kernel=k,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=dil,
                        groups=self.groups,
                        activation=self.activation,
                        dropout=self.dropout,
                        bias=self.bias,
                        debug=self.debugMode,
                    )
                )
                self.pairs.append((k, g))

        self.outputChannels = self.outputChannelsBranch * len(self.branches)

        self.modelParams = sum(p.numel() for p in self.parameters())
        self.modelTrainableParams = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Compute convolution with different kernel widths and different gap sizes.
        Output of each branch is L_out = L_in - k + 1 and output shape is (B,C_out,L_out).
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

        for block in self.branches:
            y, m = block(x, mask)
            outputs.append(y)
            masks.append(m)
            self._debugBranch(y, m)

        Lmin = min(y.size(-1) for y in outputs)
        outputs = [y[..., :Lmin] for y in outputs]
        masks = [m[..., :Lmin] for m in masks]

        outputs = torch.cat(outputs, dim=1)
        masksOut = masks[0]
        for m in masks[1:]:
            masksOut = (masksOut | m)

        self._debugOut(outputs, masksOut)

        if self.debugMode and self.forwardDebugLimit > self.forwardDebugCounter:
            self.forwardDebugCounter += 1

        return outputs, masksOut
    
    def print(self) -> None:
        """
        Prints member variables of the class and number of model parameters and trainable model parameters.
        """

        Helpers.colourPrint(Types.Colours.BLUE, "Multiple Gap Kernel Convolution Parameters:")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Input Channels: {self.inputChannels}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Output Channels per Gap Kernel: {self.outputChannelsBranch}")
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
                f"[MultiGapKernelConvolution] Input x.shape={tuple(x.shape)}-dtype={x.dtype}, mask.shape={tuple(mask.shape)}-dtype={mask.dtype}\n"
                f"[MultiGapKernelConvolution] mask sum per-batch={mask.sum(dim=1).detach().cpu().tolist()[:4]}"
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
                f"[MultiGapKernelConvolution] Branch x.shape={tuple(b.shape)}-dtype={b.dtype}, mask.shape={tuple(m.shape)}-dtype={m.dtype}\n"
                f"[MultiGapKernelConvolution] mask sum per-batch={m.sum(dim=1).detach().cpu().tolist()[:4]}"
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
                f"[MultiGapKernelConvolution] Output out.shape={tuple(out.shape)}-dtype={out.dtype}, mask.shape={tuple(m.shape)}-dtype={m.dtype}\n"
                f"[MultiGapKernelConvolution] mask sum per-batch={m.sum(dim=1).detach().cpu().tolist()[:4]}"
            )