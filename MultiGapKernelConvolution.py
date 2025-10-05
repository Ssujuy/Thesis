import torch
import torch.nn as nn
from Convolution import ConvolutionBlock

import Types

# ---------------------------------------------
# Multiple Gapped Kernel Convultion
# ---------------------------------------------

class MultiGapKernelConvolution(nn.Module):
    """
    Parallel Convolution Blocks with different kernel widths and different gap sizes (dilation),\n
    to capture short motifs (codons, start/stop) and longer context resembles g-gap for DNA sequences.

    Attributes
    ----------
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
        Compute convolution with different kernel widths and different gap sizes (dillation).\n
        Output of each branch is L_out = L_in - k + 1 and output shape is (B,C_out,L_out).\n
        Because different kernel widths produce different L_out we crop to minimum, which is L_min = 498.\n
        Mask first is updated from each convolution, cropped to L_min and combined with or statement for each mask.\n
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
        debug: bool                 = Types.DEFAULT_DEBUG_MODE
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

        self.branches = nn.ModuleList()
        self.pairs = []

        for k in kernelList:
            for g in gapList:
                dil = g + 1
                self.branches.append(
                    ConvolutionBlock(
                        inputChannels=self.inputChannels,
                        outputChannels=self.outputChannelsGKernel,
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Compute convolution with different kernel widths and different gap sizes.\n
        Output of each branch is L_out = L_in - k + 1 and output shape is (B,C_out,L_out).\n
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

        for block in self.branches:
            y, m = block(x, mask)
            outputs.append(y)
            masks.append(m)

        Lmin = min(y.size(-1) for y in outputs)
        outputs = [y[..., :Lmin] for y in outputs]
        masks = [m[..., :Lmin] for m in masks]

        outputs = torch.cat(outputs, dim=1)
        masksOut = masks[0]
        for m in masks[1:]:
            masksOut = (masksOut & m)

        return outputs, masksOut