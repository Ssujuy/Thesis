import torch
import torch.nn as nn
from Convolution import ConvolutionBlock

import Types

# ---------------------------------------------
# Multiple Gapped Kernel Convultion
# ---------------------------------------------

class MultiGapKernelConvolution(nn.Module):
    """
    """
    def __init__(
        self,
        inputChannels,
        outputChannelsBranch:int,
        kernelList: list            = [3,4,5,6,7,11],
        gapList: list               = [1,2,3],
        stride: int                 = Types.DEFAULT_CONVOLUTION_STRIDE,
        padding                     = Types.DEFAULT_CONVOLUTION_PADDING,
        groups: int                 = Types.DEFAULT_CONVOLUTION_GROUPS,
        activation: str             = Types.DEFAULT_CONVOLUTION_ACTIVATION,
        dropout: float              = Types.DEFAULT_TEMPORAL_DROPOUT,
        debug: bool                 = Types.DEFAULT_DEBUG_MODE
        ):
        super().__init__()

        self.inputChannels = inputChannels
        self.outputChannelsBranch = outputChannelsBranch
        self.kernelList = kernelList
        self.gapList = gapList
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.activation = activation
        self.dropout = dropout
        self.debugMode = debug

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
                        debug=self.debugMode,
                    )
                )
                self.pairs.append((k, g))

        self.outputChannels = self.outputChannelsBranch * len(self.branches)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        
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