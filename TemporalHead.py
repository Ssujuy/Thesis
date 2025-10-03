import torch
import torch.nn as nn
from Convolution import ConvolutionBlock
from Residual import ResidualBlock

import Types,Helpers

# ---------------------------------------------
# Temporal Head
# ---------------------------------------------

class TemporalHead(nn.Module):
    """
    Temporal Head Class implements a size reduction of the input channel using a Convolution Block,\n
    creates a Residual Block stack with multiple dilations and computes Global Average and Max Pooling.\n
    Assumes sequences are right-padded and `mask` marks valid positions: [B, L] with 1=valid, 0=pad.

    Attributes
    ----------
    forwardDebugOnce : bool
        Ensures forward will only print logs once to avoid overflowing output

    debugMode : bool
        Turns debug mode on when true (more information).

    hiddenChannels: int
        Size of reduction's output and Residual stack's output.

    kernelReduction : int
        Kernel width for reduction block.
    
    kernelResidual : int
        Kernel width for residual stack.

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

    multipleDilation : bool
        Flag to activate different dilations for each Residual stack item.

    residualBlocks : int
        Number of Residual Blocks included in the Residucal stack.

    reduce : ConvolutionBlock
        Convolution Block used to reduce input channels.

    residualBlocks : nn.Sequential
        Stack of Residual Blocks.

    modelParams : int
        Total of model's parameters.

    modelTrainableParams : int
        Total of model's trainable parameters, requires gradient.

    Methods
    ----------
    forward(x : Tensor, mask : Tensor) -> tuple[Tensor, Tensor]:
        Passes input x through convolution, batch normalization, activation and dropout.\n
        Returns a Tensor with the output of size outputChannels.

    print():
        Prints member variables of the class and number of model parameters and trainable model parameters.
    """

    def __init__(
        self,
        inputChannels,
        hiddenChannels: int         = Types.DEFAULT_TEMPORAL_HIDDEN_CHANNELS,
        kernelResidual: int         = Types.DEFAULT_TEMPORAL_KERNEL_RESIDUAL,
        kernelReduction: int        = Types.DEFAULT_TEMPORAL_KERNEL_REDUCTION,
        stride: int                 = Types.DEFAULT_CONVOLUTION_STRIDE,
        padding                     = Types.DEFAULT_CONVOLUTION_PADDING,
        groups: int                 = Types.DEFAULT_CONVOLUTION_GROUPS,
        activation: str             = Types.DEFAULT_CONVOLUTION_ACTIVATION,
        dropout: float              = Types.DEFAULT_TEMPORAL_DROPOUT,
        bias: bool                  = Types.DEFAULT_CONVOLUTION_BIAS,
        multipleDilation: bool      = Types.DEFAULT_TEMPORAL_MULTI_DILATION,
        residualBlocks: int         = Types.DEFAULT_RESIDUAL_BLOCKS_NMB,
        debug: bool                 = Types.DEFAULT_DEBUG_MODE
        ):
        """
        Constructs a Convolution Block for input reduction, a stack of Residual block of size residualBlocks\n
        and initializes member variables.

        Parameters
        ----------
        inputChannels: int
            Size of input.

        hiddenChannels: int
            Size of output for reduction.

        kernelReduction : int
            Kernel width for reduction block.
        
        kernelResidual : int
            Kernel width for residual stack..

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

        multipleDilation : bool
            Flag to activate different dilations for each Residual stack item.

        residualBlocks : int
            Number of Residual Blocks included in the Residucal stack.

        debug : bool
            Turns debug mode on when true (more information).
        """
        super().__init__()

        self.debugMode = debug
        self.forwardDebugOnce = debug
        self.inputChannels = inputChannels
        self.hiddenChannels = hiddenChannels
        self.kernelResidual = kernelResidual
        self.kernelReduction = kernelReduction
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.activation = activation
        self.dropout = dropout
        self.bias = bias
        self.multipleDilation = multipleDilation
        self.residualBlocks = residualBlocks

        self.reduce = ConvolutionBlock(
            self.inputChannels,
            self.hiddenChannels,
            self.kernelReduction,
            stride=self.stride,
            padding=self.padding,
            dilation=Types.DEFAULT_CONVOLUTION_DILATION,
            groups=self.groups,
            activation=self.activation,
            dropout=self.dropout,
            bias=self.bias,
            debug=self.debugMode
        )

        blocks = []

        for i in range(self.residualBlocks):
            dilation = 2**i if self.multipleDilation else Types.DEFAULT_CONVOLUTION_DILATION
            
            blocks.append(
                ResidualBlock(
                    self.hiddenChannels,
                    self.hiddenChannels,
                    self.kernelResidual,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=dilation,
                    groups=self.groups,
                    activation=self.activation,
                    dropout=self.dropout,
                    bias=self.bias,
                    debug=self.debugMode
                )
            )       
        
        self.residualBlocks = nn.ModuleList(blocks)

        self.modelParams = sum(p.numel() for p in self.parameters())
        self.modelTrainableParams = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Passes input Tensor to reduce (Conv Block) and project to smaller size.\n
        Then passes reduced input to Residual Blocks sequentially for residualBlocks number.\n
        Mask is also passed to both reduction and residual block.\n
        Finally, calculates global max pooling and global average pooling, then concatenates.

        Parameters
        ----------
        x : Tensor
            Input Tensor, shape [B, C, L] feature map.

        mask : Tensor
            Mask Tensor with shape [B, L], 1 for valid tokens, 0 for right-padded tail.

        Return
        ----------
        Tensor
            Concatenation of global max pooling and global average pooling, with shape [B, 2*C] (global-max ⊕ masked-global-avg).
        """

        self._debugIn(x, mask)

        x, m = self.reduce(x, mask)

        self._debugReduce(x)

        for block in self.residualBlocks:
            x, m = block(x, m)

        self._debugRes(x)

        gmp = Helpers.globalMaxPooling(x, m)
        gap = Helpers.globalAveragePooling(x, m)

        self._debugPooling(gmp,gap)

        out = torch.cat([gap,gmp], dim=1)
        self._debugOut(out)
        return out

    def print(self) -> None:
        """
        Prints member variables of the class and number of model parameters and trainable model parameters.
        """

        print("Temporal Head Parameters:")
        print(f" - Input Channels: {self.inputChannels}")
        print(f" - Hidden Channels: {self.hiddenChannels}")
        print(f" - Kernel for Residual Block: {self.kernelResidual}")
        print(f" - Kernel for size Reduction: {self.kernelReduction}")
        print(f" - Stride: {self.stride}")
        print(f" - Padding: {self.padding}")
        print(f" - Groups: {self.groups}")
        print(f" - Activation: {self.activation}")
        print(f" - Dropout: {self.dropout}")
        print(f" - Change Dilation for Residual Block: {self.multipleDilation}")
        print(f" - Number of Residual blocks: {self.residualBlocks}")
        print(f" - Model Paramters: {self.modelParams}")
        print(f" - Model Trainable Parameters: {self.modelTrainableParams}")

    def _debugIn(self, x: torch.Tensor, mask: torch.Tensor) -> None:
        """
        Prints shape of input Tensor and mask Tensor in forward. Only works once for forward and when debugMode is True.

        Parameters
        ----------
        x : Tensor
            Input Tensor.
        
        mask : Tensor
            Mask Tensor of input.
        """

        if self.forwardDebugOnce and self.debugMode:
            print(f"[TemporalHead] in x shape={tuple(x.shape)} mask shape={tuple(mask.shape)} "
            f"mask sum per-batch={mask.sum(dim=1).detach().cpu().tolist()[:4]}")
    
    def _debugReduce(self, x: torch.Tensor) -> None:
        """
        Prints shape of input Tensor and mask Tensor in forward after reduction.\n
        Only works once for forward and when debugMode is True.

        Parameters
        ----------
        x : Tensor
            Input Tensor.
        """

        if self.forwardDebugOnce and self.debugMode:
            print(f"[TemporalHead] after reduce shape={tuple(x.shape)}")

    def _debugRes(self, x: torch.Tensor) -> None:
        """
        Prints shape of input Tensor and mask Tensor in forward after Residual stack.\n
        Only works once for forward and when debugMode is True.

        Parameters
        ----------
        x : Tensor
            Input Tensor.
        """

        if self.forwardDebugOnce and self.debugMode:
            print(f"[TemporalHead] after residualBlocks shape={tuple(x.shape)}")
    
    def  _debugPooling(self, gmp: torch.Tensor, gap: torch.Tensor) -> None:
        """
        Prints type and shape of Gloab Max and Global Average Pooling in forward.\n
        Only works once for forward and when debugMode is True.

        Parameters
        ----------
        gmp : Tensor
            Gloabal Max Pooling Tensor.
        
        gap : Tensor
            Gloabal Average Pooling Tensor.
        """

        if self.forwardDebugOnce and self.debugMode:
            print(f"[TemporalHead] GMP type={type(gmp)} shape={tuple(gmp.shape)} "
            f"GAP type={type(gap)} shape={tuple(gap.shape)}")
    
    def _debugOut(self, out: torch.Tensor) -> None:
        """
        Prints shape of forward output. Only works once for forward and when debugMode is True.

        Parameters
        ----------
        out : Tensor
            Output Tensor.
        """

        if self.forwardDebugOnce and self.debugMode:
            print(f"[TemporalHead] out shape={tuple(out.shape)}")
            self.forwardDebugOnce = False
