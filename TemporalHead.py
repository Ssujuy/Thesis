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
    forwardDebugLimit : int
        Limit for times debug logs are printed in forward.
    
    forwardDebugCounter : int
        Counter for debug logs in forward.

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
        dropout: float              = Types.DEFAULT_CONVOLUTION_DROPOUT,
        bias: bool                  = Types.DEFAULT_CONVOLUTION_BIAS,
        multipleDilation: bool      = Types.DEFAULT_TEMPORAL_MULTI_DILATION,
        residualBlocks: int         = Types.DEFAULT_RESIDUAL_BLOCKS_NMB,
        debug: bool                 = Types.DEFAULT_DEBUG_MODE,
        forwardDebugLimit: int      = Types.DEFAULT_FORWARD_DEBUG_LIMIT
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

        forwardDebugLimit : int
            Limit for times debug logs are printed in forward.
        """
        super().__init__()

        self.debugMode = debug
        self.forwardDebugCounter = 0
        self.forwardDebugLimit = forwardDebugLimit
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

        if not isinstance(mask, torch.Tensor) or not isinstance(x, torch.Tensor):
            raise TypeError("Input x or mask arguments given is not a Tensor")

        if x is None or mask is None:
            raise ValueError("Input x or mask tensor is None")

        self._debugIn(x, mask)

        x, m = self.reduce(x, mask)

        self._debugReduce(x, m)

        for block in self.residualBlocks:
            x, m = block(x, m)

        self._debugResidual(x, m)

        gmp = Helpers.globalMaxPooling(x, m)
        gap = Helpers.globalAveragePooling(x, m)

        self._debugPooling(gmp,gap)

        out = torch.cat([gap,gmp], dim=1)
        self._debugOut(out)

        if self.debugMode and self.forwardDebugLimit > self.forwardDebugCounter:
            self.forwardDebugCounter += 1

        return out

    def print(self) -> None:
        """
        Prints member variables of the class and number of model parameters and trainable model parameters.
        """

        Helpers.colourPrint(Types.Colours.BLUE, "Temporal Head Parameters:")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Input Channels: {self.inputChannels}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Hidden Channels: {self.hiddenChannels}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Kernel for Residual Block: {self.kernelResidual}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Kernel for size Reduction: {self.kernelReduction}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Stride: {self.stride}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Padding: {self.padding}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Groups: {self.groups}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Activation: {self.activation}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Dropout: {self.dropout}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Change Dilation for Residual Block: {self.multipleDilation}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Number of Residual blocks: {self.residualBlocks}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Model Paramters: {self.modelParams}")
        Helpers.colourPrint(Types.Colours.BLUE, f" - Model Trainable Parameters: {self.modelTrainableParams}")

    def _debugIn(self, x: torch.Tensor, mask: torch.Tensor) -> None:
        """
        Prints shape of input Tensor and mask Tensor in forward.\n
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
                f"[TemporalHead] Input x.shape={tuple(x.shape)}-dtype={x.dtype}, mask.shape={tuple(mask.shape)}-dtype={mask.dtype}\n" 
                f"[TemporalHead] mask sum per-batch={mask.sum(dim=1).detach().cpu().tolist()[:4]}"
            )

    def _debugReduce(self, x: torch.Tensor, mask: torch.Tensor) -> None:
        """
        Prints shape of input Tensor and mask Tensor in forward after reduction.\n
        Prints will occur until limit is reached and debugMode is True.

        Parameters
        ----------
        x : Tensor
            Input Tensor.

        mask : Tensor
            Mask Tensor.
        """

        if self.debugMode and self.forwardDebugLimit > self.forwardDebugCounter:
            Helpers.colourPrint(
                Types.Colours.PURPLE,
                f"[TemporalHead] after reduce x.shape={tuple(x.shape)}-dtype={x.dtype}, mask.shape={tuple(mask.shape)}-dtype={mask.dtype}\n" 
                f"[TemporalHead] mask sum per-batch={mask.sum(dim=1).detach().cpu().tolist()[:4]}"
            )

    def _debugResidual(self, x: torch.Tensor, mask: torch.Tensor) -> None:
        """
        Prints shape of input Tensor and mask Tensor in forward after Residual stack.\n
        Prints will occur until limit is reached and debugMode is True.

        Parameters
        ----------
        x : Tensor
            Input Tensor.

        mask : Tensor
            Mask Tensor.
        """

        if self.debugMode and self.forwardDebugLimit > self.forwardDebugCounter:
            Helpers.colourPrint(
                Types.Colours.PURPLE,
                f"[TemporalHead] after residualBlocks x.shape={tuple(x.shape)}-dtype={x.dtype}, mask.shape={tuple(mask.shape)}-dtype={mask.dtype}\n"
                f"[TemporalHead] mask sum per-batch={mask.sum(dim=1).detach().cpu().tolist()[:4]}"
            )

    
    def  _debugPooling(self, gmp: torch.Tensor, gap: torch.Tensor) -> None:
        """
        Prints type and shape of Gloab Max and Global Average Pooling in forward.\n
        Prints will occur until limit is reached and debugMode is True.

        Parameters
        ----------
        gmp : Tensor
            Gloabal Max Pooling Tensor.
        
        gap : Tensor
            Gloabal Average Pooling Tensor.
        """

        if self.debugMode and self.forwardDebugLimit > self.forwardDebugCounter:
            Helpers.colourPrint(
                Types.Colours.PURPLE,
                f"[TemporalHead] GMP shape={tuple(gmp.shape)}\n"
                f"[TemporalHead] GAP shape={tuple(gap.shape)}"
            )

    
    def _debugOut(self, out: torch.Tensor) -> None:
        """
        Prints shape of forward output. Prints will occur until limit is reached and debugMode is True.

        Parameters
        ----------
        out : Tensor
            Output Tensor.
        """

        if self.debugMode and self.forwardDebugLimit > self.forwardDebugCounter:
            Helpers.colourPrint(Types.Colours.PURPLE, f"[TemporalHead] Output shape={tuple(out.shape)}")
