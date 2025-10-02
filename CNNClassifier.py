import Types, Helpers
import torch,os,random
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrScheduler
from torch.utils.data import DataLoader, ConcatDataset, Subset
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.model_selection import StratifiedKFold


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

        print("Convolution Block Parameters:")
        print(f" - Input Channels: {self.inputChannels}")
        print(f" - Output Channels: {self.outputChannels}")
        print(f" - Kernel: {self.kernel}")
        print(f" - Stride: {self.stride}")
        print(f" - Padding: {self.padding}")
        print(f" - Groups: {self.groups}")
        print(f" - Activation: {self.activation}")
        print(f" - Dropout: {self.dropout}")
        print(f" - Bias: {self.bias}")
        print(f" - Debug Mode: {self.debugMode}")
        print(f" - Model Parameters: {self.modelParams}")
        print(f" - Model Trainable Parameters: {self.modelTrainableParams}")

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
        Size of input

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
        Finally, the features of shape (B, C_out, L_min) are concatenated and create (B, C_out x 5, L_min)

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

# class MultiGapKernelConvolution(nn.Module):
#     """
#     """
#     def __init__(
#         self,
#         inputChannels,
#         outputChannelsBranch:int,
#         kernelList: list          = Types.DEFAULT_TEMPORAL_KERNEL_RESIDUAL,
#         gapList: list        = Types.DEFAULT_TEMPORAL_KERNEL_REDUCTION,
#         stride: int                 = Types.DEFAULT_CONVOLUTION_STRIDE,
#         padding                     = Types.DEFAULT_CONVOLUTION_PADDING,
#         groups: int                 = Types.DEFAULT_CONVOLUTION_GROUPS,
#         activation: str             = Types.DEFAULT_CONVOLUTION_ACTIVATION,
#         dropout: float              = Types.DEFAULT_TEMPORAL_DROPOUT,
#         debug: bool                 = Types.DEFAULT_DEBUG_MODE
#         ):
#         super().__init__()

#         self.inputChannels = inputChannels
#         self.outputChannelsBranch = outputChannelsBranch
#         self.kernelList = kernelList
#         self.gapList = gapList
#         self.stride = stride
#         self.padding = padding
#         self.groups = groups
#         self.activation = activation
#         self.dropout = dropout
#         self.debugMode = debug

#         self.branches = nn.ModuleList()
#         self.pairs = []

#         for k in kernelList:
#             for g in gapList:
#                 dil = g + 1
#                 self.branches.append(
#                     ConvolutionBlock(
#                         inputChannels=self.inputChannels,
#                         outputChannels=self.outputChannelsBranch,
#                         kernel=k,
#                         stride=self.stride,
#                         padding=self.padding,
#                         dilation=dil,
#                         groups=self.groups,
#                         activation=self.activation,
#                         dropout=self.dropout,
#                         debug=self.debugMode,
#                     )
#                 )
#                 self.pairs.append((k, g))

#         self.outputChannels = self.outputChannelsBranch * len(self.branches)

#     def forward(self, x: torch.Tensor, mask: torch.Tensor):
#         # Collect per-branch outputs and masks
#         ys, ms = [], []
#         for block in self.branches:
#             y, m = block(x, mask)    # y:[B,C,L_k,g], m:[B,L_k,g]
#             ys.append(y); ms.append(m)

#         # Align lengths: crop all to the minimum L
#         Lmin = min(y.size(-1) for y in ys)
#         ys = [y[..., :Lmin] for y in ys]
#         ms = [m[..., :Lmin] for m in ms]

#         # Concatenate channels and AND the masks across branches
#         y_cat = torch.cat(ys, dim=1)    # [B, C_total, Lmin]
#         m_cat = ms[0]
#         for m in ms[1:]:
#             m_cat = (m_cat & m)

#         return y_cat, m_cat

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

# ---------------------------------------------
# The CNNClassifier for smORFs
# ---------------------------------------------

class SmORFCNN(nn.Module):
    """
    Multi branch Classifier that takes dnabert 6 embeddings, CLS embeddings and mean embeddings of size [B,1536,1].\n
    Uses onehot encoded sequences of size [B, 4, 512] that are passed through a Multi Kernel Convolution and then are\n
    concatenated creating [B, C_out * len(kernelList), L_min]. The features from Multi Kernel are passed to Temporal Head,\n
    which reduces the feature's size [B, reduces, 498] then passes them through residual mapping 2 residual blocks and computes\n
    Global Average Pooling and Global Max Pooling [B, C_out * 2].

    
    Attributes
    ----------
    onehotInputChannels : int
        Size of onehot encoded sequences.

    embeddingsInputChannels : int
        Size of dnabert6 embeddings.

    featuresPath: str
        Path to pyTorch file storing onehot encoded sequences as Tensors [B,4,Length], masked Tensors\n
        for valid=1, padded=0 positions and dnabert6 embeddings

    temporalHead : bool
        Activates Temporal Head Block.
    
    multikernel : bool
        Activates Multi Kernel Block.

    onehotKernelList : list
        List of different kernel sizes.
    
    outputChannelsKernel : int
        Size of C_out, output channels per kernel convolution block.

    temporalHeadOutputChannels : int
        Size of C_out, output channels of temporal head block (* 2).

    residualBlocks : int
        Number of residual blocks used in Temporal head.

    classes : int
        Number of classes produced at the end of classifier.

    layer1Output : int
        Output size of layer1 in classifier.

    layer2Output : int
        Output size of layer2 in classifier.

    classifierDropout : float
        Dropout between each layer in classifer.

    learningRate : float
        Initial Learning Rate passed in AdamW optimizer.

    weightDecay : float
        Weight decay parameter passed in optimizer.

    minLearningRate : float
        Minimum Learning Rate parameter passed in scheduler.

    schedulerFactor : float
        Factor parameter passed in scheduler.
    
    schedulerWarmup : float
        Warmup parameter passed in scheduler.
    
    threshold : float
        threshold used to differentiate negative from positive smORFs, from sigmoid probs.

    maxGradNorm : float
        Maximum grad norm used to clip grad norm.

    seed : int
        Random see of the model.

    deterministic : bool
        Enables model to use only deterministic algorithms, helps reproducibillity.

    device : str
        Device used by model.

    trainBatchSize : int
        Batch size used for training DataLoader.

    valBatchSize : int
        Batch size used for validation DataLoader.

    testBatchSize : int
        Batch size used for testing DataLoader.

    trainSplit : float
        Split of the total Dataset used for training.

    valSplit : float
        plit of the total Dataset used for validation.

    testSplit : float
        plit of the total Dataset used for testing.

    epochs : int
        Number of epochs for fit.
    
    Methods
    ----------
    initializeDataset() -> None:
        Uses Helpers function loadFeaturesFromPt in order to load all tensors from pyTorch file in a TensorDataset.\n
        Then, uses Helpers toDataLoaders function, in order to split the Dataset in training, validation and testing DataLoaders.\n
        Finally prints some minor information for its split and a label distribution for each DataLoader and sum.
    
    forward(xOnehot: Tensor, xEmbeddings: Tensor, maskOnehot: Tensor) -> Tensor:
        Takes dnabert6 embeddings and appends them to features list.\n
        Passes onehot encoded sequences ([B,4,512]) along with their mask through ([B,512,1]) through Multiple Kernel Convolution,\n
        then through Temporal Head and appends them to the features list as well.\n
        Finally, creates a features tensor with all features drawn from all branches and passes them to the classifer.\n
        Logits are returned.


    trainEpoch(epochIndex: int) -> dict:
        Sets the model to training mode with self.train(), initializes loss function BCEWithLogits.\n
        Iterates the trainingDataLoader and for each batch, moves inputs to device and uses model's forward function.\n
        Then, computes probabillities and loss, calls back propagation, clips grad norm and uses optimizer and scheduler step.\n
        Finally, calculates runningLoss and computes all epoch metrics and returns them as a dict. ("loss", "acc", "precision", "recall", "f1", etc.).

    validateEpoch(epochIndex: int) -> dict:
        Uses torch.no_grad(), sets the model to evaluation mode using self.eval(), initializes loss function BCEWithLogits.\n
        Iterates the validationDataLoader and for each batch, moves inputs to device and uses model's forward function.\n
        Then, computes probabillities and loss. Finally, calculates runningLoss and computes all epoch metrics\n
        and returns them as a dict. ("loss", "acc", "precision", "recall", "f1", etc.), along with epoch ROC AUC.


    test() -> dict:
        Uses torch.no_grad(), sets the model to evaluation mode using self.eval(), initializes loss function BCEWithLogits.\n
        Iterates the validationDataLoader and for each batch, moves inputs to device and uses model's forward function.\n
        Then, computes probabillities and loss. Finally, calculates runningLoss and computes all epoch metrics\n 
        and returns them as a dict. ("loss", "acc", "precision", "recall", "f1", etc.), along with epoch ROC AUC.

    fit(epochs: int):
        Moves model to device, initializes scheduler and keeps dictionaries for training and validation metrics.\n
        For each epoch trainEpoch and validateEpoch functions are called, their metrics are saved, along with lr and best state.\n
        Finally after the epochs are finished test function is called to test the model and curves for acc,loss,f1 and auc are printed.

    kFoldCrossValidation()

    print():
        Prints member variables of the class and number of model parameters and trainable model parameters.
    """
    def __init__(
        self,
        onehotInputChannels: int,
        embeddingsInputChannels: int,
        featuresPath: str,
        temporalHead: bool                      = Types.DEFAULT_SMORFCNN_TEMPORAL_HEAD,
        multiKernel: bool                       = Types.DEFAULT_SMORFCNN_MULTI_KERNEL,
        onehotKernelList: list                  = Types.DEFAULT_SMORFCNN_ONEHOT_KERNEL_LIST,
        outputChannelsKernel: int               = Types.DEFAULT_SMORFCNN_OUTPUT_CHANNELS_KERNEL,
        temporalHeadOutputChannels: int         = Types.DEFAULT_SMORFCNN_OUTPUT_CHANNELS_TEMPORAL,
        residualBlocks: int                     = Types.DEFAULT_SMORFCNN_RESIDUAL_BLOCKS,
        classes: int                            = Types.DEFAULT_SMORFCNN_CLASSES,
        layer1Output: int                       = Types.DEFAULT_SMORFCNN_CLASSIFIER_L1_OUTPUT,
        layer2Output: int                       = Types.DEFAULT_SMORFCNN_CLASSIFIER_L2_OUTPUT,
        classifierDropout: float                = Types.DEFAULT_SMORFCNN_CLASSIFIER_DROPOUT,
        learningRate: float                     = Types.DEFAULT_SMORFCNN_LEARNING_RATE,
        weightDecay: float                      = Types.DEFAULT_SMORFCNN_WEIGHT_DECAY,
        minLearningRate: float                  = Types.DEFAULT_SMORFCNN_MINIMUM_LEARNING_RATE,
        schedulerFactor: float                  = Types.DEFAULT_SMORFCNN_SCHEDULER_FACTOR,
        schedulerWarmup: float                  = Types.DEFAULT_SMORFCNN_SCHEDULER_WARMUP,
        threshold: float                        = Types.DEFAULT_SMORFCNN_THRESHOLD,
        maxGradNorm: float                      = Types.DEFAULT_SMORFCNN_MAX_GRAD_NORM,
        seed: int                               = Types.DEFAULT_SMORFCNN_SEED,
        deterministic: bool                     = Types.DEFAULT_SMORFCNN_DETERMINISTIC,
        device: str                             = Types.DEFAULT_SMORFCNN_DEVICE,
        trainBatchSize: int                     = Types.DEFAULT_SMORFCNN_TRAIN_BATCH_SIZE,
        valBatchSize: int                       = Types.DEFAULT_SMORFCNN_VALIDATION_BATCH_SIZE,
        testBatchSize: int                      = Types.DEFAULT_SMORFCNN_TEST_BATCH_SIZE,
        trainSplit: float                       = Types.DEFAULT_SMORFCNN_TRAIN_SPLIT,
        valSplit: float                         = Types.DEFAULT_SMORFCNN_VALIDATION_SPLIT,
        testSplit: float                        = Types.DEFAULT_SMORFCNN_TEST_SPLIT,
        epochs: int                             = Types.DEFAULT_SMORFCNN_EPOCHS,
        debug: bool                             = Types.DEFAULT_DEBUG_MODE
    ):
        """
        Constructs the complete smORF Classifier, using multi branched Multiple Kernel Convolution and DNABERT6 embeddings.\n
        Also, initializes model's parameters and the final 2-layer classifier.

        Parameters
        ----------
        onehotInputChannels : int
            Size of onehot encoded sequences.

        embeddingsInputChannels : int
            Size of dnabert6 embeddings.

        featuresPath: str
            Path to pyTorch file storing onehot encoded sequences as Tensors [B,4,Length], masked Tensors\n
            for valid=1, padded=0 positions and dnabert6 embeddings

        temporalHead : bool
            Activates Temporal Head Block.
        
        multikernel : bool
            Activates Multi Kernel Block.

        onehotKernelList : list
            List of different kernel sizes.
        
        outputChannelsKernel : int
            Size of C_out, output channels per kernel convolution block.

        temporalHeadOutputChannels : int
            Size of C_out, output channels of temporal head block (* 2).

        residualBlocks : int
            Number of residual blocks used in Temporal head.

        classes : int
            Number of classes produced at the end of classifier.

        layer1Output : int
            Output size of layer1 in classifier.

        layer2Output : int
            Output size of layer2 in classifier.

        classifierDropout : float
            Dropout between each layer in classifer.

        learningRate : float
            Initial Learning Rate passed in AdamW optimizer.

        weightDecay : float
            Weight decay parameter passed in optimizer.

        minLearningRate : float
            Minimum Learning Rate parameter passed in scheduler.

        schedulerFactor : float
            Factor parameter passed in scheduler.
        
        schedulerWarmup : float
            Warmup parameter passed in scheduler.
        
        threshold : float
            threshold used to differentiate negative from positive smORFs, from sigmoid probs.

        maxGradNorm : float
            Maximum grad norm used to clip grad norm.

        seed : int
            Random see of the model.

        deterministic : bool
            Enables model to use only deterministic algorithms, helps reproducibillity.

        device : str
            Device used by model.

        trainBatchSize : int
            Batch size used for training DataLoader.

        valBatchSize : int
            Batch size used for validation DataLoader.

        testBatchSize : int
            Batch size used for testing DataLoader.

        trainSplit : float
            Split of the total Dataset used for training.

        valSplit : float
            plit of the total Dataset used for validation.

        testSplit : float
            plit of the total Dataset used for testing.

        epochs : int
            Number of epochs for fit.
        """
        super().__init__()

        self.debugMode = debug
        self.forwardDebugOnce = debug
        self.seed = seed
        self.deterministic = deterministic
        self.device = device
        self.featuresPath = featuresPath

        self._initializeEnvironment()

        self.onehotInputChannels = onehotInputChannels
        self.embeddingsInputChannels =  embeddingsInputChannels

        self.temporalHead = temporalHead
        self.multiKernel = multiKernel
        self.onehotKernelList = onehotKernelList
        self.outputChannelsKernel = outputChannelsKernel
        self.temporalHeadOutputChannels = temporalHeadOutputChannels
        self.residualBlocks = residualBlocks
        self.layer1Output = layer1Output
        self.layer2Output = layer2Output
        self.classifierDropout = classifierDropout
        self.classes = classes
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.eps = Types.DEFAULT_SMORFCNN_EPS
        self.betas = Types.DEFAULT_SMORFCNN_BETAS
        self.minLearningRate = minLearningRate
        self.schedulerFactor = schedulerFactor
        self.schedulerWarmpup = schedulerWarmup
        self.threshold = threshold
        self.maxGradNorm = maxGradNorm
        self.trainBatchSize = trainBatchSize
        self.validationBatchSize = valBatchSize
        self.testBatchSize = testBatchSize
        self.trainSplit = trainSplit
        self.validationSplit = valSplit
        self.testSplit = testSplit
        self.epochs = epochs

        self.onehotMultiKernelClass = None
        self.onehotTemporalClass = None

        if self.multiKernel:
            self.onehotMultiKernelClass = MultiKernelConvolution(
                inputChannels=self.onehotInputChannels,
                outputChannelsKernel=self.outputChannelsKernel,
                kernelList=self.onehotKernelList
            )
            
        if self.temporalHead:
            self.onehotTemporalClass = TemporalHead(
                self.onehotMultiKernelClass.outputChannels,
                hiddenChannels=self.temporalHeadOutputChannels,
                residualBlocks=self.residualBlocks
            )

        self.fusedDim = self._calculateFusedDim()

        self.classifier = nn.Sequential(
            nn.Linear(self.fusedDim, self.layer1Output),
            nn.GELU(),
            nn.Dropout(self.classifierDropout),

            nn.Linear(self.layer1Output, self.layer2Output),
            nn.GELU(),
            nn.Dropout(self.classifierDropout),

            nn.Linear(self.layer2Output, self.classes)
        )

        self.optimizer = self._optimizerInit()

        self.modelParams = sum(p.numel() for p in self.parameters())
        self.modelTrainableParams = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self._debugInit()

        self.trainDataLoader = None
        self.validationDataLoader = None
        self.testDataLoader = None

        self.to(self.device)

    def _initializeEnvironment(self):

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        os.environ["PYTHONHASHSEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        torch.manual_seed(self.seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        if self.deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def initializeDataset(self) -> None:
        """
        Uses Helpers function loadFeaturesFromPt in order to load all tensors from pyTorch file in a TensorDataset.\n
        Then, uses Helpers toDataLoaders function, in order to split the Dataset in training, validation and testing DataLoaders.\n
        Finally prints some minor information for its split and a label distribution for each DataLoader and sum.
        """
        tensorDataset = Helpers.loadFeaturesFromPt(self.featuresPath)

        self.trainDataLoader, self.validationDataLoader, self.testDataLoader = Helpers.toDataloaders(
            tensorDataset,
            self.trainSplit,
            self.validationSplit,
            self.testSplit,
            self.trainBatchSize,
            self.validationBatchSize,
            self.testBatchSize,
            self.seed
        )

        self._debugDataLoader()

        Helpers.printDataloader("Training", self.trainDataLoader)
        Helpers.printDataloader("Validation", self.validationDataLoader)
        Helpers.printDataloader("Testing", self.testDataLoader)

        Helpers.plotLabelDistribution(self.trainDataLoader, self.validationDataLoader, self.testDataLoader)

    def _calculateFusedDim(self) -> int:

        temporalHeadDim = (2 * self.temporalHeadOutputChannels) * int(self.temporalHead)

        if self.temporalHead:
            poolingNoTemporalDim = 0
        else:
            onehotChannels = (self.onehotMultiKernelClass.outputChannels
                            if self.multiKernel else self.onehotInputChannels)
            embedChannels  = (self.embeddingsMultiKernelClass.outputChannels
                            if self.multiKernel else self.embeddingsInputChannels)
            poolingNoTemporalDim = 2 * (onehotChannels + embedChannels)

        multiKernelDim = 0

        return 2560 

    def _optimizerInit(self) -> torch.optim.Optimizer:

        decay, noDecay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or name.endswith(".bias"):
                noDecay.append(param)     # norms & biases
            else:
                decay.append(param)        # weight matrices/kernels

        parameterGroups = [
            {"params": decay,    "weight_decay": self.weightDecay},
            {"params": noDecay, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(parameterGroups, lr=self.learningRate, betas=self.betas, eps=self.eps)

    @staticmethod
    def _best_threshold_for_f1(probs: torch.Tensor, targets: torch.Tensor):  # NEW
        # probs, targets on CPU 1D
        p = probs.view(-1).float().numpy()
        y = targets.view(-1).long().numpy()
        # candidate thresholds: sorted unique probabilities (plus endpoints)
        thr = np.unique(p)
        thr = np.concatenate(([0.0], thr, [1.0]))
        # vectorized F1 sweep
        best_f1, best_t = 0.0, 0.5
        # To keep it simple and fast, sample at most 2048 thresholds
        if thr.size > 2048:
            thr = np.linspace(0.0, 1.0, 2048, dtype=np.float64)
        for t in thr:
            yhat = (p >= t).astype(np.int32)
            tp = (yhat & (y == 1)).sum()
            fp = (yhat & (y == 0)).sum()
            fn = ((1 - yhat) & (y == 1)).sum()
            denom = (2*tp + fp + fn)
            f1 = (2*tp) / denom if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return best_t, best_f1


    def _schedulerInit(self) -> torch.optim.lr_scheduler:

        if self.trainDataLoader is None:
            AttributeError("Traindataloader must be initialized to calculate size")

        stepsPerEpoch = len(self.trainDataLoader)
        totalSteps = stepsPerEpoch * self.epochs
        warmupSteps = int(self.schedulerWarmpup * totalSteps)
        cosineSteps = totalSteps - warmupSteps

        warmup = lrScheduler.LinearLR(self.optimizer, start_factor=self.schedulerFactor, total_iters=warmupSteps)

        cosine = lrScheduler.CosineAnnealingLR(self.optimizer, T_max=cosineSteps, eta_min=self.minLearningRate)

        return lrScheduler.SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[warmupSteps])    

    def forward(self, xOnehot: torch.Tensor, xEmbeddings:  torch.Tensor, maskOnehot: torch.Tensor) -> torch.Tensor:
        """
        Takes dnabert6 embeddings and appends them to features list.\n
        Passes onehot encoded sequences ([B,4,512]) along with their mask through ([B,512,1]) through Multiple Kernel Convolution,\n
        then through Temporal Head and appends them to the features list as well.\n
        Finally, creates a features tensor with all features drawn from all branches and passes them to the classifer.\n
        Logits are returned.

        Parameters
        ----------
        xOnehot : Tensor
            Onehot encoded sequences of shape [B,4,512].

        xEmbeddings : Tensor
            DNABERT6 embeddings of shape [B,1536,1].

        maskOnehot : Tensor
            Mask used to differentiate valid from padded positions, shape [B,512,1].

        Return
        ----------
        Tensor
            Logits Tensor after the classifier's output.
        """
        self._debugIn(xOnehot, xEmbeddings, maskOnehot)

        features = []
        inputOneHot = xOnehot
        inputEmbeddings = xEmbeddings

        if xOnehot is None:
            raise AttributeError("Did not receive onehot encoded input!")
        
        if xEmbeddings is None:
            raise AttributeError("Did not receive dnabert6 embeddings input!")

        if self.multiKernel:
            inputOneHot, maskMKC = self.onehotMultiKernelClass(inputOneHot, maskOnehot)

        if self.temporalHead:
            inputOneHot = self.onehotTemporalClass(inputOneHot, maskMKC)
        
        self._debugOnehot(inputOneHot)

        features.append(inputOneHot)

        features.append(inputEmbeddings.squeeze(-1))

        if not features:
            raise RuntimeError("Failed to produce any features!")

        fused = features[0] if len(features) == 1 else torch.cat(features, dim=1)

        if fused.size(1) != self.fusedDim:
            raise ValueError(f"Expected fused dimension {self.fusedDim} , got {fused.size(1)}")

        fused = fused.squeeze(-1)

        logits = self.classifier(fused)

        self._debugLogits(logits)

        return logits.squeeze(-1) if self.classes == 1 else logits

    def trainEpoch(self, epochIndex: int) -> dict:
        """
        Sets the model to training mode with self.train(), initializes loss function BCEWithLogits.\n
        Iterates the trainingDataLoader and for each batch, moves inputs to device and uses model's forward function.\n
        Then, computes probabillities and loss, calls back propagation, clips grad norm and uses optimizer and scheduler step.\n
        Finally, calculates runningLoss and computes all epoch metrics and returns them as a dict. ("loss", "acc", "precision", "recall", "f1", etc.).

        Parameters
        ----------
        epochIndex : int
            Current epoch number.

        Return
        ----------
        dict
            Dictionary of all epoch metrics during triaing ("acc", "loss", precision, etc.).
        """
        self.train()

        n = 0
        runningLoss = 0.0
        probabilities = []
        targets = []

        lossFunction = torch.nn.BCEWithLogitsLoss()

        print(f"Starting training for epoch {epochIndex}")

        try:
            iterator = tqdm(
                    enumerate(self.trainDataLoader),
                    total=len(self.trainDataLoader),
                    desc=f"Epoch {epochIndex}",
                    leave=False
                )
        except Exception:
            iterator = enumerate(self.trainDataLoader)

        for i, batch in iterator:
            xOnehot, maskOnehot, xEmbed, maskEmbed, y = batch

            self._debugInEpoch(xOnehot, maskOnehot, xEmbed, maskEmbed, y, "Training", i, epochIndex)

            xOnehot = xOnehot.to(self.device)
            maskOnehot = maskOnehot.to(self.device)
            xEmbed = xEmbed.to(self.device)
            maskEmbed = maskEmbed.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            outputs = self(xOnehot, xEmbed, maskOnehot)

            self._debugOutEpoch(outputs, "Training", i, epochIndex)

            loss = lossFunction(outputs, y.float())
            probs = torch.sigmoid(outputs)

            loss.backward()

            # self._debugStats(
            #     probs,
            #     loss,
            #     maxGradNorm,
            #     "Training",
            #     i,
            #     epochIndex
            # )

            torch.nn.utils.clip_grad_norm_(self.parameters(), self.maxGradNorm)

            self.optimizer.step()
            self.scheduler.step()
            batchSize = y.size(0)
            n += batchSize
            runningLoss += loss.item() * batchSize
            probabilities.append(probs.detach().view(-1).cpu())
            targets.append(y.detach().long().view(-1).cpu())

            if 'tqdm' in locals():
                iterator.set_postfix(loss=runningLoss / n)

        probabilities = torch.cat(probabilities, dim=0)
        targets = torch.cat(targets, dim=0)

        self._debugFinal(probabilities, targets, runningLoss, n, "Training", epochIndex)

        return Helpers.computeEpochMetrics(probabilities, targets, runningLoss, n, self.threshold, epochIndex)

    @torch.no_grad()
    def validateEpoch(self, epochIndex: int) -> dict:
        """
        Uses torch.no_grad(), sets the model to evaluation mode using self.eval(), initializes loss function BCEWithLogits.\n
        Iterates the validationDataLoader and for each batch, moves inputs to device and uses model's forward function.\n
        Then, computes probabillities and loss. Finally, calculates runningLoss and computes all epoch metrics\n
        and returns them as a dict. ("loss", "acc", "precision", "recall", "f1", etc.).

        Parameters
        ----------
        epochIndex : int
            Current epoch number.

        Return
        ----------
        dict
            Dictionary of all epoch metrics during validation ("acc", "loss", precision, etc.), along with ROC AUC.
        """
        self.eval()

        n = 0
        runningLoss = 0.0
        probabilities = []
        targets = []

        lossFunction = torch.nn.BCEWithLogitsLoss()

        print(f"Starting validation for epoch {epochIndex}")

        iterator = tqdm(
                enumerate(self.validationDataLoader),
                total=len(self.validationDataLoader),
                desc=f"Epoch {epochIndex}",
                leave=False
            )

        for j, batch in iterator:

            xOnehot, maskOnehot, xEmbed, maskEmbed, y = batch

            self._debugInEpoch(xOnehot, maskOnehot, xEmbed, maskEmbed, y, "Validation", j, epochIndex)

            xOnehot = xOnehot.to(self.device)
            maskOnehot = maskOnehot.to(self.device)
            xEmbed = xEmbed.to(self.device)
            maskEmbed = maskEmbed.to(self.device)
            y = y.to(self.device)

            outputs = self(xOnehot, xEmbed, maskOnehot)

            self._debugOutEpoch(outputs, "Validation", j, epochIndex)

            loss = lossFunction(outputs, y.float())
            probs = torch.sigmoid(outputs)

            batchSize = y.size(0)
            n += batchSize
            runningLoss += loss.item() * batchSize
            probabilities.append(probs.detach().cpu())
            targets.append(y.detach().cpu())

            if 'tqdm' in locals():
                iterator.set_postfix(loss=runningLoss / n)

        probabilities = torch.cat(probabilities, dim=0)
        targets = torch.cat(targets, dim=0)

        self._debugFinal(probabilities, targets, runningLoss, n, "Validation", epochIndex)

        metrics = Helpers.computeEpochMetrics(probabilities, targets, runningLoss, n, self.threshold, epochIndex)


        best_t, best_f1 = self._best_threshold_for_f1(probabilities.cpu(), targets.cpu())   # NEW
        # metrics["best_threshold"] = best_t
        # metrics["best_f1"] = best_f1
        print(f"[VAL] best threshold={best_t:.4f}  best F1={best_f1:.4f}") 

        metrics = metrics | Helpers.computeEpochROC(probabilities, targets, epochIndex)

        return metrics

    @torch.no_grad()
    def test(self) -> dict:
        """
        Uses torch.no_grad(), sets the model to evaluation mode using self.eval(), initializes loss function BCEWithLogits.\n
        Iterates the validationDataLoader and for each batch, moves inputs to device and uses model's forward function.\n
        Then, computes probabillities and loss. Finally, calculates runningLoss and computes all epoch metrics\n 
        and returns them as a dict. ("loss", "acc", "precision", "recall", "f1", etc.), along with epoch ROC AUC.

        Return
        ----------
        dict
            Dictionary of all epoch metrics during validation ("acc", "loss", precision, etc.), along with ROC AUC.
        """
        self.eval()

        n = 0
        runningLoss = 0.0
        probabilities = []
        targets = []

        lossFunction = torch.nn.BCEWithLogitsLoss()

        print(f"Starting testing")

        iterator = tqdm(
                enumerate(self.testDataLoader),
                total=len(self.testDataLoader),
                desc=f"{self.testDataLoader}",
                leave=False
            )

        for k, batch in iterator:

            xOnehot, maskOnehot, xEmbed, maskEmbed, y = batch

            self._debugInEpoch(xOnehot, maskOnehot, xEmbed, maskEmbed, y, "Testing", k, 1)

            xOnehot = xOnehot.to(self.device)
            maskOnehot = maskOnehot.to(self.device)
            xEmbed = xEmbed.to(self.device)
            maskEmbed = maskEmbed.to(self.device)
            y = y.to(self.device)

            outputs = self(xOnehot, xEmbed, maskOnehot)

            self._debugOutEpoch(outputs, "Testing", k, 1)

            loss = lossFunction(outputs, y.float())
            probs = torch.sigmoid(outputs)

            batchSize = y.size(0)
            n += batchSize
            runningLoss += loss.item() * batchSize
            probabilities.append(probs.detach().cpu())
            targets.append(y.detach().cpu()) 

            if 'tqdm' in locals():
                iterator.set_postfix(loss=runningLoss / n)

        probabilities = torch.cat(probabilities, dim=0)
        targets = torch.cat(targets, dim=0)

        self._debugFinal(probabilities, targets, runningLoss, n, "Testing", 1)

        metrics = Helpers.computeEpochMetrics(probabilities, targets, runningLoss, n, self.threshold, 0)

        metrics = metrics | Helpers.computeEpochROC(probabilities, targets, 0)

        return metrics

    def fit(self, epochs: int):
        """
        Moves model to device, initializes scheduler and keeps dictionaries for training and validation metrics.\n
        For each epoch trainEpoch and validateEpoch functions are called, their metrics are saved, along with lr and best state.\n
        Finally after the epochs are finished test function is called to test the model and curves for acc,loss,f1 and auc are printed.

        Parameters
        ----------
        epochIndex : int
            Current epoch number.
        """
        self.to(self.device)

        self.scheduler = self._schedulerInit()

        bestF1 = -1.0
        bestEpoch = 0
        bestState: dict[str, torch.Tensor] | None = None

        trainingMetrics = {
            "loss": [],
            "acc": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "TP": [],
            "TN": [],
            "FP": [],
            "FN": [],
        }

        validationMetrics = {
            "loss": [],
            "acc": [],
            "precision": [],
            "recall": [],
            "learningRate": [],
            "f1": [],
            "TP": [],
            "TN": [],
            "FP": [],
            "FN": [],
            "auc": [],
            "fpr": [],
            "tpr": []
        }

        epochIter = tqdm(
            enumerate(range(1, epochs + 1), start=1),
            total=epochs,
            desc="Epochs",
            leave=True
        )

        for epoch,_ in epochIter:

            epochTrainMetrics = self.trainEpoch(epochIndex=epoch)

            epochValMetrics = self.validateEpoch(epochIndex=epoch)

            for key in epochTrainMetrics:
                trainingMetrics[key].append(epochTrainMetrics[key])

            for key in epochValMetrics:
                validationMetrics[key].append(epochValMetrics[key])

            epochLR = self.optimizer.param_groups[0]["lr"]
            validationMetrics["learningRate"].append(epochLR)
            print(f"[Scheduler] epoch={epoch} val_loss={epochValMetrics["loss"]:.6f} lr={epochLR:.3e}")

            # ---- 5) Save best-by-val-F1 weights ----
            if validationMetrics["f1"][epoch - 1] > bestF1:
                bestF1 = validationMetrics["f1"][epoch - 1]
                bestEpoch = epoch
                # Keep a CPU copy; avoids GPU memory bloat and is device-agnostic to reload
                bestState = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

            try:
                epochIter.set_postfix_str(
                    f"F1 {validationMetrics['f1']:.4f}, AUC {validationMetrics.get('roc_auc', float('nan')):.4f}, LR {epochLR:.2e}"
                )
            except Exception:
                pass

        # ---- 7) Restore the best checkpoint (by val F1) ----
        if bestState is not None:
            self.load_state_dict(bestState)
            self.to(self.device)
        
        testMetrics = self.test()

        Helpers.printFitSummary(trainingMetrics, validationMetrics)

        Helpers.plotFitCurves(trainingMetrics, validationMetrics)

        Helpers.plotROCCurve(validationMetrics, bestEpoch)

        Helpers.plotConfusionPie(trainingMetrics, validationMetrics, testMetrics, epochs)

        return trainingMetrics, validationMetrics

    def kFoldCrossValidation(self, k: int = Types.DEFAULT_SMORFCNN_KFOLD):
        """
        """
        fullDataset = ConcatDataset([self.trainDataLoader.dataset, self.validationDataLoader.dataset])

        # ---------- B) Extract labels directly (no type branching) ----------
        # Assumptions:
        #   trainLoader.dataset           -> Subset
        #   trainLoader.dataset.dataset   -> TensorDataset
        #   trainLoader.dataset.indices   -> index list/array into the base TensorDataset
        #   base TensorDataset.tensors[-1] is labels [N]
        tSubset = self.trainDataLoader.dataset
        valSubset = self.validationDataLoader.dataset

        tBase = tSubset.dataset          # TensorDataset
        valBase = valSubset.dataset          # TensorDataset

        tIdx = torch.as_tensor(tSubset.indices, dtype=torch.long)
        valIdx = torch.as_tensor(valSubset.indices, dtype=torch.long)

        tLabels = tBase.tensors[-1].index_select(0, tIdx)  # [N_tr]
        valLabels = valBase.tensors[-1].index_select(0, valIdx)  # [N_va]

        labels = torch.cat([tLabels, valLabels], dim=0).detach().cpu().long().numpy()  # [N_tr+N_va]

        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.seed)
        splits = list(splitter.split(np.arange(len(labels)), labels))

        foldMetrics = []
        bestFoldF1 = -1.0
        bestFoldState = None
        bestFold = None

        iterator = tqdm(enumerate(splits, start=1),total=k,desc=f"{k}-Fold CV",leave=True)

        for foldIndex, (trainIndex, valIndex) in iterator:

            print(f"\n=== Fold {foldIndex}/{k}: train={len(trainIndex)}  val={len(valIndex)} ===")

            trainDataset = Subset(fullDataset, indices=trainIndex)
            valDataset = Subset(fullDataset, indices=valIndex)

            trainDataloader = DataLoader(
                trainDataset,
                batch_size=self.trainBatchSize,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                drop_last=False
            )
            validationDataloader = DataLoader(
                valDataset,
                batch_size=self.validationBatchSize,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )

            # # ---------- F) Reinitialize model for an independent fold run ----------
            # def _init(m):
            #     if hasattr(m, "reset_parameters"):
            #         m.reset_parameters()
            # self.apply(_init)
            # self.to(self.device)

            _ = self.fit(
                trainLoader=trainDataloader,
                valLoader=validationDataloader,
                epochs=self.epochs
            )

            metrics = self.validateEpoch(
                validationData=validationDataloader,
                epochIndex=foldIndex,
            )

            foldMetrics.append(metrics)

            if metrics["f1"] > bestFoldF1:
                bestFoldF1 = metrics["f1"]
                bestFold = foldIndex
                bestFoldState = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

            if 'tqdm' in locals():
                iterator.set_postfix_str(f"F1 {metrics['f1']:.4f}  AUC {metrics.get('roc_auc', float('nan')):.4f}")

        if bestFoldState is not None:
            self.load_state_dict(bestFoldState)
            self.to(self.device)
            print(f"\nRestored best fold #{bestFold} (val F1={bestFoldF1:.4f})")

        summary = Helpers.kFoldSummary(foldMetrics)

        Helpers.printKFoldMetrics(foldMetrics, summary)

        Helpers.plotMeanROC(foldMetrics, summary)

        return foldMetrics, summary

    def saveModel() -> None:
        return
    
    def _debugInit(self):

        if self.debugMode:
            print(f"[INIT] total params={self.modelParams} trainable={self.modelTrainableParams}")
            print(f"[INIT] optimizer param_groups sizes={[len(g['params']) for g in self.optimizer.param_groups]}")
            print(f"[INIT] device={self.device} multiKernel={self.multiKernel} temporalHead={self.temporalHead}")
            print(f"[INIT] onehot kernels={self.onehotKernelList}")
            print(f"[INIT] fusedDim={self.fusedDim} classifier={self.classifier}")
    
    def _debugDataLoader(self):

        if self.debugMode:
            tIdx  = set(self.trainDataLoader.dataset.indices)
            vIdx  = set(self.validationDataLoader.dataset.indices)
            teIdx = set(self.testDataLoader.dataset.indices)
            print(f"[SPLIT] |train|={len(tIdx)} |val|={len(vIdx)} |test|={len(teIdx)} "
                f"overlap train∩val={len(tIdx & vIdx)} train∩test={len(tIdx & teIdx)} val∩test={len(vIdx & teIdx)}")

    def _debugIn(self, xOnehot, xEmbeddings, maskOnehot):
        if self.forwardDebugOnce and self.debugMode:
            print(f"[SmORFCNN.forward] xOnehot shape={tuple(xOnehot.shape)} "
                    f"min/max/mean=({xOnehot.detach().min().item():.3f}/{xOnehot.detach().max().item():.3f}/{xOnehot.detach().float().mean().item():.3f})")
            print(f"[SmORFCNN.forward] xEmbeddings shape={tuple(xEmbeddings.shape)} "
                    f"min/max/mean=({xEmbeddings.detach().min().item():.3f}/{xEmbeddings.detach().max().item():.3f}/{xEmbeddings.detach().float().mean().item():.3f})")
            print(f"[SmORFCNN.forward] maskOnehot shape={tuple(maskOnehot.shape)} sum per-batch={maskOnehot.sum(dim=1).detach().cpu().tolist()[:8]}")

    def _debugOnehot(self, inputOneHot):
        if self.forwardDebugOnce and self.debugMode:
            print(f"[SmORFCNN.forward] onehot features shape={tuple(inputOneHot.shape)}")

    def _debugEmbeddings(self, inputEmbeddings):
        if self.forwardDebugOnce and self.debugMode:
            print(f"[SmORFCNN.forward] embeddings features shape={tuple(inputEmbeddings.shape)}")
    
    def _debugLogits(self, logits):
        if self.forwardDebugOnce and self.debugMode:
            print(f"[SmORFCNN.forward] logits shape={tuple(logits.shape)} "
                f"min/max/mean=({logits.detach().min().item():.3f}/{logits.detach().max().item():.3f}/{logits.detach().float().mean().item():.3f}) "
                f"sample={logits.detach().view(-1)[:8].cpu().tolist()}")
            self.forwardDebugOnce = False

    def _debugInEpoch(self, xOnehot, maskOnehot, xEmbed, maskEmbed, y, func, index, epochIndex):
        if index == 0 and epochIndex == 1 and self.debugMode:
            print(f"[{func} Epoch-{epochIndex}] batch0 shapes: xOnehot={tuple(xOnehot.shape)} maskOnehot={tuple(maskOnehot.shape)} "
                f"xEmbed={tuple(xEmbed.shape)} maskEmbed={tuple(maskEmbed.shape)} y={tuple(y.shape)} "
                f"y pos rate={(y.sum().item()/max(1,y.numel())):.3f}")
            print(f"[{func} Epoch-{epochIndex}] xEmbed stats: min={xEmbed.min().item():.3f} max={xEmbed.max().item():.3f} mean={xEmbed.float().mean().item():.3f} "
                f"len(T)={xEmbed.shape[-1]}")
    
    def _debugOutEpoch(self, outputs, func, index, epochIndex):
        if index == 0 and epochIndex == 1 and self.debugMode:
            print(f"[{func} Epoch-{epochIndex}] outputs shape={tuple(outputs.shape)} "
                f"min/max/mean=({outputs.detach().min().item():.3f}/{outputs.detach().max().item():.3f}/{outputs.detach().float().mean().item():.3f})")
            
    # def _debugStats(self, probs, loss, maxGradNorm, func, index, epochIndex):

    #     if index == 0 and epochIndex == 1 and self.debugMode:

    #         print(f"[{func} Epoch-{epochIndex}] total_grad_norm(before clip)={total_norm:.4f} "
    #             f"clipped={bool(total_norm > maxGradNorm)}")
    #         print(f"[{func} Epoch-{epochIndex}] classifier[0].weight grad |mean|="
    #             f"{self.classifier[0].weight.grad.abs().mean().item():.6f}")

    #         total_norm_sq = 0.0
    #         for p in self.parameters():
    #             if p.grad is not None:
    #                 gn = p.grad.data.norm(2).item()
    #                 total_norm_sq += gn*gn

    #         print(f"[{func} Epoch-{epochIndex}] grad_norm(L2)={total_norm_sq**0.5:.4f} loss={loss.item():.6f} "
    #                 f"probs sample={probs.detach().view(-1)[:8].cpu().tolist()}")
    
    def _debugFinal(self, probabilities, targets, runningLoss, n, func, epochIndex):
        if epochIndex == 0 and self.debugMode:
            print(f"[{func} Epoch{epochIndex}] epoch probs shape={tuple(probabilities.shape)} targets shape={tuple(targets.shape)} "
            f"loss_avg={runningLoss/max(1,n):.6f}")

mymodel = SmORFCNN(4,768,"features.pt",debug=False)
mymodel.initializeDataset()
mymodel.fit(10)