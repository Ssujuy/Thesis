import Types, Helpers
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionBlock(nn.Module):
    """
    Conv1d -> BatchNorm -> GELU -> (Dropout)
    - BatchNorm stabilizes training
    - GELU is a strong modern activation

    Where:
    1. kernel: number of adjacent positiona single filter looks at.
    2. stride: how far the kernel moves during each step.
    3. padding: ensures edge positions get full windows and layer shapes stay compatible.
    4. dilation: spacing between kernel taps.
    5. groups: splits the inputChannels in groups and applies independent filters.
    6. activation: activation function to use, configurable for testing.
    7. dropout: zeros activations to reduce co-adaptation and overfitting
    """

    def __init__(
            self,
            inputChannels: int,
            outputChannels: int,
            kernel: int,
            stride: int                 = Types.DEFAULT_CONVOLUTION_STRIDE,
            padding                     = Types.DEFAULT_CONVOLUTION_PADDING,
            dilation: int               = Types.DEFAULT_CONVOLUTION_DILATION,
            groups: int                 = Types.DEFAULT_CONVOLUTION_GROUPS,
            activation: str             = Types.DEFAULT_CONVOLUTION_ACTIVATION,
            dropout: float              = Types.DEFAULT_CONVOLUTION_DROPOUT
        ):
        super().__init__()

        if padding is None:
            # "Same" padding for odd kernels under given dilation.
            padding = (dilation * (kernel - 1)) // 2

        self.conv1d = nn.Conv1d(
            inputChannels,
            outputChannels,
            kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )

        self.batchNormalization = nn.BatchNorm1d(outputChannels)
        self.activation = Types.activationFunctionMapping.get(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(x)
        x = self.batchNormalization(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    
class ResidualBlock(nn.Module):
    """
    Defining 2 Convolution Block classes. This class computes a residual mapping
    y = x + F(x) introduced by ResNets. inputChannels length is equal to output channels length.
    """
    def __init__(
            self,
            inputChannels: int,
            kernel: int,
            stride: int                 = Types.DEFAULT_CONVOLUTION_STRIDE,
            padding                     = Types.DEFAULT_CONVOLUTION_PADDING,
            dilation: int               = Types.DEFAULT_CONVOLUTION_DILATION,
            groups: int                 = Types.DEFAULT_CONVOLUTION_GROUPS,
            activation: str             = Types.DEFAULT_CONVOLUTION_ACTIVATION,
            dropout: float              = Types.DEFAULT_CONVOLUTION_DROPOUT
        ):
        super().__init__()
        
        self.convolution1 = ConvolutionBlock(
            inputChannels,
            inputChannels,
            kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            activation=activation,
            dropout=dropout
        )

        self.convolution2 = ConvolutionBlock(
            inputChannels,
            inputChannels,
            kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            activation=activation,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.convolution2(self.convolution1(x))

class MultiKernelConvolution(nn.Module):
    """
    Parallel convs with different kernel widths (e.g., 3/7/11/15)
    Captures short motifs (codons, start/stop) and longer context.
    """
    def __init__(
            self,
            inputChannels: int,
            outputChannelsKernel: int   = Types.DEFAULT_MULTI_KERNEL_PER_KERNEL_OUTPUTCH,
            kernelList: list            = Types.DEFAULT_MULTI_KERNEL_KERNEL_LIST,
            stride: int                 = Types.DEFAULT_CONVOLUTION_STRIDE,
            padding                     = Types.DEFAULT_CONVOLUTION_PADDING,
            dilation: int               = Types.DEFAULT_CONVOLUTION_DILATION,
            groups: int                 = Types.DEFAULT_CONVOLUTION_GROUPS,
            activation: str             = Types.DEFAULT_CONVOLUTION_ACTIVATION,
            dropout: float              = Types.DEFAULT_CONVOLUTION_DROPOUT
        ):
        super().__init__()

        self.branches = nn.ModuleList()

        for kernel in kernelList:
            conv = ConvolutionBlock(
                inputChannels,
                outputChannelsKernel,
                kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                activation=activation,
                dropout=dropout
            )
            self.branches.append(conv)
        
        self.outputChannels = outputChannelsKernel * len(kernelList)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = []

        for branch in self.branches:
            output.append(branch(x))

        return torch.cat(output, dim=1)


class TemporalHead(nn.Module):
    """
    Reduce channels -> residual stack -> masked global pooling (max + avg).
    Assumes sequences are right-padded and `mask` marks valid positions: [B, L] with 1=valid, 0=pad.
    """
    def __init__(
        self,
        inputChannels,
        hiddenChannels,
        kernelResidual: int         = ,
        kernerReduction: int        = ,
        stride: int                 = Types.DEFAULT_CONVOLUTION_STRIDE,
        padding                     = Types.DEFAULT_CONVOLUTION_PADDING,
        groups: int                 = Types.DEFAULT_CONVOLUTION_GROUPS,
        activation: str             = Types.DEFAULT_CONVOLUTION_ACTIVATION,
        dropout: float              = 
        use_dilation: bool          = True,
        residualBlocks: int
        ):
        super().__init__()

        self.reduce = ConvolutionBlock(
            in_ch,
            hidden_ch,
            k=1,
            dropout=dropout,
            activation=act
        )

        blocks = []
        for i in range(num_blocks):
            dil = 2**i if use_dilation else 1
            blocks.append(ResidualBlock(hidden_ch, k=3, dilation=dil, dropout=dropout,
                                        act=act, act_kwargs=act_kwargs))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, C, L]   feature map
        mask: [B, L]      1 for valid tokens, 0 for right-padded tail (length==512 for all samples)
        returns: [B, 2*C] (global-max ⊕ masked-global-avg)
        """
        # refine features (length preserved)
        x = self.reduce(x)        # [B, C, L]
        x = self.blocks(x)        # [B, C, L]

        # broadcast mask over channels
        m = mask[:, None, :]                          # [B, 1, L]
        m_float = m.to(dtype=x.dtype)

        # MASKED GLOBAL MAX: padded positions -> -inf so they can’t win the max
        x_neg_inf = x.masked_fill(m == 0, float("-inf"))
        gmax = x_neg_inf.max(dim=-1).values          # [B, C]
        # guard against fully-masked rows (shouldn't occur with right-padding)
        gmax = torch.where(torch.isfinite(gmax), gmax, torch.zeros_like(gmax))

        # MASKED GLOBAL AVG: zero-out padded, divide by count of valid steps
        x_zero = x * m_float                          # zero padded positions
        denom = m_float.sum(dim=-1).clamp(min=1.0)    # [B, 1]
        gavg = x_zero.sum(dim=-1) / denom             # [B, C]

        return torch.cat([gmax, gavg], dim=1)         # [B, 2*C]

    

# ---------------------------------------------
# The CNNClassifier for smORFs
# ---------------------------------------------

class SmORFCNN(nn.Module):
    """
    Two-branch CNN for coding vs non-coding prediction.

    Inputs
    ------
    x_onehot: [B,4,L]            # optional one-hot DNA
    x_embed:  [B,E,T]            # optional DNABERT6 embeddings (channels-first)
    mask_onehot: [B,L] optional  # 1=valid
    mask_embed:  [B,T] optional  # 1=valid

    Design notes (theory links):
      - Local convs + weight sharing (translation equivariance) [T2]
      - Multi-scale kernels capture motifs & context [T1]
      - 1x1 conv to squeeze large E (NIN) [T3]
      - Dilations increase receptive field efficiently [T4]
      - Residual skips ease optimization [T5]
      - Global pooling yields fixed-size representation [T3]
    """
    def __init__(
        self,
        use_onehot: bool = True,
        use_embed: bool = True,
        in_ch_onehot: int = 4,
        embed_dim: int = 768,
        ms_kernels_onehot: Sequence[int] = (3, 7, 11, 15),
        ms_kernels_embed: Sequence[int] = (3, 7, 11),
        branch_ch: int = 64,
        head_hidden: int = 128,
        head_blocks: int = 2,
        dropout: float = 0.2,
        num_classes: int = 1
    ):
        super().__init__()
        assert use_onehot or use_embed, "Enable at least one input branch."
        self.use_onehot = use_onehot
        self.use_embed = use_embed

        if use_onehot:
            self.stem_onehot = MultiScaleStem(in_ch_onehot, out_ch_per_branch=branch_ch,
                                              kernels=ms_kernels_onehot, dropout=dropout)
            self.head_onehot = TemporalHead(self.stem_onehot.out_ch, hidden_ch=head_hidden,
                                            num_blocks=head_blocks, dropout=dropout)

        if use_embed:
            squeeze_ch = min(256, embed_dim)  # 1x1 conv squeeze [T3]
            self.squeeze_embed = ConvBNAct(embed_dim, squeeze_ch, k=1, dropout=dropout)
            self.stem_embed = MultiScaleStem(squeeze_ch, out_ch_per_branch=branch_ch,
                                             kernels=ms_kernels_embed, dropout=dropout)
            self.head_embed = TemporalHead(self.stem_embed.out_ch, hidden_ch=head_hidden,
                                           num_blocks=head_blocks, dropout=dropout)

        fused_dim = (2 * head_hidden) * (int(use_onehot) + int(use_embed))
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self,
                x_onehot: Optional[torch.Tensor] = None,
                x_embed: Optional[torch.Tensor] = None,
                mask_onehot: Optional[torch.Tensor] = None,
                mask_embed: Optional[torch.Tensor] = None,
                rc_augment: bool = False) -> torch.Tensor:

        def forward_once(x1, x2, m1, m2):
            feats = []
            if self.use_onehot:
                h = self.stem_onehot(x1)                    # [B,C,L]
                f = self.head_onehot(h, m1)                 # [B,2H]
                feats.append(f)
            if self.use_embed:
                z = self.squeeze_embed(x2)                  # [B,C,T]
                z = self.stem_embed(z)                      # [B,C',T]
                g = self.head_embed(z, m2)                  # [B,2H]
                feats.append(g)
            fused = feats[0] if len(feats) == 1 else torch.cat(feats, dim=1)
            logits = self.classifier(fused).squeeze(-1)     # [B] if num_classes=1
            return logits

        # Reverse-complement test-time augmentation on one-hot branch only.
        if rc_augment and (x_onehot is not None):
            logits_fwd = forward_once(x_onehot, x_embed, mask_onehot, mask_embed)
            x_rc = onehot_reverse_complement(x_onehot)
            logits_rc = forward_once(x_rc, x_embed, mask_onehot, mask_embed)
            return 0.5 * (logits_fwd + logits_rc)
        else:
            return forward_once(x_onehot, x_embed, mask_onehot, mask_embed)
        
