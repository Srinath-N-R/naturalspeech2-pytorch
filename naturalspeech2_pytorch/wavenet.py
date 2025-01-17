import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange

mlist = nn.ModuleList

def exists(x):
    return x is not None


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kernel_size, = self.kernel_size
        dilation, = self.dilation
        stride, = self.stride

        assert stride == 1
        self.causal_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        causal_padded_x = F.pad(x, (self.causal_padding, 0), value = 0.)
        return super().forward(causal_padded_x)


class WavenetResBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dilation,
        kernel_size = 3,
        skip_conv = False,
        dim_cond_mult = None
    ):
        super().__init__()

        self.cond = exists(dim_cond_mult)
        self.to_time_cond = None

        if self.cond:
            self.to_time_cond = nn.Linear(dim * dim_cond_mult, dim * 2)

        self.conv = CausalConv1d(dim, dim, kernel_size, dilation = dilation)
        self.res_conv = CausalConv1d(dim, dim, 1)
        self.skip_conv = CausalConv1d(dim, dim, 1) if skip_conv else None

    def forward(self, x, t = None):

        if self.cond:
            assert exists(t)
            t = self.to_time_cond(t)
            t = rearrange(t, 'b c -> b c 1')
            t_gamma, t_beta = t.chunk(2, dim = -2)

        res = self.res_conv(x)

        x = self.conv(x)

        if self.cond:
            x = x * t_gamma + t_beta

        x = x.tanh() * x.sigmoid()

        x = x + res

        skip = None
        if exists(self.skip_conv):
            skip = self.skip_conv(x)

        return x, skip


class WavenetStack(nn.Module):
    def __init__(
        self,
        dim,
        *,
        layers,
        kernel_size = 3,
        has_skip = False,
        dim_cond_mult = None
    ):
        super().__init__()
        dilations = 2 ** torch.arange(layers)

        self.has_skip = has_skip
        self.blocks = mlist([])

        for dilation in dilations.tolist():
            block = WavenetResBlock(
                dim = dim,
                kernel_size = kernel_size,
                dilation = dilation,
                skip_conv = has_skip,
                dim_cond_mult = dim_cond_mult
            )

            self.blocks.append(block)

    def forward(self, x, t):
        residuals = []
        skips = []

        if isinstance(x, Tensor):
            x = (x,) * len(self.blocks)

        for block_input, block in zip(x, self.blocks):
            residual, skip = block(block_input, t)

            residuals.append(residual)
            skips.append(skip)

        if self.has_skip:
            return torch.stack(skips)

        return residuals

class Wavenet(nn.Module):
    def __init__(
        self,
        dim,
        *,
        stacks,
        layers,
        init_conv_kernel = 3,
        dim_cond_mult = None
    ):
        super().__init__()
        self.init_conv = CausalConv1d(dim, dim, init_conv_kernel)
        self.stacks = mlist([])

        for ind in range(stacks):
            is_last = ind == (stacks - 1)

            stack = WavenetStack(
                dim,
                layers = layers,
                dim_cond_mult = dim_cond_mult,
                has_skip = is_last
            )

            self.stacks.append(stack)

        self.final_conv = CausalConv1d(dim, dim, 1)

    def forward(self, x, t = None):

        x = self.init_conv(x)

        for stack in self.stacks:
            x = stack(x, t)

        return self.final_conv(x.sum(dim = 0))