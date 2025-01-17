import math
import json

from math import expm1
from pathlib import Path
from functools import partial

import logging
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import torchaudio
from torch.cuda.amp import autocast

from naturalspeech2_pytorch.attend import Attend
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from audiolm_pytorch import SoundStream, EncodecWrapper

from beartype import beartype
from beartype.typing import Tuple, Union, Optional
from beartype.door import is_bearable


from naturalspeech2_pytorch.utils.tokenizer import Tokenizer, ESpeak
from naturalspeech2_pytorch.version import __version__

from naturalspeech2_pytorch.wavenet import Wavenet

from speechbrain.lobes.models.FastSpeech2 import EncoderPreNet

from speechbrain.lobes.models.transformer.Transformer import (
    PositionalEncoding,
    TransformerEncoder,
    get_key_padding_mask,
) 
from speechbrain.nnet import CNN, linear
from speechbrain.nnet.normalization import LayerNorm

from flamingo_pytorch import PerceiverResampler

from functools import wraps
from tqdm.auto import tqdm


# constants

mlist = nn.ModuleList

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Direct logs to the console
    ]
)

logger = logging.getLogger(__name__)

def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def divisible_by(num, den):
    return (num % den) == 0

def identity(t, *args, **kwargs):
    return t

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


# tensor helpers

def pad_or_curtail_to_length(t, length):
    if t.shape[-1] == length:
        return t

    if t.shape[-1] > length:
        return t[..., :length]

    return F.pad(t, (0, length - t.shape[-1]))

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def generate_mask_from_repeats(repeats):
    repeats = repeats.int()
    device = repeats.device

    lengths = repeats.sum(dim = -1)
    max_length = lengths.amax().item()
    cumsum = repeats.cumsum(dim = -1)
    cumsum_exclusive = F.pad(cumsum, (1, -1), value = 0.)

    seq = torch.arange(max_length, device = device)
    seq = repeat(seq, '... j -> ... i j', i = repeats.shape[-1])

    cumsum = rearrange(cumsum, '... i -> ... i 1')
    cumsum_exclusive = rearrange(cumsum_exclusive, '... i -> ... i 1')

    lengths = rearrange(lengths, 'b -> b 1 1')
    mask = (seq < cumsum) & (seq >= cumsum_exclusive) & (seq < lengths)
    return mask


def custom_collate_fn(batch):
    try:
        # Ensure consistent dimension for audio and prompt
        audio = [item['audio'].squeeze(0) for item in batch]
        prompt = [item['prompt'].squeeze(0) for item in batch]

        # text = [item['text'] for item in batch]
        # tokenizer = Tokenizer()  # Replace with your tokenizer instance
        # text = tokenizer.texts_to_tensor_ids([item['text'] for item in batch])
        phoneme = [item['phoneme'] for item in batch]
        speaker_embeddings = torch.stack([item['speaker_embeddings'] for item in batch])
        context_embeddings = torch.stack([item['context_embeddings'] for item in batch])
        
        # Pad audio and prompt to the same length
        audio = pad_sequence(audio, batch_first=True)
        prompt = pad_sequence(prompt, batch_first=True)
        segment = [item['segment'] for item in batch]

        return {
            'audio': audio,
            'segment': segment,
            'prompt': prompt,
            'phoneme': phoneme,
            'speaker_embeddings': speaker_embeddings,
            'context_embeddings': context_embeddings,
        }
    except Exception as e:
        logger.error(f"Error in custom_collate_fn: {e}")
        raise


def f0_to_coarse(f0, f0_bin = 256, f0_max = 1100.0, f0_min = 50.0):
    f0_mel_max = 1127 * torch.log(1 + torch.tensor(f0_max) / 700)
    f0_mel_min = 1127 * torch.log(1 + torch.tensor(f0_min) / 700)

    f0_mel = 1127 * (1 + f0 / 700).log()
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).int()
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse

# peripheral models

class PhonemeEncoder(nn.Module):
    def __init__(
        self,
        *,
        tokenizer,
        padding_value=0,
        enc_d_model=512,
        enc_num_layers=6,
        enc_num_head=2,
        enc_ffn_dim=1536,
        enc_k_dim=512,
        enc_v_dim=512,
        enc_dropout=0.1,
        normalize_before=False,
        ffn_type='1dcnn',
        ffn_cnn_kernel_size_list=[9, 1],
    ):
        super().__init__()

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = tokenizer
        self.padding_idx = padding_value
        n_char = len(tokenizer.vocab)
        self.enc_num_head = enc_num_head

        self.encPreNet = EncoderPreNet(
            n_char, padding_value, out_channels=enc_d_model
        )
        self.encoder = TransformerEncoder(
            num_layers=enc_num_layers,
            nhead=enc_num_head,
            d_ffn=enc_ffn_dim,
            d_model=enc_d_model,
            kdim=enc_k_dim,
            vdim=enc_v_dim,
            dropout=enc_dropout,
            activation=nn.ReLU,
            normalize_before=normalize_before,
            ffn_type=ffn_type,
            ffn_cnn_kernel_size_list=ffn_cnn_kernel_size_list,
        )

        self.sinusoidal_positional_embed_encoder = PositionalEncoding(
            enc_d_model
        )


    def forward(
        self,
        x: str,
    ):
        x = self.tokenizer.phoneme_to_tensor_ids(x, self.padding_idx)
        x = x.to(device=self._device)
        srcmask = get_key_padding_mask(x, pad_idx=self.padding_idx)
        srcmask_inverted = (~srcmask).unsqueeze(-1)

        # prenet & encoder
        x = self.encPreNet(x)
        pos = self.sinusoidal_positional_embed_encoder(x)
        x = torch.add(x, pos) * srcmask_inverted
        attn_mask = (
            srcmask.unsqueeze(-1)
            .repeat(self.enc_num_head, 1, x.shape[1])
            .permute(0, 2, 1)
            .bool()
        )
        x, _ = self.encoder(
            x, src_mask=attn_mask, src_key_padding_mask=srcmask
        )
        x = x * srcmask_inverted
        return x, srcmask_inverted


class RMSNorm(nn.Module):
    def __init__(self, dim, scale = True, dim_cond = None):
        super().__init__()
        self.cond = exists(dim_cond)
        self.to_gamma_beta = nn.Linear(dim_cond, dim * 2) if self.cond else None

        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x, cond = None):
        gamma = default(self.gamma, 1)
        out = F.normalize(x, dim = -1) * self.scale * gamma

        if not self.cond:
            return out

        assert exists(cond)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim = -1)
        gamma, beta = map(lambda t: rearrange(t, 'b d -> b 1 d'), (gamma, beta))
        return out * gamma + beta


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


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4, causal_conv = False):
    dim_inner = int(dim * mult * 2 / 3)

    conv = None
    if causal_conv:
        conv = nn.Sequential(
            Rearrange('b n d -> b d n'),
            CausalConv1d(dim_inner, dim_inner, 3),
            Rearrange('b d n -> b n d'),
        )

    return Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        conv,
        nn.Linear(dim_inner, dim)
    )


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_context = None,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        use_flash = False,
        cross_attn_include_queries = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries

        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)

        self.attend = Attend(causal = causal, dropout = dropout, use_flash = use_flash)
        self.to_q = nn.Linear(dim, dim_inner, bias = False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(self, x, context = None, mask = None):
        h, has_context = self.heads, exists(context)

        context = default(context, x)

        if has_context and self.cross_attn_include_queries:
            context = torch.cat((x, context), dim = -2)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        causal=False,
        dim_head=64,
        heads=8,
        use_flash=False,  # Flash Attention can be implemented using libraries like `xformers`.
        dropout=0.,
        ff_mult=4,
        final_norm=False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        # RMSNorm is a popular normalization introduced in the transformer community,
        # implemented in `xformers` and other libraries.
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                RMSNorm(dim),  # Libraries like `einops` can help simplify tensor manipulations here.
                Attention(
                    dim,
                    causal=causal,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=dropout,
                    use_flash=use_flash
                ),
                RMSNorm(dim),
                FeedForward(
                    dim,
                    mult=ff_mult
                )
            ]))

        # Normalization layers like RMSNorm are essential for stable training of transformers.
        self.norm = RMSNorm(dim) if final_norm else nn.Identity()

    def forward(self, x, mask=None):
        for attn_norm, attn, ff_norm, ff in self.layers:
            x = attn(attn_norm(x), mask=mask) + x
            x = ff(ff_norm(x)) + x

        return self.norm(x)


class SpeechPromptEncoder(nn.Module):
    @beartype
    def __init__(
        self,
        dim_codebook,
        dims: Tuple[int] = (256, 2048, 2048, 2048, 2048, 512, 512, 512),
        *,
        depth = 6,
        heads = 8,
        dim_head = 64,
        dropout = 0.2,
        kernel_size = 9,
        padding = 4,
        ff_mult=4,
        use_flash_attn = True

    ):
        super().__init__()

        dims = [dim_codebook, *dims]

        self.dim, self.dim_out = dims[0], dims[-1]

        dim_pairs = zip(dims[:-1], dims[1:])

        modules = []
        for dim_in, dim_out in dim_pairs:
            modules.extend([
                nn.Conv1d(dim_in, dim_out, kernel_size, padding = padding),
                nn.SiLU()
            ])

        self.conv = nn.Sequential(
            Rearrange('b n c -> b c n'),
            *modules,
            Rearrange('b c n -> b n c')
        )

        # Model
        self.transformer = Transformer(
            dim = dims[-1],
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            dropout = dropout,
            use_flash = use_flash_attn
        )


    def forward(self, x):
        assert x.shape[-1] == self.dim
        x = self.conv(x)
        x = self.transformer(x)
        return x

# duration and pitch predictor seems to be the same

class DurationPredictor(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dropout=0.0, n_units=1
    ):
        super().__init__()
        self.conv1 = CNN.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.conv2 = CNN.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.linear = linear.Linear(n_neurons=n_units, input_size=out_channels)
        self.ln1 = LayerNorm(out_channels)
        self.ln2 = LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=8, batch_first=True)

    def forward(self, x, x_mask, x_):
        x = self.relu(self.conv1(x * x_mask))
        x = self.ln1(x).to(x.dtype)
        x = self.dropout1(x)

        x = self.relu(self.conv2(x * x_mask))
        x = self.ln2(x).to(x.dtype)
        x = self.dropout2(x)

        x_ = self.relu(self.conv1(x_))
        x_ = self.ln1(x_).to(x.dtype)
        x_ = self.dropout1(x_)

        x_ = self.relu(self.conv2(x_))
        x_ = self.ln2(x_)
        x_ = self.dropout2(x_)
        
        attn_output, _ = self.attention(x, x_, x_) 
        combined = x + attn_output

        return self.linear(combined)


class DurationPitchPred(nn.Module):
    def __init__(
        self,
        enc_d_model=512,
        dur_pred_kernel_size=3,
        variance_predictor_dropout=0.5,
    ):
        super().__init__()
        self.durPred = DurationPredictor(
            in_channels=enc_d_model,
            out_channels=enc_d_model,
            kernel_size=dur_pred_kernel_size,
            dropout=variance_predictor_dropout,
        )
        self.pitchPred = DurationPredictor(
            in_channels=enc_d_model,
            out_channels=enc_d_model,
            kernel_size=dur_pred_kernel_size,
            dropout=variance_predictor_dropout,
        )
        

    def forward(
        self,
        phoneme_enc,
        mask,
        prompt_enc
    ):
        with autocast():
            # duration predictor
            predict_durations = self.durPred(phoneme_enc, mask, prompt_enc).squeeze(
                -1
            )
            predict_pitch = self.pitchPred(phoneme_enc, mask, prompt_enc)

        return predict_durations, predict_pitch



class ConditionableTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        ff_causal_conv = False,
        dim_cond_mult = None,
        cross_attn = False,
        use_flash = False
    ):
        super().__init__()
        self.dim = dim
        self.layers = mlist([])

        cond = exists(dim_cond_mult)

        maybe_adaptive_norm_kwargs = dict(scale = not cond, dim_cond = dim * dim_cond_mult) if cond else dict()
        rmsnorm = partial(RMSNorm, **maybe_adaptive_norm_kwargs)

        for _ in range(depth):
            self.layers.append(mlist([
                rmsnorm(dim),
                Attention(dim = dim, dim_head = dim_head, heads = heads, use_flash = use_flash),
                rmsnorm(dim) if cross_attn else None,
                Attention(dim = dim, dim_head = dim_head, heads = heads, use_flash = use_flash) if cross_attn else None,
                rmsnorm(dim),
                FeedForward(dim = dim, mult = ff_mult, causal_conv = ff_causal_conv)
            ]))

        self.to_pred = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim, bias = False)
        )

    def forward(
        self,
        x,
        times = None,
        context = None
    ):
        t = times

        for attn_norm, attn, cross_attn_norm, cross_attn, ff_norm, ff in self.layers:
            res = x
            x = attn_norm(x, cond = t)
            x = attn(x) + res

            if exists(cross_attn):
                assert exists(context)
                res = x
                x = cross_attn_norm(x, cond = t)
                x = cross_attn(x, context = context) + res

            res = x
            x = ff_norm(x, cond = t)
            x = ff(x) + res

        return self.to_pred(x)


class Model(nn.Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        num_media_embeds=4,
        wavenet_layers = 8,
        wavenet_stacks = 4,
        dim_cond_mult = 4,
        dim_prompt = None,
        num_latents_m = 32,   # number of latents to be perceiver resampled ('q-k-v' with 'm' queries in the paper)
        resampler_depth = 2,
        cond_drop_prob = 0.,
        condition_on_prompt= False,
        use_flash_attn=True
    ):
        super().__init__()
        self.dim = dim

        # time condition

        dim_time = dim * dim_cond_mult

        self.to_time_cond = Sequential(
            PositionalEncoding(dim),
            nn.Linear(dim, dim_time),
            nn.SiLU()
        )

        # prompt condition

        self.cond_drop_prob = cond_drop_prob # for classifier free guidance
        self.condition_on_prompt = condition_on_prompt
        self.to_prompt_cond = None

        if self.condition_on_prompt:
            self.null_prompt_cond = nn.Parameter(torch.randn(dim_time))
            self.null_prompt_tokens = nn.Parameter(torch.randn(num_latents_m, dim))

            nn.init.normal_(self.null_prompt_cond, std = 0.02)
            nn.init.normal_(self.null_prompt_tokens, std = 0.02)

            self.to_prompt_cond = Sequential(
                Reduce('b n d -> b d', 'mean'),
                nn.Linear(dim_prompt, dim_time),
                nn.SiLU()
            )

            self.perceiver_resampler = PerceiverResampler(
                dim=dim,
                depth=resampler_depth,
                num_latents=num_latents_m,
                num_media_embeds=num_media_embeds,
                dim_head=dim_head,
                heads=heads
            )

            self.proj_context = nn.Linear(dim_prompt, dim) if dim_prompt != dim else nn.Identity()

        # aligned conditioning from aligner + duration module

        self.null_cond = None
        self.cond_to_model_dim = None

        if self.condition_on_prompt:
            self.cond_to_model_dim = nn.Conv1d(dim_prompt, dim, 1)
            self.null_cond = nn.Parameter(torch.zeros(dim, 1))

        # conditioning includes time and optionally prompt

        dim_cond_mult = dim_cond_mult * (2 if condition_on_prompt else 1)

        # wavenet

        self.wavenet = Wavenet(
            dim = dim,
            stacks = wavenet_stacks,
            layers = wavenet_layers,
            dim_cond_mult = dim_cond_mult
        )

        self.transformer = ConditionableTransformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            ff_mult = ff_mult,
            ff_causal_conv = True,
            dim_cond_mult = dim_cond_mult,
            use_flash = use_flash_attn,
            cross_attn = condition_on_prompt
        )


    @property
    def device(self):
        return next(self.parameters()).device

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1.:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)

        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        times,
        prompt = None,
        cond = None,
        cond_drop_prob = None
    ):
        b = x.shape[0]
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # prepare prompt condition
        # prob should remove going forward
        if times.dim() == 1:
            times = times.unsqueeze(-1)
        t = self.to_time_cond(times)
        c = None

        if exists(self.to_prompt_cond):
            assert exists(prompt)

            prompt_cond_drop_mask = prob_mask_like((b,), cond_drop_prob, self.device)

            prompt_cond = self.to_prompt_cond(prompt)

            prompt_cond = torch.where(
                rearrange(prompt_cond_drop_mask, 'b -> b 1'),
                self.null_prompt_cond,
                prompt_cond,
            )

            t_repeated = t.squeeze(0).repeat(prompt_cond.shape[0], 1)  # Repeat along the batch dimension
            t = torch.cat((t_repeated, prompt_cond), dim=-1)

            prompt = self.proj_context(prompt)
            resampled_prompt_tokens = self.perceiver_resampler(prompt).squeeze(1)

            c = torch.where(
                rearrange(prompt_cond_drop_mask, 'b -> b 1 1'),
                self.null_prompt_tokens,
                resampled_prompt_tokens
            )

        # rearrange to channel first
        x = rearrange(x, 'b n d -> b d n')
    
        # sum aligned condition to input sequence
        if exists(self.cond_to_model_dim):
            assert exists(cond)
            cond = self.cond_to_model_dim(cond)

            cond_drop_mask = prob_mask_like((b,), cond_drop_prob, self.device)

            cond = torch.where(
                rearrange(cond_drop_mask, 'b -> b 1 1'),
                self.null_cond,
                cond
            )

            x = x + cond

        # main wavenet body

        x = self.wavenet(x, t)
        x = rearrange(x, 'b d n -> b n d')
        x = self.transformer(x, t, context = c)

        return x

# feedforward


def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def safe_div(numer, denom):
    return numer / denom.clamp(min = 1e-10)

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# noise schedules

def simple_linear_schedule(t, clip_min = 1e-9):
    return (1 - t).clamp(min = clip_min)

def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)

# converting gamma to alpha, sigma or logsnr

def gamma_to_alpha_sigma(gamma, scale = 1):
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)

def gamma_to_log_snr(gamma, scale = 1, eps = 1e-5):
    return log(gamma * (scale ** 2) / (1 - gamma), eps = eps)

# gaussian diffusion

class NaturalSpeech2(nn.Module):

    @beartype
    def __init__(
        self,
        model: Model,
        codec: Optional[Union[SoundStream, EncodecWrapper]] = None,
        *,
        tokenizer: Optional[Tokenizer] = None,
        speaker_embedding_dim=192,
        context_embedding_dim=768,
        target_sample_hz = None,
        timesteps = 1000,
        use_ddim = True,
        noise_schedule = 'sigmoid',
        objective = 'v',
        schedule_kwargs: dict = dict(),
        time_difference = 0.,
        min_snr_loss_weight = True,
        min_snr_gamma = 5,
        rvq_cross_entropy_loss_weight = 0., # default this to off until we are sure it is working. not totally sold that this is critical
        dim_codebook: int = 128,
        duration_pitch_dim: int = 512,
        pitch_emb_dim: int = 256,
        pitch_emb_pp_hidden_dim: int= 512,
        mel_hop_length = 160,
        audio_to_mel_kwargs: dict = dict(),
        scale = 1., # this will be set to < 1. for better convergence when training on higher resolution images
        duration_loss_weight = 0.1,
        pitch_loss_weight = 0.0005
    ):
        super().__init__()

        self.conditional = model.condition_on_prompt
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model and codec

        self.model = model.to(self._device)
        self.codec = codec

        assert exists(codec) or exists(target_sample_hz)

        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = None

        if exists(codec):
            self.target_sample_hz = codec.target_sample_hz
            self.seq_len_multiple_of = codec.seq_len_multiple_of

        # preparation for conditioning

        if self.conditional:
            if exists(self.target_sample_hz):
                audio_to_mel_kwargs.update(sampling_rate = self.target_sample_hz)

            self.mel_hop_length = mel_hop_length

            self.phoneme_enc = PhonemeEncoder(tokenizer=tokenizer)
            self.prompt_enc = SpeechPromptEncoder(dim_codebook=dim_codebook)
            self.duration_pitch = DurationPitchPred()
            self.pitch_emb = nn.Embedding(pitch_emb_dim, pitch_emb_pp_hidden_dim)

        # rest of ddpm

        assert not exists(codec) or model.dim == codec.codebook_dim, f'transformer model dimension {model.dim} must be equal to codec dimension {codec.codebook_dim}'

        self.dim = codec.codebook_dim if exists(codec) else model.dim

        assert objective in {'x0', 'eps', 'v'}, 'objective must be either predict x0 or noise'
        self.objective = objective

        if noise_schedule == "linear":
            self.gamma_schedule = simple_linear_schedule
        elif noise_schedule == "cosine":
            self.gamma_schedule = cosine_schedule
        elif noise_schedule == "sigmoid":
            self.gamma_schedule = sigmoid_schedule
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        # the main finding presented in Ting Chen's paper - that higher resolution images requires more noise for better training

        assert scale <= 1, 'scale must be less than or equal to 1'
        self.scale = scale

        # gamma schedules

        self.gamma_schedule = partial(self.gamma_schedule, **schedule_kwargs)

        self.timesteps = timesteps
        self.use_ddim = use_ddim

        # proposed in the paper, summed to time_next
        # as a way to fix a deficiency in self-conditioning and lower FID when the number of sampling timesteps is < 400

        self.time_difference = time_difference

        # min snr loss weight

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

        # weight of the cross entropy loss to residual vq codebooks

        self.rvq_cross_entropy_loss_weight = rvq_cross_entropy_loss_weight

        # loss weight for duration and pitch

        self.duration_loss_weight = duration_loss_weight
        self.pitch_loss_weight = pitch_loss_weight

        # Speaker and Context Embeddings
        self.speaker_encoder = nn.Linear(speaker_embedding_dim, duration_pitch_dim)
        self.context_encoder = nn.Linear(context_embedding_dim, duration_pitch_dim)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def print(self, s):
        return self.accelerator.print(s)

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    @torch.no_grad()
    def ddpm_sample(self, shape, prompt = None, time_difference = None, cond_scale = 1., cond = None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        audio = torch.randn(shape, device=device)

        x_start = None
        last_latents = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.timesteps):

            # add the time delay

            time_next = (time_next - self.time_difference).clamp(min = 0.)

            noise_cond = time

            # get predicted x0

            model_output = self.model.forward_with_cond_scale(audio, noise_cond, prompt = prompt, cond_scale = cond_scale, cond = cond)

            # get log(snr)

            gamma = self.gamma_schedule(time)
            gamma_next = self.gamma_schedule(time_next)
            gamma, gamma_next = map(partial(right_pad_dims_to, audio), (gamma, gamma_next))

            # get alpha sigma of time and next time

            alpha, sigma = gamma_to_alpha_sigma(gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(gamma_next, self.scale)

            # calculate x0 and noise

            if self.objective == 'x0':
                x_start = model_output

            elif self.objective == 'eps':
                x_start = safe_div(audio - sigma * model_output, alpha)

            elif self.objective == 'v':
                x_start = alpha * audio - sigma * model_output

            # derive posterior mean and variance

            log_snr, log_snr_next = map(gamma_to_log_snr, (gamma, gamma_next))

            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (audio * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # get noise

            noise = torch.where(
                rearrange(time_next > 0, 'b -> b 1 1 1'),
                torch.randn_like(audio),
                torch.zeros_like(audio)
            )

            audio = mean + (0.5 * log_variance).exp() * noise

        return audio

    @torch.no_grad()
    def ddim_sample(self, shape, prompt = None, time_difference = None, cond_scale = 1., cond = None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        audio = torch.randn(shape, device = device)

        x_start = None
        last_latents = None

        for times, times_next in tqdm(time_pairs, desc = 'sampling loop time step'):

            # get times and noise levels

            gamma = self.gamma_schedule(times)
            gamma_next = self.gamma_schedule(times_next)

            padded_gamma, padded_gamma_next = map(partial(right_pad_dims_to, audio), (gamma, gamma_next))

            alpha, sigma = gamma_to_alpha_sigma(padded_gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(padded_gamma_next, self.scale)

            # add the time delay

            times_next = (times_next - time_difference).clamp(min = 0.)

            # predict x0

            model_output = self.model.forward_with_cond_scale(audio, times, prompt = prompt, cond_scale = cond_scale, cond = cond)

            # calculate x0 and noise

            if self.objective == 'x0':
                x_start = model_output

            elif self.objective == 'eps':
                x_start = safe_div(audio - sigma * model_output, alpha)

            elif self.objective == 'v':
                x_start = alpha * audio - sigma * model_output

            # get predicted noise

            pred_noise = safe_div(audio - alpha * x_start, sigma)

            # calculate x next

            audio = x_start * alpha_next + pred_noise * sigma_next

        return audio

    def process_prompt(self, prompt = None):
        if not exists(prompt):
            return None

        assert self.model.condition_on_prompt

        is_raw_prompt = prompt.ndim == 2
        assert not (is_raw_prompt and not exists(self.codec)), 'codec must be passed in if one were to train on raw prompt'

        if is_raw_prompt:
            with torch.no_grad():
                self.codec.eval()
                prompt, _, _ = self.codec(prompt, curtail_from_left = True, return_encoded = True)

        return prompt


    def expand_encodings(self, phoneme_enc, attn, pitch, speaker_emb, context_emb):
        # Apply attention-based expansion
        expanded_dur = einsum('k l m n, k j m -> k j n', attn, phoneme_enc)
        pitch_emb = self.pitch_emb(rearrange(f0_to_coarse(pitch), 'b 1 t -> b t'))
        pitch_emb = rearrange(pitch_emb, 'b t d -> b d t')
        target_dtype = pitch_emb.dtype  # Use pitch_emb's dtype as the target dtype

        attn = attn.to(target_dtype) 
        expanded_pitch = einsum('k l m n, k j m -> k j n', attn, pitch_emb)

        # Add the necessary dimension for broadcasting
        speaker_emb = rearrange(speaker_emb, 'b 1 d -> b d 1')
        context_emb = rearrange(context_emb, 'b 1 d -> b d 1')

        # Combine expanded components
        try:
            expanded_combined = torch.cat((expanded_dur, expanded_pitch, speaker_emb, context_emb), dim=-1)
        except RuntimeError as e:
            raise RuntimeError(
                f"Shape mismatch during addition: expanded_dur={expanded_dur.shape}, "
                f"expanded_pitch={expanded_pitch.shape}, context_emb={context_emb.shape}"
                f"speaker_emb={speaker_emb.shape}"
            ) from e

        return expanded_combined



    @torch.no_grad()
    def sample(
        self,
        *,
        length,
        prompt=None,
        batch_size=1,
        cond_scale=1.0,
        phoneme=None,
        speaker_embeddings=None,
        context_embeddings=None
    ):
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample

        prompt_enc = cond = None

        if self.conditional:
            assert exists(prompt) and exists(phoneme)
            assert exists(speaker_embeddings) and exists(context_embeddings)

            prompt = self.process_prompt(prompt)
            prompt_enc = self.prompt_enc(prompt)
            phoneme_enc, mask = self.phoneme_enc(phoneme)

            duration, pitch = self.duration_pitch(phoneme_enc, mask, prompt_enc)
            
            pitch = rearrange(pitch, 'b n -> b 1 n')

            aln_mask = generate_mask_from_repeats(duration).float()

            speaker_emb = self.speaker_encoder(speaker_embeddings)
            context_emb = self.context_encoder(context_embeddings)

            cond = self.expand_encodings(
                rearrange(phoneme_enc, 'b n d -> b d n'),
                rearrange(aln_mask, 'b n c -> b 1 n c'),
                pitch,
                speaker_emb=speaker_emb,
                context_emb=context_emb

            )

        audio = sample_fn(
            (batch_size, length, self.dim),
            prompt=prompt_enc,
            cond=cond,
            cond_scale=cond_scale
        )

        if exists(self.codec):
            audio = self.codec.decode(audio)
            if audio.ndim == 3:
                audio = rearrange(audio, 'b 1 n -> b n')

        return audio


    def forward(
        self,
        audio,
        phoneme=None,
        segment=None,
        prompt=None,
        pitch=None,
        speaker_embeddings=None,
        context_embeddings=None,
        return_loss=True,
        *args,
        **kwargs
    ):
        batch, is_raw_audio = audio.shape[0], audio.ndim == 2
        with autocast():
            # Process speaker and context embeddings
            assert exists(speaker_embeddings), "Speaker embeddings must be provided."
            assert exists(context_embeddings), "Context embeddings must be provided."
            audio = audio.to(self._device)

            if prompt is not None:
                prompt = prompt.to(self._device)

            context_embeddings = context_embeddings.unsqueeze(1).to(self._device)
            speaker_emb = self.speaker_encoder(speaker_embeddings).to(self._device)
            context_emb = self.context_encoder(context_embeddings).to(self._device)

            # Initialize losses
            aux_loss = 0.0
            duration_loss = 0.0
            pitch_loss = 0.0

            # Conditional processing
            if self.conditional:
                assert exists(prompt), "Prompt must be provided for conditional processing."
                assert exists(phoneme), "Text must be provided for conditional processing."

                # Process prompt
                prompt = self.process_prompt(prompt)
                prompt_enc = self.prompt_enc(prompt)
                phoneme_enc, mask = self.phoneme_enc(phoneme)

                duration_pred, pitch_pred = self.duration_pitch(phoneme_enc, mask, prompt_enc)

                # Initialize tensors for pitch and alignment
                pitch = torch.zeros_like(pitch_pred)
                aln_hard = torch.zeros_like(duration_pred)

                # Iterate over each batch
                for batch_idx, batch_segments in enumerate(segment):
                    phoneme_ = phoneme[batch_idx].split()

                    # Iterate over batch segments and process them
                    for segment_idx, segment_details in batch_segments.items():
                        try:
                            # Extract relevant details
                            label = segment_details[0]
                            duration = segment_details[1].get('duration_sec', 0)
                            mean_pitch = segment_details[1].get('mean_pitch', 0)
                            
                            # Ensure segment index is within bounds and phonemes align
                            segment_idx = int(segment_idx)  # Ensure segment_idx is an integer
                            if segment_idx < len(phoneme_) and phoneme_[segment_idx] == label:
                                aln_hard[batch_idx, segment_idx] = torch.tensor(duration, dtype=aln_hard.dtype)
                                pitch[batch_idx, segment_idx] = mean_pitch
                            else:
                                # Handle cases where the phoneme doesn't match
                                print(f"Phoneme mismatch at segment {segment_idx} in batch {batch_idx}: {label} vs {phoneme_[segment_idx]}")
                        except Exception as e:
                            print(f"Error processing segment {segment_idx} in batch {batch_idx}: {e}")
                            continue

                aln_hard = aln_hard * 100
                
                aln_mask = generate_mask_from_repeats(aln_hard)
                aln_mask = aln_mask.to(torch.float)

                # Calculate losses
                duration_loss = F.l1_loss(aln_hard, duration_pred)

                pitch_loss = F.l1_loss(pitch, pitch_pred)

                # Combine auxiliary losses
                overall_duration_loss = duration_loss * self.duration_loss_weight
                overall_pitch_loss = (pitch_loss * self.pitch_loss_weight)
                aux_loss = overall_duration_loss + overall_pitch_loss


            # Handle raw audio encoding with codec
            if is_raw_audio:
                assert exists(self.codec), "Codec must be provided for raw audio input."
                with torch.no_grad():
                    self.codec.eval()
                    audio, codes, _ = self.codec(audio, return_encoded=True)

            pitch = rearrange(pitch, 'b n 1 -> b 1 n')
            # pitch_pred = rearrange(pitch_pred, 'b n 1 -> b 1 n')

            cond =  self.expand_encodings(
                rearrange(phoneme_enc, 'b n d -> b d n'),
                rearrange(aln_mask, 'b n c -> b 1 n c'),
                pitch,
                speaker_emb=speaker_emb,
                context_emb=context_emb,
            )

            target_length = audio.shape[-2]
            cond = F.interpolate(cond, size=target_length, mode='linear', align_corners=False)

            # Shape and device checks
            batch, n, d, device = *audio.shape, self.device
            assert d == self.dim, f"Codec codebook dimension {d} must match model dimensions {self.dim}"

            # Diffusion step
            times = torch.zeros((batch,), device=device).float().uniform_(0, 1.0)
            noise = torch.randn_like(audio)
            gamma = self.gamma_schedule(times)
            padded_gamma = right_pad_dims_to(audio, gamma)
            alpha, sigma = gamma_to_alpha_sigma(padded_gamma, self.scale)
            noised_audio = alpha * audio + sigma * noise

            # Predict and calculate diffusion loss
            pred = self.model(noised_audio, times, prompt=prompt_enc, cond=cond)

            if self.objective == "eps":
                target = noise
            elif self.objective == "x0":
                target = audio
            elif self.objective == "v":
                target = alpha * noise - sigma * audio

            loss = F.mse_loss(pred, target, reduction="none")
            loss = reduce(loss, "b ... -> b", "mean")

            # Min SNR loss weighting
            snr = (alpha ** 2) / (sigma ** 2)
            maybe_clipped_snr = snr.clone()
            if self.min_snr_loss_weight:
                maybe_clipped_snr.clamp_(max=self.min_snr_gamma)

            if self.objective == "eps":
                loss_weight = maybe_clipped_snr / snr
            elif self.objective == "x0":
                loss_weight = maybe_clipped_snr
            elif self.objective == "v":
                loss_weight = maybe_clipped_snr / (snr + 1)

            loss = (loss * loss_weight).mean()
        
        loss = loss + aux_loss

        return {
            "loss": loss,
            "mse_loss": loss,
            "aux_loss": aux_loss,
            "duration_loss": overall_duration_loss,
            "pitch_loss": overall_pitch_loss,
        }


# trainer
def cycle(dl):
    assert dl is not None, "DataLoader is None. Ensure it is properly initialized."
    while True:
        for data in dl:
            yield data


class CustomDataset(Dataset):
    def __init__(self, dataset_folder, max_items=None, sampling_rate=24000):
            self.dataset_folder = dataset_folder
            self.sampling_rate = sampling_rate
            # Collect files
            self.audio_files = sorted(
                [path for path in (Path(dataset_folder) / 'wav').rglob('*.wav') if not path.name.startswith('._')]
            )
            # self.text_files = sorted(
            #     [path for path in (Path(dataset_folder) / 'txt').rglob('*.txt') if not path.name.startswith('._')]
            # )
            self.phoneme_files = sorted(
                [path for path in (Path(dataset_folder) / 'phonemized').rglob('*.txt') if not path.name.startswith('._')]
            )
            self.speaker_embeddings_files = sorted(
                [path for path in (Path(dataset_folder) / 'speaker_embeddings').rglob('*.npy') if not path.name.startswith('._')]
            )
            self.context_embeddings_files = sorted(
                [path for path in (Path(dataset_folder) / 'context_embeddings').rglob('*.npy') if not path.name.startswith('._')]
            )
            self.segment_files = sorted(
                [path for path in (Path(dataset_folder) / 'segments').rglob('*.json') if not path.name.startswith('._')]
            )

            # Get the base file names (without extensions) for matching
            audio_basenames = {path.stem for path in self.audio_files}
            # text_basenames = {path.stem for path in self.text_files}
            phoneme_basenames = {path.stem for path in self.phoneme_files}
            context_basenames = {path.stem for path in self.context_embeddings_files}
            segment_basenames = {path.stem for path in self.segment_files}

            # Intersection of all file sets (excluding speaker embeddings)
            common_basenames = audio_basenames & phoneme_basenames & context_basenames & segment_basenames

            # Filter files to only include common base names
            self.audio_files = [path for path in self.audio_files if path.stem in common_basenames]
            # self.text_files = [path for path in self.text_files if path.stem in common_basenames]
            self.phoneme_files = [path for path in self.phoneme_files if path.stem in common_basenames]
            self.context_embeddings_files = [
                path for path in self.context_embeddings_files if path.stem in common_basenames
            ]
            self.segment_files = [path for path in self.segment_files if path.stem in common_basenames]

            # Apply the limit if max_items is specified
            if max_items is not None:
                self.audio_files = self.audio_files[:max_items]
                # self.text_files = self.text_files[:max_items]
                self.phoneme_files = self.phoneme_files[:max_items]
                self.context_embeddings_files = self.context_embeddings_files[:max_items]
                self.segment_files = self.segment_files[:max_items]

            logger.info(f"Dataset initialized with {len(self.audio_files)} audio files, "
                        f"{len(self.phoneme_files)} phoneme files, "
                        f"{len(self.context_embeddings_files)} context embeddings, "
                        f"{len(self.segment_files)} segments, "
                        f"{len(self.speaker_embeddings_files)} speaker embeddings.")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        try:
            audio_path = self.audio_files[idx]
            phoneme_path = self.phoneme_files[idx]
            context_embedding_path = self.context_embeddings_files[idx]
            segment_path = self.segment_files[idx]
            speaker_id = context_embedding_path.parts[-2]
            
            # Locate the speaker embedding and prompt audio
            speaker_embedding_path = next(
                (path for path in self.speaker_embeddings_files if path.stem == speaker_id),
                None  # Default to None if no match found
            )
            speaker_folder = Path(self.dataset_folder) / 'wav' / speaker_id

            # Load data
            audio, original_sample_rate = torchaudio.load(str(audio_path))

            if original_sample_rate != self.sampling_rate:
                audio = torchaudio.transforms.Resample(original_sample_rate, self.sampling_rate)(audio)
                
            with open(phoneme_path, 'r') as f:
                phoneme = f.read()
            try:
                speaker_embedding = torch.tensor(np.load(speaker_embedding_path), dtype=torch.float16)
            except:
                speaker_embedding = torch.tensor(np.zeros(shape=(192, 1), dtype=torch.float16))

            try:
                context_embedding = torch.tensor(np.load(context_embedding_path), dtype=torch.float16)
            except:
                context_embedding = torch.tensor(np.zeros(shape=(768), dtype=torch.float16))
            
            with open(segment_path, 'r') as file:
                segment = json.load(file)

            # Load and validate the prompt
            prompt_audio = self.accumulate_prompt_audio(speaker_folder)

            return {
                'audio': audio,
                'phoneme': phoneme,
                'speaker_embeddings': speaker_embedding,
                'context_embeddings': context_embedding,
                'prompt': prompt_audio,
                'segment': segment
            }
        except IndexError as e:
            logger.error(f"IndexError: {e} - Index {idx} is out of range.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error at index {idx}: {e}")
            raise

    def _is_valid_prompt(self, path):
        """
        Validate if a prompt audio file meets the desired criteria for inclusion in accumulation.
        """
        try:
            metadata = torchaudio.info(str(path))
            duration = metadata.num_frames / metadata.sample_rate
            return duration > 0  # Include all audio files with a positive duration
        except Exception as e:
            logger.warning(f"Error checking prompt validity for {path}: {e}")
            return False

    def accumulate_prompt_audio(self, speaker_folder):
        """
        Accumulate audio for a speaker up to 30 seconds and pad if necessary.
        """
        max_duration = 30  # Maximum duration in seconds
        accumulated_audio = []
        total_frames = 0
        sampling_rate = self.sampling_rate

        for audio_path in speaker_folder.rglob('*.wav'):
            if not self._is_valid_prompt(audio_path):
                continue

            try:
                audio, sr = torchaudio.load(str(audio_path))
                if sr != sampling_rate:
                    audio = torchaudio.transforms.Resample(sr, sampling_rate)(audio)

                frames = audio.shape[1]
                if total_frames + frames > max_duration * sampling_rate:
                    # Truncate audio to fit the remaining time
                    remaining_frames = max_duration * sampling_rate - total_frames
                    accumulated_audio.append(audio[:, :remaining_frames])
                    total_frames += remaining_frames
                    break
                else:
                    accumulated_audio.append(audio)
                    total_frames += frames

            except Exception as e:
                logger.warning(f"Error loading prompt audio {audio_path}: {e}")
                continue

        # Combine all accumulated audio
        if accumulated_audio:
            accumulated_audio = torch.cat(accumulated_audio, dim=1)
        else:
            # No valid audio, return silence
            accumulated_audio = torch.zeros(1, 0)

        # Pad to 30 seconds if necessary
        if accumulated_audio.shape[1] < max_duration * sampling_rate:
            padding = max_duration * sampling_rate - accumulated_audio.shape[1]
            accumulated_audio = torch.nn.functional.pad(accumulated_audio, (0, padding), mode='constant', value=0)

        return accumulated_audio
