import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from einops import rearrange

from .transformer_blocks import (Mlp, PatchEmbed, DropPath, trunc_normal_,
                                 nlc_to_nchw, nchw_to_nlc, nlc_by_nchw,
                                 MultiScaleGroupConv)

__all__ = ['fact_nano', 'fact_tiny', 'fact_small', 'fact_base']


class Focus(nn.Module):
    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right,
                       patch_bot_right), dim=1)
        return x


class ConvPosEnc(nn.Module):
    """Convolutional position encoding with DWConv."""
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.proj = nn.Conv2d(dim,
                              dim,
                              kernel_size,
                              stride=1,
                              padding=kernel_size // 2,
                              groups=dim)

    @nlc_by_nchw
    def forward(self, x):
        x = self.proj(x) + x
        return x


class ConvRelPosEnc(nn.Module):
    """Convolutional relative position encoding."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.conv = MultiScaleGroupConv(dim, num_heads=num_heads)

    def forward(self, q, v, hw_shape):
        h, w = hw_shape
        v = rearrange(v, 'b n (h w) c -> b (n c) h w', h=h, w=w)  # B, C, H, W
        v = self.conv(v)
        v = rearrange(v, 'b (n c) h w -> b n (h w) c',
                      n=q.shape[1])  # B, h, N, Ch
        return q * v


class FactorizedAttention(nn.Module):
    """Factorized self-attention."""
    def __init__(self,
                 dim,
                 num_heads=8,
                 kv_kernel_size=4,
                 kv_stride=4,
                 qkv_bias=True,
                 qk_scale=None,
                 proj_drop=0.,
                 rope=None,
                 crpe=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads)**-0.5
        self.kv_stride = kv_stride
        self.crpe = crpe
        self.rope = rope

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, dim, kv_kernel_size, kv_stride,
                            groups=dim, bias=False
                            ) if kv_kernel_size != 1 else nn.Identity()
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, hw_shape):
        B, N, C = x.shape
        x = nlc_to_nchw(x, hw_shape)

        kv = self.kv(x)
        q, k, v = self.q(x), self.k(kv), self.v(kv)
        q, k, v = [
            nchw_to_nlc(t)[0].reshape(B, -1, self.num_heads, C // self.num_heads
                                      ).transpose(1, 2).contiguous()
            for t in (q, k, v)]  # B, h, N, Ch

        kv = torch.einsum('bhnk, bhnv -> bhkv', F.softmax(k, dim=2), v)
        attn = torch.einsum('bhnk, bhkv -> bhnv', q, kv)
        x = self.scale * attn

        if self.crpe:
            x += self.crpe(q, v, hw_shape)
        x = x.transpose(1, 2).reshape(B, N, C)  # B, h, N, Ch -> B, N, C

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 mlp_ratio=4,
                 kv_kernel_size=4,
                 kv_stride=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 cpe=None,
                 crpe=None,
                 rope=None):
        super().__init__()
        self.cpe = cpe
        self.norm1 = norm_layer(dim)
        self.attn = FactorizedAttention(dim, num_heads, kv_kernel_size,
                                        kv_stride, qkv_bias, qk_scale,
                                        drop_rate, rope, crpe)

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim,
                       dim * mlp_ratio,
                       act_layer=act_layer,
                       drop=drop_rate)

    def forward(self, x, hw_shape):
        x = self.cpe(x, hw_shape)
        identity = x
        x = self.norm1(x)
        x = self.attn(x, hw_shape)
        x = identity + self.drop_path(x)

        identity = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = identity + self.drop_path(x)
        return x


class FacTLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth=2,
                 num_heads=8,
                 mlp_ratio=4,
                 kv_kernel_size=4,
                 kv_stride=4,
                 input_size=None,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 crpe=False):
        super().__init__()
        self.cpe = ConvPosEnc(dim)
        self.crpe = ConvRelPosEnc(dim) if crpe else None
        self.blocks = nn.ModuleList([
            Block(dim,
                  num_heads,
                  mlp_ratio,
                  kv_kernel_size,
                  kv_stride,
                  qkv_bias,
                  qk_scale,
                  drop_rate,
                  drop_path_rate,
                  act_layer=act_layer,
                  norm_layer=norm_layer,
                  cpe=self.cpe,
                  crpe=self.crpe) for _ in range(depth)])

    def forward(self, x, hw_shape):
        for blk in self.blocks:
            x = blk(x, hw_shape)
        x = nlc_to_nchw(x, hw_shape)
        return x


class FacT(nn.Module):
    """Factorized Self-Attention Transformer.
    """
    def __init__(self,
                 in_channels=3,
                 embed_dims=(64, 128, 256, 512),
                 depths=(2, 2, 4, 2),
                 num_heads=8,
                 mlp_ratio=4,
                 kv_kernel_sizes=(4, 4, 1, 1),
                 kv_strides=(4, 4, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 crpe=(False, False, True, True),
                 **kwargs):
        super().__init__()
        self.out_indices = out_indices
        self.stem = nn.Sequential(
            Focus(),
            nn.Conv2d(in_channels * 4, embed_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU(),
            nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU())

        self.patch_embeds = nn.ModuleList([
            PatchEmbed(embed_dims[i - 1] if i else embed_dims[i],
                       embed_dims[i], 2) for i in range(len(depths))])

        self.layers = nn.ModuleList([
            FacTLayer(embed_dims[i],
                      depth=depths[i],
                      num_heads=num_heads,
                      mlp_ratio=mlp_ratio,
                      kv_kernel_size=kv_kernel_sizes[i],
                      kv_stride=kv_strides[i],
                      input_size=int((640 + 5 * 32) / 2**(i + 2)),
                      qkv_bias=qkv_bias,
                      qk_scale=qk_scale,
                      drop_rate=drop_rate,
                      drop_path_rate=drop_path_rate,
                      act_layer=act_layer,
                      norm_layer=norm_layer,
                      crpe=crpe[i]) for i in range(len(depths))])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i, layer in enumerate(self.layers):
            x, hw_shape = self.patch_embeds[i](x)
            x = layer(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)
        return outs


def fact_nano(**kwargs):
    model = FacT(embed_dims=(48, 96, 192, 384),
                 depths=(1, 2, 2, 1),
                 **kwargs)
    return model


def fact_tiny(**kwargs):
    model = FacT(embed_dims=(64, 128, 256, 320),
                 depths=(2, 2, 4, 2),
                 **kwargs)
    return model


def fact_small(**kwargs):
    model = FacT(embed_dims=(96, 192, 384, 768),
                 depths=(2, 2, 6, 2),
                 **kwargs)
    return model


def fact_base(**kwargs):
    model = FacT(embed_dims=(96, 192, 384, 768),
                 depths=(4, 4, 24, 4),
                 **kwargs)
    return model
