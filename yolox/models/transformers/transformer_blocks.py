import torch
import torch.nn as nn

from timm.models.layers import Mlp, DropPath, trunc_normal_, to_2tuple


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, in_channels, embed_dims, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels,
                              embed_dims,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, x):
        x = self.proj(x)
        x, hw_shape = nchw_to_nlc(x)
        out = self.norm(x)
        return out, hw_shape


def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.
    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.
    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len does not match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W).contiguous()


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.
    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.
    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
        tuple: The [H, W] shape.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous(), x.shape[2:]


def nlc_by_nchw(func):
    """Operate in [N, C, H, W] manner with [N, L, C] shape tensoras both input and output.
    """
    def wrapper(obj, x, hw_shape):
        x = nlc_to_nchw(x, hw_shape)
        x = func(obj, x)
        x, _ = nchw_to_nlc(x)
        return x

    return wrapper


def nchw_by_nlc(func):
    """Operate in [N, L, C] manner with [N, C, H, W] shape tensor as both input and output.
    """
    def wrapper(obj, x):
        x, hw_shape = nchw_to_nlc(x)
        x = func(obj, x)
        x = nlc_to_nchw(x, hw_shape)
        return x

    return wrapper


class LayerNorm2d(nn.LayerNorm):
    @nchw_by_nlc
    def forward(self, x):
        return super().forward(x)


class MultiScaleGroupConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 num_heads=8,
                 depthwise=True,
                 kernel_sizes: dict = {
                     3: 2,
                     5: 3,
                     7: 3
                 }):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        assert (in_channels % num_heads == 0 and out_channels % num_heads
                == 0), 'channels not divisible by num_heads'

        self.convs = nn.ModuleList()
        self.channel_splits = []
        for k, h in kernel_sizes.items():
            c_in = h * in_channels // num_heads
            c_out = h * out_channels // num_heads
            conv = nn.Conv2d(c_in,
                             c_out,
                             kernel_size=k,
                             padding=k // 2,
                             groups=c_in if depthwise else 1)
            self.convs.append(conv)
            self.channel_splits.append(c_in)

    def forward(self, x):
        x_s = torch.split(x, self.channel_splits, dim=1)
        x = torch.cat([conv(x) for conv, x in zip(self.convs, x_s)], dim=1)
        return x


def sin_cos_pos_emb_1d(pos: int, dim: int):
    assert dim % 4 == 0
    pos = torch.arange(pos)
    omega = torch.arange(dim // 4) / (dim / 4.)
    omega = 1. / 10000**omega
    emb = torch.einsum('m, d -> md', pos, omega)  # (M, D/4)
    emb = emb.transpose(0, 1).contiguous()
    return emb.sin(), emb.cos()


def rotatory_pos_emb_2d(H, W, C):
    sin_x, cos_x = sin_cos_pos_emb_1d(W, C)  # (C/4, W)
    sin_x, cos_x = map(lambda x: x.unsqueeze(1).repeat(2, H, 1),
                       (sin_x, cos_x))
    sin_y, cos_y = sin_cos_pos_emb_1d(H, C)  # (C/4, H)
    sin_y, cos_y = map(lambda x: x.unsqueeze(2).repeat(2, 1, W),
                       (sin_y, cos_y))
    return torch.stack((sin_x, cos_x, sin_y, cos_y), dim=0)


class RotatoryPosEmb2d(nn.Module):
    def __init__(self, height, width, dim):
        super().__init__()
        emb = rotatory_pos_emb_2d(height, width, dim)
        self.sin_pos_emb = nn.Parameter(emb, requires_grad=False)
        self.C, self.H, self.W = dim, height, width

    def forward(self, x, start=0, step=1):
        _, C, H, W = x.shape
        assert C == self.C and H <= self.H and W <= self.W

        x0, x1, x2, x3 = x[:, 0::4], x[:, 1::4], x[:, 2::4], x[:, 3::4]
        sin_pos_emb = self.sin_pos_emb[..., start::step, start::step]
        sin_x, cos_x, sin_y, cos_y = sin_pos_emb[..., :H, :W]

        x01 = torch.stack((x0, x1), dim=2).flatten(1, 2)  # B, C/2, H, W
        x_10 = torch.stack((-x1, x0), dim=2).flatten(1, 2)
        x23 = torch.stack((x2, x3), dim=2).flatten(1, 2)
        x_32 = torch.stack((-x3, x2), dim=2).flatten(1, 2)

        x01_ = x01 * cos_x + x_10 * sin_x
        x23_ = x23 * cos_y + x_32 * sin_y
        return torch.stack((x01_, x23_), dim=2).flatten(1, 2)


class LeFF(nn.Module):
    """Locality-enhanced Feed-Forward Network.
    """
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.conv = nn.Conv2d(hidden_features,
                              hidden_features,
                              kernel_size=3,
                              padding=1,
                              groups=hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x, hw_shape):
        x = self.fc1(x)
        identity = x
        x = nlc_to_nchw(x, hw_shape)
        x = self.conv(x)
        x, _ = nchw_to_nlc(x)
        x = x + identity

        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
