import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F
from config import opt


def number_parameters(Net, type_size=8):
    para = sum([np.prod(list(p.size())) for p in Net.parameters()])
    return para / 1024 * type_size / 1024


class Patch_Projection(nn.Module):
    def __init__(self, in_channels, out_channels, proj_shape, patch_shape, switch_padding):
        super().__init__()
        self.switch_padding = switch_padding
        self.proj_shape = proj_shape
        self.patch_shape = patch_shape

        self.patch_merge = nn.Unfold(kernel_size=proj_shape, stride=proj_shape, padding=0)
        self.linear = nn.Linear(in_channels * proj_shape ** 3, out_channels)

    def forward(self, x):

        if x.shape[-1] % self.proj_shape:
            padding_shape = self.proj_shape - x.shape[-1] % self.proj_shape
            padding_left, padding_right = int(padding_shape / 2), int((padding_shape + 1) / 2)
            patch_padding = (padding_left, padding_right, padding_left, padding_right, padding_left, padding_right)
            x = F.pad(x, patch_padding, mode='constant', value=0)

        b, l, c, n_l, n_h, n_w = x.shape

        x = rearrange(x, 'b l c n_l n_h n_w -> (b l) c n_l n_h n_w')

        new_l, new_h, new_w = n_l // self.proj_shape, n_h // self.proj_shape, n_w // self.proj_shape

        x = rearrange(x, 'n_b c (nw_l w_l) (nw_h w_h) (nw_w w_w) -> n_b nw_l nw_h nw_w (c w_l w_h w_w)',
                      w_l=self.proj_shape, w_h=self.proj_shape, w_w=self.proj_shape)

        x = x.view(b * l, -1, new_l, new_h, new_w)

        if self.switch_padding == True and x.shape[-1] % self.patch_shape:
            padding_shape = self.patch_shape - x.shape[-1] % self.patch_shape
            padding_left, padding_right = int(padding_shape / 2), int((padding_shape + 1) / 2)
            patch_padding = (padding_left, padding_right, padding_left, padding_right, padding_left, padding_right)
            x = F.pad(x, patch_padding, mode='constant', value=0)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.linear(x)
        x = rearrange(x, '(b l) c n_l n_h n_w -> b l c n_l n_h n_w', b=b)

        return x


class Residual_Connection(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Layer_Normal(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, direction):

    mask = torch.zeros(window_size ** 3, window_size ** 3)

    if direction[0] == 1:

        mask[-displacement * window_size * window_size:, :-displacement * window_size * window_size] = float('-inf')
        mask[:-displacement * window_size * window_size, -displacement * window_size * window_size:] = float('-inf')

    elif direction[1] == 1:
        mask = rearrange(mask, '(l1 h1 w1) (l2 h2 w2) -> l1 (h1 w1) l2 (h2 w2)', l1=window_size, l2=window_size,
                         h1=window_size, h2=window_size)
        mask[:, -displacement * window_size:, :, :-displacement * window_size] = float('-inf')
        mask[:, :-displacement * window_size, :, -displacement * window_size:] = float('-inf')
        mask = rearrange(mask, 'l1 (h1 w1) l2 (h2 w2) -> (l1 h1 w1) (l2 h2 w2)', h1=window_size, h2=window_size)

    else :
        mask = rearrange(mask, '(l1 h1 w1) (l2 h2 w2) -> (l1 h1) w1 (l2 h2) w2', l1=window_size, l2=window_size,
                         h1=window_size, h2=window_size)
        mask[-displacement:, :-displacement] = float('-inf')
        mask[:-displacement, -displacement:] = float('-inf')
        mask = rearrange(mask, '(l1 h1) w1 (l2 h2) w2 -> (l1 h1 w1) (l2 h2 w2)', h1=window_size, h2=window_size)
    return mask


def Relative_Position_Maps(window_size):
    indices = torch.tensor(
        np.array([[x, y, z] for x in range(window_size) for y in range(window_size) for z in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class Feature_Shifting(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement, self.displacement), dims=(1, 2, 3))


class MSA_Block(nn.Module):
    def __init__(self, dim_seq, num_heads, dim_head, patch_shape, switch_shift, switch_position):
        super().__init__()
        dim_inner = dim_head * num_heads

        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        self.patch_shape = patch_shape
        self.switch_position = switch_position
        self.switch_shift = switch_shift

        if self.switch_shift and patch_shape is not None:
            displacement = patch_shape // 2
            self.patch_shifting = Feature_Shifting(-displacement)
            self.patch_restoration = Feature_Shifting(displacement)
            self.upper_mask = nn.Parameter(
                create_mask(window_size=patch_shape, displacement=displacement, direction=(1, 0, 0)),
                requires_grad=False)
            self.left_mask = nn.Parameter(
                create_mask(window_size=patch_shape, displacement=displacement, direction=(0, 1, 0)),
                requires_grad=False)
            self.front_mask = nn.Parameter(
                create_mask(window_size=patch_shape, displacement=displacement, direction=(0, 0, 1)),
                requires_grad=False)

        self.to_qkv = nn.Linear(dim_seq, dim_inner * 3, bias=False)

        if self.switch_position and patch_shape is not None:
            self.relative_indices = Relative_Position_Maps(patch_shape) + patch_shape - 1
            self.pos_embedding = nn.Parameter(
                torch.randn(2 * patch_shape - 1, 2 * patch_shape - 1, 2 * patch_shape - 1))
        elif patch_shape is not None:
            self.pos_embedding = nn.Parameter(torch.randn(patch_shape ** 2, patch_shape ** 2))

        self.to_out = nn.Linear(dim_inner, dim_seq)

    def forward(self, x):

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if self.patch_shape is not None:

            if self.switch_shift and self.patch_shape is not None:
                x = self.patch_shifting(x)

            b, n_l, n_h, n_w, _, h = *x.shape, self.num_heads
            nw_l, nw_h, nw_w = n_l // self.patch_shape, n_h // self.patch_shape, n_w // self.patch_shape
            q, k, v = map(lambda t:
                          rearrange(t,
                                    'b (nw_l w_l) (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_l nw_h nw_w) (w_l w_h w_w) d',
                                    h=h, w_l=self.patch_shape, w_h=self.patch_shape, w_w=self.patch_shape), qkv)
            dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

            if self.switch_position:
                dots += self.pos_embedding[
                    self.relative_indices[:, :, 0].type(torch.long), self.relative_indices[:, :, 1].type(
                        torch.long), self.relative_indices[:, :, 2].type(torch.long)]
            else:
                dots += self.pos_embedding

            if self.switch_shift:
                dots[:, :, -nw_w:] += self.upper_mask
                dots[:, :, nw_w - 1::nw_w] += self.left_mask
                dots = rearrange(dots, 'l h (l1 h1 w1) q k ->  l h (l1 w1) h1 q k', l1=nw_l, h1=nw_h, w1=nw_w)
                dots[:, :, :, nw_w - 1] += self.front_mask
                dots = rearrange(dots, ' l h (l1 w1) h1 q k -> l h (l1 h1 w1) q k', l1=nw_l)

                dots[:, :, nw_w - 1::nw_w] += self.front_mask

            attn = dots.softmax(dim=-1)

            out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
            out = rearrange(out, 'b h (nw_l nw_h nw_w) (w_l w_h w_w) d -> b (nw_l w_l) (nw_h w_h) (nw_w w_w) (h d)',
                            w_l=self.patch_shape, w_h=self.patch_shape, w_w=self.patch_shape, nw_l=nw_l, nw_h=nw_h,
                            nw_w=nw_w)

            out = self.to_out(out)

            if self.switch_shift and self.patch_shape is not None:
                out = self.patch_restoration(out)

        else:
            b, n_l, _, h = *x.shape, self.num_heads
            q, k, v = map(lambda t: rearrange(t, 'b nw_l (h d) -> b h nw_l d', h=h), qkv)
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

            attn = dots.softmax(dim=-1)

            out = einsum('b h i j, b h j d -> b h i d', attn, v)
            out = rearrange(out, 'b h nw_l d -> b nw_l (h d)')
            out = self.to_out(out)

        return out


class transformer_block(nn.Module):
    def __init__(self, dim_seq, dim_mlp, num_heads, dim_head, patch_shape, switch_shift, switch_position):
        super().__init__()
        self.patch_shape = patch_shape
        self.attention_block = Residual_Connection(Layer_Normal(dim_seq,
                                                                MSA_Block(dim_seq=dim_seq, num_heads=num_heads,
                                                                          dim_head=dim_head, patch_shape=patch_shape,
                                                                          switch_shift=switch_shift,
                                                                          switch_position=switch_position)))

        self.mlp_block = Residual_Connection(Layer_Normal(dim_seq, MLP_Block(dim=dim_seq, hidden_dim=dim_mlp)))

    def forward(self, x):
        if self.patch_shape is not None:
            b, l, c, n_l, n_h, n_w = x.shape
            x = rearrange(x, 'b l c n_l n_h n_w -> (b l) c n_l n_h n_w')
            x = self.attention_block(x)
            x = self.mlp_block(x)
            x = rearrange(x, '(b l) c n_l n_h n_w -> b l c n_l n_h n_w', b=b)
        else:
            x = self.attention_block(x)
            x = self.mlp_block(x)

        return x


class Local_Transformer_Block(nn.Module):
    def __init__(self, in_channels, dim_hidden, num_layers, num_heads, dim_head, proj_shape, patch_shape,
                 switch_position):
        super().__init__()

        self.patch_partition = Patch_Projection(in_channels=in_channels, out_channels=dim_hidden, proj_shape=proj_shape,
                                                patch_shape=patch_shape, switch_padding=True)

        self.layers = nn.ModuleList([])

        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                transformer_block(dim_seq=dim_hidden, num_heads=num_heads, dim_mlp=dim_hidden * 2,
                                  dim_head=dim_head, patch_shape=patch_shape, switch_shift=False,
                                  switch_position=switch_position),
                transformer_block(dim_seq=dim_hidden, num_heads=num_heads, dim_mlp=dim_hidden * 2,
                                  dim_head=dim_head, patch_shape=patch_shape, switch_shift=True,
                                  switch_position=switch_position)
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for patch_block, shift_block in self.layers:
            x = patch_block(x)
            x = shift_block(x)
        return x.permute(0, 1, 5, 2, 3, 4)


class Local_Transformer_Structure(nn.Module):
    def __init__(self, *, in_channels=1, dim_hidden=(96, 128, 128), num_layers=(1, 2, 1), num_heads=(3, 4, 4),
                 dim_head=32, proj_shape=2, patch_shape=4, switch_position=True):
        super().__init__()

        self.stage1 = Local_Transformer_Block(in_channels=in_channels,
                                              dim_hidden=dim_hidden[0], num_layers=num_layers[0],
                                              num_heads=num_heads[0], dim_head=dim_head,
                                              proj_shape=proj_shape, patch_shape=patch_shape,
                                              switch_position=switch_position)

        self.stage2 = Local_Transformer_Block(in_channels=dim_hidden[0],
                                              dim_hidden=dim_hidden[1], num_layers=num_layers[1],
                                              num_heads=num_heads[1], dim_head=dim_head,
                                              proj_shape=proj_shape, patch_shape=patch_shape,
                                              switch_position=switch_position)

        self.stage3 = Local_Transformer_Block(in_channels=dim_hidden[1],
                                              dim_hidden=dim_hidden[2], num_layers=num_layers[2],
                                              num_heads=num_heads[2], dim_head=dim_head,
                                              proj_shape=proj_shape, patch_shape=patch_shape,
                                              switch_position=switch_position)

        self.stage4 = Patch_Projection(in_channels=dim_hidden[2], out_channels=dim_hidden[2], proj_shape=proj_shape,
                                       patch_shape=patch_shape, switch_padding=False)

    def forward(self, img):
        b, l, n_l, n_h, n_w = img.shape
        x = rearrange(img, 'b (l c) n_l n_h n_w -> b l c n_l n_h n_w', c=1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x.view(b, l, -1)


class Global_Transformer_Structure(nn.Module):
    def __init__(self, dim_seq=1024, num_heads=4, dim_head=32, num_encoders=4):
        super().__init__()

        dim_trans = num_heads * dim_head
        self.order_embedding = nn.Parameter(torch.randn(1, 30, dim_seq))
        self.to_trans, self.to_seq = nn.Linear(dim_seq, dim_trans), nn.Linear(dim_trans, dim_seq)

        self.layers = nn.ModuleList([])
        for _ in range(num_encoders):
            self.layers.append(transformer_block(dim_seq=dim_trans, num_heads=num_heads,
                                                 dim_mlp=dim_seq * 2, dim_head=dim_head,
                                                 patch_shape=None, switch_shift=False,
                                                 switch_position=False)
                               )

    def forward(self, img):

        x = img + self.order_embedding
        x = self.to_trans(x)
        for layer in self.layers:
            x = layer(x)
        x = self.to_seq(x)
        return x


class Softmax_Classify(nn.Module):
    def __init__(self, hidden_size, num_linear, num_class):
        super().__init__()

        tmp_hidden_size = hidden_size

        self.layers = nn.ModuleList([])
        for _ in range(num_linear - 1):
            self.layers.append(nn.Linear(int(tmp_hidden_size), int(tmp_hidden_size / 2)))
            tmp_hidden_size /= 2

        self.layers.append(nn.Linear(int(tmp_hidden_size), num_class))

        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        b, l, n = x.shape
        x = rearrange(x, 'b l n -> (b l) n')
        for layer in self.layers:
            x = layer(x)
        x = self.soft_max(x)
        x = rearrange(x, '(b l) n -> b l n', b=b)
        return x


class trnet_pp(nn.Module):
    def __init__(self, *, in_channels=1, local_proj_shape=2, local_dim_hidden=(96, 128, 128),
                 local_num_layers=(1, 2, 1),
                 local_num_heads=(3, 4, 4), local_head_dim=32, local_patch_shape=4, local_switch_position=True,
                 global_dim_seq=1024, global_num_heads=4, global_head_dim=32, global_num_encoders=4,
                 CLS_num_linear=3, CLS_num_class=2):
        super().__init__()

        self.local_transformer = Local_Transformer_Structure(in_channels=in_channels,
                                                             dim_hidden=local_dim_hidden, num_layers=local_num_layers,
                                                             num_heads=local_num_heads, dim_head=local_head_dim,
                                                             proj_shape=local_proj_shape, patch_shape=local_patch_shape,
                                                             switch_position=local_switch_position)

        self.global_transformer = Global_Transformer_Structure(num_encoders=global_num_encoders, dim_seq=global_dim_seq,
                                                               num_heads=global_num_heads, dim_head=global_head_dim)

        self.softmax_classify = Softmax_Classify(hidden_size=global_dim_seq, num_linear=CLS_num_linear,
                                                 num_class=CLS_num_class)

        # print('local_transformer', number_parameters(self.local_transformer))
        # print('global_transformer', number_parameters(self.global_transformer))
        # print('softmax_classify', number_parameters(self.softmax_classify))

    def forward(self, img):
        x = self.local_transformer(img)
        x = self.global_transformer(x)
        x = self.softmax_classify(x)

        return x


def test_model():
    inputtest = torch.randn([1, 30, 25, 25, 25])
    model = trnet_pp(in_channels=opt.in_channels,
                     local_proj_shape=opt.local_proj_shape, local_dim_hidden=opt.local_dim_hidden,
                     local_num_layers=opt.local_num_layers, local_num_heads=opt.local_num_heads,
                     local_head_dim=opt.local_head_dim, local_patch_shape=opt.local_patch_shape,
                     local_switch_position=opt.local_switch_position,

                     global_dim_seq=opt.global_dim_seq, global_num_heads=opt.global_num_heads,
                     global_head_dim=opt.global_head_dim, global_num_encoders=opt.global_num_encoders,

                     CLS_num_linear=opt.CLS_num_linear, CLS_num_class=opt.CLS_num_class)
    print(number_parameters(model))
    logits = model(inputtest)
    print(logits.shape)
    return

