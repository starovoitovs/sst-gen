"""
https://github.com/imics-lab/tts-gan/
"""
from torch import nn, optim
import torch
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
from torch.nn import functional as F

from genhack.utils import DEVICE


class TTSGAN(nn.Module):

    def __init__(self, n_latent_dim, n_dim, seq_length, patch_size, emb_size, depth, n_classes, n_heads, forward_dropout_rate, attention_dropout_rate, forward_expansion, *args, **kwargs):
        super().__init__()
        self.n_dim = n_dim
        self.n_latent_dim = n_latent_dim
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.depth = depth
        self.n_classes = n_classes
        self.n_heads = n_heads
        self.forward_dropout_rate = forward_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.forward_expansion = forward_expansion

        self.generator = Generator(n_channels=n_dim, n_latent_dim=n_latent_dim, seq_length=seq_length, patch_size=patch_size, depth=depth, emb_size=emb_size, n_heads=n_heads, attention_dropout_rate=attention_dropout_rate, forward_dropout_rate=forward_dropout_rate, forward_expansion=forward_expansion)
        self.discriminator = Discriminator(n_channels=n_dim, seq_length=seq_length, patch_size=patch_size, emb_size=emb_size, n_heads=n_heads, depth=depth, n_classes=n_classes, attention_dropout_rate=attention_dropout_rate, forward_dropout_rate=forward_dropout_rate, forward_expansion=forward_expansion)

    def forward(self, inputs):
        inputs, time = inputs
        return [inputs.reshape(-1, self.n_dim, 1, self.seq_length)]

    def loss(self, *args, **kwargs):

        # train discriminator
        real_samples = args[0]
        real_validity = self.discriminator(real_samples)

        z = torch.randn((len(real_samples), self.n_latent_dim), device=DEVICE)
        fake_samples = self.generator(z)
        fake_validity = self.discriminator(fake_samples)

        # soft labels
        real_label = torch.full((real_samples.shape[0],), 0.9, device=DEVICE)
        fake_label = torch.full((real_samples.shape[0],), 0.1, device=DEVICE)
        real_validity = nn.Sigmoid()(real_validity.view(-1))
        fake_validity = nn.Sigmoid()(fake_validity.view(-1))
        d_real_loss = nn.BCELoss()(real_validity, real_label)
        d_fake_loss = nn.BCELoss()(fake_validity, fake_label)
        d_loss = d_real_loss + d_fake_loss

        # train generator
        real_label = torch.full((fake_validity.shape[0],), 1., device=DEVICE)
        fake_validity = nn.Sigmoid()(fake_validity.view(-1))
        g_loss = nn.BCELoss()(fake_validity.view(-1), real_label)

        # the order of the losses must coincide with the order of optimizers
        # convention: discriminator, generator
        return {'d_loss': d_loss, 'g_loss': g_loss}

    def sample(self, noise):
        return torch.squeeze(self.generator(noise))


class Generator(nn.Module):

    def __init__(self, seq_length, patch_size, n_channels, n_latent_dim, emb_size, n_heads, depth, forward_dropout_rate, attention_dropout_rate, forward_expansion):
        super().__init__()
        self.n_channels = n_channels
        self.n_latent_dim = n_latent_dim
        self.seq_length = seq_length
        self.emb_size = emb_size
        self.patch_size = patch_size
        self.depth = depth
        self.n_heads = n_heads
        self.attention_dropout_rate = attention_dropout_rate
        self.forward_dropout_rate = forward_dropout_rate
        self.forward_expansion = forward_expansion

        self.l1 = nn.Linear(self.n_latent_dim, self.seq_length * self.emb_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_length, self.emb_size))
        self.blocks = GenTransformerEncoder(
            depth=self.depth,
            emb_size=self.emb_size,
            n_heads=self.n_heads,
            attention_dropout_rate=self.attention_dropout_rate,
            forward_dropout_rate=self.forward_dropout_rate,
            forward_expansion=self.forward_expansion,
        )

        self.deconv = nn.Sequential(nn.Conv2d(self.emb_size, self.n_channels, 1, 1, 0))

    def forward(self, z):
        x = self.l1(z).view(-1, self.seq_length, self.emb_size)
        x = x + self.pos_embed
        H, W = 1, self.seq_length
        x = self.blocks(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        output = self.deconv(x.permute(0, 3, 1, 2))
        output = output.view(-1, self.n_channels, H, W)
        return output


class GenTransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 n_heads,
                 attention_dropout_rate,
                 forward_expansion,
                 forward_dropout_rate):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, n_heads, attention_dropout_rate),
                nn.Dropout(attention_dropout_rate)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, attention_dropout_rate=forward_dropout_rate),
                nn.Dropout(attention_dropout_rate)
            )
            ))


class GenTransformerEncoder(nn.Sequential):
    def __init__(self, depth, **kwargs):
        super().__init__(*[GenTransformerEncoderBlock(**kwargs) for _ in range(depth)])


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, attention_dropout_rate):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(attention_dropout_rate),
            nn.Linear(expansion * emb_size, emb_size),
        )


class DisTransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 n_heads,
                 attention_dropout_rate,
                 forward_expansion,
                 forward_dropout_rate):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, n_heads, attention_dropout_rate),
                nn.Dropout(attention_dropout_rate)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, attention_dropout_rate=forward_dropout_rate),
                nn.Dropout(attention_dropout_rate)
            )
            ))


class DisTransformerEncoder(nn.Sequential):
    def __init__(self, depth, **kwargs):
        super().__init__(*[DisTransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes

        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return out


class PatchEmbeddingLinear(nn.Module):
    # what are the proper parameters set here?
    def __init__(self, n_channels, patch_size, emb_size, seq_length):
        # self.patch_size = patch_size
        super().__init__()
        # change the conv2d parameters here
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=1, s2=patch_size),
            nn.Linear(patch_size * n_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((seq_length // patch_size) + 1, emb_size))

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # position
        x += self.positions
        return x


class Discriminator(nn.Sequential):
    def __init__(self, n_channels, patch_size, emb_size, seq_length, depth, n_classes, n_heads, forward_expansion, attention_dropout_rate, forward_dropout_rate):
        super().__init__(
            PatchEmbeddingLinear(n_channels, patch_size, emb_size, seq_length),
            DisTransformerEncoder(depth=depth, emb_size=emb_size, n_heads=n_heads, forward_expansion=forward_expansion, attention_dropout_rate=attention_dropout_rate, forward_dropout_rate=forward_dropout_rate),
            ClassificationHead(emb_size, n_classes)
        )
