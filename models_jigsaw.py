# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg, Block
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from utils import get_2d_sincos_pos_embed
from munch import Munch

import numpy as np


class JigsawVisionTransformer(VisionTransformer):
    def __init__(self, mask_ratio, use_jigsaw, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_ratio = mask_ratio
        self.use_jigsaw = use_jigsaw
        self.num_patches = self.patch_embed.num_patches

        if self.use_jigsaw:
            self.jigsaw = nn.Sequential(
                *[
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.ReLU(),
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.ReLU(),
                    nn.Linear(self.embed_dim, self.num_patches),
                ]
            )
            self.target = torch.arange(self.num_patches)

    def forward_cls(self, x):
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

    def random_masking(self, x, target, mask_ratio=0.0):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim

        # keep the first subset
        len_keep = int(L * (1 - mask_ratio))
        ids_keep = target[:, :len_keep]  # N, len_keep
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        target_masked = ids_keep

        return x_masked, target_masked

    def forward_jigsaw(self, x, target):
        # masking: length -> length * mask_ratio
        x, target = self.random_masking(x, target, self.mask_ratio)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        x = self.jigsaw(x[:, 1:])
        return x, target

    def infer_jigsaw(self, x):
        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        x = self.jigsaw(x[:, 1:])
        return x.reshape(-1, self.num_patches)

    def forward(self, x, target):
        x = self.patch_embed(x)
        # pred_cls = self.forward_cls(x)
        if self.use_jigsaw:
            # if train, use forward_jigsaw, otherwise use forwar_cls
            # if self.training:
            pred_jigsaw, targets_jigsaw = self.forward_jigsaw(x, target)
            outs = Munch()
            # dim: [N * num_patches, num_patches]
            outs.pred_jigsaw = pred_jigsaw
            # dim: [N * num_patches]
            outs.gt_jigsaw = targets_jigsaw
            # else:
            #     pred_jigsaw = self.infer_jigsaw(x)
            #     outs = Munch()
            #     outs.pred_jigsaw = pred_jigsaw

        else:
            raise NotImplementedError("You must use jigsaw!")
        return outs


class JigsawViTR(JigsawVisionTransformer):
    def __init__(self, reconst_depth, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reconst_embed = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, self.embed_dim),
            requires_grad=False,
        )
        # fixed sin-cos embedding
        self.reconst_blocks = nn.ModuleList(
            [
                # Assuming the Block class is defined somewhere in your codebase
                Block(self.embed_dim, num_heads=16, mlp_ratio=4.0, qkv_bias=True)
                for _ in range(reconst_depth)  # 8 is the depth, you can adjust this
            ]
        )
        self.reconst_norm = nn.LayerNorm(self.embed_dim)
        self.reconst_pred = nn.Linear(
            self.embed_dim,
            self.patch_embed.patch_size[0] ** 2 * 3,
            bias=True,
        )
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio=0.0):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]  # N, len_keep
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        target_masked = ids_keep

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, target_masked, mask, ids_restore

    def forward_encoder(self, x):
        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_jigsaw(self, x, target):
        x = self.jigsaw(x[:, 1:])
        return x.reshape(-1, self.num_patches), target.reshape(-1)

    def forward_reconst(self, x, ids_restore):
        # add pos embed
        x = self.reconst_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.reconst_blocks:
            x = blk(x)
        x = self.reconst_norm(x)

        # predictor projection
        x = self.reconst_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        return x

    def reconst_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = loss.sum(dim=-1).mean()
        # loss = (loss * mask).sum() / mask.sum()
        # mean loss on removed patches
        return loss

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward(self, x):
        imgs = x
        x = self.patch_embed(x)
        x, target, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        x = self.forward_encoder(x)
        # Abovei is the same as MAE.forward_encoder
        # pred_cls = self.forward_cls(x)
        if self.use_jigsaw:
            # if train, use forward_jigsaw, otherwise use forwar_cls
            if self.training:
                reconst = self.forward_reconst(x, ids_restore)
                rec_loss = self.reconst_loss(imgs, reconst, mask)
                pred_jigsaw, targets_jigsaw = self.forward_jigsaw(x, target)
                outs = Munch()
                # dim: [N * num_patches, num_patches]
                outs.pred_jigsaw = pred_jigsaw
                # dim: [N * num_patches]
                outs.gt_jigsaw = targets_jigsaw
                outs.rec_loss = rec_loss
            else:
                pred_jigsaw = self.infer_jigsaw(x)
                outs = Munch()
                outs.pred_jigsaw = pred_jigsaw
        else:
            raise NotImplementedError("You must use jigsaw!")
        return outs


@register_model
def jigsaw_tiny_patch16_224(
    mask_ratio=0.5, use_jigsaw=True, pretrained=False, **kwargs
):
    model = JigsawVisionTransformer(
        mask_ratio=mask_ratio,
        use_jigsaw=use_jigsaw,
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def jigsaw_small_patch16_224(
    mask_ratio=0.5, use_jigsaw=True, pretrained=False, **kwargs
):
    model = JigsawVisionTransformer(
        mask_ratio=mask_ratio,
        use_jigsaw=use_jigsaw,
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def jigsaw_base_patch16_224(
    mask_ratio=0.5, use_jigsaw=True, pretrained=False, **kwargs
):
    model = JigsawVisionTransformer(
        mask_ratio=mask_ratio,
        use_jigsaw=use_jigsaw,
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def jigsaw_base_patch16_384(
    mask_ratio=0.5, use_jigsaw=True, pretrained=False, **kwargs
):
    model = JigsawVisionTransformer(
        mask_ratio=mask_ratio,
        use_jigsaw=use_jigsaw,
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def jigsaw_base_patch56_336(
    mask_ratio=0.5, use_jigsaw=True, pretrained=False, **kwargs
):
    model = JigsawVisionTransformer(
        mask_ratio=mask_ratio,
        use_jigsaw=use_jigsaw,
        img_size=336,
        patch_size=56,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def jigsaw_small_patch56_336(
    mask_ratio=0.5, use_jigsaw=True, pretrained=False, **kwargs
):
    model = JigsawVisionTransformer(
        mask_ratio=mask_ratio,
        use_jigsaw=use_jigsaw,
        img_size=336,
        patch_size=56,
        embed_dim=768,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def jigsaw_tiny_patch56_336(
    mask_ratio=0.5, use_jigsaw=True, pretrained=False, **kwargs
):
    model = JigsawVisionTransformer(
        mask_ratio=mask_ratio,
        use_jigsaw=use_jigsaw,
        img_size=336,
        patch_size=56,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def jigsaw_r_base_patch56_336(
    mask_ratio=0.5, use_jigsaw=True, pretrained=False, **kwargs
):
    model = JigsawViTR(
        reconst_depth=4,
        mask_ratio=mask_ratio,
        use_jigsaw=use_jigsaw,
        img_size=336,
        patch_size=56,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def jigsaw_base_patch112_336(
    mask_ratio=0.0, use_jigsaw=True, pretrained=False, **kwargs
):
    model = JigsawVisionTransformer(
        mask_ratio=mask_ratio,
        use_jigsaw=use_jigsaw,
        img_size=336,
        patch_size=112,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def jigsaw_base_patch168_336(
    mask_ratio=0.0, use_jigsaw=True, pretrained=False, **kwargs
):
    model = JigsawVisionTransformer(
        mask_ratio=mask_ratio,
        use_jigsaw=use_jigsaw,
        img_size=336,
        patch_size=168,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def jigsaw_tiny_patch168_336(
    mask_ratio=0.0, use_jigsaw=True, pretrained=False, **kwargs
):
    model = JigsawVisionTransformer(
        mask_ratio=mask_ratio,
        use_jigsaw=use_jigsaw,
        img_size=336,
        patch_size=168,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


if __name__ == "__main__":
    net = jigsaw_base_patch16_224(mask_ratio=0.5, use_jigsaw=True, pretrained=False)
    net = net.cuda()
    img = torch.cuda.FloatTensor(6, 3, 224, 224)
    with torch.no_grad():
        outs = net(img)
