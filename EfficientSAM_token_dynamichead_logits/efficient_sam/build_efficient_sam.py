# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .efficient_sam import build_efficient_sam


def build_efficient_sam_vitt(task_num, scale_num):
    return build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
	    task_num=task_num,
	    scale_num=scale_num,
        checkpoint="/Data4/HATs/EfficientSAM_token_dynamichead_logits/weights/efficient_sam_vitt.pt",
    ).eval()


def build_efficient_sam_vits(task_num, scale_num):
    return build_efficient_sam(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
	    task_num=task_num,
	    scale_num=scale_num,
        checkpoint="/Data4/HATs/EfficientSAM_token_dynamichead_logits/weights/efficient_sam_vits.pt",
    ).eval()
