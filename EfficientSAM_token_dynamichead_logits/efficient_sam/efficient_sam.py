# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, List, Tuple, Type

import torch
import torch.nn.functional as F

from torch import nn, Tensor

from .efficient_sam_decoder import MaskDecoder, PromptEncoder
from .efficient_sam_encoder import ImageEncoderViT
from .two_way_transformer import TwoWayAttentionBlock, TwoWayTransformer


in_place = True
class EfficientSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        decoder_max_num_input_points: int,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [0.485, 0.456, 0.406],
        pixel_std: List[float] = [0.229, 0.224, 0.225],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.decoder_max_num_input_points = decoder_max_num_input_points
        self.mask_decoder = mask_decoder
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(1, 3, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(pixel_std).view(1, 3, 1, 1), False
        )

        self.cls_token_MLP = nn.Conv2d(384,256,kernel_size=1, stride=1, padding=0)
        self.sls_token_MLP = nn.Conv2d(384,256,kernel_size=1, stride=1, padding=0)


        self.fusionConv = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            # conv3x3x3(256, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0), weight_std=self.weight_std)
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=1, padding=(0, 0), dilation=1, bias=False)
        )

        self.GAP = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )

        self.controller = nn.Conv2d(256 + 384 + 384, 162, kernel_size=1, stride=1, padding=0)
        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=in_place),
            # nn.Conv3d(32, 8, kernel_size=1)
            nn.Conv2d(32, 8, kernel_size=(1, 1))
        )

        self.upsamplex2 = nn.Upsample(scale_factor=(2, 2))
        self.x1_resb = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1))
        self.x2_resb = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1))

    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1), dilation=1, multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(16, inplanes),
                nn.ReLU(inplace=in_place),
                # conv3x3x3(inplanes, planes, kernel_size=(1, 1, 1), stride=stride, padding=0,
                #           weight_std=self.weight_std),
                nn.Conv2d(inplanes, planes, kernel_size=(1, 1), stride=stride, padding=(0, 0), dilation=1,
                          bias=False)
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=False))
        # self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=False))

        return nn.Sequential(*layers)

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 2, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 2)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x


    @torch.jit.export
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        # batched_points: torch.Tensor,
        # batched_point_labels: torch.Tensor,
        cls_token: torch.Tensor,
        sls_token: torch.Tensor,
        multimask_output: bool,
        input_h: int,
        input_w: int,
        output_h: int = -1,
        output_w: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks given image embeddings and prompts. This only runs the decoder.

        Arguments:
          image_embeddings: A tensor of shape [B, C, H, W] or [B*max_num_queries, C, H, W]
          batched_points: A tensor of shape [B, max_num_queries, num_pts, 2]
          batched_point_labels: A tensor of shape [B, max_num_queries, num_pts]
        Returns:
          A tuple of two tensors:
            low_res_mask: A tensor of shape [B, max_num_queries, 256, 256] of predicted masks
            iou_predictions: A tensor of shape [B, max_num_queries] of estimated IOU scores
        """

        # batch_size, max_num_queries, num_pts, _ = batched_points.shape
        # num_pts = batched_points.shape[2]
        # rescaled_batched_points = self.get_rescaled_pts(batched_points, input_h, input_w)
        #
        # if num_pts > self.decoder_max_num_input_points:
        #     rescaled_batched_points = rescaled_batched_points[
        #         :, :, : self.decoder_max_num_input_points, :
        #     ]
        #     batched_point_labels = batched_point_labels[
        #         :, :, : self.decoder_max_num_input_points
        #     ]
        # elif num_pts < self.decoder_max_num_input_points:
        #     rescaled_batched_points = F.pad(
        #         rescaled_batched_points,
        #         (0, 0, 0, self.decoder_max_num_input_points - num_pts),
        #         value=-1.0,
        #     )
        #     batched_point_labels = F.pad(
        #         batched_point_labels,
        #         (0, self.decoder_max_num_input_points - num_pts),
        #         value=-1.0,
        #     )

        # sparse_embeddings = self.prompt_encoder(
        #     rescaled_batched_points.reshape(
        #         batch_size * max_num_queries, self.decoder_max_num_input_points, 2
        #     ),
        #     batched_point_labels.reshape(
        #         batch_size * max_num_queries, self.decoder_max_num_input_points
        #     ),
        # )
        #
        # sparse_embeddings = sparse_embeddings.view(
        #     batch_size,
        #     max_num_queries,
        #     sparse_embeddings.shape[1],
        #     sparse_embeddings.shape[2],
        # )

        x = self.fusionConv(image_embeddings)
        x_feat = self.GAP(x)
        x_cond = torch.cat([x_feat, cls_token.unsqueeze(0).permute([0, 3, 1, 2]).repeat(x_feat.shape[0], 1, 1, 1), sls_token.unsqueeze(0).permute([0, 3, 1, 2]).repeat(x_feat.shape[0], 1, 1, 1)], 1)
        params = self.controller(x_cond)
        params.squeeze_(-1).squeeze_(-1)

        cls_emb = self.cls_token_MLP(cls_token.unsqueeze(0).permute([0,3,1,2])).permute([0,2,3,1]).squeeze(1)
        sls_emb = self.sls_token_MLP(sls_token.unsqueeze(0).permute([0,3,1,2])).permute([0,2,3,1]).squeeze(1)

        toekn_embeddings = torch.cat([cls_emb, sls_emb], 1)

        _, upscaled_embedding, iou_predictions = self.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=toekn_embeddings,
            multimask_output=False,
        )

        x = self.upsamplex2(upscaled_embedding)
        x = self.x2_resb(x)
        x = self.upsamplex2(x)
        x = self.x1_resb(x)

        head_inputs = self.precls_conv(x)
        N, _, H, W = head_inputs.size()
        head_inputs = head_inputs.reshape(1, -1, H, W)

        weight_nums, bias_nums = [], []
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 2)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(2)
        weights, biases = self.parse_dynamic_params(params, 8, weight_nums, bias_nums)

        logits = self.heads_forward(head_inputs, weights, biases, N)
        logits = logits.reshape(-1, 2, H, W)

        # softmax_fuc = torch.nn.Softmax(1)
		#
        # low_res_masks = softmax_fuc(logits)[:,1:2,:,:]
		#
        # _, num_predictions, low_res_size, _ = low_res_masks.shape
		#
        # batch_size = low_res_masks.shape[0]
        # max_num_queries = 1
        # if output_w > 0 and output_h > 0:
        #     output_masks = F.interpolate(
        #         low_res_masks, (output_h, output_w), mode="bicubic"
        #     )
        #     output_masks = torch.reshape(
        #         output_masks,
        #         (batch_size, max_num_queries, num_predictions, output_h, output_w),
        #     )
        # else:
        #     output_masks = torch.reshape(
        #         low_res_masks,
        #         (
        #             batch_size,
        #             max_num_queries,
        #             num_predictions,
        #             low_res_size,
        #             low_res_size,
        #         ),
        #     )
        # iou_predictions = torch.reshape(
        #     iou_predictions, (batch_size, max_num_queries, num_predictions)
        # )
        # sorted_ids = torch.argsort(iou_predictions, dim=-1, descending=True)
        # iou_predictions = torch.take_along_dim(iou_predictions, sorted_ids, dim=2)
        # output_masks = torch.take_along_dim(
        #     output_masks, sorted_ids[..., None, None], dim=2
        # )
		#
        # binary_mask_final = torch.cat([1 - output_masks.squeeze(1), output_masks.squeeze(1)], 1)

        return logits#, iou_predictions

    def get_rescaled_pts(self, batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * self.image_encoder.img_size / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * self.image_encoder.img_size / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )

    @torch.jit.export
    def get_image_embeddings(self, batched_images, task_id, scale_id) -> torch.Tensor:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_images: A tensor of shape [B, 3, H, W]
        Returns:
          List of image embeddings each of of shape [B, C(i), H(i), W(i)].
          The last embedding corresponds to the final layer.
        """
        batched_images = self.preprocess(batched_images)
        return self.image_encoder(batched_images, task_id, scale_id)

    def forward(
        self,
        batched_images: torch.Tensor,
        task_id: torch.Tensor,
        scale_id: torch.Tensor,
        # batched_points: torch.Tensor,
        # batched_point_labels: torch.Tensor,
        scale_to_original_image_size: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_images: A tensor of shape [B, 3, H, W]
          batched_points: A tensor of shape [B, num_queries, max_num_pts, 2]
          batched_point_labels: A tensor of shape [B, num_queries, max_num_pts]

        Returns:
          A list tuples of two tensors where the ith element is by considering the first i+1 points.
            low_res_mask: A tensor of shape [B, 256, 256] of predicted masks
            iou_predictions: A tensor of shape [B, max_num_queries] of estimated IOU scores
        """
        batch_size, _, input_h, input_w = batched_images.shape
        image_embeddings, cls_token, sls_token = self.get_image_embeddings(batched_images, task_id, scale_id)
        return self.predict_masks(
            image_embeddings,
            # batched_points,
            # batched_point_labels,
            cls_token = cls_token,
            sls_token = sls_token,
            multimask_output=True,
            input_h=input_h,
            input_w=input_w,
            output_h=input_h if scale_to_original_image_size else -1,
            output_w=input_w if scale_to_original_image_size else -1,
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        if (
            x.shape[2] != self.image_encoder.img_size
            or x.shape[3] != self.image_encoder.img_size
        ):
            x = F.interpolate(
                x,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
            )
        return (x - self.pixel_mean) / self.pixel_std


def build_efficient_sam(encoder_patch_embed_dim, encoder_num_heads, task_num, scale_num, checkpoint=None):
    img_size = 512   #ori 1024
    encoder_patch_size = 16
    encoder_depth = 12
    encoder_mlp_ratio = 4.0
    encoder_neck_dims = [256, 256]
    decoder_max_num_input_points = 6
    decoder_transformer_depth = 2
    decoder_transformer_mlp_dim = 2048
    decoder_num_heads = 8
    decoder_upscaling_layer_dims = [64, 32]
    num_multimask_outputs = 3
    iou_head_depth = 3
    iou_head_hidden_dim = 256
    activation = "gelu"
    normalization_type = "layer_norm"
    normalize_before_activation = False

    assert activation == "relu" or activation == "gelu"
    if activation == "relu":
        activation_fn = nn.ReLU
    else:
        activation_fn = nn.GELU

    image_encoder = ImageEncoderViT(
        img_size=img_size,
        patch_size=encoder_patch_size,
        in_chans=3,
        patch_embed_dim=encoder_patch_embed_dim,
        normalization_type=normalization_type,
        depth=encoder_depth,
        num_heads=encoder_num_heads,
        mlp_ratio=encoder_mlp_ratio,
        neck_dims=encoder_neck_dims,
        act_layer=activation_fn,
	    task_num=task_num,
	    scale_num=scale_num,
    )

    image_embedding_size = image_encoder.image_embedding_size
    encoder_transformer_output_dim = image_encoder.transformer_output_dim

    sam = EfficientSam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=encoder_transformer_output_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(img_size, img_size),
        ),
        decoder_max_num_input_points=decoder_max_num_input_points,
        mask_decoder=MaskDecoder(
            transformer_dim=encoder_transformer_output_dim,
            transformer=TwoWayTransformer(
                depth=decoder_transformer_depth,
                embedding_dim=encoder_transformer_output_dim,
                num_heads=decoder_num_heads,
                mlp_dim=decoder_transformer_mlp_dim,
                activation=activation_fn,
                normalize_before_activation=normalize_before_activation,
            ),
            num_multimask_outputs=num_multimask_outputs,
            activation=activation_fn,
            normalization_type=normalization_type,
            normalize_before_activation=normalize_before_activation,
            iou_head_depth=iou_head_depth - 1,
            iou_head_hidden_dim=iou_head_hidden_dim,
            upscaling_layer_dims=decoder_upscaling_layer_dims,
        ),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        sam.load_state_dict(state_dict["model"], strict = False)
    return sam

class NoBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1, weight_std=False):
        super(NoBottleneck, self).__init__()
        self.weight_std = weight_std
        self.gn1 = nn.GroupNorm(16, inplanes)
        # self.conv1 = conv3x3x3(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=(1,1,1),
        #                         dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), stride=stride, padding=(1,1),dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=in_place)

        self.gn2 = nn.GroupNorm(16, planes)
        # self.conv2 = conv3x3x3(planes, planes, kernel_size=(3, 3, 3), stride=1, padding=(1,1,1),
        #                         dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=1, padding=(1,1),dilation=1, bias=False)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.gn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out
