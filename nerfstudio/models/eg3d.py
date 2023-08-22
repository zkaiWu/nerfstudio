# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TensorRF implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type, cast

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes,
                                         TrainingCallbackLocation)
from nerfstudio.field_components.encodings import (NeRFEncoding,
                                                   TensorCPEncoding,
                                                   TensorVMEncoding,
                                                   TriplaneEncoding)
from nerfstudio.field_components.field_heads import FieldHeadNames
# from nerfstudio.fields.tensorf_field import TensoRFField
from nerfstudio.fields.eg3d_field import Eg3dField
from nerfstudio.model_components.losses import (MSELoss, distortion_loss,
                                                tv_loss)
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (AccumulationRenderer,
                                                   DepthRenderer, RGBRenderer)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc


@dataclass
class Eg3dModelConfig(ModelConfig):
    """TensoRF model config"""

    _target: Type = field(default_factory=lambda: Eg3dModel)
    """target class to instantiate"""
    triplane_resolution: int = 512 
    """triplane render resolution"""
    """specifies a list of iteration step numbers to perform upsampling"""
    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {
            "rgb": 1.0,
            "rgb_coarse": 1.0,
            "plane_tv": 0.0001,
            'distortion_coarse': 0.001,
            'distortion_fine': 0.001,
        }
    )
    """Loss specific weights."""
    num_samples: int = 256 
    """Number of samples in field evaluation"""
    num_uniform_samples: int = 512 
    """Number of samples in density evaluation"""
    appearance_dim: int = 48
    """Number of channels for triplane encoding"""
    decoder_hidden_dim: int = 128 
    """Number of hidden units for decoder"""
    decoder_output_dim: int = 48 
    """Number of channels for decoder output"""
    regularization: Literal["none", "l1", "tv"] = "l1"
    """Regularization method used in tensorf paper"""


class Eg3dModel(Model):
    """TensoRF Model

    Args:
        config: TensoRF configuration to instantiate model
    """

    config: Eg3dModelConfig

    def __init__(
        self,
        config: Eg3dModelConfig,
        **kwargs,
    ) -> None:
        self.triplane_resolution = config.triplane_resolution
        self.appearance_dim = config.appearance_dim
        self.decoder_hidden_dim = config.decoder_hidden_dim
        self.decoder_output_dim = config.decoder_output_dim
        super().__init__(config=config, **kwargs)


    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()
        # setting up fields

        self.triplane_encoding = TriplaneEncoding(
            resolution=self.triplane_resolution,
            num_components=self.appearance_dim,
        )

        self.field = Eg3dField(
            self.scene_box.aabb,
            feature_encoding=self.triplane_encoding,
            appearance_dim=self.appearance_dim,
            decoder_hidden_dim=self.decoder_hidden_dim,
            decoder_output_dim=self.decoder_output_dim,
        ) 

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_uniform_samples, single_jitter=True)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_samples, single_jitter=True, include_original=False)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # colliders
        if self.config.enable_collider:
            self.collider = AABBBoxCollider(scene_box=self.scene_box)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}

        # import pdb; pdb.set_trace()
        # for name, param in self.field.osg_decoder.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        param_groups["fields"] = (
            list(self.field.osg_decoder.parameters())
        )
        param_groups["encodings"] = list(self.field.feature_encoding.parameters()) + list(
            self.field.feature_encoding.parameters()
        )

        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        field_outputs_coarse = self.field(ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(rgb=field_outputs_coarse[FieldHeadNames.RGB], weights=weights_coarse)
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)
        # acc_mask = torch.where(coarse_accumulation < 0.0001, False, True).reshape(-1)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

        # fine field:
        field_outputs_fine = self.field(
            ray_samples_pdf
        )

        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])

        rgb = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        depth = self.renderer_depth(weights_fine, ray_samples_pdf)
        accumulation = self.renderer_accumulation(weights_fine)

        rgb = rgb[:, :3]
        rgb = torch.where(accumulation < 0, colors.WHITE.to(rgb.device), rgb)
        accumulation = torch.clamp(accumulation, min=0)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "rgb_coarse": rgb_coarse,
            "accumulation_coarse": accumulation_coarse,
            "depth_coarse": depth_coarse,
            "weights_coarse": weights_coarse,
            "weights_fine": weights_fine,
            "ray_sampels_coarse": ray_samples_uniform,
            "ray_sampels_fine": ray_samples_pdf,
        }
        return outputs

    def get_metrics_dict(self, outputs, batch):
        image = batch["image"].to(self.device)

        metrics_dict = {
            "psnr": self.psnr(outputs["rgb"], image)
        }
        if self.training:
        #     metrics_dict["interlevel"] = interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"])
            metrics_dict["distortion_coarse"] = distortion_loss([outputs["weights_coarse"]], outputs["ray_sampels_coarse"])
            metrics_dict["distortion_fine"] = distortion_loss([outputs["weights_fine"]], outputs["ray_sampels_fine"])

        #     prop_grids = [p.grids.plane_coefs for p in self.proposal_networks]
            field_grids = [g.plane_coef for g in [self.field.feature_encoding]]

            metrics_dict["plane_tv"] = space_tv_loss(field_grids)
        #     metrics_dict["plane_tv_proposal_net"] = space_tv_loss(prop_grids)

        #     if len(self.config.grid_base_resolution) == 4:
        #         metrics_dict["l1_time_planes"] = l1_time_planes(field_grids)
        #         metrics_dict["l1_time_planes_proposal_net"] = l1_time_planes(prop_grids)
        #         metrics_dict["time_smoothness"] = time_smoothness(field_grids)
        #         metrics_dict["time_smoothness_proposal_net"] = time_smoothness(prop_grids)

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        image = batch["image"].to(self.device)

        loss_dict = {}
        loss_dict['rgb'] = self.rgb_loss(image, outputs["rgb"])
        loss_dict['rgb_coarse'] = self.rgb_loss(image, outputs["rgb_coarse"])
        if self.training:
            for key in self.config.loss_coefficients:
                if key in metrics_dict:
                    loss_dict[key] = metrics_dict[key].clone()

            loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)

        return loss_dict



        # if self.config.regularization == "l1":
        #     l1_parameters = []
        #     for parameter in self.field.feature_encoding.parameters():
        #         l1_parameters.append(parameter.view(-1))
        #     loss_dict["l1_reg"] = torch.abs(torch.cat(l1_parameters)).mean()
        # elif self.config.regularization == "tv":
        #     density_plane_coef = self.field.density_encoding.plane_coef
        #     color_plane_coef = self.field.color_encoding.plane_coef
        #     assert isinstance(color_plane_coef, torch.Tensor) and isinstance(
        #         density_plane_coef, torch.Tensor
        #     ), "TV reg only supported for TensoRF encoding types with plane_coef attribute"
        #     loss_dict["tv_reg_density"] = tv_loss(density_plane_coef)
        #     loss_dict["tv_reg_color"] = tv_loss(color_plane_coef)
        # elif self.config.regularization == "none":
        #     pass
        # else:
        #     raise ValueError(f"Regularization {self.config.regularization} not supported")

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        # return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        assert self.config.collider_params is not None
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = cast(torch.Tensor, self.ssim(image, rgb))
        lpips = self.lpips(image, rgb)

        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "lpips": float(lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": acc, "depth": depth}
        return metrics_dict, images_dict


def compute_plane_tv(t: torch.Tensor, only_w: bool = False) -> float:
    """Computes total variance across a plane.

    Args:
        t: Plane tensor
        only_w: Whether to only compute total variance across w dimension

    Returns:
        Total variance
    """
    _, h, w = t.shape
    w_tv = torch.square(t[..., :, 1:] - t[..., :, : w - 1]).mean()

    if only_w:
        return w_tv

    h_tv = torch.square(t[..., 1:, :] - t[..., : h - 1, :]).mean()
    return h_tv + w_tv


def space_tv_loss(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes total variance across each spatial plane in the grids.

    Args:
        multi_res_grids: Grids to compute total variance over

    Returns:
        Total variance
    """

    total = 0.0
    num_planes = 0
    for grids in multi_res_grids:
        if len(grids) == 3:
            spatial_planes = {0, 1, 2}
        else:
            spatial_planes = {0, 1, 3}

        for grid_id, grid in enumerate(grids):
            if grid_id in spatial_planes:
                total += compute_plane_tv(grid)
            else:
                # Space is the last dimension for space-time planes.
                total += compute_plane_tv(grid, only_w=True)
            num_planes += 1
    return total / num_planes