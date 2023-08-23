# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Implementation of K-Planes (https://sarafridov.github.io/K-Planes/).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes,
                                         TrainingCallbackLocation)
from nerfstudio.field_components.encodings import (KPlanesEncoding,
                                                   TriplaneEncoding)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.eg3d_field import Eg3dField
from nerfstudio.fields.kplanes_importance_field import KPlanesImportanceField
from nerfstudio.model_components.losses import (MSELoss, distortion_loss,
                                                interlevel_loss)
from nerfstudio.model_components.ray_samplers import (
    PDFSampler, ProposalNetworkSampler, UniformLinDispPiecewiseSampler,
    UniformSampler)
from nerfstudio.model_components.renderers import (AccumulationRenderer,
                                                   DepthRenderer, RGBRenderer)
from nerfstudio.model_components.scene_colliders import (AABBBoxCollider,
                                                         NearFarCollider)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc


@dataclass
class Eg3dModelConfig(ModelConfig):
    """K-Planes Model Config"""

    _target: Type = field(default_factory=lambda: Eg3dModel)

    near_plane: float = 2.0
    """How far along the ray to start sampling."""

    far_plane: float = 6.0
    """How far along the ray to stop sampling."""

    grid_base_resolution: int = 256
    """Base grid resolution."""

    grid_feature_dim: int = 32
    """Dimension of feature vectors stored in grid."""

    is_contracted: bool = False
    """Whether to use scene contraction (set to true for unbounded scenes)."""

    linear_decoder: bool = False
    """Whether to use a linear decoder instead of an MLP."""

    linear_decoder_layers: Optional[int] = 1
    """Number of layers in linear decoder"""

    num_importance_samples: Optional[int] = 512 
    """Number of samples per ray for each proposal network."""

    num_samples: Optional[int] = 512
    """Number of samples per ray used for rendering."""

    single_jitter: bool = False
    """Whether use single jitter or not for the proposal networks."""

    appearance_embedding_dim: int = 0
    """Dimension of appearance embedding. Set to 0 to disable."""

    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""

    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """The background color as RGB."""

    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {
            "rgb": 1.0,
            "rgb_coarse": 1.0,
            "plane_tv": 0.01,
            'distortion_coarse': 0.01,
            'distortion_fine': 0.01,
        }
    )
    """Loss coefficients."""
    use_viewdirs: bool = True
    """whether to use viewdirs to rgb net"""
    use_tcnn: bool = True 


class Eg3dModel(Model):
    config: Eg3dModelConfig
    """K-Planes model with PDF sampler.

    Args:
        config: K-Planes configuration to instantiate model
    """

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.is_contracted:
            scene_contraction = SceneContraction(order=float("inf"))
        else:
            scene_contraction = None

        self.config.grid_base_resolution = [self.config.grid_base_resolution] * 3
        # Fields
        # self.field = KPlanesImportanceField(
        #     self.scene_box.aabb,
        #     num_images=self.num_train_data,
        #     grid_base_resolution=self.config.grid_base_resolution,
        #     grid_feature_dim=self.config.grid_feature_dim,
        #     concat_across_scales=True,
        #     multiscale_res=[1],
        #     spatial_distortion=scene_contraction,
        #     appearance_embedding_dim=self.config.appearance_embedding_dim,
        #     use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        #     linear_decoder=self.config.linear_decoder,
        #     linear_decoder_layers=self.config.linear_decoder_layers,
        #     reduce='sum',
        #     use_viewdirs=self.config.use_viewdirs
        # )

        self.triplane_encoding = TriplaneEncoding(
            resolution=self.config.grid_base_resolution[0],
            num_components=self.config.grid_feature_dim,
        )
        # self.triplane_encoding = KPlanesEncoding(self.config.grid_base_resolution, num_components=self.config.grid_feature_dim, reduce='sum')

        self.field = Eg3dField(
            self.scene_box.aabb,
            feature_encoding=self.triplane_encoding,
            appearance_dim=self.config.grid_feature_dim,
            decoder_hidden_dim=64,
            decoder_output_dim=3,
            use_viewdirs=self.config.use_viewdirs,
            use_tcnn=self.config.use_tcnn
        ) 

        if self.config.is_contracted:
            self.initial_sampler = UniformLinDispPiecewiseSampler(num_samples=self.config.num_samples, single_jitter=self.config.single_jitter)
        else:
            self.initial_sampler = UniformSampler(num_samples=self.config.num_samples, single_jitter=self.config.single_jitter)

        # Using PDF sampler
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples, single_jitter=self.config.single_jitter, include_original=False)

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
        # self.collider = AABBBoxCollider(scene_box=self.scene_box)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.temporal_distortion = len(self.config.grid_base_resolution) == 4  # for viewer

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {
            "fields": list(self.field.parameters()),
        }
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):

        ray_samples_uniform = self.initial_sampler(ray_bundle)
        # dens, _ = self.field.get_density(ray_samples_uniform)
        field_outputs_coarse = self.field(ray_samples_uniform)
        density_coarse = field_outputs_coarse[FieldHeadNames.DENSITY]
        weights_coarse = ray_samples_uniform.get_weights(density_coarse)
        rgb_coarse = self.renderer_rgb(rgb=field_outputs_coarse[FieldHeadNames.RGB], weights=weights_coarse)
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)
        # coarse_accumulation = self.renderer_accumulation(weights_fine)
        # acc_mask = torch.where(coarse_accumulation < 0.0001, False, True).reshape(-1)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

        field_outputs_fine = self.field(ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])

        rgb = self.renderer_rgb(rgb=field_outputs_fine[FieldHeadNames.RGB], weights=weights_fine)
        depth = self.renderer_depth(weights=weights_fine, ray_samples=ray_samples_pdf)
        accumulation = self.renderer_accumulation(weights=weights_fine)


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
            # field_grids = [g.plane_coefs for g in self.field.grids]
            # field_grids = [g.plane_coefs for g in [self.field.feature_encoding]]
            field_grids = [g.plane_coef for g in [self.field.feature_encoding]]

            metrics_dict["plane_tv"] = space_tv_loss(field_grids)
        #     metrics_dict["plane_tv_proposal_net"] = space_tv_loss(prop_grids)

        #     if len(self.config.grid_base_resolution) == 4:
        #         metrics_dict["l1_time_planes"] = l1_time_planes(field_grids)
        #         metrics_dict["l1_time_planes_proposal_net"] = l1_time_planes(prop_grids)
        #         metrics_dict["time_smoothness"] = time_smoothness(field_grids)
        #         metrics_dict["time_smoothness_proposal_net"] = time_smoothness(prop_grids)

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)

        loss_dict = {}
        loss_dict['rgb'] = self.rgb_loss(image, outputs['rgb'])
        loss_dict['rgb_coarse'] = self.rgb_loss(image, outputs['rgb_coarse'])
        if self.training:
            for key in self.config.loss_coefficients:
                if key in metrics_dict:
                    loss_dict[key] = metrics_dict[key].clone()

            loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)

        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(outputs["depth"], accumulation=outputs["accumulation"])

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        # all of these metrics will be logged as scalars
        metrics_dict = {
            "psnr": float(self.psnr(image, rgb).item()),
            "ssim": float(self.ssim(image, rgb)),
            "lpips": float(self.lpips(image, rgb))
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        # for i in range(self.config.num_proposal_iterations):
        #     key = f"prop_depth_{i}"
        #     prop_depth_i = colormaps.apply_depth_colormap(
        #         outputs[key],
        #         accumulation=outputs["accumulation"],
        #     )
        #     images_dict[key] = prop_depth_i

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


def l1_time_planes(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes the L1 distance from the multiplicative identity (1) for spatiotemporal planes.

    Args:
        multi_res_grids: Grids to compute L1 distance over

    Returns:
         L1 distance from the multiplicative identity (1)
    """
    time_planes = [2, 4, 5]  # These are the spatiotemporal planes
    total = 0.0
    num_planes = 0
    for grids in multi_res_grids:
        for grid_id in time_planes:
            total += torch.abs(1 - grids[grid_id]).mean()
            num_planes += 1

    return total / num_planes


def compute_plane_smoothness(t: torch.Tensor) -> float:
    """Computes smoothness across the temporal axis of a plane

    Args:
        t: Plane tensor

    Returns:
        Time smoothness
    """
    _, h, _ = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., : h - 1, :]  # [c, h-1, w]
    second_difference = first_difference[..., 1:, :] - first_difference[..., : h - 2, :]  # [c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()


def time_smoothness(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes smoothness across each time plane in the grids.

    Args:
        multi_res_grids: Grids to compute time smoothness over

    Returns:
        Time smoothness
    """
    total = 0.0
    num_planes = 0
    for grids in multi_res_grids:
        time_planes = [2, 4, 5]  # These are the spatiotemporal planes
        for grid_id in time_planes:
            total += compute_plane_smoothness(grids[grid_id])
            num_planes += 1

    return total / num_planes
