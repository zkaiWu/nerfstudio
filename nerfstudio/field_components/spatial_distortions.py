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

"""Space distortions."""

import abc
from typing import Optional, Union

import torch
from functorch import jacrev, vmap
from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.utils.math import Gaussians


class SpatialDistortion(nn.Module):
    """Apply spatial distortions"""

    @abc.abstractmethod
    def forward(self, positions: Union[Float[Tensor, "*bs 3"], Gaussians]) -> Union[Float[Tensor, "*bs 3"], Gaussians]:
        """
        Args:
            positions: Sample to distort

        Returns:
            Union: distorted sample
        """


class SceneContraction(SpatialDistortion):
    """Contract unbounded space using the contraction was proposed in MipNeRF-360.
        We use the following contraction equation:

        .. math::

            f(x) = \\begin{cases}
                x & ||x|| \\leq 1 \\\\
                (2 - \\frac{1}{||x||})(\\frac{x}{||x||}) & ||x|| > 1
            \\end{cases}

        If the order is not specified, we use the Frobenius norm, this will contract the space to a sphere of
        radius 2. If the order is L_inf (order=float("inf")), we will contract the space to a cube of side length 4.
        If using voxel based encodings such as the Hash encoder, we recommend using the L_inf norm.

        Args:
            order: Order of the norm. Default to the Frobenius norm. Must be set to None for Gaussians.

    """

    def __init__(self, order: Optional[Union[float, int]] = None) -> None:
        super().__init__()
        self.order = order

    def forward(self, positions):
        def contract(x):
            mag = torch.linalg.norm(x, ord=self.order, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

        if isinstance(positions, Gaussians):
            means = contract(positions.mean.clone())

            def contract_gauss(x):
                return (2 - 1 / torch.linalg.norm(x, ord=self.order, dim=-1, keepdim=True)) * (
                    x / torch.linalg.norm(x, ord=self.order, dim=-1, keepdim=True)
                )

            jc_means = vmap(jacrev(contract_gauss))(positions.mean.view(-1, positions.mean.shape[-1]))
            jc_means = jc_means.view(list(positions.mean.shape) + [positions.mean.shape[-1]])

            # Only update covariances on positions outside the unit sphere
            mag = positions.mean.norm(dim=-1)
            mask = mag >= 1
            cov = positions.cov.clone()
            cov[mask] = jc_means[mask] @ positions.cov[mask] @ torch.transpose(jc_means[mask], -2, -1)

            return Gaussians(mean=means, cov=cov)

        return contract(positions)


class NDC(SpatialDistortion):
    """

        Use NDC Mapping, according to H, W, focal, near plane, and coordinate

    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, height: Tensor, width: Tensor, f_x: Tensor, f_y: Tensor, ray_bundle: RayBundle) -> RayBundle:

        near = 1.0
        ori = ray_bundle.origins
        dir = ray_bundle.directions

        # height_bundle = height[camera_idx[..., 0]]
        # width_bundle = width[camera_idx[..., 0]]
        # f_x_bundle = f_x[camera_idx[..., 0]]
        # f_y_bundle = f_y[camera_idx[..., 0]]

        height_bundle = height.unsqueeze(0).repeat(ori.shape[0], 1)
        width_bundle = width.unsqueeze(0).repeat(ori.shape[0], 1)
        f_x_bundle = f_x.unsqueeze(0).repeat(ori.shape[0], 1)
        f_y_bundle = f_y.unsqueeze(0).repeat(ori.shape[0], 1)

        # Shift ray origins to near plane
        t = -(near + ori[..., 2:3]) / dir[..., 2:3]
        # ori = ori + t[...,None] * dir 
        ori = ori + t * dir
        
        # Projection
        o0 = -1./(width_bundle/(2.*f_x_bundle)) * ori[..., 0:1] / ori[..., 2:3]
        o1 = -1./(height_bundle/(2.*f_y_bundle)) * ori[..., 1:2] / ori[..., 2:3]
        o2 = 1. + 2. * near / ori[..., 2:3]

        d0 = -1./(width_bundle/(2.*f_x_bundle)) * (dir[..., 0:1]/dir[..., 2:3] - ori[..., 0:1]/ori[..., 2:3])
        d1 = -1./(height_bundle/(2.*f_y_bundle)) * (dir[..., 1:2]/dir[..., 2:3] - ori[..., 1:2]/ori[..., 2:3])
        d2 = -2. * near / ori[..., 2:3]
        
        ori = torch.cat([o0,o1,o2], -1)
        dir = torch.cat([d0,d1,d2], -1)

        ray_bundle.origins = ori
        ray_bundle.directions = dir

        ray_bundle.nears = 0.0 * torch.ones((ray_bundle.origins.shape[0], 1), device=ray_bundle.origins.device)
        ray_bundle.fars = 1.0 * torch.ones((ray_bundle.origins.shape[0], 1), device=ray_bundle.origins.device)

        return ray_bundle 