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

"""TensoRF Field"""


from typing import Dict, Optional

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import (Encoding, Identity,
                                                   SHEncoding)
from nerfstudio.field_components.field_heads import (FieldHeadNames,
                                                     RGBFieldHead)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.base_field import Field


class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            raise ValueError(f'Unsupported activation function: {self.activation}, only linear activator is supported.')
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'


class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim, decoder_output_dim, decoder_geofeat_dim, use_viewdirs, use_tcnn):
        super().__init__()
        self.hidden_dim = hidden_dim 
        self.use_viewdirs = use_viewdirs
        self.decoder_geofeat_dim = decoder_geofeat_dim


        if self.use_viewdirs:
            if use_tcnn:
                self.sigma_net = tcnn.Network(
                    n_input_dims=n_features,
                    n_output_dims=self.decoder_geofeat_dim + 1,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": self.hidden_dim,
                        "n_hidden_layers": 1,
                    },
                )
                self.direction_encoder = tcnn.Encoding(
                        n_input_dims=3,
                        encoding_config={
                            "otype": "SphericalHarmonics",
                            "degree": 4,
                        },
                    ) 
                color_feature_dim = decoder_geofeat_dim + self.direction_encoder.n_output_dims 
                self.color_net = tcnn.Network(
                    n_input_dims=color_feature_dim,
                    n_output_dims=decoder_output_dim,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": 64,
                        "n_hidden_layers": 2,
                    },
                )
            else:
                print('not use tcnn')
                self.sigma_net = torch.nn.Sequential(
                    FullyConnectedLayer(n_features, self.hidden_dim),
                    # torch.nn.Softplus(),
                    torch.nn.ReLU(),
                    FullyConnectedLayer(self.hidden_dim, self.hidden_dim),
                    torch.nn.ReLU(),
                    FullyConnectedLayer(self.hidden_dim, self.decoder_geofeat_dim + 1), 
                ) 
                self.direction_encoder = SHEncoding(levels=4)
                color_feature_dim = decoder_geofeat_dim + self.direction_encoder.get_out_dim()
                self.color_net = torch.nn.Sequential(
                    FullyConnectedLayer(color_feature_dim, self.hidden_dim),
                    torch.nn.ReLU(),
                    FullyConnectedLayer(self.hidden_dim, self.hidden_dim),
                    torch.nn.ReLU(),
                    FullyConnectedLayer(self.hidden_dim, self.hidden_dim),
                    torch.nn.ReLU(),
                    FullyConnectedLayer(self.hidden_dim, decoder_output_dim), 
                )
        else:
            self.net = torch.nn.Sequential(
                FullyConnectedLayer(n_features, self.hidden_dim),
                torch.nn.Softplus(),
                FullyConnectedLayer(self.hidden_dim, 1 + decoder_output_dim),
            )

    def forward(self, x, ray_directions=None):
        # Aggregate features
        # x = x.mean(1)
        N, M, C = x.shape
        x = x.view(N*M, C)
        if self.use_viewdirs: 
            geo_features = self.sigma_net(x)
            geo_features, sigma = torch.split(geo_features, [self.decoder_geofeat_dim, 1], dim=-1)
            direction_enc = self.direction_encoder(ray_directions)
            color_features = [direction_enc, geo_features.view(-1, self.decoder_geofeat_dim)]
            color_features = torch.cat(color_features, dim=-1)
            rgb = self.color_net(color_features)
            rgb = rgb.view(N, M, -1)
            rgb = torch.sigmoid(rgb)*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
            sigma = sigma.view(N, M, -1)
        else:
            x = self.net(x)
            x = x.view(N, M, -1)
            rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
            sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}


class Eg3dField(Field):
    """Eg3d Field"""

    def __init__(
        self,
        aabb: Tensor,
        # the aabb bounding box of the dataset
        feature_encoding: Encoding = Identity(in_dim=3),
        # the encoding method used for appearance encoding outputs
        appearance_dim: int = 32,
        # the number of dimensions for the appearance embedding
        decoder_hidden_dim: int = 64,
        # the number of dimensions for the decoder hidden layer
        decoder_output_dim: int = 3,
        # the number of dimensions for the decoder output
        decoder_geofeat_dim: int = 15,
        # the number of dimensions for the decoder geo features
        use_viewdirs: bool = True,
        # whether to use view directions
        use_tcnn: bool = True,
        # whether to use tcnn
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.feature_encoding = feature_encoding
        self.decoder_hidden_dim = decoder_hidden_dim
        self.appearance_dim = appearance_dim
        self.decoder_output_dim = decoder_output_dim
        self.decoder_geofeat_dim = decoder_geofeat_dim
        self.use_viewdirs = use_viewdirs
        self.use_tcnn = use_tcnn

        # self.osg_decoder = torch.nn.Sequential(
        #     FullyConnectedLayer(self.appearance_dim, self.hidden_dim),
        #     torch.nn.Softplus(),
        #     FullyConnectedLayer(self.hidden_dim, 1 + self.decoder_output_dim)
        # )
        self.osg_decoder = OSGDecoder(self.appearance_dim, self.decoder_hidden_dim, self.decoder_output_dim, self.decoder_geofeat_dim, self.use_viewdirs, self.use_tcnn)


    def get_density(self, ray_samples: RaySamples) -> Tensor:
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1
        triplane_feature = self.feature_encoding(positions)
        directions = ray_samples.frustums.directions.reshape(-1, 3)
        out = self.osg_decoder(triplane_feature, directions)
        sigma = out['sigma']
        density = F.softplus(sigma - 1)
        # density = trunc_exp(sigma - 1)
        return density

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> Tensor:
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1
        triplane_features = self.feature_encoding(positions)
        directions = ray_samples.frustums.directions.reshape(-1, 3)
        out = self.osg_decoder(triplane_features, directions)
        rgb = out['rgb']
        return rgb

    def get_density_and_outputs(self, ray_samples: RaySamples) -> Dict:
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1
        triplane_features = self.feature_encoding(positions)
        directions = ray_samples.frustums.directions.reshape(-1, 3)
        out = self.osg_decoder(triplane_features, directions)
        rgb = out['rgb']
        sigma = out['sigma']
        # density = trunc_exp(sigma - 1)
        density = F.softplus(sigma - 1)
        return {'rgb': rgb, 'density': density}


    def forward(
        self,
        ray_samples: RaySamples,
        compute_normals: bool = False,
        mask: Optional[Tensor] = None,
        bg_color: Optional[Tensor] = None,
    ) -> Dict[FieldHeadNames, Tensor]:
        if compute_normals is True:
            raise ValueError("Surface normals are not currently supported with TensoRF")
        if mask is not None and bg_color is not None:
            base_density = torch.zeros(ray_samples.shape)[:, :, None].to(mask.device)
            base_rgb = bg_color.repeat(ray_samples[:, :, None].shape)
            if mask.any():
                input_rays = ray_samples[mask, :]
                # density = self.get_density(input_rays)
                # rgb = self.get_outputs(input_rays, None)
                out = self.get_density_and_outputs(input_rays)
                density = out['density']
                rgb = out['rgb']
                base_density[mask] = density
                base_rgb[mask] = rgb
                base_density.requires_grad_()
                base_rgb.requires_grad_()

            density = base_density
            rgb = base_rgb
        else:
            # density = self.get_density(ray_samples)
            # rgb = self.get_outputs(ray_samples, None)
            out = self.get_density_and_outputs(ray_samples)
            density = out['density']
            rgb = out['rgb']

        return {FieldHeadNames.DENSITY: density, FieldHeadNames.RGB: rgb}
