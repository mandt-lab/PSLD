# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# pylint: skip-file

from . import utils, layers, layerspp, normalization
from util import register_module
import torch.nn as nn
import functools
import torch
import numpy as np

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@register_module(category="clf_fn", name="ncsnpp_clf")
class NCSNppClassifier(nn.Module):
    """NCSN++ Classifier model for Class conditional generation. Same as the NCSN++ but removes the upsampling stage"""

    def __init__(self, config):
        super().__init__()
        self.config = config.model
        self.act = act = get_act(self.config.clf_fn)

        self.nf = nf = self.config.clf_fn.nf
        ch_mult = self.config.clf_fn.ch_mult
        self.num_res_blocks = num_res_blocks = self.config.clf_fn.num_res_blocks
        self.attn_resolutions = attn_resolutions = self.config.clf_fn.attn_resolutions
        dropout = self.config.clf_fn.dropout
        resamp_with_conv = self.config.clf_fn.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [
            config.data.image_size // (2**i) for i in range(num_resolutions)
        ]

        self.noise_cond = (
            noise_cond
        ) = self.config.clf_fn.noise_cond  # noise-conditional
        fir = self.config.clf_fn.fir
        fir_kernel = self.config.clf_fn.fir_kernel
        self.skip_rescale = skip_rescale = self.config.clf_fn.skip_rescale
        self.resblock_type = resblock_type = self.config.clf_fn.resblock_type.lower()
        self.progressive = progressive = self.config.clf_fn.progressive.lower()
        self.progressive_input = (
            progressive_input
        ) = self.config.clf_fn.progressive_input.lower()
        self.embedding_type = embedding_type = self.config.clf_fn.embedding_type.lower()
        init_scale = self.config.clf_fn.init_scale
        assert progressive in ["none", "output_skip", "residual"]
        assert progressive_input in ["none", "input_skip", "residual"]
        assert embedding_type in ["fourier", "positional"]
        combine_method = self.config.clf_fn.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        modules = []
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            assert (
                config.training.continuous
            ), "Fourier features are only used for continuous training."

            modules.append(
                layerspp.GaussianFourierProjection(
                    embedding_size=nf, scale=self.config.clf_fn.fourier_scale
                )
            )
            embed_dim = 2 * nf

        elif embedding_type == "positional":
            embed_dim = nf

        else:
            raise ValueError(f"embedding type {embedding_type} unknown.")

        if noise_cond:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(
            layerspp.AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale
        )

        Downsample = functools.partial(
            layerspp.Downsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if progressive_input == "input_skip":
            self.pyramid_downsample = layerspp.Downsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive_input == "residual":
            pyramid_downsample = functools.partial(
                layerspp.Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        if resblock_type == "ddpm":
            ResnetBlock = functools.partial(
                ResnetBlockDDPM,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )

        elif resblock_type == "biggan":
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )

        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")

        # Downsampling block

        channels = self.config.clf_fn.in_ch
        if progressive_input != "none":
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == "input_skip":
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == "cat":
                        in_ch *= 2

                elif progressive_input == "residual":
                    modules.append(
                        pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch)
                    )
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        # Construct classifier
        self.n_cls = config.model.clf_fn.n_cls
        last_res = all_resolutions[-1]
        modules.append(nn.Linear(in_ch * last_res**2, self.n_cls, bias=False))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond):
        # TODO: Add label and other forms of conditioning here!
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0
        if self.embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif self.embedding_type == "positional":
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            # used_sigmas = self.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f"embedding type {self.embedding_type} unknown.")

        if self.noise_cond:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != "none":
            input_pyramid = x

        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                if self.progressive_input == "input_skip":
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == "residual":
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.0)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        # Apply Classifier
        h = torch.flatten(h, start_dim=1)
        h = modules[m_idx](h)

        assert h.shape[-1] == self.n_cls
        return h
