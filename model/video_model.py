# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
from torch import nn
import compressai
from .layers import SubpelConv2x, DepthConvBlock, ResidualBlockUpsample, ResidualBlockWithStride2
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

#qp_shift = [0, 8, 4]
#extra_qp = max(qp_shift)

g_ch_src_d = 8 * 8
g_ch_recon = 128
g_ch_y = 64
g_ch_z = 64
g_ch_d = 128

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv2 = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )

    def forward(self, x, quant):
        x1, ctx_t = self.forward_part1(x, quant)
        ctx = self.forward_part2(x1)
        return ctx, ctx_t

    def forward_part1(self, x, quant):
        x1 = self.conv1(x)
        ctx_t = x1 * quant
        return x1, ctx_t

    def forward_part2(self, x1):
        ctx = self.conv2(x1)
        return ctx


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_src_d, g_ch_d, 1)
        self.conv2 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv3 = DepthConvBlock(g_ch_d, g_ch_d)
        self.down = nn.Conv2d(g_ch_d, g_ch_y, 3, stride=2, padding=1)

        self.fuse_conv1_flag = False

    def forward(self, x, ctx, quant_step):
        feature = F.pixel_unshuffle(x, 8)
        return self.forward_torch(feature, ctx, quant_step)


    def forward_torch(self, feature, ctx, quant_step):
        feature = self.conv1(feature)
        feature = self.conv2(torch.cat((feature, ctx), dim=1))
        feature = self.conv3(feature)
        feature = feature * quant_step
        feature = self.down(feature)
        return feature

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = SubpelConv2x(g_ch_y, g_ch_d, 3, padding=1)
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv2 = nn.Conv2d(g_ch_d, g_ch_d, 1)

    def forward(self, x, ctx, quant_step,):
        return self.forward_torch(x, ctx, quant_step)
    def forward_torch(self, x, ctx, quant_step):
        feature = self.up(x)
        feature = self.conv1(torch.cat((feature, ctx), dim=1))
        feature = self.conv2(feature)
        feature = feature * quant_step
        return feature


class ReconGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_d,     g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
        )
        self.head = nn.Conv2d(g_ch_recon, g_ch_src_d, 1)

    def forward(self, x, quant_step):
        return self.forward_torch(x, quant_step)

    def forward_torch(self, x, quant_step):
        out = self.conv(x)
        out = out * quant_step
        out = self.head(out)
        out = F.pixel_shuffle(out, 8)
        out = torch.clamp(out, 0., 1.)
        return out



class HyperEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y, g_ch_z),
            ResidualBlockWithStride2(g_ch_z, g_ch_z),
            ResidualBlockWithStride2(g_ch_z, g_ch_z),
        )

    def forward(self, x):
        return self.conv(x)


class HyperDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlockUpsample(g_ch_z, g_ch_z),
            ResidualBlockUpsample(g_ch_z, g_ch_z),
            DepthConvBlock(g_ch_z, g_ch_y),
        )

    def forward(self, x):
        return self.conv(x)


class PriorFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            nn.Conv2d(g_ch_y * 3, g_ch_y, 1),
        )

    def forward(self, x):
        return self.conv(x)

class SpatialPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 4, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            nn.Conv2d(g_ch_y * 3, g_ch_y * 2, 1),
        )

    def forward(self, x):
        return self.conv(x)


class RefFrame():
    def __init__(self):
        self.frame = None
        self.feature = None
        self.poc = None

def get_padding_size(height, width, p=64):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    padding_right = new_w - width
    padding_bottom = new_h - height
    return padding_right, padding_bottom
def pad_for_y(y):
    _, _, H, W = y.size()
    padding_r, padding_b = get_padding_size(H, W, 4)
    y_pad = F.pad(y, (0, padding_r, 0, padding_b), mode="replicate")
    
    return y_pad
    
class DMC(nn.Module):
    def __init__(self):
        super().__init__()
        self.z_channel=g_ch_z
        self.entropy_bottleneck = EntropyBottleneck(self.z_channel)
        self.gaussian_conditional = GaussianConditional(None)
        self.feature_adaptor_i = DepthConvBlock(g_ch_src_d, g_ch_d)
        self.feature_adaptor_p = nn.Conv2d(g_ch_d, g_ch_d, 1)
        self.feature_extractor = FeatureExtractor()

        self.encoder = Encoder()
        self.hyper_encoder = HyperEncoder()
        self.hyper_decoder = HyperDecoder()
        self.temporal_prior_encoder = ResidualBlockWithStride2(g_ch_d, g_ch_y * 2)
        self.y_prior_fusion = PriorFusion()
        self.y_spatial_prior = SpatialPrior()
        self.decoder = Decoder()
        self.recon_generation_net = ReconGeneration()

        self.q_encoder = nn.Parameter(torch.ones((64, g_ch_d, 1, 1)))
        self.q_decoder = nn.Parameter(torch.ones((64, g_ch_d, 1, 1)))
        self.q_feature = nn.Parameter(torch.ones((64, g_ch_d, 1, 1)))
        self.q_recon = nn.Parameter(torch.ones((64, g_ch_recon, 1, 1)))

        self.ref_frame = None
        self.curr_poc = 0


    def feature_i(self,frame):

        ttt=F.pixel_unshuffle(frame, 8)

        return self.feature_adaptor_i(ttt)

    def feature_p(self,feature):
        return self.feature_adaptor_p(feature)

    def res_prior_param_decoder(self, z_hat, ctx_t):
        hierarchical_params = self.hyper_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(ctx_t)
        _, _, H, W = temporal_params.shape
        hierarchical_params = hierarchical_params[:, :, :H, :W].contiguous()
        params = self.y_prior_fusion(torch.cat((hierarchical_params, temporal_params), dim=1))
        return params

    def forward(self, x, recon, p_feature, qp):

        q_encoder = self.q_encoder[qp:qp+1, :, :, :]
        q_decoder = self.q_decoder[qp:qp+1, :, :, :]
        q_feature = self.q_feature[qp:qp+1, :, :, :]
        q_recon = self.q_recon[qp:qp+1, :, :, :]

        if recon is None:
            f_feature=self.feature_p(p_feature)
        else:
            f_feature=self.feature_i(recon)

        ctx, ctx_t = self.feature_extractor(f_feature, q_feature)

        y = self.encoder(x, ctx, q_encoder)

        hyper_inp = pad_for_y(y)

        z = self.hyper_encoder(hyper_inp)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        params = self.res_prior_param_decoder(z_hat, ctx_t)

        _, _, yH, yW = y.shape
        params = params[:, :, :yH, :yW].contiguous()

        y_hat, y_likelihoods = self.gaussian_conditional(y, params)

        feature = self.decoder(y_hat, ctx, q_decoder)
        x_hat = self.recon_generation_net(feature, q_recon)#.clamp_(0, 1)

        return x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods, feature

    def get_i_feature(self, recon):
        return self.feature_i(recon)
    def get_p_feature(self, p_feature):
        return self.feature_p(p_feature)

    def compress(self, x, f_feature, qp):
        q_encoder = self.q_encoder[qp:qp+1, :, :, :]
        q_decoder = self.q_decoder[qp:qp+1, :, :, :]
        q_feature = self.q_feature[qp:qp+1, :, :, :]
        q_recon = self.q_recon[qp:qp+1, :, :, :]

        ctx, ctx_t = self.feature_extractor(f_feature, q_feature)
        y = self.encoder(x, ctx, q_encoder)
        hyper_inp = pad_for_y(y)
        z = self.hyper_encoder(hyper_inp)

        z_strings = self.entropy_bottleneck.compress(z)

        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.res_prior_param_decoder(z_hat, ctx_t)

        _, _, yH, yW = y.shape
        params = params[:, :, :yH, :yW].contiguous()
        indexes = self.gaussian_conditional.build_indexes(params)
        
        y_strings = self.gaussian_conditional.compress(y, indexes)
        
        #y_hat, y_likelihoods = self.gaussian_conditional(y, params)
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, z_hat.dtype)

        feature = self.decoder(y_hat, ctx, q_decoder)

        return y_strings, z_strings, feature, z.size()[-2:], y.shape


    def decompress(self, y_strings, z_strings, f_feature, zshape, yshape, qp):

        q_encoder = self.q_encoder[qp:qp+1, :, :, :]
        q_decoder = self.q_decoder[qp:qp+1, :, :, :]
        q_feature = self.q_feature[qp:qp+1, :, :, :]
        q_recon = self.q_recon[qp:qp+1, :, :, :]

        ctx, ctx_t = self.feature_extractor(f_feature, q_feature)

        z_hat = self.entropy_bottleneck.decompress(z_strings, zshape)

        params = self.res_prior_param_decoder(z_hat, ctx_t)
        
        _, _, yH, yW = yshape
        params = params[:, :, :yH, :yW].contiguous()
        
        indexes = self.gaussian_conditional.build_indexes(params)

        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, z_hat.dtype)
        
        feature = self.decoder(y_hat, ctx, q_decoder)
        x_hat = self.recon_generation_net(feature, q_recon)#.clamp_(0, 1)

        return x_hat, feature
