"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            #norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False) #BORDE
        elif param_free_norm_type == 'syncbatch':
            #self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False) #no multi-cloud
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False) #SPADE
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        #self.mlp_shared = nn.Sequential(
        #    nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
        #    nn.ReLU()
        #)
        #self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        #self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class BORDE(nn.Module):
    def __init__(self, config_text, fin, fout, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        self.param_free_norm = nn.InstanceNorm2d(fin, affine=False) #BORDE

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 64 #in BORDE less than SPADE

        pw = ks // 2

        #boundary
        self.mlp_shared1 = nn.Sequential(
            nn.Conv2d(1, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma1 = nn.Conv2d(nhidden, fin, kernel_size=ks, padding=pw)
        self.mlp_beta1 = nn.Conv2d(nhidden, fin, kernel_size=ks, padding=pw)
        self.after1 = nn.Sequential(
            nn.Conv2d(fin, fout, kernel_size=1, padding=0),
            nn.ReLU()
        )

        #foreground
        self.mlp_shared2 = nn.Sequential(
            nn.Conv2d(1, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma2 = nn.Conv2d(nhidden, fout, kernel_size=ks, padding=pw)
        self.mlp_beta2 = nn.Conv2d(nhidden, fout, kernel_size=ks, padding=pw)
        self.after2 = nn.Sequential(
            nn.Conv2d(fout, fout, kernel_size=1, padding=0),
            nn.ReLU()
        )

        #edema
        self.mlp_shared3 = nn.Sequential(
            nn.Conv2d(1, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma3 = nn.Conv2d(nhidden, fout, kernel_size=ks, padding=pw)
        self.mlp_beta3 = nn.Conv2d(nhidden, fout, kernel_size=ks, padding=pw)
        self.after3 = nn.Sequential(
            nn.Conv2d(fout, fout, kernel_size=1, padding=0),
            nn.ReLU()
        )

        #tumorcore
        self.mlp_shared4 = nn.Sequential(
            nn.Conv2d(2, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma4 = nn.Conv2d(nhidden, fout, kernel_size=ks, padding=pw)
        self.mlp_beta4 = nn.Conv2d(nhidden, fout, kernel_size=ks, padding=pw)
        self.after4 = nn.Sequential(
            nn.Conv2d(fout, fout, kernel_size=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x, segmap):

        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')

        normalized = self.param_free_norm(x)
        #boundary
        actv1 = self.mlp_shared1(segmap[:,0,:,:].unsqueeze(1))
        gamma1 = self.mlp_gamma1(actv1)
        beta1 = self.mlp_beta1(actv1)
        out1 = normalized * (1 + gamma1) + beta1
        after = self.after1(out1)

        normalized = self.param_free_norm(after)
        #foreground
        actv2 = self.mlp_shared2(segmap[:,1,:,:].unsqueeze(1))
        gamma2 = self.mlp_gamma2(actv2)
        beta2 = self.mlp_beta2(actv2)
        out2 = normalized * (1 + gamma2) + beta2
        after = self.after2(out2)

        #edema
        if(1 == torch.unique(segmap[:,2,:,:]).size()[0]):
            normalized = self.param_free_norm(after)
            #foreground
            actv3 = self.mlp_shared3(segmap[:,2,:,:].unsqueeze(1))
            gamma3 = self.mlp_gamma3(actv3)
            beta3 = self.mlp_beta3(actv3)
            out3 = normalized * (1 + gamma3) + beta3
            after = self.after3(out3)

        #tumorcore
        if(1 == torch.unique(segmap[:,3:,:,:]).size()[0]):
            normalized = self.param_free_norm(after)
            #foreground
            actv4 = self.mlp_shared4(segmap[:,3:,:,:])
            gamma4 = self.mlp_gamma4(actv4)
            beta4 = self.mlp_beta4(actv4)
            out4 = normalized * (1 + gamma4) + beta4
            after = self.after4(out4)

        return after