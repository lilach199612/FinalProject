import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from Clustering import *

import random

import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def group_norm(input, group, running_mean, running_var, weight=None, bias=None,
               use_input_stats=True, momentum=0.1, eps=1e-5):
    r"""Applies Group Normalization for channels in the same group in each data sample in a
    batch.
    See :class:`~torch.nn.GroupNorm1d`, :class:`~torch.nn.GroupNorm2d`,
    :class:`~torch.nn.GroupNorm3d` for details.
    """
    if not use_input_stats and (running_mean is None or running_var is None):
        raise ValueError('Expected running_mean and running_var to be not None when use_input_stats=False')

    b, c = input.size(0), input.size(1)
    if weight is not None:
        weight = weight.repeat(b)
    if bias is not None:
        bias = bias.repeat(b)

    def _instance_norm(input, group, running_mean=None, running_var=None, weight=None,
                       bias=None, use_input_stats=None, momentum=None, eps=None):
        # Repeat stored stats and affine transform params if necessary
        if running_mean is not None:
            running_mean_orig = running_mean
            running_mean = running_mean_orig.repeat(b)
        if running_var is not None:
            running_var_orig = running_var
            running_var = running_var_orig.repeat(b)
        eps = 1e-5
        G = group
        x_np = input
        #if not input.is_cuda:
        #    input = input.cuda()
        gpu_copy = input
        out_final =gpu_copy
        # NO PYTORCH EQUIVALENT YET WE CAN USEhttps://discuss.pytorch.org/t/equivalent-to-numpys-nan-to-num/52448
        #x_np = np.nan_to_num(x_np)
        # At this point we have just stored our input on CPU with Numpy
        Nabs, C, H, W = x_np.size()
        #N = Nabs
        # N refers to the __? What the batch size?
        # res_x is the original shape of our input
        #res_x = torch.zeros((N, C, H, W))
        # this is a looser form where we are just by channel dimensions
        #x_np_new = torch.zeros((C,2))
        # we loop through the channels
        #TODO DOUBLE CHECK THIS IS RIGHT!!!
        N = 1
        #for n in range(Nabs):
        #xnp_new = x_np[n, :, :, :]
        #xnp_new = x_np
        #x_np = torch.unsqueeze(x_np, dim=0)
        #temp = torch.reshape(xnp_new, (C, N * H * W))
        temp = (torch.transpose(x_np, dim0=1, dim1=0)).contiguous().view(C, -1)
        img = torch.zeros((C, 2))
        #img[:, 0], img[:, 1]= torch.std_mean(temp, dim=1)
        img[:, 0] = torch.std(temp, dim=1)

        img[:, 1] = torch.mean(temp, dim=1)
        # TODO we could use that other function to replace
        #x_np_new = np.nan_to_num(x_np_new,posinf=0,neginf=0)
        #Data = data_preparation(n_cluster=G, data=img[:, :])
        '''con = C//G
        Data = torch.zeros((C, ))
        count = -1
        for i in range(0, C, con):
            if (i+con) < C:
                count += 1
                Data[i:i+con] = count

            else:
                Data[i:C] = count'''

        #print(Data.shape)

        # for Random Channel Selection
        Data = torch.randint(G, (C,))
        #print(Data2.shape)
        # Up to here we are all correct, as long as we are clustering in the right basis
        #print(Data)
        count = 0
        Common_mul = 1
        Common_add = 0
        for val in range(G):
            # The problem is arising when we assign everybody to the same group
            inx = torch.nonzero((Data == val))
            if inx.size()[0] >  0:
                tmp = torch.zeros((x_np.shape[0], inx.size()[0], x_np.shape[2], x_np.shape[3]))
            else:
                continue
            flat_inx = torch.flatten(inx)
            if gpu_copy.is_cuda:
                temp_flat_inx = flat_inx.cuda()
                tmp = tmp.cuda()
            else:
                temp_flat_inx = flat_inx
            tmp[:, :, :, :] = torch.index_select(x_np,1,temp_flat_inx)
            #print(tmp.std())
            out_final_tmp = Stat_torch(tmp)
            out_final_tmp[out_final_tmp != out_final_tmp] = 0
            if out_final.is_cuda:
                out_temp = out_final_tmp.cuda()
            else:
                out_temp = out_final_tmp
            #tmp_tensor = torch.unsqueeze(out_final[:, :, :, :], dim=0)
            out_final[:, :, :, :] = out_final.index_copy_(1, temp_flat_inx, out_temp)



        return out_final.view(b, c, *input.size()[2:])
    return _instance_norm(input, group, running_mean=running_mean,
                          running_var=running_var, weight=weight, bias=bias,
                          use_input_stats=use_input_stats, momentum=momentum,
                          eps=eps)


class _HoogiNorm(_BatchNorm):
    def __init__(self, num_features, num_groups=5, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=False):
        self.num_groups = num_groups
        self.track_running_stats = track_running_stats
        super(_HoogiNorm, self).__init__(int(num_features / num_groups), eps,
                                        momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)

        return group_norm(
            input, self.num_groups, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)
''''def Stat_torch(IN):
    t = torch.zeros((IN.shape[0], IN.shape[1], IN.shape[2], IN.shape[3]))
    my_temp = torch.nn.functional.normalize(IN.view(IN.shape[0],IN.shape[1]*IN.shape[2]*IN.shape[3]),dim=1)
    return my_temp.view(*t.size())'''

def Stat_torch(IN, eps = 1e-4):
    '''IN_no_grad = IN.detach()
    IN_no_grad.requires_grad = False'''
    #t = torch.zeros((IN.shape[0], IN.shape[1], IN.shape[2], IN.shape[3]))
    '''my_temp = torch.zeros_like(IN)
    my_temp.requires_grad = True'''
    if IN.shape[1] == 1:
        my_temp = IN
    else:
        mu = IN.mean()
        #mu = torch.unsqueeze(torch.mean(IN, dim=1), dim=1)
        if torch.isnan(torch.sum(torch.unsqueeze(torch.std(IN, dim=1), dim=1))):
            Std = IN.std()
            #Std = torch.unsqueeze(torch.std(IN, dim=1), dim=1)
        else:
            Std = IN.std()
            #Std = torch.unsqueeze(torch.std(IN, dim=1), dim=1)
        #my_temp = torch.div((IN.add(- mu)), (Std + 1e-4))
        my_temp = (IN - mu)/(Std + eps)
        #print(my_temp.mean())

    #my_temp = torch.nn.functional.normalize(IN.view(IN.shape[0],IN.shape[1]*IN.shape[2]*IN.shape[3]),dim=1)
    return my_temp



