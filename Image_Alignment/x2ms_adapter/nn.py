#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
from collections import OrderedDict

import mindspore.nn
import mindspore.ops as ops

from .nn_functional import adaptive_avg_pool2d


class AdaptiveAvgPool2d(mindspore.nn.Cell):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def construct(self, input):
        return adaptive_avg_pool2d(input, self.output_size)


class BatchNorm2d(mindspore.nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,
                 device=None, dtype=None):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine)

    @property
    def bias(self):
        return self.beta

    @property
    def weight(self):
        return self.gamma

    @property
    def running_mean(self):
        return self.moving_mean

    @property
    def running_var(self):
        return self.moving_variance


class Conv2d(mindspore.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        if isinstance(stride, (tuple, list)) and len(stride) == 1:
            stride = stride[0]
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, pad_mode='pad',
                         dilation=dilation, group=groups, has_bias=bias)

    @property
    def groups(self):
        return self.group

    def construct(self, input):
        if input.dtype == mindspore.float64:
            input = ops.Cast()(input, mindspore.float32)
        return super().construct(input)


class GroupNorm(mindspore.nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None):
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)


class Linear(mindspore.nn.Dense):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, has_bias=bias)


class MaxPool2d(mindspore.nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        if stride is None:
            stride = kernel_size
        if kernel_size == 2 * padding + 1:
            super().__init__(kernel_size=kernel_size, stride=stride, pad_mode="same")
        elif padding == 0:
            super().__init__(kernel_size=kernel_size, stride=stride, pad_mode="valid")
        else:
            raise NotImplementedError("Unsupported padding value")


class ReLU(mindspore.nn.ReLU):
    def __init__(self, inplace=False):
        super().__init__()


class Sequential(mindspore.nn.SequentialCell):
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            super().__init__(args[0])
        else:
            super().__init__(list(args))

    def add_module(self, name, module):
        self.append(module)


class ModuleList(mindspore.nn.CellList):
    def __init__(self, modules=None):
        if not modules:
            super().__init__([])
        else:
            super().__init__(modules)


class LogSoftmax(mindspore.nn.Cell):
    def __init__(self, dim=0):
        super().__init__()
        self.softmax = mindspore.ops.Softmax(dim)
        self.log_softmax = mindspore.ops.LogSoftmax(dim)

    def construct(self, x):
        if x.dim() <= 2:
            return self.log_softmax(x)
        x = self.softmax(x)
        x = mindspore.ops.functional.log(x)
        return x


class AvgPool2d(mindspore.nn.AvgPool2d):
    def __init__(self, kernel_size, stride=None, padding=0,
                 ceil_mode=False, count_include_pad=True, divisor_override=None):
        ms_stride = stride
        if ms_stride is None:
            ms_stride = kernel_size
        pad_mode = 'valid'
        if padding > 0:
            pad_mode = 'same'
        super().__init__(kernel_size=kernel_size, stride=ms_stride, pad_mode=pad_mode)


class Dropout2d(mindspore.nn.Dropout):
    def __init__(self, p=0.5, inplace=False):
        super().__init__(keep_prob=1-p)


class Dropout(mindspore.nn.Dropout):
    def __init__(self, p=0.5, inplace=False):
        super().__init__(keep_prob=1-p)


class Embedding(mindspore.nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0,
                 scale_grad_by_freq=False, sparse=False, _weight=None, device=None, dtype=None):
        super().__init__(vocab_size=num_embeddings, embedding_size=embedding_dim)

    @property
    def embedding_dim(self):
        return self.embedding_size


class Upsample(mindspore.nn.ResizeBilinear):
    """
    Only support mode='bilinear'
    """
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def construct(self, x):
        return super().construct(x, size=self.size, scale_factor=self.scale_factor, align_corners=self.align_corners)


class Identity(mindspore.nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        return ops.Identity()(x)


class ZeroPad2d(mindspore.nn.Cell):
    def __init__(self, padding):
        if isinstance(padding, int):
            padding = tuple(padding for _ in range(4))
        elif isinstance(padding, (list, tuple)) and len(padding) == 4:
            padding = tuple(padding)
        else:
            raise ValueError(f'Invalid arg \'padding\': {padding}')
        self.padding = padding
        super().__init__()

    def construct(self, x):
        padding = list([0, 0] for _ in range(x.dim()))
        padding[-1] = self.padding[0:2]
        padding[-2] = self.padding[2:4]
        padding = tuple(tuple(elem) for elem in padding)
        return ops.Pad(padding)(x)


class NLLLoss(mindspore.nn.LossBase):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super().__init__(reduction=reduction)
        self.reduction = reduction
        self.nll_loss = mindspore.ops.NLLLoss(reduction=reduction)
        self.one_hot = mindspore.ops.OneHot(-1)
        self.on_value = mindspore.Tensor(1.0, mindspore.float32)
        self.off_value = mindspore.Tensor(0.0, mindspore.float32)
        self.ignore_index = ignore_index

    def construct(self, input, target):
        if input.dim() == 2:
            ones = mindspore.ops.Ones()
            weight = ones(input.shape[1], mindspore.float32)
            if self.ignore_index >= 0:
                weight[self.ignore_index] = 0
            return self.nll_loss(input, target.astype(mindspore.int32), weight)[0]
        if input.dim() == 3:
            _target = self.one_hot(target, input.shape[1], self.on_value, self.off_value).transpose(0, 2, 1)
            if self.reduction == "sum":
                return self.reduce_sum(-(input * _target))
            else:
                _input = input
                if self.ignore_index >= 0:
                    _input[:, self.ignore_index] = 0
                    loss = self.reduce_sum(-(_input * _target)) / mindspore.ops.count_nonzero(
                        target - self.ignore_index)
                else:
                    loss = self.reduce_sum(-(_input * _target)) / target.size
                return loss
        else:
            raise NotImplementedError(f"Unsupported NLLLoss input dim: {input.dim()}")


class LambdaCell(mindspore.nn.Cell):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def construct(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class LayerNorm(mindspore.nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        super().__init__(normalized_shape, epsilon=eps)

    @property
    def weight(self):
        return self.gamma

    @weight.setter
    def weight(self, weight):
        self.gamma = weight

    @property
    def bias(self):
        return self.beta

    @bias.setter
    def bias(self, bias):
        self.beta = bias


class Softmax(mindspore.nn.Softmax):
    def __init__(self, dim=None):
        if dim is None:
            dim = -1
        super().__init__(axis=dim)
