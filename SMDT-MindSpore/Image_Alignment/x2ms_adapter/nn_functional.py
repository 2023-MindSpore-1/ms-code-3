#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
import numpy as np
import mindspore
import mindspore.ops as ops


def relu(input, inplace=False):
    relu_func = ops.ReLU()
    return relu_func(input)


def softmax(input, dim=None, _stacklevel=3, dtype=None):
    soft_max = ops.Softmax(axis=dim)
    if dtype:
        return soft_max(input).astype(dtype)
    return soft_max(input)


def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
               divisor_override=None):
    avg_pool2d_func = ops.AvgPool(kernel_size=kernel_size, strides=kernel_size)
    return avg_pool2d_func(input)


def sigmoid(input):
    return ops.Sigmoid()(input)


def dropout(input, p=0.5, training=True, inplace=False):
    if not training:
        return input
    dropout_func = ops.Dropout(1 - p)
    output, _ = dropout_func(input)
    return output


def adaptive_avg_pool2d(input, output_size):
    if output_size == (1, 1) or output_size == 1:
        return ops.ReduceMean(keep_dims=True)(input, tuple(range(2, len(input.shape))))
    else:
        return ops.AdaptiveAvgPool2D(output_size)(input)


def gelu(input):
    return ops.GeLU()(input)


def pad(input, pad, mode='constant', value=0):
    if not isinstance(pad, (list, tuple)) or len(pad) % 2 != 0 or len(pad) // 2 > input.dim():
        raise ValueError(f'Invalid arg \'pad\' {pad}')
    new_pad = list((0, 0) for _ in range(input.dim()))
    for i in range(len(pad) // 2):
        new_pad[-1 - i] = (pad[2 * i], pad[2 * i + 1])
    new_pad = tuple(new_pad)
    return ops.Pad(new_pad)(input)


def one_hot(tensor, num_classes=-1):
    if num_classes == -1:
        num_classes = int(tensor.asnumpy().max().item()) + 1
    return ops.OneHot()(tensor, num_classes,
                        mindspore.Tensor(1.0, mindspore.float32),
                        mindspore.Tensor(0.0, mindspore.float32)).astype(mindspore.int64)


def conv2d(data, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    out_channel = weight.shape[0]
    kernel_size = (weight.shape[2], weight.shape[3])
    if isinstance(padding, (list, tuple)) and len(padding) == 2:
        padding = (padding[0], padding[1]) * 2
    op_conv2d = ops.Conv2D(out_channel, kernel_size, mode=1, pad_mode="pad", pad=padding,
                           stride=stride, dilation=dilation, group=groups)
    return op_conv2d(data, weight)
