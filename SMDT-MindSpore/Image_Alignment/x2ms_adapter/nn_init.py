#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

import mindspore
import mindspore.ops as ops
from mindspore.common.initializer import initializer, Constant, Normal, One, Uniform, Zero,\
    HeNormal, HeUniform, XavierUniform


def _assign_value(tensor, init):
    value = initializer(init, tensor.shape, mindspore.float32)
    value.init_data()
    if isinstance(tensor, mindspore.Parameter):
        tensor.set_data(value)
    else:
        ops.Assign()(tensor, value)
    return tensor


def constant_(tensor, val):
    return _assign_value(tensor, Constant(val))


def normal_(tensor, mean=0.0, std=1.0):
    # see also: https://en.wikipedia.org/wiki/Normal_distribution
    return _assign_value(tensor, Normal(sigma=std, mean=mean))


def ones_(tensor):
    return _assign_value(tensor, One())


def uniform_(tensor, a=0.0, b=1.0):
    if a + b != 0.0:
        print(f'[Warning] Uniform initializer in MindSpore does not support bound ({a}, {b}), '
              'will use default argument.')
        return _assign_value(tensor, Uniform())
    else:
        return _assign_value(tensor, Uniform(scale=b))


def zeros_(tensor):
    return _assign_value(tensor, Zero())


def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    return _assign_value(tensor, HeNormal(negative_slope=a, mode=mode, nonlinearity=nonlinearity))


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    return _assign_value(tensor, HeUniform(negative_slope=a, mode=mode, nonlinearity=nonlinearity))


def xavier_uniform_(tensor, gain=1.0):
    return _assign_value(tensor, XavierUniform(gain=gain))

