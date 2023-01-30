#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

import numpy as np

import mindspore
import mindspore.context as context
import mindspore.numpy
import mindspore.common.initializer


def cat(tensors, dim=0, *, out=None):
    return mindspore.ops.Concat(dim)(tensors)


def flatten(input_tensor, start_dim=0, end_dim=-1):
    shape_tuple = input_tensor.shape
    _start_dim = start_dim if start_dim >= 0 else (start_dim + input_tensor.dim())
    _end_dim = end_dim if end_dim >= 0 else (end_dim + input_tensor.dim())
    new_dim = 1
    for idx in range(start_dim, _end_dim + 1):
        new_dim *= shape_tuple[idx]
    new_shape_list = list(shape_tuple[0:start_dim])
    new_shape_list.append(new_dim)
    new_shape_list.extend(shape_tuple[(_end_dim + 1):input_tensor.dim()])
    reshape = mindspore.ops.Reshape()
    return reshape(input_tensor, tuple(new_shape_list))


def from_numpy(ndarray):
    return mindspore.Tensor.from_numpy(np.ascontiguousarray(ndarray))


def zeros(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if dtype is None:
        dtype = mindspore.float32
    return mindspore.ops.Zeros()(_tuple(size), dtype)


def ones(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if dtype is None:
        dtype = mindspore.float32
    return mindspore.ops.Ones()(_tuple(size), dtype)


def arange(start, end=None, step=1, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if end is None:
        return mindspore.numpy.arange(start, step=step, dtype=dtype)
    else:
        return mindspore.numpy.arange(start, stop=end, step=step, dtype=dtype)


def tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False):
    return mindspore.Tensor(data, dtype=dtype)


def matmul(input, other, out=None):
    return mindspore.ops.matmul(input, other)


def tanh(input, out=None):
    return mindspore.ops.Tanh()(input)


def sin(input, out=None):
    return mindspore.ops.Sin()(input)


def cos(input, out=None):
    return mindspore.ops.Cos()(input)


def pow(input, exponent, out=None):
    return mindspore.ops.Pow()(input, exponent)


def clamp(input, min, max, out=None):
    return mindspore.ops.clip_by_value(input, min, max)


def normal(mean, std, *, generator=None, out=None):
    if isinstance(mean, float):
        input_shape = std.shape
        mean = mindspore.Tensor(mean, mindspore.float32)
    elif isinstance(std, float):
        input_shape = mean.shape
        std = mindspore.Tensor(std, mindspore.float32)
    else:
        input_shape = mean.shape
    mean = mean.astype(mindspore.float32)
    std = std.astype(mindspore.float32)
    return mindspore.ops.normal(input_shape, mean, std)


def mm(input, mat2, *, out=None):
    return mindspore.ops.matmul(input, mat2)


def split(tensor, split_size_or_sections, dim=0):
    if isinstance(split_size_or_sections, int) and tensor.shape[dim] % split_size_or_sections == 0:
        split_ops = mindspore.ops.Split(dim, int(tensor.shape[dim] / split_size_or_sections))
        return split_ops(tensor)
    raise NotImplementedError("not implement split input parameter")


def _tuple(size):
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        return tuple(size[0])
    return size


def as_tensor(data, dtype=None, device=None):
    if isinstance(data, mindspore.Tensor) and data.dtype == dtype:
        return data

    return mindspore.Tensor(data, dtype=dtype)


def dot(input, tensor):
    return mindspore.ops.tensor_dot(input, tensor, axes=1)


def sum(*args, **kwargs):
    def _sum(input, dim=None, keepdim=False, dtype=None):
        if dtype:
            input = input.astype(dtype)
        if context.get_context('device_target') == 'Ascend':
            input_type = input.dtype
            input = input.astype(mindspore.float32)
            if input_type == mindspore.bool_:
                return input.sum(axis=dim, keepdims=keepdim)
            return input.sum(axis=dim, keepdims=keepdim).astype(input_type)
        return input.sum(axis=dim, keepdims=keepdim)
    return _sum(*args, **kwargs)


def argmax(input, dim=None, keepdim=False):
    return input.argmax(axis=dim)


def sigmoid(data, out=None):
    return mindspore.ops.Sigmoid()(data)


def rand(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    shape = _tuple(size)
    data = mindspore.Tensor(np.random.rand(*shape), dtype=dtype)
    return data


def floor(data, out=None):
    return mindspore.ops.Floor()(data)


class Device:
    def __init__(self, *args, **kwargs):
        pass

    @property
    def type(self):
        return mindspore.context.get_context('device_target')

    def __format__(self, format_spec):
        device_target = mindspore.context.get_context('device_target')
        device_id = mindspore.context.get_context('device_id')
        return f'{device_target}:{device_id}'


class LongTensor(mindspore.Tensor):
    def __init__(self, *args, **kwargs):
        param = args[0]
        if isinstance(param, tuple):
            super().__init__(dtype=mindspore.int64, shape=param, init=mindspore.common.initializer.Zero())
            self.init_data()
        else:
            super().__init__(input_data=param, dtype=mindspore.int64)


class Tensor(mindspore.Tensor):
    def __init__(self, data, *, dtype=None, device=None, requires_grad=False, pin_memory=False):
        super().__init__(input_data=data, dtype=dtype)


class Generator:
    def __init__(self, *args, **kwargs):
        pass

    def manual_seed(self, seed):
        pass
