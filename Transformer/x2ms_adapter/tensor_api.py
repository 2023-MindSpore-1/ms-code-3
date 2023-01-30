#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
import itertools
import numbers
import mindspore
import mindspore.ops as ops
import mindspore.numpy as ms_np
import numpy as np

from . import torch_base_api

classic_tensor_format = mindspore.Tensor.__format__


def tensor_format(self, format_spec):
    if self.dim() > 0:
        return classic_tensor_format(self, format_spec)
    return self.asnumpy().item().__format__(format_spec)


def t(self):
    dim = len(self.shape)
    perm = (1, 0)
    if dim == 1:
        perm = (0,)
    return ops.transpose(self, perm)


def topk(self, k, dim=-1, largest=True, sorted=True):
    return ops.TopK(sorted)(self, k)


def eq(self, other):
    equal = ops.Equal()
    return equal(self, other)


def float(self):
    return mindspore.Tensor(np.float32(self.asnumpy()))


def permute(self, *axis):
    return self.transpose(*axis)


def numpy(self):
    return self.asnumpy()


def contiguous(self):
    return self


def scatter_(self, dim, index, src):
    def get_value(src_value, tensor_index):
        if isinstance(src_value, mindspore.Tensor):
            return src_value[tensor_index]
        if isinstance(src_value, numbers.Number):
            return src_value
        return src_value

    shape_tuple = index.shape
    list_range = []
    for shape in shape_tuple:
        list_range.append(range(shape))
    for tensor_idx_tuple in itertools.product(*list_range):
        idx_list = list(tensor_idx_tuple)
        idx_list[dim] = index[tuple(tensor_idx_tuple)].asnumpy().item()
        self[tuple(idx_list)] = get_value(src, tuple(tensor_idx_tuple))

    return self


def unsqueeze(self, dim):
    expand_dims = mindspore.ops.ExpandDims()
    return expand_dims(self, dim)


def type(self, dtype=None, non_blocking=False, **kwargs):
    if dtype is None:
        return str(self.dtype)
    if dtype == torch_base_api.LongTensor:
        return self.astype(dtype=mindspore.int64)
    if isinstance(dtype, mindspore.common.Type):
        return self.astype(dtype=dtype)
    raise NotImplementedError(f'Unsupported tensor dtype {dtype}')


def to(self, *args, **kwargs):
    if args:
        param = args[0]
        if isinstance(param, mindspore.common.Type):
            return self.astype(dtype=param)
        if isinstance(param, mindspore.Tensor):
            return self.astype(dtype=param.dtype)
    if len(args) > 1:
        return self.astype(dtype=args[1])
    if kwargs.get('dtype'):
        return self.astype(dtype=kwargs.get('dtype'))
    if kwargs.get('other'):
        return self.astype(dtype=kwargs.get('other').dtype)
    return self


def add(self, other, *, alpha=1):
    if alpha != 1:
        raise NotImplementedError('alpha parameter is not supported!')
    return mindspore.ops.Add()(self, other)


def sub(self, other, *, alpha=1):
    if alpha != 1:
        raise NotImplementedError('alpha parameter is not supported!')
    return mindspore.ops.Sub()(self, other)


def mul(self, value):
    return mindspore.ops.Mul()(self, value)


def exp(self, out=None):
    return mindspore.ops.Exp()(self)


def div(self, value):
    return mindspore.ops.Div()(self, value)


def inplace_add(self, other, *, alpha=1):
    if alpha != 1:
        raise NotImplementedError('alpha parameter is not supported!')
    result = mindspore.ops.Add()(self, other)
    mindspore.ops.Assign()(self, result)
    return self


def inplace_sub(self, other, *, alpha=1):
    if alpha != 1:
        raise NotImplementedError('alpha parameter is not supported!')
    result = mindspore.ops.Sub()(self, other)
    mindspore.ops.Assign()(self, result)
    return self


def inplace_mul(self, value):
    result = mindspore.ops.Mul()(self, value)
    mindspore.ops.Assign()(self, result)
    return self


def inplace_div(self, value):
    result = mindspore.ops.Div()(self, value)
    mindspore.ops.Assign()(self, result)
    return self


@property
def property_data(self):
    return self


@property
def property_device(self):
    return torch_base_api.Device()


def sum(obj, *args, **kwargs):
    if isinstance(obj, (mindspore.Tensor, mindspore.Parameter)):
        return torch_base_api.sum(obj, *args, **kwargs)
    else:
        return obj.sum(*args, **kwargs)


def tensor_size(tensor, axis=None):
    if axis is None:
        return tensor.shape
    return tensor.shape[axis]


def size(obj, *args, **kwargs):
    if isinstance(obj, mindspore.Tensor):
        return tensor_size(obj, *args, **kwargs)
    else:
        return obj.size(*args, **kwargs)


def item(obj):
    if isinstance(obj, mindspore.Tensor):
        return obj.asnumpy().item()
    return obj.item()


def nelement(self):
    return self.size


def tensor_repeat(obj, *sizes):
    if obj.dtype == mindspore.bool_:
        tensor = obj.astype(mindspore.int32)
        return ms_np.tile(tensor, sizes) > 0
    return ms_np.tile(obj, sizes)


def repeat(obj, *args, **kwargs):
    if not isinstance(obj, (mindspore.Tensor, mindspore.Parameter)):
        return obj.repeat(*args, **kwargs)
    return tensor_repeat(obj, *args, **kwargs)


def tensor_mean(tensor, dim=None, keepdim=False):
    return tensor.mean(axis=dim, keep_dims=keepdim)


def mean(obj, *args, **kwargs):
    if not isinstance(obj, (mindspore.Tensor, mindspore.Parameter)):
        return obj.mean(*args, **kwargs)
    return tensor_mean(obj, *args, **kwargs)


def tensor_std(tensor, dim=None, unbiased=True, keepdim=False):
    return tensor.std(axis=dim, ddof=1, keepdims=keepdim)


def std(obj, *args, **kwargs):
    if not isinstance(obj, (mindspore.Tensor, mindspore.Parameter)):
        return obj.std(*args, **kwargs)
    return tensor_std(obj, *args, **kwargs)


def tensor_transpose(tensor, dim0, dim1):
    dim = tensor.dim()
    _dim0 = dim0 if dim0 >= 0 else (dim0 + dim)
    _dim1 = dim1 if dim1 >= 0 else (dim1 + dim)
    dim_list = list(range(dim))
    dim_list[_dim0] = _dim1
    dim_list[_dim1] = _dim0
    return tensor.transpose(*dim_list)


def transpose(obj, *args, **kwargs):
    if not isinstance(obj, (mindspore.Tensor, mindspore.Parameter)):
        return obj.transpose(*args, **kwargs)
    return tensor_transpose(obj, *args, **kwargs)


def inplace_copy(self, value):
    mindspore.ops.Assign()(self, value)
    return self


def masked_fill(self, mask, value):
    broadcast_to = mindspore.ops.BroadcastTo(self.shape)
    mask = mask.astype(mindspore.int32)
    reverse_mask = (mask == 0).astype(mindspore.int32)
    mask = broadcast_to(mask)
    reverse_mask = broadcast_to(reverse_mask)
    return self * reverse_mask + mask * value


def tensor_argmax(tensor, dim=None, keepdim=False):
    return tensor.argmax(axis=dim)


def argmax(obj, *args, **kwargs):
    if not isinstance(obj, (mindspore.Tensor, mindspore.Parameter)):
        return obj.argmax(*args, **kwargs)
    return tensor_argmax(obj, *args, **kwargs)


def sigmoid(self):
    return torch_base_api.sigmoid(self)


def tensor_max(tensor, dim=None, keepdim=False):
    if not dim:
        return tensor.max(axis=dim, keepdims=keepdim)
    else:
        max_ops = mindspore.ops.ArgMaxWithValue(axis=dim, keep_dims=keepdim)
        index, output = max_ops(tensor)
        return output, index


def max(obj, *args, **kwargs):
    if not isinstance(obj, (mindspore.Tensor, mindspore.Parameter)):
        return obj.max(*args, **kwargs)
    return tensor_max(obj, *args, **kwargs)


def long(self, memory_format=None):
    return mindspore.Tensor(self.asnumpy().astype(np.int64))


def numel(obj):
    if not isinstance(obj, (mindspore.Tensor, mindspore.Parameter)):
        return obj.numel()
    return obj.size


def expand(self, *sizes):
    broadcast_to = mindspore.ops.BroadcastTo(sizes)
    return broadcast_to(self)


def flatten(obj, *args, **kwargs):
    if not isinstance(obj, (mindspore.Tensor, mindspore.Parameter)):
        return obj.flatten(*args, **kwargs)
    return torch_base_api.flatten(obj, *args, **kwargs)


def tensor_flatten(self, start_dim=0, end_dim=-1):
    shape_tuple = self.shape
    _start_dim = start_dim if start_dim >= 0 else (start_dim + self.dim())
    _end_dim = end_dim if end_dim >= 0 else (end_dim + self.dim())
    new_dim = 1
    for idx in range(start_dim, _end_dim + 1):
        new_dim *= shape_tuple[idx]
    new_shape_list = list(shape_tuple[0:start_dim])
    new_shape_list.append(new_dim)
    new_shape_list.extend(shape_tuple[(_end_dim + 1):self.dim()])
    reshape = mindspore.ops.Reshape()
    return reshape(self, tuple(new_shape_list))
