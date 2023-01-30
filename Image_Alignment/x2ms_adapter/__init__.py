#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

from distutils.version import LooseVersion

import PIL
import numpy
import mindspore
import mindspore.default_config
import mindspore.nn
import mindspore.communication
import mindspore.numpy
import mindspore.context as context
from .save_load import save, load, load_state_dict
from . import tensor_api
from .context import x2ms_context
from .torch_base_api import arange, cat, cos, clamp, Device, from_numpy, flatten,\
    LongTensor, matmul, mm, normal, ones, pow, sin, tanh, tensor, Tensor, zeros, split, as_tensor, dot, sum, argmax,\
    Generator, sigmoid, rand, floor

__all__ = ["save", "load", "load_state_dict", "arange", "cat", "cos", "clamp", "Device", "from_numpy", "flatten",
           "LongTensor", "matmul", "mm", "normal", "ones", "pow", "sin", "tanh", "tensor", "Tensor", "zeros",
           "split", 'as_tensor', 'argmax', 'Generator', 'sigmoid', 'rand', 'floor']


if LooseVersion(PIL.__version__) < LooseVersion('8.2.0'):
    print("The Pillow of an earlier version has "
          "CVE-2019-19911, CVE-2020-5310, CVE-2020-5311, CVE-2020-5312, CVE-2020-5313, CVE-2021-25289, "
          "CVE-2020-10177, CVE-2020-10378, CVE-2020-10379, CVE-2020-10994, CVE-2020-11538, CVE-2020-15999, "
          "CVE-2020-35653, CVE-2020-35654, CVE-2020-35655, CVE-2021-25289, CVE-2021-25290, CVE-2021-25291, "
          "CVE-2021-25292, CVE-2021-25293, CVE-2021-27921, CVE-2021-27922, CVE-2021-27923, CVE-2021-25287, "
          "CVE-2021-25288, CVE-2021-28675, CVE-2021-28676, CVE-2021-28677, CVE-2021-28678 vulnerabilities, "
          "which affect the use of MindSpore. Upgrade it to 8.2.0 or later.")
    exit(1)

# if LooseVersion(numpy.__version__) < LooseVersion('1.22.0'):
#     print("The Numpy of an earlier version has CVE-2021-41496 vulnerability, which affects the use of MindSpore. "
#           "Upgrade it to 1.22.0 or later.")
#     exit(1)


def modules(self):
    return (m[1] for m in self.cells_and_names())


def cell_to(self, *args, **kwargs):
    if args:
        param = args[0]
        if param in (mindspore.float16, mindspore.float32):
            return self.to_float(dst_type=param)
        if isinstance(param, mindspore.Tensor) and param.dtype in (mindspore.float16, mindspore.float32):
            return self.to_float(dst_type=param.dtype)
    if len(args) > 1:
        param = args[1]
        if param in (mindspore.float16, mindspore.float32):
            return self.to_float(dst_type=param)
    if kwargs.get('dtype') in (mindspore.float16, mindspore.float32):
        return self.to_float(dst_type=kwargs.get('dtype'))
    if kwargs.get('other') and kwargs.get('other').dtype in (mindspore.float16, mindspore.float32):
        return self.to_float(dst_type=kwargs.get('other').dtype)
    return self


def register_buffer(self, name, tensor):
    setattr(self, name, mindspore.Parameter(tensor, requires_grad=False))


def zero_grad(self, *args, **kwargs):
    pass


def named_children(self):
    return self.name_cells().items()


mindspore.Tensor.t = tensor_api.t
mindspore.Tensor.topk = tensor_api.topk
mindspore.Tensor.eq = tensor_api.eq
mindspore.Tensor.float = tensor_api.float
mindspore.Tensor.permute = tensor_api.permute
mindspore.Tensor.numpy = tensor_api.numpy
mindspore.Tensor.contiguous = tensor_api.contiguous
mindspore.Tensor.__format__ = tensor_api.tensor_format
mindspore.Tensor.scatter_ = tensor_api.scatter_
mindspore.Tensor.unsqueeze = tensor_api.unsqueeze
mindspore.Tensor.type = tensor_api.type
mindspore.Tensor.to = tensor_api.to
mindspore.Tensor.add = tensor_api.add
mindspore.Tensor.sub = tensor_api.sub
mindspore.Tensor.mul = tensor_api.mul
mindspore.Tensor.div = tensor_api.div
mindspore.Tensor.exp = tensor_api.exp
mindspore.Tensor.add_ = tensor_api.inplace_add
mindspore.Tensor.sub_ = tensor_api.inplace_sub
mindspore.Tensor.mul_ = tensor_api.inplace_mul
mindspore.Tensor.div_ = tensor_api.inplace_div
mindspore.Tensor.data = tensor_api.property_data
mindspore.Tensor.device = tensor_api.property_device
mindspore.Tensor.nelement = tensor_api.nelement
mindspore.Tensor.masked_fill = tensor_api.masked_fill
mindspore.Tensor.sigmoid = tensor_api.sigmoid
mindspore.Tensor.long = tensor_api.long
mindspore.Tensor.expand = tensor_api.expand
mindspore.Parameter.copy_ = tensor_api.inplace_copy


mindspore.nn.Cell.modules = modules
mindspore.nn.Cell.to = cell_to
mindspore.nn.Cell.register_buffer = register_buffer
mindspore.nn.Cell.zero_grad = zero_grad
mindspore.nn.Cell.named_children = named_children


def state_dict(obj, destination=None, prefix='', keep_vars=False):
    if isinstance(obj, mindspore.nn.Cell):
        return obj.parameters_dict()
    return {}


def init_process_group(backend, init_method=None, timeout=-1, world_size=-1, rank=-1, store=None,
                       group_name=''):
    """
       Stub function for torch.distributed.init_process_group.
    """
    if backend in ['hccl', 'HCCL']:
        context.set_context(device_target='Ascend')
    if backend in ['nccl', 'NCCL']:
        context.set_context(device_target='GPU')
    mindspore.communication.init()


def cuda_device_count():
    """
       Stub function for torch.cuda.device_count.
       if init can not be called for twice, call release before init again
    """
    try:
        return mindspore.communication.get_group_size()
    # only RuntimeError when is not init, we consider it not run distributedly.
    except RuntimeError:
        return 1


def get_rank():
    """
       Stub function for torch.cuda.device_count.
       if init can not be called for twice, call release before init again
    """
    try:
        return mindspore.communication.get_rank()
    # only RuntimeError when is not init, we consider it not run distributedly.
    except RuntimeError:
        return -1


def get_local_rank():
    """
       Stub function for torch.cuda.device_count.
       if init can not be called for twice, call release before init again
    """
    try:
        return mindspore.communication.get_local_rank()
    # only RuntimeError when is not init, we consider it not run distributedly.
    except RuntimeError:
        return -1


def cuda_set_device(device):
    if isinstance(device, int):
        mindspore.context.set_context(device_id=device)
    else:
        raise NotImplementedError(f'unsupported device type {type(device)}')


def is_cuda_available():
    """
       Stub function for torch.cuda.is_available.
       get the info from default_config.
    """
    device_targets = mindspore.default_config.__device_target__
    return 'gpu' in device_targets


def get_params(model, recurse=True):
    if isinstance(model, mindspore.nn.Cell):
        return model.trainable_params(recurse)
    return model.parameters()


def named_parameters(model, prefix='', recurse=True):
    if isinstance(model, mindspore.nn.Cell):
        return list(param for param in model.parameters_and_names(prefix, recurse) if param[1].requires_grad)
    return model.named_parameters()


def named_modules(model, prefix=''):
    if isinstance(model, mindspore.nn.Cell):
        return model.cells_and_names(prefix)
    return model.named_modules()


def forward(obj, *args, **kwargs):
    if isinstance(obj, mindspore.nn.Cell):
        return obj.construct(*args, **kwargs)
    return obj.forward(*args, **kwargs)


def train_one_step_cell(model, optimizer):
    if x2ms_context.train_one_step_model:
        return x2ms_context.train_one_step_model
    if x2ms_context.amp_opt_level is None:
        wrapped_model = mindspore.nn.TrainOneStepCell(model, optimizer)
    else:
        wrapped_model = mindspore.amp.build_train_network(model, optimizer, level=x2ms_context.amp_opt_level)
    x2ms_context.train_one_step_model = wrapped_model
    return wrapped_model


def wrapped_model_forward(model, data, target=None):
    loss = model(data) if target is None else model(data, target)
    if isinstance(loss, (tuple, list)):
        return loss[0]
    return loss
