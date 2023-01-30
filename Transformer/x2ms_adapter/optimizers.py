#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

from types import GeneratorType
from collections import namedtuple

import mindspore.nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from .context import x2ms_context

OptimizerInfo = namedtuple('OptimizerInfo', ['instance', 'func_caller'])


class OptimAdaptorMixIn:
    def zero_grad(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass


class Adam(mindspore.nn.Adam, OptimAdaptorMixIn):
    def __init__(self, params, **kwargs):
        mindspore.nn.Adam.__init__(self, params, **kwargs)

    def construct(self, gradients):
        if x2ms_context.clip_grad_norm is not None:
            gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
        return super().construct(gradients)


class SGD(mindspore.nn.SGD, OptimAdaptorMixIn):
    def __init__(self, params, **kwargs):
        mindspore.nn.SGD.__init__(self, params, **kwargs)

    def construct(self, gradients):
        if x2ms_context.clip_grad_norm is not None:
            gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
        return super().construct(gradients)


class RMSprop(mindspore.nn.RMSProp, OptimAdaptorMixIn):
    def __init__(self, params, **kwargs):
        mindspore.nn.RMSProp.__init__(self, params, **kwargs)

    def construct(self, gradients):
        if x2ms_context.clip_grad_norm is not None:
            gradients = mindspore.ops.composite.clip_by_global_norm(gradients, x2ms_context.clip_grad_norm)
        return super().construct(gradients)


class FuncCaller:
    def __init__(self, func, *args, **kwargs):
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def get_call(self, *args, **kwargs):
        args = (*args, self._args)
        self._kwargs.update(kwargs)
        return self._func(*args, **self._kwargs)


class OptimRegister:
    def __init__(self):
        self._func = None
        self._register_info = {}
        self._lr_scheduler = None

    @staticmethod
    def _generator_to_list(params):
        if isinstance(params, GeneratorType):
            params = list(params)
        return params

    def adam(self, params, lr=0.001, betas=(0.9, 0.999),
             eps=1e-8, weight_decay=0, amsgrad=False):
        params = self._generator_to_list(params)
        kwargs = {
            "learning_rate": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay,
        }
        optimizer_instance = Adam(params, **kwargs)
        self._register_info[str(Adam.__name__)] = \
            OptimizerInfo(optimizer_instance, FuncCaller(Adam, *params, **kwargs))
        return optimizer_instance

    def sgd(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        params = self._generator_to_list(params)
        kwargs = {
            "learning_rate": lr,
            "momentum": momentum,
            "dampening": dampening,
            "nesterov": nesterov,
            "weight_decay": weight_decay,
        }
        optimizer_instance = SGD(params, **kwargs)
        self._register_info[str(SGD.__name__)] = \
            OptimizerInfo(optimizer_instance, FuncCaller(SGD, *params, **kwargs))
        return optimizer_instance

    def rmsprop(self, params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.0, centered=False):
        params = self._generator_to_list(params)
        kwargs = {
            "learning_rate": lr,
            "momentum": momentum,
            "epsilon": eps,
            "centered": centered,
            "weight_decay": weight_decay,
        }
        optimizer_instance = RMSprop(params, **kwargs)
        self._register_info[str(RMSprop.__name__)] = \
            OptimizerInfo(optimizer_instance, FuncCaller(RMSprop, *params, **kwargs))
        return optimizer_instance

    def get_instance(self):
        if len(self._register_info) == 0:
            raise RuntimeError('No optimizer instance has been created.')
        elif len(self._register_info) > 1:
            raise NotImplementedError('More than one optimizer instances have been created.')
        return next(iter(self._register_info.values())).instance


class OptimizerParamGroupsModifier:
    def __init__(self, optimizer):
        self._optimizer = optimizer

    def __setitem__(self, key, value):
        if key == 'lr':
            ms_lr = self._optimizer.get_lr()
            ms_lr.set_data(Tensor(value, mstype.float32))
            self._optimizer.learning_rate = ms_lr
        else:
            print("WARN: Not support modify key `{}`, ignored".format(key))

    def __getitem__(self, key):
        if key == 'lr':
            ms_lr = self._optimizer.get_lr()
            return ms_lr.asnumpy().tolist()
        else:
            print("WARN: Not support get key `{}`, return none".format(key))


@property
def get_param_groups(self):
    modifier = OptimizerParamGroupsModifier(self)
    return [modifier]


mindspore.nn.Optimizer.param_groups = get_param_groups
optim_register = OptimRegister()
