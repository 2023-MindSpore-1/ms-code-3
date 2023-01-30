#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.


class Function:
    @property
    def saved_tensors(self):
        return []

    @staticmethod
    def forward(ctx, *args, **kwargs):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        pass

    @classmethod
    def apply(cls, *args, **kwargs):
        return cls.forward(Function(), *args, **kwargs)

    def mark_dirty(self, *args):
        pass

    def mark_non_differentiable(self, *args):
        pass

    def save_for_backward(self, *tensors):
        pass

    def set_materialize_grads(self, value):
        pass

    def mark_shared_storage(self, *pairs):
        pass
