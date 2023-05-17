#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
from .context import x2ms_context


def pair(x):
    if isinstance(x, (tuple, list)):
        return x
    return x, x


class SummaryWriter(object):
    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix=''):
        pass

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        pass

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        pass

    def close(self):
        pass


def amp_initialize(models, optimizers=None, enabled=True, opt_level="O1", cast_model_type=None,
                   patch_torch_functions=None, keep_batchnorm_fp32=None, master_weights=None, loss_scale=None,
                   cast_model_outputs=None, num_losses=1, verbosity=1, min_loss_scale=None, max_loss_scale=2.**24):
    if opt_level == "O1":
        print("[X2MindSpore]: MindSpore does not support O1, use O2 instead.")
        x2ms_context.amp_opt_level = "O2"
    else:
        x2ms_context.amp_opt_level = opt_level
    if optimizers is None:
        return models
    return models, optimizers


def clip_grad_norm(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False):
    x2ms_context.clip_grad_norm = max_norm


def amp_master_params(optimizer):
    return optimizer.trainable_params(True)
