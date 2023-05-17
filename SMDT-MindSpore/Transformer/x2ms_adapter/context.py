#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

class Context:
    def __init__(self):
        self.amp_opt_level = None
        self.train_one_step_model = None
        self.clip_grad_norm = None


x2ms_context = Context()
