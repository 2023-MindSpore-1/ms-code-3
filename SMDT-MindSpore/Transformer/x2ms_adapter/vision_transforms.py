#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

import PIL

import mindspore.dataset.vision


class Resize(mindspore.dataset.vision.py_transforms.Resize):
    interpolation_map = {
        PIL.Image.BILINEAR: mindspore.dataset.vision.Inter.BILINEAR,
        PIL.Image.NEAREST: mindspore.dataset.vision.Inter.NEAREST,
        PIL.Image.BICUBIC: mindspore.dataset.vision.Inter.BICUBIC,
    }

    def __init__(self, size, interpolation=PIL.Image.BILINEAR, max_size=None, antialias=None):
        interpolation = self.interpolation_map.get(interpolation, mindspore.dataset.vision.Inter.BILINEAR)
        super().__init__(size=size, interpolation=interpolation)
