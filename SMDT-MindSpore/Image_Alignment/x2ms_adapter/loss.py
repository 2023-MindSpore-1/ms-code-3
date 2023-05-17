#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

import mindspore


class CrossEntropyLoss(mindspore.nn.SoftmaxCrossEntropyWithLogits):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        m_reduction = 'none'
        if size_average is None and reduce is None:
            m_reduction = reduction
        elif reduce is not None:
            if reduce is True:
                m_reduction = 'sum' if size_average is False else 'mean'
        super().__init__(reduction=m_reduction, sparse=True)

    def construct(self, logits, labels):
        if labels.dtype not in (mindspore.int32, mindspore.int64):
            labels = labels.astype(mindspore.int64)

        logits_shape, labels_shape = logits.shape, labels.shape
        if len(logits_shape) == 4:
            # shape of pytorch logits: (N, C, H, W) -> mindspore (N, C)
            # shape of pytorch labels: (N, H, W) -> mindspore (N,)
            logits = mindspore.ops.Transpose()(logits, (0, 2, 3, 1))
            new_n = logits_shape[0] * logits_shape[2] * logits_shape[3]
            logits_shape = (new_n, logits_shape[1])
            labels_shape = (new_n,)
        logits = logits.reshape(logits_shape)
        labels = labels.reshape(labels_shape)

        return super().construct(logits, labels)


class BCEWithLogitsLoss(mindspore.nn.BCEWithLogitsLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super().__init__(reduction=reduction, weight=weight, pos_weight=pos_weight)
