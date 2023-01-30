#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
import mindspore.nn


class StepLR(mindspore.nn.learning_rate_schedule.LearningRateSchedule):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        super().__init__()
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.learn_rate = mindspore.Tensor(optimizer.get_lr(), mindspore.float32)
        self.step_value = -1
        if last_epoch >= 0:
            for i in range(last_epoch + 1):
                self.step()

    def step(self):
        self.step_value += 1
        lr = self.construct(self.step_value)
        self.optimizer.learning_rate = lr
        return lr

    def construct(self, global_step):
        if self.step_value != 0 and self.step_value % self.step_size == 0:
            self.learn_rate *= self.gamma
        return mindspore.Tensor(self.learn_rate, mindspore.float32)

    def get_lr(self):
        return (self.learn_rate,)


class LambdaLR(mindspore.nn.learning_rate_schedule.LearningRateSchedule):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
        super().__init__()
        self.optimizer = optimizer
        self.init_lr = mindspore.Tensor(optimizer.get_lr(), mindspore.float32)
        self.lr_lambda = lr_lambda
        self.step_value = last_epoch

    def step(self):
        self.step_value += 1
        lr = mindspore.Tensor(self.lr_lambda(self.step_value), mindspore.float32) * self.init_lr
        self.optimizer.learning_rate = lr
        return lr

    def construct(self, global_step):
        return mindspore.Tensor(self.lr_lambda(self.step_value), mindspore.float32) * self.init_lr

    def get_lr(self):
        return (self.lr_lambda(self.step_value) * self.init_lr,)


class CosineAnnealingLR(mindspore.nn.CosineDecayLR):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False):
        min_lr = 0.0
        max_lr = float(optimizer.get_lr().asnumpy())
        decay_steps = T_max
        super().__init__(min_lr=min_lr, max_lr=max_lr, decay_steps=decay_steps)
        self.optimizer = optimizer
        self.step_value = -1

    def step(self):
        self.step_value += 1
        lr = super().construct(mindspore.Tensor(self.step_value))
        self.optimizer.learning_rate = lr
        return lr

    def construct(self, global_step):
        return super().construct(mindspore.Tensor(self.step_value))


class MultiStepLR(mindspore.nn.learning_rate_schedule.LearningRateSchedule):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        super().__init__()
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.learn_rate = mindspore.Tensor(optimizer.get_lr(), mindspore.float32)
        self.step_value = -1
        if last_epoch >= 0:
            for i in range(last_epoch + 1):
                self.step()

    def step(self):
        self.step_value += 1
        lr = self.construct(self.step_value)
        self.optimizer.learning_rate = lr
        return lr

    def construct(self, global_step):
        if self.step_value in self.milestones:
            self.learn_rate *= self.gamma
        return mindspore.Tensor(self.learn_rate, mindspore.float32)

    def get_lr(self):
        return (self.learn_rate,)
