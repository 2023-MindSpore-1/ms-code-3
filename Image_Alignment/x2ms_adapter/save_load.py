#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
from numbers import Number
import mindspore
import mindspore.nn as nn


def save(obj, f, pickle_module=None, pickle_protocol=None, _use_new_zipfile_serialization=False):
    """
    Function replace torch.save
    """
    if isinstance(f, str):
        if not f.endswith('.ckpt'):
            f += '.ckpt'
    if isinstance(obj, nn.Cell):
        mindspore.save_checkpoint(obj, f)
    if isinstance(obj, dict):
        _SaveLoadDict.save(obj, f)


def load(f, map_location=None, pickle_module=None, **pickle_load_args):
    """
    Loads checkpoint info from a specified file.
    """
    if isinstance(f, str):
        if not f.endswith('.ckpt'):
            f += '.ckpt'
    return _SaveLoadDict.load(f)


def load_state_dict(obj, state_dict, strict=True):
    """
    Stub function for torch.nn.module.load_state_dict
    Loads parameters into network.
    The parameter strict will be set False, to avoid defects caused by deleting functions such as nn.DataParallel.
    Returns:
       List, parameters not loaded in the network.
    """
    param_not_load = []
    if isinstance(obj, mindspore.nn.Cell):
        param_not_load = mindspore.load_param_into_net(obj, state_dict, strict_load=False)

    return param_not_load


class _SaveLoadDict(object):
    SUPPORT_MEMBER_TYPE = [Number, str, mindspore.Tensor, mindspore.Parameter, dict, bool]
    _VALUE_SUFFIX = "_x2ms_value"
    _STR_SUFFIX = ".x2ms_str"
    _SAVE_HEAD = "x2ms_dict"

    @staticmethod
    def save(save_obj, file_name):
        if _SaveLoadDict._is_save_parameter_dict(save_obj):
            param_list = list({'name': k, 'data': v} for k, v in save_obj.items())
            mindspore.save_checkpoint(param_list, file_name)
        else:
            _SaveLoadDict._save_dict(save_obj, file_name)

    @staticmethod
    def load(file_name):
        load_dict = mindspore.load_checkpoint(file_name)
        if _SaveLoadDict._is_load_x2ms_dict(load_dict):
            return _SaveLoadDict._load_dict(load_dict)
        return load_dict

    @staticmethod
    def _is_save_parameter_dict(save_obj):
        return all(isinstance(member, mindspore.Parameter) for member in save_obj.values())

    @staticmethod
    def _is_load_x2ms_dict(load_obj):
        return _SaveLoadDict._SAVE_HEAD in load_obj.keys()

    @staticmethod
    def _save_dict(save_obj, file_name):
        param_list = []
        param_list.append({"name": _SaveLoadDict._SAVE_HEAD, "data": mindspore.Tensor(0)})
        for key, value in save_obj.items():
            for support_type in _SaveLoadDict.SUPPORT_MEMBER_TYPE:
                if isinstance(value, support_type):
                    getattr(_SaveLoadDict, f"_save_dict_{support_type.__name__.lower()}")(param_list, key, value)
                    break
        mindspore.save_checkpoint(param_list, file_name)

    @staticmethod
    def _save_dict_dict(param_list, save_name, save_obj):
        if not _SaveLoadDict._is_save_parameter_dict(save_obj):
            raise TypeError(f"Does not support to saving type of {save_name}.")
        param_list.append({"name": f"{save_name}.dict", "data": mindspore.Tensor(len(save_obj))})
        param_list.extend(list({'name': f"{save_name}.{k}", 'data': v} for k, v in save_obj.items()))

    @staticmethod
    def _save_dict_number(param_list, save_name, save_obj: Number):
        _SaveLoadDict._save_single_value(param_list, save_name, "number", mindspore.Tensor(save_obj))

    @staticmethod
    def _save_dict_str(param_list, save_name, save_obj):
        param_list.append({"name": f"{save_name}.str", "data": mindspore.Tensor(1)})
        param_list.append({"name": f"{save_obj}{_SaveLoadDict._STR_SUFFIX}", "data": mindspore.Tensor(0)})

    @staticmethod
    def _save_dict_tensor(param_list, save_name, save_obj: mindspore.Tensor):
        _SaveLoadDict._save_single_value(param_list, save_name, "tensor", save_obj)

    @staticmethod
    def _save_dict_parameter(param_list, save_name, save_obj: mindspore.Parameter):
        _SaveLoadDict._save_single_value(param_list, save_name, "parameter", save_obj)

    @staticmethod
    def _save_dict_bool(param_list, save_name, save_obj):
        _SaveLoadDict._save_single_value(param_list, save_name, "bool", mindspore.Tensor(save_obj))

    @staticmethod
    def _save_single_value(param_list, save_name, save_type, save_obj):
        param_list.append({"name": f"{save_name}.{save_type}", "data": mindspore.Tensor(1)})
        param_list.append({"name": f"{save_name}{_SaveLoadDict._VALUE_SUFFIX}", "data": save_obj})

    @staticmethod
    def _load_dict(load_dict):
        param_dict = {}
        param_iter = iter(load_dict)
        next(param_iter)
        try:
            while True:
                key = next(param_iter)
                length = load_dict.get(key).asnumpy().item()
                data_type = key.split(".")[-1]
                if getattr(_SaveLoadDict, f"_load_dict_{data_type}"):
                    value = getattr(_SaveLoadDict, f"_load_dict_{data_type}")(load_dict, param_iter, length, key)
                    param_dict[".".join(key.split(".")[:-1])] = value
        except StopIteration:
            return param_dict

    @staticmethod
    def _load_dict_number(load_dict, iterator, length, save_name):
        return load_dict.get(next(iterator)).asnumpy().item()

    @staticmethod
    def _load_dict_str(load_dict, iterator, length, save_name):
        result_str = next(iterator)
        return ".".join(result_str.split(".")[:-1])

    @staticmethod
    def _load_dict_bool(load_dict, iterator, length, save_name):
        return load_dict.get(next(iterator)).asnumpy().item()

    @staticmethod
    def _load_dict_tensor(load_dict, iterator, length, save_name):
        return load_dict.get(next(iterator))

    @staticmethod
    def _load_dict_parameter(load_dict, iterator, length, save_name):
        return load_dict.get(next(iterator))

    @staticmethod
    def _load_dict_dict(load_dict, iterator, length, save_name):
        result_dict = {}
        real_save_name = ".".join(save_name.split(".")[:-1])
        for _ in range(length):
            key = next(iterator)
            real_name = key[len(real_save_name) + 1:]
            parameter = load_dict.get(key)
            parameter.name = real_name
            result_dict[real_name] = parameter
        return result_dict
