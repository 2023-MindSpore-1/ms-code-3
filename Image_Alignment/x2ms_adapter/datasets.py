#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

import numpy as np
import PIL.Image

import mindspore
import mindspore.dataset as datasets
import mindspore.dataset.vision.py_transforms as v_transforms
import mindspore.dataset.transforms.py_transforms as transforms

from mindspore.communication.management import get_rank, get_group_size, context
from mindspore.dataset import MappableDataset, BatchDataset


def _dataset_len(self):
    return self.get_dataset_size()


mindspore.dataset.Dataset.__len__ = _dataset_len


class RawDatasetWrapper:
    def __init__(self, dataset):
        self.dataset = dataset
        sample = dataset[0]
        self.is_dict = isinstance(sample, dict)

        if self.is_dict:
            self.column_names = list(sample.keys())
        else:
            self.column_names = list('column_' + str(i) for i in range(len(sample)))

    @staticmethod
    def _to_numpy_array(data):
        if isinstance(data, mindspore.Tensor):
            return data.asnumpy()
        else:
            return np.asarray(data)

    def __getitem__(self, item):
        output = self.dataset[item]
        if isinstance(output, dict):
            output = output.values()
        output = tuple(self._to_numpy_array(value) for value in output)
        return output

    def __len__(self):
        return len(self.dataset)


class BatchDatasetWrapper(mindspore.dataset.BatchDataset):
    def __init__(self, dataset: RawDatasetWrapper, batch_size=1):
        self._is_dict = dataset.is_dict
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode == context.ParallelMode.DATA_PARALLEL:
            super().__init__(mindspore.dataset.GeneratorDataset(dataset, dataset.column_names, shard_id=get_rank(),
                                                                num_shards=get_group_size()), batch_size=batch_size)
        else:
            super().__init__(mindspore.dataset.GeneratorDataset(dataset, dataset.column_names), batch_size=batch_size)

    def __iter__(self):
        if self._is_dict:
            return self.create_dict_iterator()
        else:
            return self.create_tuple_iterator()


def _create_batch_dataset_wrapper(dataset, batch_size):
    wrapped_dataset = RawDatasetWrapper(dataset)
    wrapped_dataset = BatchDatasetWrapper(wrapped_dataset, batch_size=batch_size)
    return wrapped_dataset


def _is_cifar100(dataset):
    child_dataset = dataset
    while True:
        if isinstance(child_dataset, MappableDataset):
            return isinstance(child_dataset, datasets.Cifar100Dataset)
        if not child_dataset.children:
            break
        child_dataset = child_dataset.children[0]
    return False


def _del_cifar100_column(col_1, col_2, col_3, batch_info):
    return col_1, col_2,


def _batch_dataset(dataset, batch_size):
    if _is_cifar100(dataset):
        return dataset.batch(batch_size, per_batch_map=_del_cifar100_column,
                             input_columns=['image', 'fine_label', 'coarse_label'],
                             output_columns=['image', 'label'])
    return dataset.batch(batch_size)


def data_loader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None,
                pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None):
    if not isinstance(dataset, mindspore.dataset.Dataset):
        dataset = _create_batch_dataset_wrapper(dataset, batch_size)
    else:
        dataset = _batch_dataset(dataset, batch_size)
    child_dataset = dataset
    while True:
        if isinstance(child_dataset, MappableDataset):
            setattr(child_dataset, 'shuffle_flag', shuffle)
            if sampler:
                setattr(child_dataset, 'sampler', sampler)
            setattr(child_dataset, 'num_parallel_workers', num_workers)
        if isinstance(child_dataset, BatchDataset):
            setattr(child_dataset, 'drop_remainder', drop_last)
        if not child_dataset.children:
            break
        child_dataset = child_dataset.children[0]
    return dataset


class Subset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths, generator=None):
    if isinstance(dataset, mindspore.dataset.Dataset):
        return dataset.split(lengths, randomize=True)

    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = np.random.permutation(np.arange(sum(lengths))).tolist()
    split_datasets = []
    offset = 0
    for length in lengths:
        split_datasets.append(Subset(dataset, indices[offset: offset + length]))
        offset += length

    return tuple(split_datasets)


def _ensure_numpy_array(data):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, PIL.Image.Image):
        return np.asarray(data)
    elif isinstance(data, mindspore.Tensor):
        return data.asnumpy()
    else:
        raise NotImplementedError(f'Unsupported data type {type(data)}')


def uint_to_int(data):
    if data.dtype == np.uint32:
        return data.astype(np.int32)
    return data


def image_folder(root, transform=None, target_transform=None, loader=None, is_valid_file=None):
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    if parallel_mode == context.ParallelMode.DATA_PARALLEL:
        ms_dataset = datasets.ImageFolderDataset(dataset_dir=root, shard_id=get_rank(), num_shards=get_group_size())
    else:
        ms_dataset = datasets.ImageFolderDataset(dataset_dir=root)
    transform_to_add = [v_transforms.Decode(), v_transforms.ToPIL()]
    ms_dataset = _map_transform(ms_dataset, transform, transform_to_add, 'image')
    target_transform_to_add = [uint_to_int]
    ms_dataset = _map_transform(ms_dataset, target_transform, target_transform_to_add, 'label')
    return ms_dataset


def cifar10(root, train=True, transform=None, target_transform=None, download=False):
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    if parallel_mode == context.ParallelMode.DATA_PARALLEL:
        ms_dataset = datasets.Cifar10Dataset(dataset_dir=root, usage='train' if train else 'test',
                                             shard_id=get_rank(), num_shards=get_group_size())
    else:
        ms_dataset = datasets.Cifar10Dataset(dataset_dir=root, usage='train' if train else 'test')
    transform_to_add = [v_transforms.ToPIL()]
    ms_dataset = _map_transform(ms_dataset, transform, transform_to_add, 'image')
    target_transform_to_add = [uint_to_int]
    ms_dataset = _map_transform(ms_dataset, target_transform, target_transform_to_add, 'label')
    return ms_dataset


def cifar100(root, train=True, transform=None, target_transform=None, download=False):
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    if parallel_mode == context.ParallelMode.DATA_PARALLEL:
        ms_dataset = datasets.Cifar100Dataset(dataset_dir=root, usage='train' if train else 'test',
                                             shard_id=get_rank(), num_shards=get_group_size())
    else:
        ms_dataset = datasets.Cifar100Dataset(dataset_dir=root, usage='train' if train else 'test')
    transform_to_add = [v_transforms.ToPIL()]
    ms_dataset = _map_transform(ms_dataset, transform, transform_to_add, 'image')
    target_transform_to_add = [uint_to_int]
    ms_dataset = _map_transform(ms_dataset, target_transform, target_transform_to_add, 'fine_label')
    return ms_dataset


def mnist(root, train=True, transform=None, target_transform=None, download=False):
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    if parallel_mode == context.ParallelMode.DATA_PARALLEL:
        ms_dataset = datasets.MnistDataset(dataset_dir=root, usage='train' if train else 'test',
                                           shard_id=get_rank(), num_shards=get_group_size())
    else:
        ms_dataset = datasets.MnistDataset(dataset_dir=root, usage='train' if train else 'test')
    if transform:
        ms_dataset = ms_dataset.map(operations=transform, input_columns='image')
    return ms_dataset


def _map_transform(ms_dataset, transform, transform_to_add, input_columns):
    if transform:
        if isinstance(transform, list):
            transform_to_add.extend(transform)
        if isinstance(transform, transforms.Compose):
            transform_to_add.extend(transform.transforms)
    transform_to_add.append(_ensure_numpy_array)
    ms_dataset = ms_dataset.map(operations=transform_to_add, input_columns=input_columns)
    return ms_dataset


class DistributedSampler(datasets.DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode == context.ParallelMode.DATA_PARALLEL:
            super().__init__(num_shards=get_group_size(), shard_id=get_rank(), shuffle=shuffle)
        else:
            super().__init__(num_shards=None, shard_id=None, shuffle=shuffle)


class RandomSampler(mindspore.dataset.RandomSampler):
    def __init__(self, data_source, replacement=False, num_samples=None, generator=None):
        super().__init__(replacement=replacement, num_samples=num_samples)


class SequentialSampler(mindspore.dataset.SequentialSampler):
    def __init__(self, indices, generator=None):
        super().__init__()
