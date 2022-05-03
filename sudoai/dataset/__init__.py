#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""Dataset module
@author: Aymen Jemi (jemix) <jemiaymen@gmail.com> at SUDO-AI

Datasets for natural language processing, sudoai.dataset
is a lightweight and extensible library to easily share
and access datasets for Natural Language Processing (NLP).

Tip:
    If you want use compressed algo, :obj:`ZipAlgo.LZMA`
    always be the best solution.

Examples:
    These examples illustrate how to use DataType in DatasetInfo class.

    Text data file:

    >>> info = DatasetInfo(id='sa', data_path='data.txt', data_type=DataType.TEXT)
    >>> info.data_type
    <DataType.TEXT: '1'>

    Excel data file:

    >>> info = DatasetInfo(id='sa', data_path='data.xlsx', data_type=DataType.EXCEL)
    >>> info.data_type
    <DataType.TEXT: '4'>

"""

from ..dataset.core import (
    DatasetError,
    DataType,
    DatasetInfo,
    Dataset,
    DatasetType,
    CustomDataset
)

__all__ = ['DatasetError',
           'DataType',
           'DatasetInfo',
           'DatasetType',
           'Dataset',
           'CustomDataset']
