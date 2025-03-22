#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
@author: Aymen Jemi (jemix) <jemiaymen@gmail.com>

MIT License

Copyright (c) 2021 Aymen Jemi SUDO-AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


import os
import pickle
from ..utils import datapath, DEVICE
from enum import Enum, unique

import bz2
import gzip
import lzma
import mmap
from tqdm.auto import tqdm
import json


@unique
class ZipAlgo(str, Enum):
    BZ2 = "1"
    GZIP = "2"
    LZMA = "3"


def convert_to_unicode(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def load_dataset(id: str):

    config = load_dataset_config(id)

    dataset_path = os.path.join(datapath(id.lower()),
                                config['version'],
                                'dataset.pt')

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f'dataset not exist please check name ({id}) or version ({config["version"]})'
        )
    dataset = None

    if config['is_ziped']:
        if config['algo'] == ZipAlgo.LZMA:
            with lzma.open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
        if config['algo'] == ZipAlgo.GZIP:
            with gzip.open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
        if config['algo'] == ZipAlgo.BZ2:
            with bz2.open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
    else:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)

    if config['device'] != DEVICE:
        dataset.set_device()
    
    return dataset


def to_fasttext_format(in_path: str, out_path: str, sep='\t'):
    if not os.path.exists(in_path):
        raise FileExistsError(
            f'input data file not found please check [{in_path}]')

    with open(in_path, "r+b") as f, open(out_path, mode='w', encoding='utf8') as wf:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            for line in tqdm(iter(m.readline, b""), desc='to fasttext format ', unit=' lines'):
                line = convert_to_unicode(line)
                text, label = line.split(sep)
                label = label.translate(str.maketrans(
                    {'\t': '', '\r': '', '\n': '', ' ': ''})
                )
                label = '__label__' + label
                _out = f'{label} \t {text} \n'
                wf.write(_out)


def save_dataset_config(id: str, version: str, is_ziped: bool, algo: ZipAlgo = ZipAlgo.LZMA):
    config = {
        'id': id,
        'version': version,
        'is_ziped': is_ziped,
        'algo': algo,
        'device': DEVICE
    }

    config_path = os.path.join(datapath(id), 'dataset_config.json')

    if os.path.exists(config_path):
        os.unlink(config_path)

    with open(config_path, 'w') as json_file:
        json.dump(config, json_file)


def load_dataset_config(id: str):

    config_path = os.path.join(datapath(id), 'dataset_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError('config file not found !')

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config
