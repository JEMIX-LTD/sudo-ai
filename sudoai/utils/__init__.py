#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""utils module
@author: Aymen Jemi (jemix) <jemiaymen@gmail.com> at SUDO-AI

"""
from .params import (
    DEVICE,
    ARABIC_PATTERN,
    LATIN_PATTERN,
    LATIN_LOWER_PATTERN,
    LATIN_LETTRE,
    LATIN_LOWER_LETTRE,
    ARABIC_LETTRE,
    MAX_LENGTH,
    SOC_TOKEN,
    EOC_TOKEN,
    SOC_CHAR,
    EOC_CHAR,
    MAX_WORDS
)
from .utils import datapath
from .text import load_tokenizer
from .datasets import load_dataset, ZipAlgo, to_fasttext_format, save_dataset_config, load_dataset_config
from .models import (
    load_config,
    save_config,
    load_model,
    save_checkpoint,
    load_checkpoint,
    checkpoint_to_model,
    check_model
)
from .langage import lid
from .io import InputOutput
from .logs import log

__all__ = ['DEVICE',
           'ARABIC_PATTERN',
           'LATIN_PATTERN',
           'LATIN_LOWER_PATTERN',
           'LATIN_LETTRE',
           'LATIN_LOWER_LETTRE',
           'ARABIC_LETTRE',
           'MAX_LENGTH',
           'SOC_TOKEN',
           'EOC_TOKEN',
           'SOC_CHAR',
           'EOC_CHAR',
           'MAX_WORDS',
           'datapath',
           'load_tokenizer',
           'load_dataset',
           'load_config',
           'save_config',
           'load_model',
           'save_checkpoint',
           'load_checkpoint',
           'checkpoint_to_model',
           'check_model',
           'InputOutput',
           'load_dataset',
           'save_dataset_config',
           'load_dataset_config',
           'lid',
           'log',
           'ZipAlgo',
           'to_fasttext_format']
