#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""Hypertuning module
@author: Aymen Jemi (jemix) <jemiaymen@gmail.com> at SUDO-AI

sudoai.hypertuning is a lightweight and extensible library to easily
try different parameters for Natural Language Processing (NLP) model.

Warning:

    To try hypertuning for dataset, it must be exist in
    drive or local storage.

See Also:
    For more information check quickstart docs http://sudoai.tech/quickstart

Examples:
    These examples illustrate how to use hypertuning modules.

    Word to Label classification:

    >>> token_hypertuning(dataset_id='w2l_dataset_model', test_mode=True, n_experience=2)

    Word to Word model.

    >>> w2w_hypertuning('t_dataset_model', n_experience=2, test_mode=True)


"""


from ..hypertuning.core import (
    seq2label_hypertuning,
    token_hypertuning,
    w2w_hypertuning
)

__all__ = ['seq2label_hypertuning',
           'token_hypertuning',
           'w2w_hypertuning']
