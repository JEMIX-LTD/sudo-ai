#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""Trainer module
@author: Aymen Jemi (jemix) <jemiaymen@gmail.com> at SUDO-AI

sudoai.trainer is a lightweight and extensible library to
easily train model for Natural Language Processing (NLP).

Tip:
    If you want use compressed algo, :obj:`ZipAlgo.LZMA`
    always be the best solution.

Warning:

    To try trainer dataset must be exist in
    drive or local storage.

See Also:
    For more information check quickstart docs http://sudoai.tech/quickstart

Examples:
    These examples illustrate how to use training modules.

    Word to Label classification:

    >>> trainer = TokenClassificationTrainer(
    >>>     id='w2l_dataset_model',
    >>>     hidden_size=12,
    >>>     lr=0.002,
    >>>     momentum=0.2,
    >>>     loss='nll',
    >>>     optimizer='rmsprop',
    >>>     do_eval=True,
    >>>     do_shuffle=True,
    >>>     epochs=1)
    >>> trainer(hyperparam=True, test_mode=True)



"""


from ..trainer.core import (
    ModelError,
    Trainer,
    Word2WordTrainer,
    Seq2LabelTrainer,
    TokenClassificationTrainer,
    HybridXMLTrainer,
    LogisticTrainer,

)

__all__ = ['ModelError',
           'Trainer',
           'Word2WordTrainer',
           'Seq2LabelTrainer',
           'TokenClassificationTrainer',
           'HybridXMLTrainer',
           'LogisticTrainer']
