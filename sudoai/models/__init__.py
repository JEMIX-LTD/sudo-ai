#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""Models module
@author: Aymen Jemi (jemix) <jemiaymen@gmail.com> at SUDO-AI

Models of deep learning for natural language processing (NLP).

Warnings:
    When you subclass :obj:`BasicModule`, you should
    overwrite __init__() and forward().

See Also:
    For more information check quickstart docs http://sudoai.tech/quickstart

Examples:
    These examples illustrate how to use sudoai Models.

    Word model:

    >>> model_w2w = Word2Word(vocab_src=58,vocab_target=847,hidden_size=128)
    >>> model_tc = Word2Label(n_class=2,hidden_size=128,vocab_size=587)

    Sequence model:

    >>> model = Seq2Label(n_class=5,vocab_size=125,hidden_size=128)

"""

import os

import torch
import torch.nn as nn

from ..utils import datapath, save_config


class BasicModule(nn.Module):
    """Base class for all neural network modules.

    Your models should also subclass this class.
    All subclasses should overwrite __ini__() and forward().


    Attributes:
        name (str): Model identifier.
        version (str): Model version.

    Raises:
        NotImplementedError: When forward not overwrited.
    """

    def __init__(self, name: str = 'default', version: str = '0.1.0'):
        """Create BasicModel class

        Args:
            name (str, optional): Model identifier. Defaults to 'default'.
            version (str, optional): Model version. Defaults to '0.1.0'.
        """
        super().__init__()
        self.version = version
        self.name = name

    def forward(self):
        """Defines the computation performed at every call.

        Should be overridden by all subclasses.

        Raises:
            NotImplementedError: When forward() not overwrited.
        """
        raise NotImplementedError('you should overwrite forward()')

    def load(self):
        """Load saved model with id and version.

        Raises:
            FileExistsError: When model fine not exist.
        """

        model_path = os.path.join(
            datapath(self.name),
            self.version,
            'model.bin'
        )
        if os.path.exists(model_path):
            raise FileExistsError(
                'model not found check version or model name'
            )
        self.load_state_dict(torch.load(model_path))

    def save(self):
        """Save model with id and version.

        Returns:
            str: Path of saved model.
        """
        folder_name = datapath(self.name)
        folder_name = os.path.join(folder_name, self.version)

        os.makedirs(folder_name, exist_ok=True)

        model_path = os.path.join(folder_name, 'model.bin')

        torch.save(self.state_dict(), model_path)
        save_config(self, True)
        return folder_name
