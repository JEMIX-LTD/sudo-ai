#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""@author: Aymen Jemi (jemix) <jemiaymen@gmail.com>

Copyright (c) 2021 Aymen Jemi SUDO-AI
"""

import torch
import torch.nn as nn
from torch import optim
from torchmetrics.functional import accuracy

from ..models import BasicModule
from ..models.lstm import ExtremMutliLabelTextClassification
from ..utils import DEVICE


class HybridXMLTC(BasicModule):
    """Hybrid Attention for Extreme Multi-Label Text Classification model.

    Useful for extreme multi-label classification.

    Attributes:
        n_class (int): Number of classes.
        vocab_size (int): Vocab size.
        embedding_size (int): Embedding size.
        hidden_size (int): Value of Hidden size of NN.
        d_a (int): Same as hidden_size.
        name (str): Model id.
        verison (str): Model version.
        momentum (float): Value of Optimizer momentum if exist.
        learning_rate (float): Learn rate for current model.
        optimizer_type (str): Code of optimizer like (sgd ~> stochastic gradient descent).
        optimizer (:obj:`torch.optim`): Pytorch Optimizer.
        xmltc (:obj:`ExtremMutliLabelTextClassification`): Base model.
        multiclass(bool): If True the model is multiclass model.
        criterion (loss): Loss function.

    Tip:
        For this model optimizer must be one of ['adam', 'sgd','rmsprop'].

    """

    def __init__(self,
                 n_class: int = 3714,
                 vocab_size: int = 30001,
                 embedding_size: int = 300,
                 hidden_size: int = 256,
                 d_a: int = 256,
                 optimizer: str = 'adam',
                 name: str = 'hybrid_xmltc',
                 momentum: float = 0.0,
                 learning_rate: float = 0.01,
                 multiclass: bool = False,
                 version: str = '0.1.0'):
        """Create HybridXMLTC model.

        Args:
            n_class (int, optional): Number of classes. Defaults to 3714.
            vocab_size (int, optional): Vocab size. Defaults to 30001.
            embedding_size (int, optional): Embedding size. Defaults to 300.
            hidden_size (int, optional): Value of Hidden size of NN Defaults to 256.
            d_a (int, optional): Same as hidden_size. Defaults to 256.
            optimizer (str, optional): Code of optimizer like (sgd ~> stochastic gradient descent). Defaults to 'adam'.
            name (str, optional): Model id. Defaults to 'hybrid_xmltc'.
            momentum (float, optional): Value of Optimizer momentum. Defaults to 0.0.
            learning_rate (float, optional): Learn rate for current model. Defaults to 0.01.
            multiclass (bool, optional): If True the model is multiclass model. Defaults to False.
            version (str, optional): Model version. Defaults to '0.1.0'.

        Raises:
            ValueError: When optimizer code not exist. ['adam', 'sgd','rmsprop']
        """
        super(HybridXMLTC, self).__init__(name, version)

        self.n_class = n_class
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.d_a = d_a
        self.name = name
        self.version = version
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.multiclass = multiclass

        self.optimizer_type = optimizer.lower()

        self.xmltc = ExtremMutliLabelTextClassification(
            self.n_class,
            self.vocab_size,
            self.embedding_size,
            self.hidden_size,
            self.d_a,
            self.multiclass
        ).to(DEVICE)

        if self.optimizer_type == 'sgd':

            self.optimizer = optim.SGD(self.xmltc.parameters(),
                                       lr=self.learning_rate,
                                       momentum=momentum)

        elif self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.xmltc.parameters(),
                                        weight_decay=4e-5,
                                        lr=self.learning_rate)

        elif self.optimizer_type == 'rmsprop':
            self.optimizer = optim.RMSprop(self.xmltc.parameters(),
                                           lr=self.learning_rate,
                                           momentum=momentum)

        else:
            raise ValueError(f'optimizer not exist {self.optimizer_type} !!')

    def forward(self, input_tensor, target_tensor=None, do_train=False, threshold=0.5):
        """Defines the computation performed at every call.

        Args:
            input_tensor (torch.Tensor): Input tensor.
            target_tensor (torch.Tensor, optional): Output tensor. Defaults to None.
            do_train (bool, optional): If True train mode. Defaults to False.
            threshold (float, optional): Value of threshold.

        Raises:
            ValueError: When input_tensor is not torch.Tensor.

        Returns:
            dict: In train mode metrics (acc,loss).
        """
        if not isinstance(input_tensor, torch.Tensor):
            raise ValueError("input_tensor must be torch.Tensor")

        if do_train is False:
            with torch.no_grad():
                out = self.xmltc(input_tensor)
                result = out.argmax(dim=1)
                size = result.size(0)

                score = torch.zeros(self.n_class)
                for x in result:
                    score[x.item()] += 1
                score = score / size
                labels = torch.where(score > threshold, 1, 0)

                return {'pred': labels, 'score': score}

        if self.multiclass is True:
            self.criterion = nn.BCELoss(reduction='sum')
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='sum')

        with torch.enable_grad():
            self.optimizer.zero_grad()
            out = self.xmltc(input_tensor)

            t = torch.flatten(target_tensor).to(DEVICE)

            loss = 0
            acc = 0

            if self.multiclass is False:

                target = torch.flatten((t == 1).nonzero(as_tuple=False))
                for x in out:
                    loss += self.criterion(x.view(1, 3), target)
                    acc += accuracy(x, t, multiclass=False,
                                    threshold=threshold)
                loss.backward()
                self.optimizer.step()

                a = acc.item() / out.size(0)
                lo = loss.item() / out.size(0)

                return {'acc': a, 'loss': lo}

            for x in out:
                loss += self.criterion(x.float(), t.float())
                acc += accuracy(x, t, multiclass=True, threshold=threshold)

            loss.backward()
            self.optimizer.step()

            a = acc.item() / out.size(0)
            lo = loss.item() / out.size(0)

            return {'acc': a, 'loss': lo}
