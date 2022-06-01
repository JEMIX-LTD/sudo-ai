#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""@author: Aymen Jemi (jemix) <jemiaymen@gmail.com>

Copyright (c) 2022 Aymen Jemi JEMIX Ltd
"""

import torch
import torch.nn as nn
from torch import optim
from torchmetrics.functional import accuracy
import torch.nn.functional as F

from ..utils import DEVICE, MAX_LENGTH

from ..models import BasicModule


class LogisticRegression(BasicModule):

    def __init__(self,
                 name,
                 version='0.1.0',
                 momentum=0.5,
                 n_class=1,
                 learning_rate=0.01):

        super().__init__(name, version)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_class = n_class

        self.linear = torch.nn.Linear(MAX_LENGTH, self.n_class).to(DEVICE)

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor = None, do_train: bool = False):

        if not isinstance(input_tensor, torch.Tensor):
            raise ValueError("input_tensor must be a torch.Tensor")

        _in = self.__pad(input_tensor).to(DEVICE)

        if do_train:

            if not isinstance(target_tensor, torch.Tensor):
                raise ValueError("target_tensor must be a torch.Tensor")

            criterion = nn.BCELoss()
            optimizer = optim.SGD(self.linear.parameters(
            ), lr=self.learning_rate, momentum=self.momentum)

            optimizer.zero_grad()
            loss = 0

            self.outputs = torch.sigmoid(self.linear(_in))

            loss = criterion(self.outputs, target_tensor)
            acc = accuracy(self.outputs, target_tensor.to(dtype=torch.int), threshold=0.5)

            loss.backward()

            optimizer.step()

            return {'loss': loss.item(), 'acc': acc.item()}

        with torch.no_grad():
            self.outputs = torch.sigmoid(self.linear(_in))
            return self.outputs

    def __pad(self, tensor: torch.Tensor):
        return F.pad(input=tensor, pad=(0, MAX_LENGTH - tensor.size()[0]), mode='constant', value=0)
