#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""@author: Aymen Jemi (jemix) <jemiaymen@gmail.com>

Copyright (c) 2021 Aymen Jemi SUDO-AI
"""


import random

import torch
import torch.nn as nn
from torch import optim
from torchmetrics.functional import accuracy

from ..models import BasicModule
from ..models.gru import AttnDecoderRNN, EncoderRNN
from ..utils import DEVICE, EOC_TOKEN, MAX_LENGTH, SOC_TOKEN


class Word2Word(BasicModule):
    """Word to Word model.

    Useful for task like transliteration and translate.

    Attributes:
        optimizer_type (str): Code of optimizer like (sgd ~> stochastic gradient descent).
        loss (str): Code of loss function like (nll ~> negative log likelihood loss).
        max_length (int): Maximum word length (see sudoai.utils.MAX_LENGTH).
        learning_rate (float): Learn rate for current model.
        teacher_forcing_ratio (float): Teacher forcing ratio for current model.
        vocab_src (int): Vocab size of source language.
        vocab_target (int): Vocab size of Target language.
        name (str): Model id.
        verison (str): Model version.
        momentum (float): Value of Optimizer momentum if exist.
        drop_out (float): Value of drop out.
        hidden_size (int): Value of Hidden size of NN.
        encoder (:obj:`EncoderRNN`): Encoder.
        decoder (:obj:`AttnDecoderRNN`): Decoder with attention.
        encoder_optimizer (:obj:`torch.optim`): Optimizer for encoder.
        decoder_optmizer (:obj:`torch.optim`): Optimizer for decoder.

    Tip:
        For this model, loss must be one of ['nll', 'crossentropy'],
        optimizer must be one of ['adam', 'sgd','rmsprop'].

    """

    def __init__(self,
                 vocab_src: int,
                 vocab_target: int,
                 hidden_size: int,
                 version: str = '0.1.0',
                 name: str = 'word2word',
                 optimizer: str = 'sgd',
                 loss: str = 'crossentropy',
                 learning_rate: float = 0.01,
                 teacher_forcing_ratio: float = 0.5,
                 momentum: float = 0.0,
                 drop_out: float = 0.1):
        """Create Word To Word model.

        Args:
            vocab_src (int): Vocab size of source language.
            vocab_target (int): Vocab size of Target language.
            hidden_size (int): Value of Hidden size of NN.
            version (str, optional): Model version. Defaults to '0.1.0'.
            name (str, optional): Model id. Defaults to 'word2word'.
            optimizer (str, optional): Code of optimizer like (sgd ~> stochastic gradient descent). Defaults to 'sgd'.
            loss (str, optional): Code of loss function like (nll ~> negative log likelihood loss). Defaults to 'crossentropy'.
            learning_rate (float, optional): Learn rate for current model. Defaults to 0.01.
            teacher_forcing_ratio (float, optional): Teacher forcing ratio for current model. Defaults to 0.5.
            momentum (float, optional): Value of Optimizer momentum. Defaults to 0.0.
            drop_out (float, optional): Value of drop out. Defaults to 0.1.

        Raises:
            ValueError: When optimizer code not exist. ['adam', 'sgd','rmsprop']
            ValueError: When loss code not exist. ['nll', 'crossentropy']
        """
        super().__init__(name, version)

        self.optimizer_type = optimizer.lower()
        self.loss = loss.lower()
        self.max_length = MAX_LENGTH
        self.learning_rate = learning_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.vocab_src = vocab_src
        self.vocab_target = vocab_target
        self.name = name
        self.version = version
        self.momentum = momentum
        self.drop_out = drop_out
        self.hidden_size = hidden_size

        self.encoder = EncoderRNN(self.vocab_src, hidden_size).to(DEVICE)
        self.decoder = AttnDecoderRNN(
            hidden_size, self.vocab_target, drop_out).to(DEVICE)

        if self.optimizer_type == 'sgd':

            self.encoder_optimizer = optim.SGD(self.encoder.parameters(),
                                               lr=self.learning_rate,
                                               momentum=momentum
                                               )
            self.decoder_optimizer = optim.SGD(self.decoder.parameters(),
                                               lr=self.learning_rate,
                                               momentum=momentum
                                               )

        elif self.optimizer_type == 'adam':
            self.encoder_optimizer = optim.Adam(self.encoder.parameters(),
                                                lr=self.learning_rate)
            self.decoder_optimizer = optim.Adam(self.decoder.parameters(),
                                                lr=self.learning_rate)

        elif self.optimizer_type == 'rmsprop':
            self.encoder_optimizer = optim.RMSprop(self.encoder.parameters(),
                                                   lr=self.learning_rate,
                                                   momentum=momentum
                                                   )
            self.decoder_optimizer = optim.RMSprop(self.decoder.parameters(),
                                                   lr=self.learning_rate,
                                                   momentum=momentum
                                                   )
        else:
            raise ValueError(
                'optimizer not exist {} !!'.format(self.optimizer_type))

        if self.loss not in ['nll', 'crossentropy']:
            raise ValueError(
                'loss function not exist {} !!'.format(self.loss))

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor = None,  do_train: bool = False):
        """Defines the computation performed at every call.

        Args:
            input_tensor (torch.Tensor): Input tensor.
            target_tensor (torch.Tensor, optional): Output tensor. Defaults to None.
            do_train (bool, optional): If True train mode. Defaults to False.

        Raises:
            ValueError: When input_tensor is not torch.Tensor.
            Exception: When model in train mode and target_tensor is None.

        Returns:
            dict: In train mode metrics (acc,loss).
            list: Decoded index chars.
        """
        if not isinstance(input_tensor, torch.Tensor):
            raise ValueError("input_tensor must be a Tensor")

        if not do_train:

            with torch.no_grad():
                input_length = input_tensor.size()[0]
                encoder_hidden = self.encoder.initHidden()
                max_length = MAX_LENGTH

                encoder_outputs = torch.zeros(
                    max_length, self.encoder.hidden_size, device=DEVICE)

                for ei in range(input_length):
                    encoder_output, encoder_hidden = self.encoder(input_tensor[ei],
                                                                  encoder_hidden)
                    encoder_outputs[ei] += encoder_output[0, 0]

                decoder_input = torch.tensor(
                    [[SOC_TOKEN]], device=DEVICE)  # SOS

                decoder_hidden = encoder_hidden

                decoded_index_words = []
                decoder_attentions = torch.zeros(max_length, max_length)

                for di in range(max_length):
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    decoder_attentions[di] = decoder_attention.data
                    _, topi = decoder_output.data.topk(1)
                    if topi.item() == EOC_TOKEN:
                        break
                    else:
                        decoded_index_words.append(topi.item())

                    decoder_input = topi.squeeze().detach()

                return decoded_index_words

        if target_tensor is None:
            raise Exception('you are in train mode you must set target_tensor')
        if self.loss == 'nll':
            criterion = nn.NLLLoss()
        elif self.loss == 'crossentropy':
            criterion = nn.CrossEntropyLoss()

        input_length = input_tensor.size()[0]
        encoder_hidden = self.encoder.initHidden()
        max_length = MAX_LENGTH

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(
            self.max_length, self.encoder.hidden_size, device=DEVICE)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOC_TOKEN]], device=DEVICE)

        decoder_hidden = encoder_hidden

        decoded_index_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        use_teacher_forcing = True if random.random(
        ) < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()

                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing
                if decoder_input.item() == EOC_TOKEN:
                    break
                decoded_index_words.append(topi.item())

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOC_TOKEN:
                    break
                decoded_index_words.append(topi.item())

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        t = torch.flatten(target_tensor).to(DEVICE)

        while len(decoded_index_words) < t.shape[0]:
            decoded_index_words.append(EOC_TOKEN)

        p = torch.tensor(decoded_index_words).to(DEVICE)

        if p.shape[0] == t.shape[0]:
            acc = accuracy(p, t, subset_accuracy=True)
        else:
            acc = torch.tensor(0)

        cal_loss = loss.item() / target_length

        return {'loss': cal_loss, 'acc': acc.item()}


class Word2Label(BasicModule):
    """Word to Label model.

    Useful for task like token classification and profanity detection.

    Attributes:
        optimizer_type (str): Code of optimizer like (sgd ~> stochastic gradient descent).
        loss (str): Code of loss function like (nll ~> negative log likelihood loss).
        max_length (int): Maximum word length (see sudoai.utils.MAX_LENGTH).
        learning_rate (float): Learn rate for current model.
        teacher_forcing_ratio (float): Teacher forcing ratio for current model.
        vocab_size (int): Vocab size.
        name (str): Model id.
        verison (str): Model version.
        momentum (float): Value of Optimizer momentum if exist.
        drop_out (float): Value of drop out.
        hidden_size (int): Value of Hidden size of NN.
        encoder (:obj:`EncoderRNN`): Encoder.
        decoder (:obj:`AttnDecoderRNN`): Decoder with attention.
        encoder_optimizer (:obj:`torch.optim`): Optimizer for encoder.
        decoder_optmizer (:obj:`torch.optim`): Optimizer for decoder.
        n_class (int): Number of classes.

    Tip:
        For this model, loss must be one of ['nll', 'crossentropy'],
        optimizer must be one of ['adam', 'sgd','rmsprop'].

    """

    def __init__(self,
                 n_class,
                 hidden_size,
                 vocab_size,
                 version='0.1.0',
                 name='word2label',
                 optimizer='sgd',
                 loss='crossentropy',
                 learning_rate=0.01,
                 teacher_forcing_ratio=0.5,
                 momentum=0.0,
                 drop_out=0.1):
        """Create Word To Label model.

        Args:
            n_class (int): Number of classes.
            vocab_size (int): Vocab size.
            hidden_size (int): Value of Hidden size of NN.
            version (str, optional): Model version. Defaults to '0.1.0'.
            name (str, optional): Model id. Defaults to 'word2label'.
            optimizer (str, optional): Code of optimizer like (sgd ~> stochastic gradient descent). Defaults to 'sgd'.
            loss (str, optional): Code of loss function like (nll ~> negative log likelihood loss). Defaults to 'crossentropy'.
            learning_rate (float, optional): Learn rate for current model. Defaults to 0.01.
            teacher_forcing_ratio (float, optional): Teacher forcing ratio for current model. Defaults to 0.5.
            momentum (float, optional): Value of Optimizer momentum. Defaults to 0.0.
            drop_out (float, optional): Value of drop out. Defaults to 0.1.

        Raises:
            ValueError: When optimizer code not exist. ['adam', 'sgd','rmsprop']
            ValueError: When loss code not exist. ['nll', 'crossentropy']
        """
        super().__init__(name, version)
        self.optimizer_type = optimizer.lower()
        self.loss = loss.lower()
        self.max_length = MAX_LENGTH
        self.learning_rate = learning_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.n_class = n_class
        self.vocab_size = vocab_size
        self.name = name
        self.version = version
        self.momentum = momentum
        self.drop_out = drop_out
        self.hidden_size = hidden_size

        self.encoder = EncoderRNN(self.vocab_size,
                                  hidden_size).to(DEVICE)
        self.decoder = AttnDecoderRNN(hidden_size,
                                      self.n_class,
                                      drop_out).to(DEVICE)

        if self.optimizer_type == 'sgd':

            self.encoder_optimizer = optim.SGD(self.encoder.parameters(),
                                               lr=self.learning_rate,
                                               momentum=momentum
                                               )
            self.decoder_optimizer = optim.SGD(self.decoder.parameters(),
                                               lr=self.learning_rate,
                                               momentum=momentum
                                               )

        elif self.optimizer_type == 'adam':
            self.encoder_optimizer = optim.Adam(self.encoder.parameters(),
                                                lr=self.learning_rate)
            self.decoder_optimizer = optim.Adam(self.decoder.parameters(),
                                                lr=self.learning_rate)

        elif self.optimizer_type == 'rmsprop':
            self.encoder_optimizer = optim.RMSprop(self.encoder.parameters(),
                                                   lr=self.learning_rate,
                                                   momentum=momentum
                                                   )
            self.decoder_optimizer = optim.RMSprop(self.decoder.parameters(),
                                                   lr=self.learning_rate,
                                                   momentum=momentum
                                                   )
        else:
            raise ValueError(
                'optimizer not exist {} !!'.format(self.optimizer_type))

        if self.loss not in ['nll', 'crossentropy']:
            raise ValueError(
                'loss function not exist {} !!'.format(self.loss))

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor = None,  do_train: bool = False):
        """Defines the computation performed at every call.

        Args:
            input_tensor (torch.Tensor): Input tensor.
            target_tensor (torch.Tensor, optional): Output tensor. Defaults to None.
            do_train (bool, optional): If True train mode. Defaults to False.

        Raises:
            ValueError: When input_tensor is not torch.Tensor.
            Exception: When model in train mode and target_tensor is None.

        Returns:
            dict: In train mode metrics (acc,loss).
            list: Decoded index chars .
        """
        if not isinstance(input_tensor, torch.Tensor):
            raise ValueError("input_tensor must be a torch.Tensor")

        if not do_train:

            with torch.no_grad():
                input_length = input_tensor.size()[0]
                encoder_hidden = self.encoder.initHidden()
                max_length = MAX_LENGTH

                encoder_outputs = torch.zeros(
                    max_length, self.encoder.hidden_size, device=DEVICE)

                for ei in range(input_length):
                    encoder_output, encoder_hidden = self.encoder(input_tensor[ei],
                                                                  encoder_hidden)
                    encoder_outputs[ei] += encoder_output[0, 0]

                decoder_input = torch.tensor([[0]], device=DEVICE)

                decoder_hidden = encoder_hidden

                decoded_label = []
                decoder_attentions = torch.zeros(max_length, max_length)

                for di in range(self.n_class):
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    decoder_attentions[di] = decoder_attention.data
                    _, topi = decoder_output.data.topk(1)

                    decoded_label.append(topi.item())

                    decoder_input = topi.squeeze().detach()

                return decoded_label, decoded_label.index(1)

        if target_tensor is None:
            raise Exception('you are in train mode you must set target_tensor')
        if self.loss == 'nll':
            criterion = nn.NLLLoss()
        elif self.loss == 'crossentropy':
            criterion = nn.CrossEntropyLoss()

        encoder_hidden = self.encoder.initHidden()

        max_length = MAX_LENGTH

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(
            self.max_length, self.encoder.hidden_size, device=DEVICE)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOC_TOKEN]], device=DEVICE)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random(
        ) < self.teacher_forcing_ratio else False

        decoded_label = []
        decoder_attentions = torch.zeros(max_length, max_length)

        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)

                decoder_attentions[di] = decoder_attention.data
                _, topi = decoder_output.data.topk(1)
                decoded_label.append(topi.item())

                decoder_input = target_tensor[di]
                loss += criterion(decoder_output, target_tensor[di])

        else:

            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)

                decoder_attentions[di] = decoder_attention.data
                _, topi = decoder_output.topk(1)

                decoded_label.append(topi.item())
                decoder_input = topi.squeeze().detach()
                loss += criterion(decoder_output, target_tensor[di])

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        t = torch.flatten(target_tensor).to(DEVICE)

        while len(decoded_label) < t.shape[0]:
            decoded_label.append(0)

        p = torch.tensor(decoded_label).to(DEVICE)

        if p.shape[0] == t.shape[0]:
            acc = accuracy(p, t, subset_accuracy=True)
        else:
            acc = torch.tensor(0)

        cal_loss = loss.item() / target_length

        return {'loss': cal_loss, 'acc': acc.item()}
