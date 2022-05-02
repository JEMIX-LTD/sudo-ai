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


import json
import os

import torch

from ..utils import DEVICE, datapath


def load_config(model_name: str, version: str = '0.1.0'):
    config_path = os.path.join(datapath(model_name), version, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError('config file not found !')

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def save_config(model, override=False, is_local=False, folder_path=None):

    if model.__class__.__name__ == 'Word2Word':
        config = {
            'vocab_src': model.vocab_src,
            'vocab_target':  model.vocab_target,
            'hidden_size': model.hidden_size,
            'version': model.version,
            'name': model.name,
            'optimizer': model.optimizer_type,
            'loss': model.loss,
            'learning_rate': model.learning_rate,
            'teacher_forcing_ratio': model.teacher_forcing_ratio,
            'momentum': model.momentum,
            'drop_out': model.drop_out
        }

    elif model.__class__.__name__ in ['Word2Label', 'Seq2Label']:
        config = {
            'vocab_size': model.vocab_size,
            'n_class':  model.n_class,
            'hidden_size': model.hidden_size,
            'version': model.version,
            'name': model.name,
            'optimizer': model.optimizer_type,
            'loss': model.loss,
            'learning_rate': model.learning_rate,
            'teacher_forcing_ratio': model.teacher_forcing_ratio,
            'momentum': model.momentum,
            'drop_out': model.drop_out
        }
    else:
        raise ValueError('model not found !')

    if is_local and folder_path is not None:
        config_path = os.path.join(folder_path, 'config.json')
    else:
        config_path = os.path.join(
            datapath(model.name),
            model.version,
            'config.json'
        )

    if override:
        if os.path.exists(config_path):
            os.unlink(config_path)

    if os.path.exists(config_path):
        raise FileExistsError()

    with open(config_path, 'w') as json_file:
        json.dump(config, json_file)


def load_model(model: str,
               model_type,
               model_version='0.1.0',
               train=False):

    model_path = os.path.join(datapath(model), model_version, 'model.bin')

    if not os.path.exists(model_path):
        raise FileNotFoundError('model not found please check your path')

    config = load_config(model, model_version)

    if model_type.__name__ == 'Word2Word':
        model_result = model_type(
            hidden_size=config['hidden_size'],
            vocab_src=config['vocab_src'],
            vocab_target=config['vocab_target'],
            version=config['version'],
            name=model,
            optimizer=config['optimizer'],
            loss=config['loss'],
            learning_rate=config['learning_rate'],
            teacher_forcing_ratio=config['teacher_forcing_ratio'],
            momentum=config['momentum'],
            drop_out=config['drop_out']
        )
    elif model_type.__name__ == 'Word2Label':
        model_result = model_type(
            hidden_size=config['hidden_size'],
            n_class=config['n_class'],
            vocab_size=config['vocab_size'],
            version=config['version'],
            name=model,
            optimizer=config['optimizer'],
            loss=config['loss'],
            learning_rate=config['learning_rate'],
            teacher_forcing_ratio=config['teacher_forcing_ratio'],
            momentum=config['momentum'],
            drop_out=config['drop_out']
        )
    elif model_type.__name__ == 'Seq2Label':
        model_result = model_type(
            hidden_size=config['hidden_size'],
            n_class=config['n_class'],
            vocab_size=config['vocab_size'],
            version=config['version'],
            name=model,
            optimizer=config['optimizer'],
            loss=config['loss'],
            learning_rate=config['learning_rate'],
            teacher_forcing_ratio=config['teacher_forcing_ratio'],
            momentum=config['momentum'],
            drop_out=config['drop_out']
        )
    else:
        raise Exception('model type not found !!')

    if DEVICE == 'cpu':

        model_result.load_state_dict(torch.load(
            model_path,  map_location=torch.device('cpu')))
    else:
        model_result.load_state_dict(torch.load(
            model_path,  map_location=torch.device('cuda')))

    if train:
        model_result.train()
    else:
        model_result.eval()
    return model_result


def save_checkpoint(model, epoch: int, loss: float, optimizer: list, path: str):
    """[summary]

    Args:
        model (Model): training model
        epoch (int): number of current epoch
        loss (float): calculate loss
        optimizer (list): list of current state dict optimizer
        path (str): path to save the checkpoint
    """
    optimizer_state_dicts = [x.state_dict() for x in optimizer]

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dicts': optimizer_state_dicts,
        'loss': loss
    }, path)


def load_checkpoint(path: str, model, do_train=True):
    """load saved checkpoint

    Args:
        path (str): path of saved checkpoint
        model (Model): type of model to load
        do_train (bool, optional): train mode. Defaults to True.

    Returns:
        [type]: [description]
    """
    if DEVICE == 'cpu':
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path, map_location=torch.device('cuda'))

    model.load_state_dict(checkpoint['model_state_dict'])

    model.encoder_optimizer.load_state_dict(
        checkpoint['optimizer_state_dicts'][0])

    model.decoder_optimizer.load_state_dict(
        checkpoint['optimizer_state_dicts'][1])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    if do_train:
        model.train()
    else:
        model.eval()

    return model, epoch, loss


def checkpoint_to_model(path: str, model, do_train=True):
    model, _, _ = load_checkpoint(path, model, do_train)
    return model


def check_model(model: str, version: str = '0.1.0'):
    return os.path.exists(os.path.join(datapath(model), version, 'model.bin'))
