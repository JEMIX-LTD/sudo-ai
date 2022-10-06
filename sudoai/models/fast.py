#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""@author: Aymen Jemi (jemix) <jemiaymen@gmail.com>

Copyright (c) 2021 Aymen Jemi SUDO-AI
"""

import os

import fasttext

from ..pipeline import predict_from_ft
from ..preprocess import clean_text, strip_duplicated_letter
from ..utils import datapath
import sudoai


class FastModel():
    """Fast text model based on facebook fasttext.

    Attributes:
        train (str): Path of train file.
        valid (str): Path of validation file.
        duration (int): Time for autotune.
        auto_metric_label (str): Label for autotune adjust.
        model (:obj:`fasttext._FastText`) Fast text model.
        version (str): Model version.
        ziped (bool): If True current model zip before save.
        id (str): Model ID.
        id_trained (bool): If True current model is trained.
    """

    def __init__(self,
                 id: str,
                 train_path: str = None,
                 valid_path: str = None,
                 version: str = '0.1.0',
                 duration: int = 600,
                 is_ziped: bool = False,
                 auto_metric: str = None):
        """Create FastModel model.

        Args:
            id (str): Model ID.
            train_path (str): Path of train file.
            valid_path (str, optional): Path of validation file. Defaults to None.
            version (str, optional): Model version. Defaults to '0.1.0'.
            duration (int, optional): Time for autotune. Defaults to 600.
            is_ziped (bool, optional): If True current model zip before save. Defaults to False.
            auto_metric (str, optional): Label for autotune adjust. Defaults to None.
        """
        self.train = train_path
        self.valid = valid_path
        self.duration = duration
        self.auto_metric_label = auto_metric
        self.model = None
        self.version = version
        self.ziped = is_ziped
        self.id = id
        self.is_trained = False

    def start(self, **kwargs):
        """Start Train the model.

        Args:
            auto (bool, optional): If True model is autotune mode. Defaults to False.
            epoch (int, optional): Number of epochs. Defaults to 50.
            loss (str, optional): Loss function. Defaults to 'hs'.
            lr (float, optional): Learning rate value. Defaults to None.


        Raises:
            FileExistsError: When train data not found.
            FileExistsError: When validation data not found.
        """

        if 'auto' in kwargs:
            auto = kwargs['auto']
        else:
            auto = False

        if auto:
            if not os.path.exists(self.valid):
                raise FileExistsError('data validation not found!')
            if self.auto_metric_label is not None:
                self.model = fasttext.train_supervised(input=self.train,
                                                       autotuneValidationFile=self.valid,
                                                       autotuneMetric=f"f1:__label__{self.auto_metric_label}",
                                                       autotuneDuration=self.duration)
            else:
                self.model = fasttext.train_supervised(input=self.train,
                                                       autotuneValidationFile=self.valid,
                                                       autotuneDuration=self.duration)
        else:
            if 'epoch' in kwargs:
                epochs = kwargs['epoch']
            else:
                epochs = 50

            if 'loss' in kwargs:
                loss = kwargs['loss']
            else:
                loss = 'hs'

            if 'lr' in kwargs:
                lr = kwargs['lr']
                self.model = fasttext.train_supervised(input=self.train,
                                                       lr=lr,
                                                       loss=loss,
                                                       epoch=epochs)
            else:
                self.model = fasttext.train_supervised(input=self.train,
                                                       loss=loss,
                                                       epoch=epochs)
        sudoai.__log__.info('train model [{self.id}] done')

        if 'eval' in kwargs:
            if not os.path.exists(self.valid):
                raise FileExistsError('data validation not found!')

            self.model.test(self.valid)
            sudoai.__log__.info('eval model [{self.id}] done')

        self.is_trained = True

    def save(self, **kwargs):
        """Save current model.

        Args:
            retrain (bool, optional): In ziped mode if True retrain model.
            qnorm (bool, optional): In ziped mode if True Normalize current model.
        """
        self._is_ready()

        if self.ziped:
            model_path = os.path.join(datapath(self.id.lower()),
                                      self.version,
                                      'model.tfz')
            if 'retrain' in kwargs:
                retrain = kwargs['retrain']
            else:
                retrain = False
            if 'qnorm' in kwargs:
                qnorm = kwargs['qnorm']
            else:
                qnorm = False

            self.quantize(retrain, qnorm)

            self.model.save_model(model_path)
        else:
            model_path = os.path.join(datapath(self.id.lower()),
                                      self.version,
                                      'model.bin')
            self.model.save_model(model_path)

        sudoai.__log__.info(f'model [{self.id}] saved')

    def quantize(self, retrain: bool = True, qnorm: bool = True):
        """Quantize the model reducing the size of the model and it's memory footprint.

        Args:
            retrain (bool, optional): Retrain mode. Defaults to True.
            qnorm (bool, optional): Normalize current model. Defaults to True.
        """
        self._is_ready()
        self.model.quantize(input=self.train, qnorm=qnorm, retrain=retrain)

        sudoai.__log__.info(f'model [{self.id}] compressed')

    @classmethod
    def load(self, id: str, version: str = '0.1.0', is_ziped=False):
        """Load saved model with id and version.

        Args:
            id (str): Model ID.
            version (str, optional): Model version. Defaults to '0.1.0'.
            is_ziped (bool, optional): If True the model is compressed. Defaults to False.

        Raises:
            FileNotFoundError: When model file not found.

        Returns:
            :obj:`FastModel`: FastModel class.
        """
        if is_ziped:
            model_path = os.path.join(datapath(id.lower()),
                                      version,
                                      'model.tfz')
        else:
            model_path = os.path.join(datapath(id.lower()),
                                      version,
                                      'model.bin')

        if not os.path.exists(model_path):
            raise FileNotFoundError('fasttext model file not found')

        self.model = fasttext.load_model(model_path)

        return self

    def predict(self, input: str, **kwargs):
        """Predict class from input.

        Args:
            input (str): Input text.
            clean (bool, optional): If True clean the input text. Defaults to False.
            strip_duplicated (bool, optional): If True stripes duplicated chars. Defaults to False.
            norm (bool, optional): If True normalize the prediction. Defaults to False.
            k (int, optional): Number of classes to predicted. Defaults to 1.
            threshold (float, optional): Minimum score for prediction. Defaults to 0.0.


        Returns:
            list: Labels predicted.
            Tuple: Labels predicted and scores for each prediction.
        """
        self._is_ready()

        if 'clean' in kwargs:
            clean = kwargs['clean']
        else:
            clean = False

        if 'strip_duplicated' in kwargs:
            duplicated = kwargs['strip_duplicated']
        else:
            duplicated = False
        _in = self._parse(input, clean, duplicated)

        if 'norm' in kwargs:
            norm = kwargs['norm']
        else:
            norm = False

        if 'k' in kwargs:
            k = kwargs['k']
        else:
            k = 1

        if 'threshold' in kwargs:
            threshold = kwargs['threshold']
        else:
            threshold = 0.5

        if norm:
            return predict_from_ft(self.model.predict(_in, k, threshold))

        return self.model.predict(_in, k, threshold)

    def _parse(self, input: str, clean: bool = False, strip_duplicated: bool = False):
        """Parse text clean and strip duplicated chars (preprocessing).

        Args:
            input (str): Input text.
            clean (bool, optional): Clean text. Defaults to False.
            strip_duplicated (bool, optional): Strip duplicated characters. Defaults to False.

        Returns:
            str: Text parsed.
        """
        output = input
        if clean:
            output = clean_text(output)
        if strip_duplicated:
            output = strip_duplicated_letter(output)
        return output

    def __call__(self, input: str, **kwargs):
        """Predict class from input.

        Args:
            input (str): Input text.
            clean (bool, optional): If True clean the input text. Defaults to False.
            strip_duplicated (bool, optional): If True stripes duplicated chars. Defaults to False.
            norm (bool, optional): If True normalize the prediction. Defaults to False.
            k (int, optional): Number of classes to predicted. Defaults to 1.
            threshold (float, optional): Minimum score for prediction. Defaults to 0.0.


        Returns:
            list: Labels predicted.
            Tuple: Labels predicted and scores for each prediction.
        """
        return self.predict(input, kwargs)

    def _is_ready(self):
        """Check if current model is trained or not.

        Raises:
            ValueError: When model is not trained.
        """
        if self.is_trained is False:
            raise ValueError('model not trained yet please call start method')
