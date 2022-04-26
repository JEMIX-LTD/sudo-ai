#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""Pipeline module
@author: Aymen Jemi (jemix) <jemiaymen@gmail.com> at SUDO-AI

"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Dict

import fasttext

from ..models.seq import Seq2Label
from ..models.word import Word2Label, Word2Word
from ..utils import (InputOutput,
                     check_model,
                     datapath,
                     load_model,
                     load_tokenizer)


class PipelineException(Exception):
    """Exception raised in Pipeline class.

    Args:
        reason (str): Reason why pipeline not working.
        msg (str, optional): Human readable string describing the exception. Default to 'pipeline error'

    Attributes:
        reason (str): Reason why pipeline not working.
        msg (str): Human readable string describing the exception.
    """

    def __init__(self, reason: str, msg="pipeline error"):
        self.msg = msg
        self.reason = reason
        super().__init__(self.msg)

    def __str__(self):
        return f'{self.msg} ~> {self.reason}'


@unique
class ModelType(str, Enum):
    """Enumerate for models types, accepted by Pipeline class.

    Attributes:
        WORD_TO_LABEL (str): Word2Label model (1) .
        WORD_TO_WORD  (str): Word2Label model (2).
        SEQ_TO_LABEL (str): Seq2Label model (3).
        SEQ_TO_SEQ (str): For Seq2Seq model (4).
        FASTTEXT (str): FastText model (5).

    """
    WORD_TO_LABEL: str = "1"
    WORD_TO_WORD: str = "2"
    SEQ_TO_LABEL: str = "3"
    SEQ_TO_SEQ: str = "4"
    FASTTEXT: str = "5"


@dataclass
class PipelineConfig():
    """This is a data class definition, to identify all pipeline info.

    A :obj:`PipelineConfig` class definition contains identificaiton and
    allocation information for :obj:`Pipeline`.


    Attributes:
        id (str): Identifier for Model and Tokenizer(s).
        version (str, optional): Model and Pipeline version. Defaults to '0.1.0'
        desscription (str, optional): Pipeline description. Defaults to 'default description'
        is_two_tokenizer (bool, optional): If model have two tokenizer (word2word). Defaults to False
        model_type (:obj:`ModelType`, optional): Model type. Defaults to 'WORD_TO_LABEL'
        i2l (Dict[int, str], optional): Index to label dict. Defaults to {0: "bad", 1: "good"}
    """
    id: str = None
    version: str = '0.1.0'
    description: str = "default description"
    i2l: Dict[int, str] = field(
        default_factory=lambda: ({0: "bad", 1: "good"})
    )
    is_two_tokenizer: bool = False
    model_type: ModelType = ModelType.WORD_TO_LABEL


def load_pipeline_config(id: str, version: str = '0.1.0'):
    """Load pipeline config from local storage with id and version.

    Args:
        id (str): Pipeline config identifier.
        version (str, optional): Pipeline version. Defaults to '0.1.0'.

    Raises:
        FileNotFoundError: When pipeline config file not exist.
        TypeError: When pipeline config file is not correct.

    Returns:
        :obj:`PipelineConfig`: Saved pipeline config.
    """
    id = id.lower()
    config_path = os.path.join(datapath(id), version, 'pipeline.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError('config_pipeline file not found !')

    with open(config_path, 'r') as f:
        config = json.load(f)
    try:
        outputs = PipelineConfig(
            config['id'],
            config['version'],
            config['description'],
            config['i2l'],
            config['is_two_tokenizer'],
            config['model_type']
        )
        return outputs
    except KeyError:
        raise TypeError(
            'pipeline config is not conform check your pipeline config file'
        )


def save_pipeline_config(config: PipelineConfig, override=False):
    """Save pipeline config in local storage with id and version.

    Args:
        config (PipelineConfig): Pipeline config to save it.
        override (bool, optional): If pipeline config file exist ecrase it. Defaults to False.

    Raises:
        FileExistsError: When pipeline config file exist and override is False.
    """
    _config = {
        'id': config.id,
        'version': config.version,
        'description':  config.description,
        'i2l': config.i2l,
        'is_two_tokenizer': config.is_two_tokenizer,
        'model_type': config.model_type
    }
    config_path = os.path.join(
        datapath(config.id),
        config.version,
        'pipeline.json'
    )

    if override:
        if os.path.exists(config_path):
            os.unlink(config_path)

    if os.path.exists(config_path):
        raise FileExistsError()

    with open(config_path, 'w') as json_file:
        json.dump(_config, json_file)


def predict_from_ft(inputs):
    """Normalize predicted result from fasttext.

    Args:
        inputs (tuple(tuple(str),tuple(float))): FastText predict result.

    Returns:
        str: Predicted label.
        list: List of predicted labels.
    """
    if len(inputs[0]) > 1:
        output = []
        for _input in inputs[0]:
            output.append(_input.replace('__label__', ''))
    else:
        label = inputs[0][0]
        output = label.replace('__label__', '')
    return output


class Pipeline():
    """Pipeline with a :obj:`PipelineConfig`.

    Create nlp pipeline with pipeline config file.

    Attributes:
        config (:obj:`PipelineConfig`): Pipeline configuration class.
        model (Model): Current model.
        src_tokenizer (Tokenizer): Source tokenizer.
        target_tokenizer (Tokenizer): Target tokenizer.
        tokenizer (Tokenizer): Tokenizer.
        io (:obj:`InputOutput`): To Download models and tokenizer if not exists.
        kwargs (Dict(str,any)): Additional arguments.

    """

    def __init__(self, config: PipelineConfig = None, **kwargs):
        """Create Pipeline class.

        Args:
            config (:obj:`PipelineConfig`, optional): Pipeline configuration class. Defaults to None.
            k (int, optional): Number of labels (classes). Defaults to 1
            threshold (float, optional): Precision ratio. Defaults to 0.0
            compressed (bool, optional): If True FastText model is compressed. Defaults to False.

        Raises:
            PipelineException: When config args is not :obj:`PipelineConfig`.
        """
        self.model = None
        self.tokenizer = None
        self.src_tokenizer = None
        self.target_tokenizer = None
        self.config = None
        self.io = InputOutput()
        self.kwargs = kwargs

        if config is not None:
            if isinstance(config, PipelineConfig):
                self.config = config
            else:
                raise PipelineException(
                    'type error', 'config is not PipelineConfig type'
                )
        elif 'id' in kwargs and isinstance(kwargs['id'], str):
            if 'version' in kwargs and isinstance(kwargs['version'], str):
                self.config = load_pipeline_config(
                    kwargs['id'],
                    kwargs['version']
                )
            else:
                self.config = load_pipeline_config(kwargs['id'])

        self.load()

    def load(self):
        """Load model and tokenizer from Pipeline config.

        Raises:
            PipelineException: When config file not found.
            PipelineException: When model is None.
            PipelineException: When tokenizer(s) is None.
        """
        if self.config.model_type == ModelType.FASTTEXT:
            model_path = os.path.join(datapath(self.config.id),
                                      self.config.version,
                                      'model.bin')
            model_path_compressed = os.path.join(datapath(self.config.id),
                                                 self.config.version,
                                                 'model.ftz')
            if not os.path.exists(model_path) and not os.path.exists(model_path_compressed):
                raise PipelineException(
                    'Model not found',
                    'check your model file .bin or .ftz'
                )
            if 'compressed' in self.kwargs and os.path.exists(model_path_compressed):
                self.model = fasttext.load_model(model_path_compressed)
            else:
                self.model = fasttext.load_model(model_path)

            return

        if not check_model(self.config.id, self.config.version):
            self.io.download_from_drive(self.config.id)

        self.model = load_model(self.config.id,
                                self._get_model_type(),
                                self.config.version)

        if not self.config.is_two_tokenizer:
            self.tokenizer = load_tokenizer(self.config.id)
        else:
            self.src_tokenizer = load_tokenizer(f'{self.config.id}/src')
            self.target_tokenizer = load_tokenizer(f'{self.config.id}/target')

        if self.model is None:
            raise PipelineException('Model not found', 'Error loading model')
        if self.tokenizer is None and (self.src_tokenizer is None or self.target_tokenizer is None):
            raise PipelineException(
                'tokenizer not found', 'Error loading tokenizer')

    def _parse(self, inputs):
        """Preprocess inputs

        Args:
            inputs (str): Inputs to work with.

        Returns:
            str: Clean inputs.
        """
        if self.config.model_type == ModelType.FASTTEXT:
            return inputs

        if not self.config.is_two_tokenizer:
            return self.tokenizer(inputs)
        else:
            return self.src_tokenizer(inputs)

    def _forward(self, inputs):
        """Predict process

        Args:
            inputs (str): Inputs to predict.
        """
        if self.config.model_type == ModelType.FASTTEXT:

            if 'k' in self.kwargs:
                k = self.kwargs['k']
            else:
                k = 1
            if 'threshold' in self.kwargs:
                threshold = self.kwargs['threshold']
            else:
                threshold = 0.0

            return predict_from_ft(self.model.predict(text=inputs, k=k, threshold=threshold))

        if not self.config.is_two_tokenizer:
            output = self.model(inputs)
            if self.config.i2l is not None:
                if len(output) > 1:
                    return self.config.i2l[str(output[1])]
                return self.config.i2l[output]
            else:
                return output
        else:
            ids = self.model(inputs)
            output = self.target_tokenizer.decode(ids)
            if self.config.model_type == ModelType.WORD_TO_WORD:
                return ''.join(output)
            elif self.config.model_type == ModelType.SEQ_TO_SEQ:
                return ' '.join(output)

    def _get_model_type(self):
        """Get model from Pipeline configuration.

        Returns:
            :obj:`Word2Label`: Word To Label model.
            :obj:`Seq2Label`: Sequence to Label model.
            :obj:`Word2Word`: Word To Word model.
            None: If model type not in (w2w,s2l,w2l)
        """
        if self.config.model_type == ModelType.WORD_TO_LABEL:
            return Word2Label
        elif self.config.model_type == ModelType.SEQ_TO_LABEL:
            return Seq2Label
        elif self.config.model_type == ModelType.WORD_TO_WORD:
            return Word2Word

        return None

    def predict(self, inputs):
        """Predict method.

        Args:
            inputs (str) Input for predictions.

        Raises:
            PipelineException: When inputs is None.
        """
        return self(inputs)

    def __call__(self, **kwargs):
        """Predict method.

        Args:
            inputs (str, optional) Input for predictions. Defaults to None.

        Raises:
            PipelineException: When inputs is None.
        """
        if 'inputs' in kwargs:
            _input = self._parse(kwargs['inputs'])
            return self._forward(_input)
        else:
            raise PipelineException(
                'param inputs not found', 'to predict you must enter inputs'
            )

    def save(self, override: bool = False):
        """Save Pipeline.

        Args:
            override (bool, optional): If pipeline config file exist ecrase it. Defaults to False.
        """
        save_pipeline_config(self.config, override)
