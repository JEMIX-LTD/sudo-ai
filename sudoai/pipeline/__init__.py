#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""Pipeline module
@author: Aymen Jemi (jemix) <jemiaymen@gmail.com> at SUDO-AI

Pipeline for natural language processing, sudoai.pipeline
is a lightweight library to run easily any model.

Tip:
    compressed arg work only on FastText model.


Examples:
    These examples illustrate how to use sudoai Pipeline.

    Create and save pipeline:

    >>> config = PipelineConfig(id='lid',model_type=ModelType.FASTTEST)
    >>> nlp = Pipeline(config=config, compressed=True)
    >>> nlp(inputs='chbik blid sa7ebi')
    'aeb'
    >>> nlp(inputs='I ask always and no one answer me')
    'eng'
    >>> nlp.save(True)

"""

from ..pipeline.core import (Pipeline,
                             PipelineConfig,
                             ModelType,
                             PipelineException,
                             predict_from_ft,
                             save_pipeline_config,
                             load_pipeline_config)

__all__ = ['Pipeline',
           'PipelineConfig',
           'ModelType',
           'PipelineException',
           'predict_from_ft',
           'save_pipeline_config',
           'load_pipeline_config']
