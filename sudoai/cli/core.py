#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""Commands module
@author: Aymen Jemi (jemix) <jemiaymen@gmail.com> at SUDO-AI

Commande line tools for sudoai.
"""

from ..utils import InputOutput, checkpoint_to_model, load_model
import argparse
import sudoai
from ..models.word import Word2Word, Word2Label
from ..models.seq import Seq2Label


def io():
    """sudoai-io Commande line to Download and Upload pipelines, dataset and models.

    Args:
        --upload (bool, optional): To upload pipeline, dataset and model. Defaults to False.
        --download (bool, optional): To download pipeline, dataset and model. Defaults to True.
        --id (str): Unique ID of pipeline, dataset or model.

    Warning:
        If you set --upload and --download to True, an exception raised.

    Raises:
        TypeError: When you set --upload and --download to True.
        TypeError: When you don't set --upload and --download.

    Examples:
        $ sudoai-io --download=True --id=trans
        $ sudoai-io --upload=True --id=ttd
        $ sudoai-io --id=ttd
    """

    parser = argparse.ArgumentParser(
        prog="sudoai-io Download and Upload pipelines, dataset and models.",
        description="Arguments for sudoai-io",
        usage="sudoai-io --upload=True --id=[pipeline_id]",
        allow_abbrev=True
    )

    parser.add_argument(
        "--upload",
        type=bool,
        default=False,
        help="upload commande",
    )

    parser.add_argument(
        "--download",
        type=bool,
        default=True,
        help="download commande",
    )

    parser.add_argument(
        "--id",
        type=str,
        help="Unique ID of pipeline, dataset or model",
        required=True
    )

    args = parser.parse_args()

    if args.upload is False and args.download is False:
        raise TypeError("--upload or --download must chose one")

    if args.upload is True and args.download is True:
        raise TypeError(
            "--upload and --download is set to True (you should set just one to True)")

    _io = InputOutput()

    if args.upload is True:
        sudoai.__log__.info(f'upload in drive id : {args.id}')
        _io.upload_in_drive(args.id)

    if args.download is True:
        sudoai.__log__.info(f'download from drive id : {args.id}')
        _io.download_from_drive(args.id)


def ch2m():
    """sudoai-ch2m Commande line to generate model from checkpoint.

    Args:
        --chp (str): Path of checkpoint.
        --id (str): Unique ID of model.
        --mtype (str, optional): Model Type. Defaults to w2w.
        --train (bool, optional): Train mode. Defaults to False.
        --version (str, optional): Model version. Defaults to '0.1.0'.

    Examples:
        $ sudoai-ch2m --chp=checkpoint-1.chp --id=trans --train=False
        $ sudoai-ch2m --chp=checkpoint-2.chp --id=trans
    """
    parser = argparse.ArgumentParser(
        prog="sudoai-ch2m Commande line to generate model from checkpoint.",
        description="Generate model from checkpoint file.",
        usage="sudoai-ch2m --chp=[path_of_checkpoint] --id=[id_of_model] --mtype=[w2w] --train=[False] --version=[0.1.1]",
        allow_abbrev=True
    )

    parser.add_argument(
        "--chp",
        type=str,
        help="path of checkpoint",
        required=True
    )

    parser.add_argument(
        "--id",
        type=str,
        help="ID of model",
        required=True
    )

    parser.add_argument(
        "--mtype",
        type=str,
        help="type of model",
        default='w2w',
        choices=['w2w', 'w2l', 's2l']
    )

    parser.add_argument(
        "--train",
        type=bool,
        help="Train mode",
        default=False
    )

    parser.add_argument(
        "--version",
        type=str,
        help="model version",
        default="0.1.0"
    )

    args = parser.parse_args()

    if args.mtype == 'w2w':
        t = Word2Word

    if args.mtype == 'w2l':
        t = Word2Label

    if args.mtype == 's2l':
        t = Seq2Label

    model = load_model(args.id, t, args.version, args.train)

    model = checkpoint_to_model(args.chp, model, args.train)

    model.save()
