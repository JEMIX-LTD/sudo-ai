# -*- coding: utf-8 -*-

"""
@author: Aymen Jemi (jemix) <jemiaymen@gmail.com>

Copyright (c) 2021 Aymen Jemi SUDO-AI
"""
import sudoai
from ax.service.ax_client import AxClient

from ..trainer import (Seq2LabelTrainer,
                       TokenClassificationTrainer,
                       Word2WordTrainer)


def seq2label_hypertuning(dataset_id: str,
                          n_experience: int = 30,
                          dataset_version: str = '0.1.0',
                          test_mode: bool = False):
    """Hyper tuning method for Seq2Label model.

    Args:
        dataset_id (str): Dataset unique id.
        n_experience (int, optional): Number of experiences. Defaults to 30.
        dataset_version (str, optional): Dataset version. Defaults to '0.1.0'.
        test_mode (bool, optional): Test mode to try with small amount of data. Defaults to False.

    Note:
        For more information check quickstart docs http://sudoai.tech/quickstart

    Returns:
        tuple: (best_parameters, metrics).

    """
    def evaluate_seq2label_train(parameters, dataset_id, dataset_version='0.1.0'):
        trainer = Seq2LabelTrainer(
            epochs=1,
            id=dataset_id,
            version=dataset_version,
            hidden_size=parameters["hidden_size"],
            lr=parameters["learning_rate"],
            momentum=parameters["momentum"],
            loss=parameters["loss"],
            optimizer=parameters["optimizer"],
            do_eval=True,
            do_shuffle=True
        )
        return trainer(hyperparam=True, test_mode=test_mode)

    ax_client = AxClient()
    ax_client.create_experiment(
        name="tokenclassification_torch_test_experiment",
        parameters=[
            {
                "name": "learning_rate",
                "type": "range",
                "bounds": [1e-6, 0.4],
                "value_type": "float",
                "log_scale": True
            },
            {
                "name": "momentum",
                "type": "range",
                "bounds": [0.001, 1.0],
            },
            {
                "name": "hidden_size",
                "type": "choice",
                "values": [16, 32, 64, 128],
            },
            {
                "name": "loss",
                "type": "choice",
                "values": ['nll', 'crossentropy'],
            },
            {
                "name": "optimizer",
                "type": "choice",
                "values": ['sgd', 'adam', 'rmsprop'],
            },
        ],
        objective_name="val_loss",
        minimize=True,
    )

    for _ in range(n_experience):
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(
            trial_index=trial_index, raw_data=evaluate_seq2label_train(parameters, dataset_id, dataset_version))

    best_parameters, metrics = ax_client.get_best_parameters()

    sudoai.__log__.info(f'best params :  {best_parameters}')
    sudoai.__log__.info(f'best metrics values : {metrics}')

    return best_parameters, metrics


def token_hypertuning(dataset_id: str,
                      n_experience: int = 30,
                      dataset_version: str = '0.1.0',
                      test_mode: bool = False):
    """Hyper tuning method for token classification (word2label) model.

    Args:
        dataset_id (str): Dataset unique id.
        n_experience (int, optional): Number of experiences. Defaults to 30.
        dataset_version (str, optional): Dataset version. Defaults to '0.1.0'.
        test_mode (bool, optional): Test mode to try with small amount of data. Defaults to False.

    Note:
        For more information check quickstart docs http://sudoai.tech/quickstart

    Returns:
        tuple: (best_parameters, metrics).

    """

    def evaluate_token_train(parameters, dataset_id, dataset_version='0.1.0'):
        trainer = TokenClassificationTrainer(
            id=dataset_id,
            version=dataset_version,
            hidden_size=parameters["hidden_size"],
            lr=parameters["learning_rate"],
            momentum=parameters["momentum"],
            loss=parameters["loss"],
            optimizer=parameters["optimizer"],
            do_eval=True,
            do_shuffle=True,
            epochs=1
        )
        return trainer(hyperparam=True, test_mode=test_mode)

    ax_client = AxClient()
    ax_client.create_experiment(
        name="tokenclassification_torch_test_experiment",
        parameters=[
            {
                "name": "learning_rate",
                "type": "range",
                "bounds": [1e-6, 0.4],
                "value_type": "float",
                "log_scale": True
            },
            {
                "name": "momentum",
                "type": "range",
                "bounds": [0.001, 1.0],
            },
            {
                "name": "hidden_size",
                "type": "choice",
                "values": [16, 32, 64, 128],
            },
            {
                "name": "loss",
                "type": "choice",
                "values": ['nll', 'crossentropy'],
            },
            {
                "name": "optimizer",
                "type": "choice",
                "values": ['sgd', 'adam', 'rmsprop'],
            },
        ],
        objective_name="val_loss",
        minimize=True,
    )

    for _ in range(n_experience):
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(
            trial_index=trial_index, raw_data=evaluate_token_train(parameters, dataset_id, dataset_version))

    best_parameters, metrics = ax_client.get_best_parameters()

    sudoai.__log__.info(f'best params :  {best_parameters}')
    sudoai.__log__.info(f'best metrics values : {metrics}')

    return best_parameters, metrics


def w2w_hypertuning(dataset_id: str,
                    n_experience: int = 30,
                    dataset_version: str = '0.1.0',
                    test_mode: bool = False):
    """Hyper tuning method for (word2word) model.

    Args:
        dataset_id (str): Dataset unique id.
        n_experience (int, optional): Number of experiences. Defaults to 30.
        dataset_version (str, optional): Dataset version. Defaults to '0.1.0'.
        test_mode (bool, optional): Test mode to try with small amount of data. Defaults to False.

    Note:
        For more information check quickstart docs http://sudoai.tech/quickstart

    Returns:
        tuple: (best_parameters, metrics).

    """
    def evaluate_w2w_train(parameters, dataset_id, dataset_version='0.1.0', test_mode=False):

        trainer = Word2WordTrainer(
            id=dataset_id,
            version=dataset_version,
            hidden_size=parameters["hidden_size"],
            lr=parameters["learning_rate"],
            momentum=parameters["momentum"],
            loss=parameters["loss"],
            optimizer=parameters["optimizer"],
            do_eval=True,
            do_shuffle=True,
            epochs=1
        )
        return trainer(hyperparam=True, test_mode=test_mode)

    ax_client = AxClient()
    ax_client.create_experiment(
        name="ttd_torch_test_experiment",
        parameters=[
            {
                "name": "learning_rate",
                "type": "range",
                "bounds": [1e-6, 1e-4],
                "value_type": "float",
                "log_scale": True
            },
            {
                "name": "momentum",
                "type": "range",
                "bounds": [0.1, 0.5],
            },
            {
                "name": "hidden_size",
                "type": "choice",
                "values": [256, 512],
            },
            {
                "name": "loss",
                "type": "choice",
                "values": ['nll', 'crossentropy'],
            },
            {
                "name": "optimizer",
                "type": "choice",
                "values": ['adam', 'rmsprop'],
            },
        ],
        objective_name="val_loss",
        minimize=True,
    )

    for _ in range(n_experience):
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(
            trial_index=trial_index, raw_data=evaluate_w2w_train(parameters, dataset_id, dataset_version, test_mode))

    best_parameters, metrics = ax_client.get_best_parameters()

    sudoai.__log__.info(f'best params :  {best_parameters}')
    sudoai.__log__.info(f'best metrics values : {metrics}')

    return best_parameters, metrics
