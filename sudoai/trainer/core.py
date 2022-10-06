#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""@author: Aymen Jemi (jemix) <jemiaymen@gmail.com>

Copyright (c) 2021 Aymen Jemi SUDO-AI
"""

import sudoai
import os
import random
import shutil
from datetime import datetime

import wandb
from tqdm.auto import tqdm

from ..dataset import DatasetError,  DatasetType
from ..models.seq import Seq2Label
from ..models.word import Word2Label, Word2Word
from ..models.xmltc import HybridXMLTC
from ..models.regression import LogisticRegression
from ..utils import DEVICE, load_checkpoint, load_dataset, save_checkpoint
import torch
import uuid


class ModelError(Exception):
    """Exception raised in Trainer class.

    Args:
        model (:obj:`torch.nn.Module`): Current Model object.
        message (str, optional): Human readable string describing the exception. Defaults to "Model error.".

    Attributes:
        model (:obj:`torch.nn.Module`): Current Model object.
        message (str): Human readable string describing the exception. Defaults to "Model error.".

    """

    def __init__(self, model, message="Model error."):
        self.model = model
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.model} ~> {self.message}'


class Trainer():
    """The base class sudoai.trainer.Trainer representing a Trainer.

    All Trainers that represent a training process should subclass it.
    All subclasses should overwrite __init__(), supporting the new attributes.
    Subclasses could also optinally overwrite steps(), which is the main process
    for training steps.

    Attributes:
        id (str): Dataset identifier.
        version (str, optional): Model version. Defaults to '0.1.0'.
        teacher_forcing_ratio (float, optional): Teacher Forcing ratio for acceleration. Defaults to 0.5.
        hidden_size (int, optional): Hidden size of the model. Defaults to 512.
        print_every (int, optional): Log frequency result by steps (log result every 30000 steps). Defaults to 30000.
        lr (float, optional): Learn rate value. Defaults to 0.0001.
        epochs (int, optional): Number of epochs. Defaults to 2.
        drop_out (float, optional): Drop out value. Defaults to 0.1.
        do_eval (bool, optional): If True evaluate. Defaults to False.
        do_save (bool, optional): If True save the model when training ends. Defaults to False.
        do_shuffle (bool, optional): If True shuffle the dataset. Defaults to True.
        do_save_checkpoint (bool, optional): If True save checkpoint every epoch. Defaults to False.
        continue_from_checkpoint (str, optional): Path of checkpoint as start point. Defaults to None.
        loss (str, optional): Loss function. Defaults to 'nll'.
        optimizer (str, optional): Optimizer function. Defaults to 'rmsprop'.
        momentum (float, optional): Momentum value for optimizer. Defaults to 0.0.
        wandb (wandb, optional): Wandb class to track logs. Defaults to None.
        wandb_key (str, optional): Wandb api key. Defaults to None.
        save_runs (bool, optional): If True save runs. Defaults to False.
        base_path (str, optional): Base path to save checkpoint and model. Defaults to None.

    See Also:
        For more information about wandb check pytorch docs https://docs.wandb.ai/guides/integrations/pytorch

    """

    def __init__(self,
                 id: str,
                 version: str = '0.1.0',
                 teacher_forcing_ratio: float = 0.5,
                 hidden_size: int = 512,
                 print_every: int = 100,
                 lr: float = 0.0001,
                 epochs: int = 2,
                 drop_out: float = 0.1,
                 do_eval: bool = False,
                 do_save: bool = False,
                 do_shuffle: bool = True,
                 do_save_checkpoint: bool = False,
                 continue_from_checkpoint: str = None,
                 loss: str = 'nll',
                 optimizer: str = 'rmsprop',
                 momentum: float = 0.0,
                 wandb: wandb = None,
                 wandb_key: str = None,
                 save_runs: bool = True,
                 base_path: str = None) -> None:
        """Create a Trainer class.

        Args:
            id (str): Dataset identifier.
            version (str, optional): Model version. Defaults to '0.1.0'.
            teacher_forcing_ratio (float, optional): Teacher Forcing ratio for acceleration. Defaults to 0.5.
            hidden_size (int, optional): Hidden size of the model. Defaults to 512.
            print_every (int, optional): Log frequency result by steps (log result every 30000 steps). Defaults to 30000.
            lr (float, optional): Learn rate value. Defaults to 0.0001.
            epochs (int, optional): Number of epochs. Defaults to 2.
            drop_out (float, optional): Drop out value. Defaults to 0.1.
            do_eval (bool, optional): If True evaluate. Defaults to False.
            do_save (bool, optional): If True save the model when training ends. Defaults to False.
            do_shuffle (bool, optional): If True shuffle the dataset. Defaults to True.
            do_save_checkpoint (bool, optional): If True save checkpoint every epoch. Defaults to False.
            continue_from_checkpoint (str, optional): Path of checkpoint as start point. Defaults to None.
            loss (str, optional): Loss function. Defaults to 'nll'.
            optimizer (str, optional): Optimizer function. Defaults to 'rmsprop'.
            momentum (float, optional): Momentum value for optimizer. Defaults to 0.0.
            wandb (wandb, optional): Wandb class to track logs. Defaults to None.
            wandb_key (str, optional): Wandb api key. Defaults to None.
            base_path (str, optional): Base path to save checkpoint and model. Defaults to None.
        """

        self.id = id
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.hidden_size = hidden_size
        self.print_every = print_every
        self.learning_rate = lr
        self.epochs = epochs
        self.do_eval = do_eval
        self.drop_out = drop_out
        self.do_save = do_save
        self.do_shuffle = do_shuffle
        self.do_save_checkpoint = do_save_checkpoint
        self.continue_from_checkpoint = continue_from_checkpoint
        self.loss = loss
        self.optimizer = optimizer
        self.momentum = momentum
        self.base_path = base_path
        self.model = None
        self.version = version
        self.dataset = None
        self.wandb = wandb
        self.wandb_key = wandb_key
        self.save_runs = save_runs

        self.test_mode = False
        self.num_workers = 1

    def save(self) -> str:
        """Save model with id and version.

        Returns:
            str: Path of model saved to.
        """
        path = self.model.save()
        sudoai.__log__.info(f'model {self.id} saved')
        return path

    def save_local(self, run_id, epoch):
        self.model.local_save(run_id, epoch)

    def start(self, hyperparam: bool = False, test_mode: bool = False, log_history: bool = False):
        """Entry point to begin the training process.

        Args:
            hyperparam (bool, optional): If True process in HyperParameter mode. Defaults to False.
            test_mode (bool, optional): If True process in test mode
            (Test mode to try with small amount of data). Defaults to False.
            log_history (bool, optional): If True logs all steps for reporting. Defaults to False.

        Warnings:
            If your dataset is too small (less then 100) and you chose test mode,
            an Exception raised.

        Raises:
            ModelError: when model is None.

        Returns:
            dict: accuracy and loss.
            list: Log History.
        """
        if self.model is None:
            raise ModelError(self.model, "Model not loaded check your model")

        self.model.train()
        self.model.to(DEVICE)
        history = []

        self.test_mode = test_mode

        if self.wandb is not None and isinstance(self.wandb, dict) and ('project' in self.wandb and 'entity' in self.wandb):

            if self.wandb_key is not None:
                os.environ['WANDB_API_KEY'] = self.wandb_key
            else:
                wandb.login()
            if self.model.__class__.__name__ == 'Word2Word':
                config = {
                    'vocab_src': self.model.vocab_src,
                    'vocab_target':  self.model.vocab_target,
                    'hidden_size': self.model.hidden_size,
                    'version': self.model.version,
                    'name': self.model.name,
                    'optimizer': self.model.optimizer_type,
                    'loss': self.model.loss,
                    'learning_rate': self.model.learning_rate,
                    'teacher_forcing_ratio': self.model.teacher_forcing_ratio,
                    'momentum': self.model.momentum,
                    'drop_out': self.model.drop_out
                }
            elif self.model.__class__.__name__ in ['Word2Label', 'Seq2Label']:
                config = {
                    'vocab_size': self.model.vocab_size,
                    'n_class':  self.model.n_class,
                    'hidden_size': self.model.hidden_size,
                    'version': self.model.version,
                    'name': self.model.name,
                    'optimizer': self.model.optimizer_type,
                    'loss': self.model.loss,
                    'learning_rate': self.model.learning_rate,
                    'teacher_forcing_ratio': self.model.teacher_forcing_ratio,
                    'momentum': self.model.momentum,
                    'drop_out': self.model.drop_out
                }
            else:
                config = {}

            wandb.init(project=self.wandb['project'],
                       entity=self.wandb['entity'],
                       config=config)
            wandb.watch(self.model, log="all")

        if(self.test_mode):

            # t_dataset = CustomDataset(self.dataset.get_train(t=1000))
            # train = DataLoader(t_dataset,
            #                    shuffle=self.do_shuffle,
            #                    num_workers=self.num_workers,
            #                    multiprocessing_context='spawn'
            #                    )
            # v_dataset = CustomDataset(self.dataset.get_valid(t=200))
            # valid = DataLoader(v_dataset,
            #                    shuffle=self.do_shuffle,
            #                    num_workers=0
            #                    )

            train = self.dataset.get_train(t=1000)
            valid = self.dataset.get_valid(t=200)
        else:

            # t_dataset = CustomDataset(self.dataset.train)
            # train = DataLoader(t_dataset,
            #                    shuffle=self.do_shuffle,
            #                    num_workers=self.num_workers,
            #                    multiprocessing_context='spawn'
            #                    )
            # v_dataset = CustomDataset(self.dataset.valid)
            # valid = DataLoader(v_dataset,
            #                    shuffle=self.do_shuffle,
            #                    num_workers=0
            #                    )

            # train = self.dataset.train
            # valid = self.dataset.valid

            train, valid = self.dataset.get_dataloader()

        if hyperparam:
            return self.steps(train, valid, 1, hyperparam, log_history)

        run_id = str(uuid.uuid4())

        for num_epoch in range(1, self.epochs + 1):

            if log_history:
                history.append(self.steps(
                    train, valid, num_epoch, hyperparam, log_history)
                )
            else:
                self.steps(train, valid, num_epoch, hyperparam, log_history)
            if self.do_save:
                self.save()

            if self.save_runs:
                self.save_local(run_id, str(num_epoch))

        if self.do_save:
            self.model.eval()
            path = self.save()
            if self.wandb is not None:
                model = os.path.join(path, 'model.bin')
                config = os.path.join(path, 'config.json')
                shutil.copy(model, os.path.join(
                    wandb.run.dir, "model.bin"))
                shutil.copy(config, os.path.join(
                    wandb.run.dir, 'config.json'))
                wandb.finish()

        if log_history:
            return history

    def evaluate(self, n=10) -> None:
        """Visual evaluation process for random input from dataset

        Args:
            n (int, optional): number of item to evaluate. Defaults to 10.
        """
        self.model.eval()
        self.model.cpu()

        size = len(self.dataset)

        for _ in range(n):
            number = random.randrange(1, size)
            pair = self.dataset(idx=number, plain=True, val=True)
            en_data = self.dataset(number, val=True)
            en = en_data[0]
            en = en.cpu()
            outputs = self.model(en)

            sudoai.__log__.info(
                f'{pair[0]} = {pair[1]} |PREDICTION| {outputs}'
            )

    def steps(self, train, valid, num_epoch: int = 1, hyperparam: bool = False, log_history: bool = False):
        """Training steps

        Args:

            num_epoch (int, optional): Current epoch number. Defaults to 1.
            hyperparam (bool, optional): If True enter Hypertuning mode. Defaults to False.
            log_history (bool, optional): If True log history. Defaults to False.

        Returns:
            dict: accuracy and loss.
            list: Log History.
        """
        print_every = 0
        history = {'loss': {'train': [], 'eval': []},
                   'acc': {'train': [], 'eval': []}}
        all_iters = len(self.dataset)

        eval_loss_avg = None

        if self.continue_from_checkpoint is not None:
            if not os.path.exists(self.continue_from_checkpoint):
                sudoai.__log__.warning('checkpoint file not exist !')
            else:
                self.model, epoch, _loss = load_checkpoint(
                    self.continue_from_checkpoint, self.model, True)
                sudoai.__log__.info(
                    f'checkpoint loaded epoch {epoch} loss {_loss} ')
            self.continue_from_checkpoint = None

        # if self.do_shuffle:
        #     random.shuffle(train)

        progress_bar = tqdm(train, desc=f'train epoch {num_epoch}')

        if DEVICE == 'cuda':
            torch.cuda.synchronize()

        for iter in progress_bar:

            input_tensor = iter[0]
            target_tensor = iter[1]

            metrics = self.model(input_tensor,
                                 target_tensor,
                                 do_train=True)

            history['loss']['train'].append(metrics['loss'])
            history['acc']['train'].append(metrics['acc'])

            if print_every == self.print_every:
                print_every = 0
                print_loss_avg = sum(
                    history['loss']['train']) / len(history['loss']['train'])
                print_acc_avg = sum(
                    history['acc']['train']) / len(history['acc']['train'])

                a = print_acc_avg
                lo = print_loss_avg

                progress_bar.set_postfix_str(f'acc : {a:.4f} loss : {lo:.4f}')

            print_every += 1

        if DEVICE == 'cuda':
            torch.cuda.synchronize()

        train_loss_avg = sum(history['loss']['train']) / \
            len(history['loss']['train'])

        train_acc_avg = sum(history['acc']['train']) / \
            len(history['acc']['train'])

        if self.do_eval:

            # if self.do_shuffle:
            #     random.shuffle(valid)

            print_every = 0
            eval_progress_bar = tqdm(valid, desc=f'eval epoch {num_epoch}')
            for iter in eval_progress_bar:

                input_tensor = iter[0]
                target_tensor = iter[1]

                metrics = self.model(input_tensor,
                                     target_tensor,
                                     do_train=True)

                history['loss']['eval'].append(metrics['loss'])
                history['acc']['eval'].append(metrics['acc'])

                if print_every == self.print_every:
                    print_every = 0
                    print_eval_loss_avg = sum(
                        history['loss']['eval']) / len(history['loss']['eval'])
                    print_eval_acc_avg = sum(
                        history['acc']['eval']) / len(history['acc']['eval'])

                    a = print_eval_acc_avg
                    lo = print_eval_loss_avg

                    eval_progress_bar.set_postfix_str(
                        f' val_acc : {a:.4f} val_loss : {lo:.4f}')

                print_every += 1

            eval_loss_avg = sum(history['loss']['eval']) / \
                len(history['loss']['eval'])

            eval_acc_avg = sum(history['acc']['eval']) / \
                len(history['acc']['eval'])
            a = train_acc_avg
            va = eval_acc_avg
            lo = train_loss_avg
            vlo = eval_loss_avg
            sudoai.__log__.info(
                f'steps : {all_iters} acc : {a:.4f} val_acc: {va:.4f}  loss : {lo:.4f}  val_loss : {vlo:.4f}')

        else:
            a = train_acc_avg
            lo = train_loss_avg
            sudoai.__log__.info(
                f'steps : {all_iters} acc : {a:.4f} loss : {lo:.4f}')

        self.save_checkpoint(num_epoch, train_loss_avg, eval_loss_avg)

        if self.wandb is not None:
            if self.do_eval:
                wandb.log({"epoch": num_epoch,
                           "acc": train_acc_avg,
                           "val_acc": eval_acc_avg,
                           "loss": train_loss_avg,
                           "val_loss": eval_loss_avg})
            else:
                wandb.log({"epoch": num_epoch,
                           "acc": train_acc_avg,
                           "loss": train_loss_avg})

        if hyperparam:
            if self.do_eval:
                return {"acc": train_acc_avg,
                        "val_acc": eval_acc_avg,
                        "loss": train_loss_avg,
                        "val_loss": eval_loss_avg}

            return {"acc": train_acc_avg, "loss": train_loss_avg}

        if log_history:
            return history

    def save_checkpoint(self, num_epoch: int, train_loss_avg: float, eval_loss_avg: float = None) -> None:
        """Save checkpoint

        Args:
            num_epoch (int): Current epoch number.
            train_loss_avg (float): Current average training loss .
            eval_loss_avg (float, optional): Current average evaluation loss. Defaults to None.
        """
        if self.do_save_checkpoint:
            now = datetime.now()
            dt = now.strftime("%m-%d-%Y_%H-%M-%S")
            if self.do_eval:
                path_checkpoint = 'checkpoint_at_{}_epoch_{}_loss_{:.4f}_val_loss_{:.4f}.pt'.format(
                    dt, num_epoch, train_loss_avg, eval_loss_avg)
            else:
                path_checkpoint = 'checkpoint_at_{}_epoch_{}_loss_{:.4f}.pt'.format(
                    dt, num_epoch, train_loss_avg)

            if self.base_path is not None:
                path_checkpoint = self.base_path + path_checkpoint

            save_checkpoint(self.model,
                            num_epoch,
                            train_loss_avg,
                            [self.model.encoder_optimizer,
                                self.model.decoder_optimizer],
                            path_checkpoint
                            )
            sudoai.__log__.info(f'checkpoint ({path_checkpoint}) saved')

    def __call__(self, **kwargs):
        """override __str__ method.

        Entry point to begin the training process.
        Make call for self.start() method.

        Args:
            hyperparam (bool, optional): If True process in HyperParameter mode. Defaults to False.
            test_mode (bool, optional): If True process in test mode
            (Test mode to try with small amount of data). Defaults to False.
            log_history (bool, optional): If True logs all steps for reporting. Defaults to False.

        Warnings:
            If your dataset is too small (less then 100) and you chose test mode,
            an Exception raised.

        Raises:
            ModelError: when model is None.

        Returns:
            dict: accuracy and loss.
            list: Log History.
        """
        hyperparam, test_mode, log_history = False, False, False

        if 'hyperparam' in kwargs:
            hyperparam = kwargs['hyperparam']
        if 'test_mode' in kwargs:
            test_mode = kwargs['test_mode']
        if 'log_history' in kwargs:
            log_history = kwargs['log_history']
        if 'num_workers' in kwargs:
            self.num_workers = kwargs['num_workers']

        return self.start(hyperparam, test_mode, log_history)

    def __str__(self):
        """override __str__ method.

        Returns:
            str: Description of the current model and dataset in trainer class.
        """
        return f'model :{self.model} -_- dataset : {self.dataset}'


class Word2WordTrainer(Trainer):
    """Word to Word Trainer class.

    Trainer for word to word model.

    Attributes:
        id (str): Dataset identifier.
        model (:obj:`Word2Word`, optional): Word To Word model. Defaults to None.
        version (str, optional): Model version. Defaults to '0.1.0'.
        teacher_forcing_ratio (float, optional): Teacher Forcing ratio for acceleration. Defaults to 0.5.
        hidden_size (int, optional): Hidden size of the model. Defaults to 512.
        print_every (int, optional): Log frequency result by steps (log result every 30000 steps). Defaults to 30000.
        lr (float, optional): Learn rate value. Defaults to 0.0001.
        epochs (int, optional): Number of epochs. Defaults to 2.
        drop_out (float, optional): Drop out value. Defaults to 0.1.
        do_eval (bool, optional): If True evaluate. Defaults to False.
        do_save (bool, optional): If True save the model when training ends. Defaults to False.
        do_shuffle (bool, optional): If True shuffle the dataset. Defaults to True.
        do_save_checkpoint (bool, optional): If True save checkpoint every epoch. Defaults to False.
        continue_from_checkpoint (str, optional): Path of checkpoint as start point. Defaults to None.
        loss (str, optional): Loss function. Defaults to 'nll'.
        optimizer (str, optional): Optimizer function. Defaults to 'rmsprop'.
        momentum (float, optional): Momentum value for optimizer. Defaults to 0.0.
        wandb (wandb, optional): Wandb class to track logs. Defaults to None.
        wandb_key (str, optional): Wandb api key. Defaults to None.
        save_runs (bool, optional): If True save runs. Defaults to False.
        base_path (str, optional): Base path to save checkpoint and model. Defaults to None.

    See Also:
        For more information about wandb check pytorch docs https://docs.wandb.ai/guides/integrations/pytorch

    """

    def __init__(self,
                 id: str,
                 model: Word2Word = None,
                 version: str = '0.1.0',
                 teacher_forcing_ratio: float = 0.5,
                 hidden_size: int = 512,
                 print_every: int = 100,
                 lr: float = 0.0001,
                 epochs: int = 2,
                 drop_out: float = 0.1,
                 do_eval: bool = False,
                 do_save: bool = False,
                 do_shuffle: bool = True,
                 do_save_checkpoint: bool = False,
                 continue_from_checkpoint: str = None,
                 loss: str = 'nll',
                 optimizer: str = 'rmsprop',
                 momentum: float = 0.0,
                 wandb: wandb = None,
                 wandb_key: str = None,
                 save_runs: bool = True,
                 base_path: str = None) -> None:
        """Create a Word2WordTrainer class.

        Args:
            id (str): Dataset identifier.
            model (:obj:`Word2Word`, optional): Word To Word model. Defaults to None.
            version (str, optional): Model version. Defaults to '0.1.0'.
            teacher_forcing_ratio (float, optional): Teacher Forcing ratio for acceleration. Defaults to 0.5.
            hidden_size (int, optional): Hidden size of the model. Defaults to 512.
            print_every (int, optional): Log frequency result by steps (log result every 30000 steps). Defaults to 30000.
            lr (float, optional): Learn rate value. Defaults to 0.0001.
            epochs (int, optional): Number of epochs. Defaults to 2.
            drop_out (float, optional): Drop out value. Defaults to 0.1.
            do_eval (bool, optional): If True evaluate. Defaults to False.
            do_save (bool, optional): If True save the model when training ends. Defaults to False.
            do_shuffle (bool, optional): If True shuffle the dataset. Defaults to True.
            do_save_checkpoint (bool, optional): If True save checkpoint every epoch. Defaults to False.
            continue_from_checkpoint (str, optional): Path of checkpoint as start point. Defaults to None.
            loss (str, optional): Loss function. Defaults to 'nll'.
            optimizer (str, optional): Optimizer function. Defaults to 'rmsprop'.
            momentum (float, optional): Momentum value for optimizer. Defaults to 0.0.
            wandb (wandb, optional): Wandb class to track logs. Defaults to None.
            wandb_key (str, optional): Wandb api key. Defaults to None.
            base_path (str, optional): Base path to save checkpoint and model. Defaults to None.
        """

        super().__init__(teacher_forcing_ratio=teacher_forcing_ratio, wandb_key=wandb_key,
                         print_every=print_every, do_eval=do_eval, hidden_size=hidden_size,
                         lr=lr, epochs=epochs, version=version, drop_out=drop_out,
                         do_save=do_save, do_save_checkpoint=do_save_checkpoint, id=id,
                         do_shuffle=do_shuffle, optimizer=optimizer,
                         continue_from_checkpoint=continue_from_checkpoint, loss=loss,
                         momentum=momentum, base_path=base_path, wandb=wandb, save_runs=save_runs)
        self.model = model
        self.dataset = load_dataset(self.id)

        if not self.dataset.info.dataset_type == DatasetType.WORD_TO_WORD:
            raise DatasetError(self.dataset,
                               "Dataset type must be (DatasetType.WORD_TO_WORD) check your dataset")

        self.src_vocab_sise = self.dataset.src_token.vocab_size
        self.target_vocab_size = self.dataset.target_token.vocab_size

        if self.model is None:
            self.model = Word2Word(version=self.version,
                                   vocab_src=self.src_vocab_sise,
                                   vocab_target=self.target_vocab_size,
                                   hidden_size=self.hidden_size,
                                   name=self.id,
                                   optimizer=self.optimizer,
                                   loss=self.loss,
                                   learning_rate=self.learning_rate,
                                   teacher_forcing_ratio=self.teacher_forcing_ratio,
                                   momentum=self.momentum,
                                   drop_out=self.drop_out)


class Seq2LabelTrainer(Trainer):

    """Sequence to Label Trainer class.

    Trainer for sequence to label model.

    Attributes:
        id (str): Dataset identifier.
        model(:obj:`Seq2Label`, optional): Sequence To Label Model. Defaults to None.
        version (str, optional): Model version. Defaults to '0.1.0'.
        teacher_forcing_ratio (float, optional): Teacher Forcing ratio for acceleration. Defaults to 0.5.
        hidden_size (int, optional): Hidden size of the model. Defaults to 512.
        print_every (int, optional): Log frequency result by steps (log result every 30000 steps). Defaults to 30000.
        lr (float, optional): Learn rate value. Defaults to 0.0001.
        epochs (int, optional): Number of epochs. Defaults to 2.
        drop_out (float, optional): Drop out value. Defaults to 0.1.
        do_eval (bool, optional): If True evaluate. Defaults to False.
        do_save (bool, optional): If True save the model when training ends. Defaults to False.
        do_shuffle (bool, optional): If True shuffle the dataset. Defaults to True.
        do_save_checkpoint (bool, optional): If True save checkpoint every epoch. Defaults to False.
        continue_from_checkpoint (str, optional): Path of checkpoint as start point. Defaults to None.
        loss (str, optional): Loss function. Defaults to 'nll'.
        optimizer (str, optional): Optimizer function. Defaults to 'rmsprop'.
        momentum (float, optional): Momentum value for optimizer. Defaults to 0.0.
        wandb (wandb, optional): Wandb class to track logs. Defaults to None.
        wandb_key (str, optional): Wandb api key. Defaults to None.
        save_runs (bool, optional): If True save runs. Defaults to False.
        base_path (str, optional): Base path to save checkpoint and model. Defaults to None.

    See Also:
        For more information about wandb check pytorch docs https://docs.wandb.ai/guides/integrations/pytorch

    """

    def __init__(self,
                 id: str,
                 model: Seq2Label = None,
                 version: str = '0.1.0',
                 teacher_forcing_ratio: float = 0.5,
                 hidden_size: int = 512,
                 print_every: int = 100,
                 lr: float = 0.0001,
                 epochs: int = 2,
                 drop_out: float = 0.1,
                 do_eval: bool = False,
                 do_save: bool = False,
                 do_shuffle: bool = True,
                 do_save_checkpoint: bool = False,
                 continue_from_checkpoint: str = None,
                 loss: str = 'nll',
                 optimizer: str = 'rmsprop',
                 momentum: float = 0.0,
                 wandb: wandb = None,
                 wandb_key: str = None,
                 save_runs: bool = True,
                 base_path: str = None) -> None:
        """Create a Seq2LabelTrainer class.

        Args:
            id (str): Dataset identifier.
            model(:obj:`Seq2Label`, optional): Sequence To Label Model. Defaults to None.
            version (str, optional): Model version. Defaults to '0.1.0'.
            teacher_forcing_ratio (float, optional): Teacher Forcing ratio for acceleration. Defaults to 0.5.
            hidden_size (int, optional): Hidden size of the model. Defaults to 512.
            print_every (int, optional): Log frequency result by steps (log result every 30000 steps). Defaults to 30000.
            lr (float, optional): Learn rate value. Defaults to 0.0001.
            epochs (int, optional): Number of epochs. Defaults to 2.
            drop_out (float, optional): Drop out value. Defaults to 0.1.
            do_eval (bool, optional): If True evaluate. Defaults to False.
            do_save (bool, optional): If True save the model when training ends. Defaults to False.
            do_shuffle (bool, optional): If True shuffle the dataset. Defaults to True.
            do_save_checkpoint (bool, optional): If True save checkpoint every epoch. Defaults to False.
            continue_from_checkpoint (str, optional): Path of checkpoint as start point. Defaults to None.
            loss (str, optional): Loss function. Defaults to 'nll'.
            optimizer (str, optional): Optimizer function. Defaults to 'rmsprop'.
            momentum (float, optional): Momentum value for optimizer. Defaults to 0.0.
            wandb (wandb, optional): Wandb class to track logs. Defaults to None.
            wandb_key (str, optional): Wandb api key. Defaults to None.
            save_runs (bool, optional): If True save runs. Defaults to False.
            base_path (str, optional): Base path to save checkpoint and model. Defaults to None.
        """

        super().__init__(teacher_forcing_ratio=teacher_forcing_ratio, wandb_key=wandb_key,
                         print_every=print_every, do_eval=do_eval, hidden_size=hidden_size,
                         lr=lr, epochs=epochs, version=version, drop_out=drop_out,
                         do_save=do_save, do_save_checkpoint=do_save_checkpoint, id=id,
                         do_shuffle=do_shuffle, optimizer=optimizer, save_runs=save_runs,
                         continue_from_checkpoint=continue_from_checkpoint, loss=loss,
                         momentum=momentum, base_path=base_path, wandb=wandb)

        self.dataset = load_dataset(self.id)

        self.model = model

        if not self.dataset.info.dataset_type == DatasetType.SEQ_TO_LABEL:
            raise DatasetError(self.dataset,
                               "Dataset type must be [DatasetType.SEQ_TO_LABEL] check your dataset")

        self.n_class = self.dataset.n_class()
        self.vocab_size = self.dataset.token.basic.vocab_size

        if self.model is None:
            self.model = Seq2Label(
                n_class=self.n_class,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                version=self.version,
                name=self.id,
                optimizer=self.optimizer,
                loss=self.loss,
                learning_rate=self.learning_rate,
                teacher_forcing_ratio=self.teacher_forcing_ratio,
                momentum=self.momentum,
                drop_out=self.drop_out)


class TokenClassificationTrainer(Trainer):

    """Word to Label Trainer class.

    Trainer for Word to label model.

    Attributes:
        id (str): Dataset identifier.
        model(:obj:`Word2Label`, optional): Word To Label Model. Defaults to None.
        version (str, optional): Model version. Defaults to '0.1.0'.
        teacher_forcing_ratio (float, optional): Teacher Forcing ratio for acceleration. Defaults to 0.5.
        hidden_size (int, optional): Hidden size of the model. Defaults to 512.
        print_every (int, optional): Log frequency result by steps (log result every 30000 steps). Defaults to 30000.
        lr (float, optional): Learn rate value. Defaults to 0.0001.
        epochs (int, optional): Number of epochs. Defaults to 2.
        drop_out (float, optional): Drop out value. Defaults to 0.1.
        do_eval (bool, optional): If True evaluate. Defaults to False.
        do_save (bool, optional): If True save the model when training ends. Defaults to False.
        do_shuffle (bool, optional): If True shuffle the dataset. Defaults to True.
        do_save_checkpoint (bool, optional): If True save checkpoint every epoch. Defaults to False.
        continue_from_checkpoint (str, optional): Path of checkpoint as start point. Defaults to None.
        loss (str, optional): Loss function. Defaults to 'nll'.
        optimizer (str, optional): Optimizer function. Defaults to 'rmsprop'.
        momentum (float, optional): Momentum value for optimizer. Defaults to 0.0.
        wandb (wandb, optional): Wandb class to track logs. Defaults to None.
        wandb_key (str, optional): Wandb api key. Defaults to None.
        save_runs (bool, optional): If True save runs. Defaults to False.
        base_path (str, optional): Base path to save checkpoint and model. Defaults to None.

    See Also:
        For more information about wandb check pytorch docs https://docs.wandb.ai/guides/integrations/pytorch

    """

    def __init__(self,
                 id: str,
                 model: Word2Label = None,
                 version: str = '0.1.0',
                 teacher_forcing_ratio: float = 0.5,
                 hidden_size: int = 512,
                 print_every: int = 100,
                 lr: float = 0.0001,
                 epochs: int = 2,
                 drop_out: float = 0.1,
                 do_eval: bool = False,
                 do_save: bool = False,
                 do_shuffle: bool = True,
                 do_save_checkpoint: bool = False,
                 continue_from_checkpoint: str = None,
                 loss: str = 'nll',
                 optimizer: str = 'rmsprop',
                 momentum: float = 0.0,
                 wandb: wandb = None,
                 wandb_key: str = None,
                 save_runs: bool = True,
                 base_path: str = None) -> None:
        """Create a TokenClassificationTrainer class.

        Args:
            id (str): Dataset identifier.
            model(:obj:`Word2Label`, optional): Word To Label Model. Defaults to None.
            version (str, optional): Model version. Defaults to '0.1.0'.
            teacher_forcing_ratio (float, optional): Teacher Forcing ratio for acceleration. Defaults to 0.5.
            hidden_size (int, optional): Hidden size of the model. Defaults to 512.
            print_every (int, optional): Log frequency result by steps (log result every 30000 steps). Defaults to 30000.
            lr (float, optional): Learn rate value. Defaults to 0.0001.
            epochs (int, optional): Number of epochs. Defaults to 2.
            drop_out (float, optional): Drop out value. Defaults to 0.1.
            do_eval (bool, optional): If True evaluate. Defaults to False.
            do_save (bool, optional): If True save the model when training ends. Defaults to False.
            do_shuffle (bool, optional): If True shuffle the dataset. Defaults to True.
            do_save_checkpoint (bool, optional): If True save checkpoint every epoch. Defaults to False.
            continue_from_checkpoint (str, optional): Path of checkpoint as start point. Defaults to None.
            loss (str, optional): Loss function. Defaults to 'nll'.
            optimizer (str, optional): Optimizer function. Defaults to 'rmsprop'.
            momentum (float, optional): Momentum value for optimizer. Defaults to 0.0.
            wandb (wandb, optional): Wandb class to track logs. Defaults to None.
            wandb_key (str, optional): Wandb api key. Defaults to None.
            base_path (str, optional): Base path to save checkpoint and model. Defaults to None.
        """

        super().__init__(teacher_forcing_ratio=teacher_forcing_ratio, wandb_key=wandb_key,
                         print_every=print_every, do_eval=do_eval, hidden_size=hidden_size,
                         lr=lr, epochs=epochs, version=version, drop_out=drop_out,
                         do_save=do_save, do_save_checkpoint=do_save_checkpoint, id=id,
                         do_shuffle=do_shuffle, optimizer=optimizer, save_runs=save_runs,
                         continue_from_checkpoint=continue_from_checkpoint, loss=loss,
                         momentum=momentum, base_path=base_path, wandb=wandb)

        self.dataset = load_dataset(self.id)

        self.model = model

        if not self.dataset.info.dataset_type == DatasetType.WORD_TO_LABEL:
            raise DatasetError(self.dataset,
                               "Dataset type must be [DatasetType.WORD_TO_LABEL] check your dataset")

        self.n_class = self.dataset.n_class()
        self.vocab_size = self.dataset.token.vocab_size

        if self.model is None:
            self.model = Word2Label(
                n_class=self.n_class,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                version=self.version,
                name=self.id,
                optimizer=self.optimizer,
                loss=self.loss,
                learning_rate=self.learning_rate,
                teacher_forcing_ratio=self.teacher_forcing_ratio,
                momentum=self.momentum,
                drop_out=self.drop_out)


class HybridXMLTrainer(Trainer):

    """HybridXML Trainer class.

    Trainer for HybridXML model.

    Attributes:
        id (str): Dataset identifier.
        model(:obj:`HybridXML`, optional): HybridXML Model. Defaults to None.
        version (str, optional): Model version. Defaults to '0.1.0'.
        teacher_forcing_ratio (float, optional): Teacher Forcing ratio for acceleration. Defaults to 0.5.
        hidden_size (int, optional): Hidden size of the model. Defaults to 512.
        print_every (int, optional): Log frequency result by steps (log result every 30000 steps). Defaults to 30000.
        lr (float, optional): Learn rate value. Defaults to 0.0001.
        epochs (int, optional): Number of epochs. Defaults to 2.
        drop_out (float, optional): Drop out value. Defaults to 0.1.
        do_eval (bool, optional): If True evaluate. Defaults to False.
        do_save (bool, optional): If True save the model when training ends. Defaults to False.
        do_shuffle (bool, optional): If True shuffle the dataset. Defaults to True.
        do_save_checkpoint (bool, optional): If True save checkpoint every epoch. Defaults to False.
        continue_from_checkpoint (str, optional): Path of checkpoint as start point. Defaults to None.
        loss (str, optional): Loss function. Defaults to 'nll'.
        optimizer (str, optional): Optimizer function. Defaults to 'rmsprop'.
        momentum (float, optional): Momentum value for optimizer. Defaults to 0.0.
        wandb (wandb, optional): Wandb class to track logs. Defaults to None.
        wandb_key (str, optional): Wandb api key. Defaults to None.
        base_path (str, optional): Base path to save checkpoint and model. Defaults to None.
        save_runs (bool, optional): If True save runs. Defaults to False.
        multiclass (bool, optional): If True multiclass. Defaults to False.

    See Also:
        For more information about wandb check pytorch docs https://docs.wandb.ai/guides/integrations/pytorch

    """

    def __init__(self,
                 id: str,
                 model: HybridXMLTC = None,
                 version: str = '0.1.0',
                 teacher_forcing_ratio: float = 0.5,
                 hidden_size: int = 512,
                 print_every: int = 100,
                 lr: float = 0.0001,
                 epochs: int = 2,
                 drop_out: float = 0.1,
                 do_eval: bool = False,
                 do_save: bool = False,
                 do_shuffle: bool = True,
                 do_save_checkpoint: bool = False,
                 continue_from_checkpoint: str = None,
                 loss: str = 'nll',
                 optimizer: str = 'rmsprop',
                 momentum: float = 0.0,
                 wandb: wandb = None,
                 wandb_key: str = None,
                 save_runs: bool = True,
                 multiclass: bool = False,
                 base_path: str = None) -> None:
        """[summary]

        Args:
            id (str): Dataset identifier.
            model(:obj:`HybridXMLTC`, optional): HybridXMLTC Model. Defaults to None.
            version (str, optional): Model version. Defaults to '0.1.0'.
            teacher_forcing_ratio (float, optional): Teacher Forcing ratio for acceleration. Defaults to 0.5.
            hidden_size (int, optional): Hidden size of the model. Defaults to 512.
            print_every (int, optional): Log frequency result by steps (log result every 30000 steps). Defaults to 30000.
            lr (float, optional): Learn rate value. Defaults to 0.0001.
            epochs (int, optional): Number of epochs. Defaults to 2.
            drop_out (float, optional): Drop out value. Defaults to 0.1.
            do_eval (bool, optional): If True evaluate. Defaults to False.
            do_save (bool, optional): If True save the model when training ends. Defaults to False.
            do_shuffle (bool, optional): If True shuffle the dataset. Defaults to True.
            do_save_checkpoint (bool, optional): If True save checkpoint every epoch. Defaults to False.
            continue_from_checkpoint (str, optional): Path of checkpoint as start point. Defaults to None.
            loss (str, optional): Loss function. Defaults to 'nll'.
            optimizer (str, optional): Optimizer function. Defaults to 'rmsprop'.
            momentum (float, optional): Momentum value for optimizer. Defaults to 0.0.
            wandb (wandb, optional): Wandb class to track logs. Defaults to None.
            wandb_key (str, optional): Wandb api key. Defaults to None.
            base_path (str, optional): Base path to save checkpoint and model. Defaults to None.
            multiclass (bool, optional): If True multiclass. Defaults to False.

        Raises:
            DatasetError: When ddataset_type not equals :obj:`DatasetType.SEQ_TO_LABEL`.
        """

        super().__init__(teacher_forcing_ratio=teacher_forcing_ratio, wandb_key=wandb_key,
                         print_every=print_every, do_eval=do_eval, hidden_size=hidden_size,
                         lr=lr, epochs=epochs, version=version, drop_out=drop_out,
                         do_save=do_save, do_save_checkpoint=do_save_checkpoint, id=id,
                         do_shuffle=do_shuffle, optimizer=optimizer, save_runs=save_runs,
                         continue_from_checkpoint=continue_from_checkpoint, loss=loss,
                         momentum=momentum, base_path=base_path, wandb=wandb)

        self.multiclass = multiclass

        self.dataset = load_dataset(self.id)
        self.model = model

        if not self.dataset.info.dataset_type == DatasetType.SEQ_TO_LABEL:
            raise DatasetError(self.dataset,
                               "Dataset type must be [DatasetType.SEQ_TO_LABEL] check your dataset")

        self.n_class = self.dataset.n_class()
        self.vocab_size = self.dataset.token.basic.vocab_size

        if self.model is None:
            self.model = HybridXMLTC(
                n_class=self.n_class,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size + 1,
                name=self.id,
                version=self.version,
                learning_rate=self.learning_rate,
                momentum=self.momentum,
                optimizer=self.optimizer,
                multiclass=self.multiclass,
                d_a=self.hidden_size)

    def steps(self, train, num_epoch=1, hyperparam=False, log_history=False):

        print_every = 0
        history = {'loss': {'train': [], 'eval': []},
                   'acc': {'train': [], 'eval': []}}
        all_iters = train['train'][0] + train['eval'][0]

        eval_loss_avg = None

        if self.continue_from_checkpoint is not None:
            if not os.path.exists(self.continue_from_checkpoint):
                sudoai.__log__.warning('checkpoint file not exist !')
            else:
                self.model, epoch, _loss = load_checkpoint(
                    self.continue_from_checkpoint, self.model, True)
                sudoai.__log__.info(
                    f'checkpoint loaded epoch {epoch} loss {_loss} ')
            self.continue_from_checkpoint = None

        if self.do_shuffle:
            random.shuffle(train['train'][1])
            random.shuffle(train['eval'][1])

        n_iters = train['train'][0]
        progress_bar = tqdm(range(n_iters), desc=f'train epoch {num_epoch}')
        for iter in progress_bar:
            training_pair = train['train'][1][iter]
            input_tensor = training_pair[0].to(DEVICE)
            labels_tensor = training_pair[1].to(DEVICE)

            metrics = {'loss': None, 'acc': None}

            history = {'loss': {'train': [], 'eval': []},
                       'acc': {'train': [], 'eval': []}}

            metrics = self.model(input_tensor, labels_tensor, do_train=True)

            history['loss']['train'].append(metrics['loss'])
            history['acc']['train'].append(metrics['acc'])

            if print_every == self.print_every:
                print_every = 0
                print_loss_avg = sum(
                    history['loss']['train']) / len(history['loss']['train'])
                print_acc_avg = sum(
                    history['acc']['train']) / len(history['acc']['train'])

                a = print_acc_avg
                lo = print_loss_avg

                progress_bar.set_postfix_str(
                    f' acc : {a:.4f} loss : {lo:.4f}')

            print_every += 1

        train_loss_avg = sum(history['loss']['train']) / \
            len(history['loss']['train'])

        train_acc_avg = sum(history['acc']['train']) / \
            len(history['acc']['train'])

        if self.do_eval:
            eval_iters = train['eval'][0]
            print_every = 0

            eval_progress_bar = tqdm(
                range(eval_iters), desc=f'eval epoch {num_epoch}')
            for iter in eval_progress_bar:
                eval_pair = train['eval'][1][iter]
                input_tensor = eval_pair[0].to(DEVICE)
                target_tensor = eval_pair[1].to(DEVICE)

                metrics = self.model(input_tensor,
                                     target_tensor,
                                     do_train=True)

                history['loss']['eval'].append(metrics['loss'])
                history['acc']['eval'].append(metrics['acc'])

                if print_every == self.print_every:
                    print_every = 0
                    print_eval_loss_avg = sum(
                        history['loss']['eval']) / len(history['loss']['eval'])
                    print_eval_acc_avg = sum(
                        history['acc']['eval']) / len(history['acc']['eval'])

                    a = print_eval_acc_avg
                    lo = print_eval_loss_avg

                    eval_progress_bar.set_postfix_str(
                        f' val_acc : {a:.4f} val_loss : {lo:.4f}')

                print_every += 1

            eval_loss_avg = sum(history['loss']['eval']) / \
                len(history['loss']['eval'])

            eval_acc_avg = sum(history['acc']['eval']) / \
                len(history['acc']['eval'])
            a = train_acc_avg
            va = eval_acc_avg
            lo = train_loss_avg
            vlo = eval_loss_avg
            sudoai.__log__.info(
                f'steps : {all_iters} acc : {a:.4f} val_acc: {va:.4f}  loss : {lo:.4f}  val_loss : {vlo:.4f}')

        else:
            a = train_acc_avg
            lo = train_loss_avg
            sudoai.__log__.info(
                f'steps : {all_iters} acc : {a:.4f} loss : {lo:.4f}')

        self.save_checkpoint(num_epoch, train_loss_avg, eval_loss_avg)

        if self.wandb is not None:
            if self.do_eval:
                wandb.log({"epoch": num_epoch,
                           "acc": train_acc_avg,
                           "val_acc": eval_acc_avg,
                           "loss": train_loss_avg,
                           "val_loss": eval_loss_avg})
            else:
                wandb.log({"epoch": num_epoch,
                           "acc": train_acc_avg,
                           "loss": train_loss_avg})

        if hyperparam:
            if self.do_eval:
                return {"acc": train_acc_avg,
                        "val_acc": eval_acc_avg,
                        "loss": train_loss_avg,
                        "val_loss": eval_loss_avg}

            return {"acc": train_acc_avg, "loss": train_loss_avg}

        if log_history:
            return history


class LogisticTrainer(Trainer):

    def __init__(self,
                 id: str,
                 model:  LogisticRegression = None,
                 version: str = '0.1.0',
                 lr: float = 0.01,
                 epochs: int = 2,
                 do_eval: bool = False,
                 do_shuffle: bool = False,
                 do_save_checkpoint: bool = False,
                 do_save: bool = True,
                 continue_from_checkpoint: str = None,
                 momentum: float = 0.0,
                 wandb: wandb = None,
                 wandb_key: str = None,
                 save_runs: bool = True,
                 base_path: str = None
                 ) -> None:

        super().__init__(id=id,
                         version=version,
                         lr=lr,
                         epochs=epochs,
                         wandb=wandb,
                         wandb_key=wandb_key,
                         do_eval=do_eval,
                         do_shuffle=do_shuffle,
                         do_save_checkpoint=do_save_checkpoint,
                         continue_from_checkpoint=continue_from_checkpoint,
                         momentum=momentum,
                         do_save=do_save,
                         save_runs=save_runs,
                         base_path=base_path)

        self.dataset = load_dataset(self.id)
        self.model = model

        self.n_class = self.dataset.n_class()

        if self.model is None:
            self.model = LogisticRegression(
                name=self.id,
                version=self.version,
                momentum=self.momentum,
                n_class=self.n_class,
                learning_rate=self.learning_rate
            )

    def steps(self, train, valid, num_epoch: int = 1, hyperparam: bool = False, log_history: bool = False):
        """Training steps

        Args:

            num_epoch (int, optional): Current epoch number. Defaults to 1.
            hyperparam (bool, optional): If True enter Hypertuning mode. Defaults to False.
            log_history (bool, optional): If True log history. Defaults to False.

        Returns:
            dict: accuracy and loss.
            list: Log History.
        """
        print_every = 0
        history = {'loss': {'train': [], 'eval': []},
                   'acc': {'train': [], 'eval': []}}
        all_iters = len(self.dataset)

        eval_loss_avg = None

        if self.continue_from_checkpoint is not None:
            if not os.path.exists(self.continue_from_checkpoint):
                sudoai.__log__.warning('checkpoint file not exist !')
            else:
                self.model, epoch, _loss = load_checkpoint(
                    self.continue_from_checkpoint, self.model, True)
                sudoai.__log__.info(
                    f'checkpoint loaded epoch {epoch} loss {_loss} ')
            self.continue_from_checkpoint = None

        # if self.do_shuffle:
        #     random.shuffle(train)

        progress_bar = tqdm(train, desc=f'train epoch {num_epoch}')

        if DEVICE == 'cuda':
            torch.cuda.synchronize()

        for iter in progress_bar:

            input_tensor = iter[0].flatten().to(dtype=torch.float32)
            target_tensor = iter[1].flatten().to(dtype=torch.float32)

            metrics = self.model(input_tensor,
                                 target_tensor,
                                 do_train=True)

            history['loss']['train'].append(metrics['loss'])
            history['acc']['train'].append(metrics['acc'])

            if print_every == self.print_every:
                print_every = 0
                print_loss_avg = sum(
                    history['loss']['train']) / len(history['loss']['train'])
                print_acc_avg = sum(
                    history['acc']['train']) / len(history['acc']['train'])

                a = print_acc_avg
                lo = print_loss_avg

                progress_bar.set_postfix_str(f'acc : {a:.4f} loss : {lo:.4f}')

            print_every += 1

        if DEVICE == 'cuda':
            torch.cuda.synchronize()

        train_loss_avg = sum(history['loss']['train']) / \
            len(history['loss']['train'])

        train_acc_avg = sum(history['acc']['train']) / \
            len(history['acc']['train'])

        if self.do_eval:

            # if self.do_shuffle:
            #     random.shuffle(valid)

            print_every = 0
            eval_progress_bar = tqdm(valid, desc=f'eval epoch {num_epoch}')
            for iter in eval_progress_bar:

                input_tensor = iter[0].flatten().to(dtype=torch.float32)
                target_tensor = iter[1].flatten().to(dtype=torch.float32)

                metrics = self.model(input_tensor,
                                     target_tensor,
                                     do_train=True)

                history['loss']['eval'].append(metrics['loss'])
                history['acc']['eval'].append(metrics['acc'])

                if print_every == self.print_every:
                    print_every = 0
                    print_eval_loss_avg = sum(
                        history['loss']['eval']) / len(history['loss']['eval'])
                    print_eval_acc_avg = sum(
                        history['acc']['eval']) / len(history['acc']['eval'])

                    a = print_eval_acc_avg
                    lo = print_eval_loss_avg

                    eval_progress_bar.set_postfix_str(
                        f' val_acc : {a:.4f} val_loss : {lo:.4f}')

                print_every += 1

            eval_loss_avg = sum(history['loss']['eval']) / \
                len(history['loss']['eval'])

            eval_acc_avg = sum(history['acc']['eval']) / \
                len(history['acc']['eval'])
            a = train_acc_avg
            va = eval_acc_avg
            lo = train_loss_avg
            vlo = eval_loss_avg
            sudoai.__log__.info(
                f'steps : {all_iters} acc : {a:.4f} val_acc: {va:.4f}  loss : {lo:.4f}  val_loss : {vlo:.4f}')

        else:
            a = train_acc_avg
            lo = train_loss_avg
            sudoai.__log__.info(
                f'steps : {all_iters} acc : {a:.4f} loss : {lo:.4f}')

        self.save_checkpoint(num_epoch, train_loss_avg, eval_loss_avg)

        if self.wandb is not None:
            if self.do_eval:
                wandb.log({"epoch": num_epoch,
                           "acc": train_acc_avg,
                           "val_acc": eval_acc_avg,
                           "loss": train_loss_avg,
                           "val_loss": eval_loss_avg})
            else:
                wandb.log({"epoch": num_epoch,
                           "acc": train_acc_avg,
                           "loss": train_loss_avg})

        if hyperparam:
            if self.do_eval:
                return {"acc": train_acc_avg,
                        "val_acc": eval_acc_avg,
                        "loss": train_loss_avg,
                        "val_loss": eval_loss_avg}

            return {"acc": train_acc_avg, "loss": train_loss_avg}

        if log_history:
            return history
