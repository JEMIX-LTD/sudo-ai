#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""Dataset module
@author: Aymen Jemi (jemix) <jemiaymen@gmail.com> at SUDO-AI

Datasets for natural language processing, sudoai.dataset
is a lightweight and extensible library to easily share
and access datasets for Natural Language Processing (NLP).

Tip:
    If you want use compressed algo, :obj:`ZipAlgo.LZMA`
    always be the best solution.

"""

import bz2
import gzip
import lzma
import mmap
import os
import pickle
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Dict

import pandas as pd
import torch
from tqdm.auto import tqdm
import sudoai

from ..preprocess import (CharTokenizer,
                          StopWord,
                          WordTokenizer,
                          convert_to_unicode)

from ..utils import (DEVICE,
                     ZipAlgo,
                     datapath,
                     save_dataset_config,
                     load_dataset)


class DatasetError(Exception):
    """Exception raised in Dataset class.

    Args:
        dataset (:obj:`Dataset`): Current dataset object.
        message (str, optional): Human readable string describing the exception. Defaults to "Dataset error!!".

    Attributes:
        dataset (:obj:`Dataset`):  Current dataset object.
        message (str): Human readable string describing the exception.

    """

    def __init__(self, dataset, message="Dataset error!!"):
        self.dataset = dataset
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.dataset} ~> {self.message}'


@unique
class DataType(str, Enum):
    """Enumerate for accepted data types by Dataset class.

    Attributes:
        TEXT (str): For text data file (1).
        CSV (str): For csv data file (2).
        JSON (str): For json data file (3).
        EXCEL (str): For excel data file (4).

    See Also:
        For more information check quickstart docs http://sudoai.tech/quickstart

    Examples:
        These examples illustrate how to use DataType in DatasetInfo class.

        Text data file:

        >>> info = DatasetInfo(id='sa', data_path='data.txt', data_type=DataType.TEXT)
        >>> info.data_type
        <DataType.TEXT: '1'>

        Excel data file:

        >>> info = DatasetInfo(id='sa', data_path='data.xlsx', data_type=DataType.EXCEL)
        >>> info.data_type
        <DataType.TEXT: '4'>

    """
    TEXT = "1"
    CSV = "2"
    JSON = "3"
    EXCEL = "4"


@unique
class DatasetType(str, Enum):
    """Enumerate for dataset types, to work with sudoai.models accepted by Dataset class.

    Attributes:
        TEXT (str): For Text model (1) .
        WORD_TO_LABEL (str): For Word2Label model (2).
        WORD_TO_WORD (str): For Word2Word model (3).
        SEQ_TO_LABEL (str): For Seq2Label model (4).
        SEQ_TO_SEQ (str): For Seq2Seq model (4).

    See Also:
        For more information check quickstart docs http://sudoai.tech/quickstart

    Examples:
        These examples illustrate how to use DatasetType in DatasetInfo class.

        For Word2Label model:

        >>> info = DatasetInfo(id='sa', data_path='data.xlsx',dataset_type=DatasetType.WORD_TO_LABEL)
        >>> info.dataset_type
        <DatasetType.WORD_TO_LABEL: '2'>

        For Seq2Label:

        >>> info = DatasetInfo(id='sa', data_path='data.txt',dataset_type=DatasetType.SEQ_TO_LABEL)
        >>> info.dataset_type
        <DatasetType.WORD_TO_LABEL: '4'>

    """
    TEXT = "1"
    WORD_TO_LABEL = "2"
    WORD_TO_WORD = "3"
    SEQ_TO_LABEL = "4"
    SEQ_TO_SEQ = "5"


@dataclass
class DatasetInfo():
    """This is a data class definition, to identify all info in dataset.

        A `DatasetInfo` class definition contains identification and
        allocation information for `Dataset`.
        Each `DatasetInfo` class must contains unique id that contains
        `[a-zA-Z0-9]` characters without spaces.

    Attributes:
        id (str): Dataset identifier.
        data_path (str): Path of data file.
        version (str, optional): Dataset version. Defaults to "0.1.0".
        description (str, optional): Dataset description. Defaults to None.
        str_text (str, optional): Column name of text (json, excel or csv). Default to 'text'.
        str_label (str, optional): Column name of label (json, excel or csv). Default to 'label'.
        data_type (DataType, optional): Enumerate for dataset types. Default to `DataType.TEXT`.
        dataset_type (DatasetType, optional): Enumerate for dataset types. Defaults to `DatasetType.TEXT`.
        sep (str, optional): Separator between text and labels. Defaults to '\t'.
        sep_label (str, optional): Separator between labels. Defaults to ','.
        encoding (str, optional): Text encoding. Defaults to 'utf8'.
        do_lower_case (bool, optional): For lowercase text. Defaults to True.
        verbose (int, optional): Verbose value. Defaults to 1.
        max_length (int, optional): Number of max elements (tokens). Defaults to 256.
        strip_duplicated (bool, optional): To strip duplicated lettre (exp hhhhello ~> hello). Defaults to False.
        strip_punc (bool, optional): To strip punctuation. Defaults to False.
        stopword (`StopWord`, optional): StopWord class to espace stopwords. Defaults to None.
        l2i (Dict[int, str], optional): Convert label to index. Defaults to {0:'bad'} .
        i2l (Dict[str, list], optional): Convert index to label. Defaults to {'good':[1]} .

    See Also:
        For more information check the quickstart docs http://sudoai.tech/quickstart

    Examples:
        These examples illustrate how to use `DatasetInfo` class.

        >>> info = DatasetInfo(id='sa', data_path='data.xlsx', data_type=DataType.EXCEL)

        >>> info = DatasetInfo(id='sa', data_path='data.txt',dataset_type=DatasetType.WORD_TO_LABEL)

    """
    id: str
    data_path: str
    version: str = '0.1.0'
    description: str = None
    str_text: str = 'text'
    str_label: str = 'label'
    i2l: Dict[int, str] = field(default_factory=lambda: ({0: "bad"}))
    l2i: Dict[str, list] = field(
        default_factory=lambda: ({"good": [1, 0]}))
    data_type: DataType = DataType.TEXT
    dataset_type: DatasetType = DatasetType.TEXT
    sep: str = '\t'
    sep_label: str = ','
    encoding: str = 'utf8'
    verbose: int = 1
    max_length: int = 256
    do_lower_case: bool = True
    strip_duplicated: bool = False
    strip_punc: bool = False
    stopword: StopWord = None


class Dataset():
    """Defines a dataset with `DatasetInfo`.

    Create a dataset from a data file (text, csv, json or excel) .
    Don't forget to attribute unique identifian to DatasetInfo.id .

    Attributes:
        info (:obj:`DatasetInfo`): DatasetInfo with all dataset information .
        data (:obj:`list`): List of encoded tokens in `torch.tensor` .
        plain_data (:obj:`list`): List of plain text in `str`.

    See Also:
        For more information check quickstart docs http://sudoai.tech/quickstart

    Examples:
        These examples illustrate how to use Dataset class.

        Word2Label model.

        >>> info = DatasetInfo(id='sa', data_path='data.txt', dataset_type=DatasetType.WORD_TO_LABEL)
        >>> tokenizer = load_tokenizer('sa')
        >>> dataset = Dataset(info=info,tokenizer=tokenizer)

        Word2Word model.

        >>> info = DatasetInfo(id='sa', data_path='data.txt', dataset_type=DatasetType.WORD_TO_WORD)
        >>> t_src = load_tokenizer('sa/src')
        >>> t_target = load_tokenizer('sa/target')
        >>> dataset = Dataset(info=info, src_tokenizer=t_src, target_tokenizer=t_target)

    """

    def __init__(self, info: DatasetInfo, **kwargs) -> None:
        """Create a dataset class.

        Args:
            info (:obj:`DatasetInfo`): DatasetInfo with all dataset information .
            tokenizer (:obj:`WordTokenizer` or :obj:`CharTokenizer`, optional): Tokenizer class to build a dataset.
            src_tokenizer (:obj:`WordTokenizer` or :obj:`CharTokenizer`, optional): src Tokenizer class to build a dataset.
            target_tokenizer (:obj:`WordTokenizer` or :obj:`CharTokenizer`, optional): target Tokenizer class to build a dataset.

        Raises:
            DatasetError: If data file not exist.
            DatasetError: If tokenizer not set.
            DatasetError: If src_tokenizer not set or target_tokenizer not set.
            TypeError: If dataset_type not set.
        """
        self.info = info
        self.data = []
        self.plain_data = []

        if not os.path.exists(self.info.data_path):
            raise DatasetError(
                self, f'data file : [{self.info.data_path}] not exist.')

        if self.info.dataset_type == DatasetType.TEXT:
            if 'tokenizer' in kwargs:
                self.token = kwargs['tokenizer']
            if not isinstance(self.token, WordTokenizer):
                raise DatasetError(
                    self, f'wrong token {self.token}'
                )
            self.token.init(
                self.info.do_lower_case,
                self.info.strip_duplicated,
                self.info.strip_punc,
                self.info.stopword
            )

        elif self.info.dataset_type == DatasetType.WORD_TO_LABEL:
            if 'tokenizer' in kwargs:
                self.token = kwargs['tokenizer']
            if not isinstance(self.token, CharTokenizer):
                raise DatasetError(
                    self, f'wrong token {self.token}'
                )
            self.token.init(
                self.info.do_lower_case,
                self.info.strip_duplicated,
                self.info.strip_punc,
                self.info.stopword
            )

        elif self.info.dataset_type == DatasetType.WORD_TO_WORD:
            if 'src_tokenizer' in kwargs and 'target_tokenizer' in kwargs:
                self.src_token = kwargs['src_tokenizer']
                self.target_token = kwargs['target_tokenizer']

            if not isinstance(self.src_token, CharTokenizer) or not isinstance(self.target_token, CharTokenizer):
                raise DatasetError(
                    self, f'wrong src_tokenizer {self.src_token} or target_tokenizer {self.target_token} !!'
                )

            self.src_token.init(
                self.info.do_lower_case,
                self.info.strip_duplicated,
                self.info.strip_punc,
                self.info.stopword
            )

            self.target_token.init(
                self.info.do_lower_case,
                self.info.strip_duplicated,
                self.info.strip_punc,
                self.info.stopword
            )

        elif self.info.dataset_type == DatasetType.SEQ_TO_LABEL:
            if 'tokenizer' in kwargs:
                self.token = kwargs['tokenizer']
            if not isinstance(self.token, WordTokenizer):
                raise DatasetError(
                    self, f'wrong token {self.token}'
                )

            self.token.init(
                self.info.do_lower_case,
                self.info.strip_duplicated,
                self.info.strip_punc,
                self.info.stopword
            )

        elif self.info.dataset_type == DatasetType.SEQ_TO_SEQ:
            if 'src_tokenizer' in kwargs and 'target_tokenizer' in kwargs:
                self.src_token = kwargs['src_tokenizer']
                self.target_token = kwargs['target_tokenizer']

            if not isinstance(self.src_token, WordTokenizer) or not isinstance(self.target_token, WordTokenizer):
                raise DatasetError(
                    self, f'wrong src_tokenizer {self.src_token} or target_tokenizer {self.target_token} !!'
                )

            self.src_token.init(
                self.info.do_lower_case,
                self.info.strip_duplicated,
                self.info.strip_punc,
                self.info.stopword
            )

            self.target_token.init(
                self.info.do_lower_case,
                self.info.strip_duplicated,
                self.info.strip_punc,
                self.info.stopword
            )
        else:
            raise TypeError(
                f"dataset [{self.info.dataset_type}] type not exist!!"
            )

    def label_to_tensor(self, label: str) -> torch.Tensor:
        """Transform label to index and return tensor.

        Args:
            label (`str`): Label in string format.

        Raises:
            TypeError: If label not found in `self.info.l2i` .

        Returns:
            `torch.Tensor`: Torch Tensor with label index.
        """

        # clean label from special chars
        label = label.translate(str.maketrans(
            {'\t': '', '\r': '', '\n': '', ' ': ''}
        )
        )

        if label not in self.info.l2i.keys():
            raise TypeError(f'label [{label}] not found !!')

        ln = self.info.l2i[label]
        return torch.tensor(ln, device=DEVICE).view(-1, 1)

    def data_to_tensor(self, data: str) -> torch.Tensor:
        """Transform data to tokens and return tensor with index from tokens.

        Args:
            data (str): Line of data in string format.

        Returns:
            `torch.Tensor`: Torch Tensor with tokens index.
        """
        if self.info.dataset_type == DatasetType.WORD_TO_WORD or self.info.dataset_type == DatasetType.SEQ_TO_SEQ:
            return (self.src_token(data[0]), self.target_token(data[1]))
        else:
            return self.token(data)

    def build(self) -> None:
        """Build dataset .

        Note:
            If dataset_type is `TEXT` data contains str.\n
            If dataset_type is `WORD_TO_LABEL` or
            `SEQ_TO_LABEL` data contains Tuple(str, label).\n
            If dataset_type is `WORD_TO_WORD` or `SEQ_TO_SEQ` data contains Tuple(str, str).

        Returns:
            None.
        """
        if self.info.dataset_type == DatasetType.TEXT:
            self._get_text()
        elif self.info.dataset_type == DatasetType.WORD_TO_LABEL or self.info.dataset_type == DatasetType.SEQ_TO_LABEL:
            self._get_str_to_label()
        elif self.info.dataset_type == DatasetType.WORD_TO_WORD or self.info.dataset_type == DatasetType.SEQ_TO_SEQ:
            self._get_str_to_str()

        sudoai.__log__.info(f"Dataset [{self.info.id}] is ready")

    def __len__(self) -> int:
        """Size of dataset.

        Returns:
            int: Size of data.
        """
        return len(self.data)

    def __getitem__(self, idx: int, plain: bool = False):
        """Get data from index.

        Args:
            idx (int): Index.
            plain (bool, optional): If True return plain text else return indexs. Defaults to False.

        Returns:
            `torch.Tensor`: Index from tokens.
            str: Plain text.
        """
        if plain:
            return self.plain_data[idx]
        return self.data[idx]

    def __call__(self, idx: int, plain: bool = False):
        """Get data from index.

        Args:
            idx (int): Index.
            plain (bool, optional): If True return plain text else return indexs. Defaults to False.

        Returns:
            `torch.Tensor`: Index from tokens.
            str: Plain text.
        """
        return self.__getitem__(idx, plain)

    def __iter__(self):
        """Initialize the iterator.

        Returns:
            `Dataset`: current dataset.
        """
        self._current = 0
        return self

    def __next__(self):
        """Called for each iteration.

        Raises:
            StopIteration: When done.

        Returns:
            `torch.Tensor`: Index from tokens.
        """
        if self._current >= self.__len__():
            raise StopIteration

        result = self.__getitem__(self._current)
        self._current += 1
        return result

    def save(self, override: bool = False, is_ziped: bool = False, algo: ZipAlgo = ZipAlgo.LZMA) -> bool:
        """Save dataset with id and version and compressed algo if exist.

        Args:
            override (bool, optional): If True delete old dataset file. Defaults to False.
            is_ziped (bool, optional): If True compress dataset file. Defaults to False.
            algo (:obj:`ZipAlgo`, optional): Compressing algorithm (LZMA, BZ2 or GZIP). Defaults to ZipAlgo.LZMA.

        Raises:
            FileExistsError: If override is False and dataset file exists.

        Warning:
            If you set is_ziped True you must chose :obj:`ZipAlgo`.

        Returns:
            bool: True if dataset saved.
        """
        folder_name = os.path.join(
            datapath(self.info.id.lower()), self.info.version)
        dataset_path = os.path.join(folder_name, 'dataset.pt')

        os.makedirs(folder_name, exist_ok=True)

        if override:
            if os.path.exists(dataset_path):
                os.unlink(dataset_path)

        if os.path.exists(dataset_path):
            raise FileExistsError()

        if is_ziped:
            if algo == ZipAlgo.LZMA:
                with lzma.open(dataset_path, 'wb') as f:
                    pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            if algo == ZipAlgo.GZIP:
                with gzip.open(dataset_path, 'wb') as f:
                    pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            if algo == ZipAlgo.BZ2:
                with bz2.BZ2File(dataset_path, 'wb') as f:
                    pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            with open(dataset_path, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        save_dataset_config(self.info.id.lower(),
                            self.info.version,
                            is_ziped,
                            algo)

        sudoai.__log__.info(f'dataset saved on : {dataset_path}')
        return True

    def _get_text(self) -> None:
        """Get texts from data file. """

        if self.info.data_type == DataType.TEXT:
            with open(self.info.data_path, mode='r+b') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as fm:

                    if self.info.verbose == 1:
                        lines = tqdm(iter(fm.readline, b""),
                                     desc='build dataset',
                                     unit=' lines')
                    else:
                        lines = iter(fm.readline, b"")

                    for line in lines:
                        if len(line.split(' ')) >= self.info.max_length:
                            continue
                        self.plain_data.append(convert_to_unicode(line))
                        self.data.append(self.data_to_tensor(line))

        elif self.info.data_type == DataType.CSV:
            data = pd.read_csv(self.info.data_path,
                               encoding=self.info.encoding, sep=self.info.sep)
            data = data.iterrows()
            if self.info.verbose == 1:
                data = tqdm(data, desc='build dataset', unit=' lines')

            for x in data:
                line = x[1][self.info.str_text]
                if len(line.split(' ')) >= self.info.max_length:
                    continue
                self.plain_data.append(convert_to_unicode(line))
                self.data.append(self.data_to_tensor(line))

        elif self.info.data_type == DataType.JSON:
            data = pd.read_json(self.info.data_path,
                                encoding=self.info.encoding)
            data = data.iterrows()
            if self.info.verbose == 1:
                data = tqdm(data, desc='build dataset', unit=' lines')

            for x in data:
                line = x[1][self.info.str_text]
                if len(line.split(' ')) >= self.info.max_length:
                    continue

                self.plain_data.append(convert_to_unicode(line))
                self.data.append(self.data_to_tensor(line))

        elif self.info.data_type == DataType.EXCEL:
            data = pd.read_excel(self.info.data_path)
            data = data.iterrows()
            if self.info.verbose == 1:
                data = tqdm(data, desc='build dataset', unit=' lines')

            for x in data:
                line = x[1][self.info.str_text]
                if len(line.split(' ')) >= self.info.max_length:
                    continue

                self.plain_data.append(convert_to_unicode(line))
                self.data.append(self.data_to_tensor(line))

    def _get_str_to_label(self) -> None:
        """Get texts and labels from data file. """

        if self.info.data_type == DataType.TEXT:
            with open(self.info.data_path, mode='r+b') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as fm:

                    if self.info.verbose == 1:
                        lines = tqdm(iter(fm.readline, b""),
                                     desc='build dataset',
                                     unit=' lines')
                    else:
                        lines = iter(fm.readline, b"")

                    for line in lines:
                        line = convert_to_unicode(line)

                        text, label = line.split(self.info.sep)
                        label = label.translate(str.maketrans(
                            {'\t': '', '\r': '', '\n': '', ' ': ''})
                        )
                        if len(text.split(' ')) >= self.info.max_length:
                            continue
                        self.plain_data.append((text, label))
                        self.data.append((self.data_to_tensor(text),
                                          self.label_to_tensor(label)))

        elif self.info.data_type == DataType.CSV:
            data = pd.read_csv(self.info.data_path,
                               encoding=self.info.encoding, sep=self.info.sep)
            data = data.iterrows()
            if self.info.verbose == 1:
                data = tqdm(data, desc='build dataset', unit=' lines')

            for x in data:
                text, label = x[1][self.info.str_text], x[1][self.info.str_label]

                label = label.translate(str.maketrans(
                    {'\t': '', '\r': '', '\n': '', ' ': ''})
                )

                if len(text.split(' ')) >= self.info.max_length:
                    continue
                self.plain_data.append((text, label))
                self.data.append((self.data_to_tensor(text),
                                  self.label_to_tensor(label)))

        elif self.info.data_type == DataType.JSON:
            data = pd.read_json(self.info.data_path,
                                encoding=self.info.encoding)
            data = data.iterrows()
            if self.info.verbose == 1:
                data = tqdm(data, desc='build dataset', unit=' lines')

            for x in data:
                text, label = x[1][self.info.str_text], x[1][self.info.str_label]
                if len(text.split(' ')) >= self.info.max_length:
                    continue

                label = label.translate(str.maketrans(
                    {'\t': '', '\r': '', '\n': '', ' ': ''})
                )

                self.plain_data.append((text, label))
                self.data.append((self.data_to_tensor(text),
                                  self.label_to_tensor(label)))

        elif self.info.data_type == DataType.EXCEL:
            data = pd.read_excel(self.info.data_path)
            data = data.iterrows()
            if self.info.verbose == 1:
                data = tqdm(data, desc='build dataset', unit=' lines')

            for x in data:
                text, label = x[1][self.info.str_text], x[1][self.info.str_label]
                if len(text.split(' ')) >= self.info.max_length:
                    continue

                label = label.translate(str.maketrans(
                    {'\t': '', '\r': '', '\n': '', ' ': ''})
                )

                self.plain_data.append((text, label))
                self.data.append((self.data_to_tensor(text),
                                  self.label_to_tensor(label)))

    def _get_str_to_str(self) -> None:
        """Get src texts and target text from data file. """

        if self.info.data_type == DataType.TEXT:
            with open(self.info.data_path, mode='r+b') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as fm:

                    if self.info.verbose == 1:
                        lines = tqdm(iter(fm.readline, b""),
                                     desc='build dataset',
                                     unit=' lines')
                    else:
                        lines = iter(fm.readline, b"")

                    for line in lines:
                        line = convert_to_unicode(line)

                        src, target = line.split(self.info.sep)

                        if len(src.split(' ')) >= self.info.max_length or len(target.split(' ')) >= self.info.max_length:
                            continue

                        self.plain_data.append((src, target))
                        self.data.append(self.data_to_tensor((src, target)))

        elif self.info.data_type == DataType.CSV:
            data = pd.read_csv(self.info.data_path,
                               encoding=self.info.encoding, sep=self.info.sep)
            data = data.iterrows()
            if self.info.verbose == 1:
                data = tqdm(data, desc='build dataset', unit=' lines')

            for x in data:
                src, target = x[1][self.info.str_text], x[1][self.info.str_label]

                if len(src.split(' ')) >= self.info.max_length or len(target.split(' ')) >= self.info.max_length:
                    continue

                self.plain_data.append((src, target))
                self.data.append(self.data_to_tensor((src, target)))

        elif self.info.data_type == DataType.JSON:
            data = pd.read_json(self.info.data_path,
                                encoding=self.info.encoding)
            data = data.iterrows()
            if self.info.verbose == 1:
                data = tqdm(data, desc='build dataset', unit=' lines')

            for x in data:
                src, target = x[1][self.info.str_text], x[1][self.info.str_label]

                if len(src.split(' ')) >= self.info.max_length or len(target.split(' ')) >= self.info.max_length:
                    continue

                self.plain_data.append((src, target))
                self.data.append(self.data_to_tensor((src, target)))

        elif self.info.data_type == DataType.EXCEL:
            data = pd.read_excel(self.info.data_path)
            data = data.iterrows()
            if self.info.verbose == 1:
                data = tqdm(data, desc='build dataset', unit=' lines')

            for x in data:
                src, target = x[1][self.info.str_text], x[1][self.info.str_label]

                if len(src.split(' ')) >= self.info.max_length or len(target.split(' ')) >= self.info.max_length:
                    continue

                self.plain_data.append((src, target))
                self.data.append(self.data_to_tensor((src, target)))

    @classmethod
    def load(cls, id: str):
        """Load saved dataset.

        Args:
            id (str): Dataset unique id.

        Returns:
            :obj:`Dataset`: Dataset from id.
        """
        return load_dataset(id)

    def __str__(self) -> str:
        """override __str__ method.

        Returns:
            str: Description of the current dataset.
        """
        return f'[{self.info.id} : {self.info.version}]'

    def n_class(self) -> int:
        """Get number of label classes.

        Returns:
            int: Number of label classes.
        """
        return len(self.info.l2i)
