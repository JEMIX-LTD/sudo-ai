#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""Preprocess.Text module
@author: Aymen Jemi (jemix) <jemiaymen@gmail.com> at SUDO-AI

"""

import itertools
import mmap
import os
import pickle
import re
import unicodedata
from collections import Counter

import torch
from tqdm.auto import tqdm

from ..utils import (DEVICE,
                     EOC_CHAR,
                     EOC_TOKEN,
                     LATIN_LOWER_PATTERN,
                     SOC_CHAR,
                     SOC_TOKEN,
                     datapath,
                     load_tokenizer)


class InputTypeError(Exception):
    """Exception raised for errors in the input.

    Args:
        str_or_list (str | list): input str_or_list which caused the error.
        message (str, optinal): Human readable string describing the exception. Defaults to 'Is not str or list'.

    Attributes:
        str_or_list (str | list): input str_or_list which caused the error.
        message (str): Human readable string describing the exception.

    """

    def __init__(self, str_or_list, message="Is not str or list ."):
        self.str_or_list = str_or_list
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.str_or_list} -> {self.message}'


class NotTrainedError(Exception):
    """Exception raised when tokenizer not trained yet.

    Args:
        obj (:obj:`BasicTokenizer`) class which caused the error.
        message (str, optinal): Human readable string describing the exception. Defaults to 'Is not trained yet'

    Attributes:
        obj (:obj:`BasicTokenizer`) class which caused the error.
        message (str): Human readable string describing the exception.
    """

    def __init__(self, obj, message="Is not trained yet"):
        self.obj = obj
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{ type(self.obj)} -> {self.message}'


def convert_to_unicode(text):
    """Convert input to unicode text.

    Args:
        text (str | bytes): Input data.

    Raises:
        ValueError: When input text is not in (str, bytes)

    Returns:
        str: Unicode text.
    """
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def clean_text(text: str) -> str:
    """Clean text.

    Args:
        text (str): Input text.

    Returns:
        str: Clean text.
    """
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def whitespace_tokenize(text: str) -> list:
    """Tokenizer text with spaces.

    Args:
        text (str): Input data text.

    Returns:
        list: Output tokens.
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def strip_duplicated_letter(text: str, tokens: bool = False):
    """Clean text from duplicated lettre.

    Args:
        text (str): Input data text.
        tokens (bool, optional): If True return tokens not str. Defaults to False.

    Examples:
        These examples illustrate how to use strip_duplicated_lettre().

        >>> strip_duplicated_letter('helllllo word')
        'hello word'
        >>> strip_duplicated_letter('hellllllllo dude', True)
        ['hello', 'dude']

    Returns:
        list: Tokens.
        str: Output clean data from duplicated lettre.
    """
    output = []
    for token in whitespace_tokenize(text):
        count = Counter(token).most_common()[0][1]
        if count > 2:
            output.append(re.sub(r'(\w)\1+', r'\1\1', token))
        else:
            output.append(token)

    if tokens:
        return output
    return ' '.join(output)


def strip_accents(text: str) -> str:
    """Clean text from accents.

    Args:
        text (str): Input data text.

    Returns:
        str: Output text cleaned from accents.
    """
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def strip_punc(text: str) -> str:
    """Clean text from punctuation.

    Args:
        text (str): Input data text.

    Returns:
        str: Output text cleaned from punctuation.
    """
    text = convert_to_unicode(text)
    output = []
    for char in text:
        if _is_punctuation(char):
            continue
        output.append(char)
    return ''.join(output)


def _is_whitespace(char: str) -> bool:
    """Check if character is whitespace.

    Args:
        char (str): Input character.

    Returns:
        bool: Is whitespace or not.
    """
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char: str) -> bool:
    """Check if character is control.

    Args:
        char (str): Input character.

    Returns:
        bool: Is control or not.
    """
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_char_accepted(char: str) -> bool:
    """Check if character accepted or not.

    Args:
        char (str): Input character.

    Returns:
        bool: Is accepted or not.
    """
    if char == '\t' or char == '\n' or char == '\r':
        return False
    cat = unicodedata.category(char)
    if cat == 'Cc' or cat == 'Cf' or cat == 'Mn':
        return False
    return True


def _to_unicode_strip_accents(text, do_lower_case: bool = False) -> str:
    """Conver input to unicode text and clean'it from accents.

    Args:
        text (str | bytes): Input data.
        do_lower_case (bool, optional): Lower case text if True. Defaults to False.

    Raises:
        ValueError: When input text is not in (str, bytes)

    Returns:
        str: Unicode text;
    """
    if isinstance(text, str):
        text = unicodedata.normalize("NFD", text)
        if do_lower_case:
            text = text.lower()
    elif isinstance(text, bytes):
        text = text.decode("utf8", "ignore")
        text = unicodedata.normalize("NFD", text)
        if do_lower_case:
            text = text.lower()
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))
    return text


def _is_punctuation(char: str) -> bool:
    """Check if character is punctuation or not.

    Args:
        char (str): Input character.

    Returns:
        bool: Is punctuation or not.
    """
    cp = ord(char)

    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True

    if cp == 247 or cp == 215 or cp == 166 or cp == 1600:
        return True

    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class StopWord():
    """StopWord class to escape stopwords.

    Attributes:
        path (str): Path of stopwords file.
        id (str): Stopwords file id.
        words (set(str)): All stopwords.


    """

    def __init__(self, path: str = None, id: str = None):
        """Create StopWord class.

        Args:
            path (str, optional): Path of stopwords file. Defaults to None.
            id (str, optional): Stopword file id. Defaults to None.
        """
        self.path = path
        self.id = id
        self.words = set()
        self.load()

    def load(self):
        """Load stopwords with path or id.

        Raises:
            FileNotFoundError: When stopwords file not found.
            FileNotFoundError: When stopwords id is wrong.
            Exception: When stopwords path is None and id is None.
        """
        if self.path is not None:
            if not os.path.exists(self.path):
                raise FileNotFoundError(
                    f'stop words file not exist [{self.path}]')
            with open(self.path, "r", encoding='utf8', errors="ignore") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                    words = m.read().decode('utf8').splitlines()
                    self.words = set(words)
        elif id is not None:
            word_path = os.path.join(datapath(id), 'stop_words.txt')
            if not os.path.exists(word_path):
                raise FileNotFoundError(
                    f'stop words file not exist [{word_path}]')
            with open(word_path, "r", encoding='utf8', errors="ignore") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                    words = m.read().decode('utf8').splitlines()
                    self.words = set(words)

        else:
            raise Exception('path or id one must be set')

    def is_stopword(self, word: str) -> bool:
        """Check if word is a stopword.

        Args:
            word (str): Input word.

        Examples:
            These examples illustrate how to use StopWord class.

            >>> sw = StopWord(path='../data/ttd/src.txt')
            >>> sw('alik')
            True
            >>> sw.is_stopword('not')
            False

        Returns:
            bool: Result if is stopword or not.
        """
        return word in self.words

    def __call__(self, word: str) -> bool:
        """Check if word is a stopword.

        Args:
            word (str): Input word.

        Examples:
            These examples illustrate how to use StopWord class.

            >>> sw = StopWord(path='../data/ttd/src.txt')
            >>> sw('alik')
            True
            >>> sw.is_stopword('not')
            False

        Returns:
            bool: Result if is stopword or not.
        """
        return self.is_stopword(word)


def unique_words(path: str,
                 clean: bool = False,
                 do_lower_case: bool = True,
                 strip_duplicated: bool = False,
                 stopword: StopWord = None) -> set:
    """Get unique words from text file.

    Args:
        path (str): Path of text data.
        clean (bool, optional): If True clean text. Defaults to False.
        do_lower_case (bool, optional): If True lower case text. Defaults to True.
        strip_duplicated (bool, optional): If True clean text from duplicated characters. Defaults to False.
        stopword (:obj:`StopWord`, optional): StopWord class to escape stopwords . Defaults to None.

    Raises:
        FileExistsError: When text file not exist.

    Returns:
        set: Unique words.
    """
    if not os.path.exists(path):
        raise FileExistsError('data file not found!')
    words = []
    output = set()

    file_stats = os.stat(path)
    size = file_stats.st_size / (1024 * 1024)

    big_file = size > 512

    if big_file:
        for line in tqdm(open(path, "r", encoding='utf8', errors="ignore"), desc='process big file', unit=' lines'):
            if clean:
                line = clean_text(line)

            if strip_duplicated:
                line = strip_duplicated_letter(line)

            for word in whitespace_tokenize(line):
                if stopword is not None and isinstance(stopword, StopWord):
                    if stopword(word):
                        continue

                if do_lower_case:
                    word = strip_accents(word.lower())
                words.append(word)
    else:
        with open(path, "r", encoding='utf8', errors="ignore") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                lines = m.read().decode('utf8').splitlines()
                for line in tqdm(lines, desc='process file', unit=' lines'):

                    if clean:
                        line = clean_text(line)
                    if strip_duplicated:
                        line = strip_duplicated_letter(line)

                    for word in whitespace_tokenize(line):
                        if stopword is not None and isinstance(stopword, StopWord):
                            if stopword(word):
                                continue

                        if do_lower_case:
                            word = strip_accents(word.lower())
                        words.append(word)
    output = set(words)
    return output


def unique_words_with_pattern(path,
                              pattern=LATIN_LOWER_PATTERN,
                              do_lower_case=True,
                              clean=False,
                              strip_duplicated=False,
                              stopword=None):
    pattern = pattern
    do_lower = do_lower_case
    clean = clean
    duplicated = strip_duplicated
    stopword = stopword

    def split_iter(string):
        if clean:
            string = clean_text(string)

        if duplicated:
            string = strip_duplicated_letter(string)

        if do_lower:
            string = strip_accents(string.lower())

        if stopword is not None and isinstance(stopword, StopWord):
            output = []
            for x in string.split():
                if stopword(x):
                    continue
                output.append(x)
            string = " ".join(output)

        return (x.group(0) for x in re.finditer(pattern, string))

    with open(path, encoding='utf8', mode="r") as f:
        return set(itertools.chain.from_iterable(map(split_iter, f)))


def word_frequency(path: str,
                   word_freq: Counter = None,
                   do_lower_case: bool = True,
                   clean: bool = False,
                   strip_duplicated: bool = False,
                   stopword: StopWord = None) -> Counter:
    """Get word frequency from text file.

    Args:
        path (str): Path of text data.
        word_freq (:obj:`Counter`, optional): Old Counter to update it with new Counter. Defaults to None.
        do_lower_case (bool, optional): If True lower case text. Defaults to True.
        clean (bool, optional): If True clean text. Defaults to False.
        strip_duplicated (bool, optional): If True clean text from duplicated characters. Defaults to False.
        stopword (:obj:`StopWord`, optional): StopWord class to escape stopwords . Defaults to None.

    Raises:
        FileExistsError: When data file not exist.

    Returns:
        :obj:`Counter`: Words frequency.
    """
    if not os.path.exists(path):
        raise FileExistsError('data file not found!')
    words = []

    with open(path, "r+b") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            for line in tqdm(iter(m.readline, b""), desc='train tokenizer ', unit=' lines'):
                line = convert_to_unicode(line)
                if clean:
                    line = clean_text(line)
                if strip_duplicated:
                    line = strip_duplicated_letter(line)

                for word in whitespace_tokenize(line):
                    if stopword is not None and isinstance(stopword, StopWord):
                        if stopword(word):
                            continue

                    if do_lower_case:
                        word = strip_accents(word.lower())
                    words.append(word)

    c = Counter(words)
    if word_freq is not None and isinstance(word_freq, Counter):
        word_freq.update(c)
        return word_freq

    return c


def unique_chars(path: str, do_lower_case: bool = False):
    """Get unique characters from data file.

    Args:
        path (str): Path of text data.
        do_lower_case (bool, optional): If True lower case text. Defaults to True.

    Raises:
        FileExistsError: When data file not exist.

    Returns:
         set(str): Unique characters.
    """
    if not os.path.exists(path):
        raise FileExistsError('data file not found!')

    chars = set()
    output = set()

    with open(path, mode='r+b') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as fm:
            for line in tqdm(iter(fm.readline, b""), desc='train chartokenizer', unit=' lines'):
                line = _to_unicode_strip_accents(line, do_lower_case)
                for char in line:
                    chars.add(char)

    for char in chars:
        if _is_char_accepted(char):
            output.add(char)

    return sorted(output)


class BasicTokenizer():
    """The base class sudoai.preprocess.BasicTokenizer representing a Tokenizer.

    Attributes:
        lower_case (bool): If True lower case text.
        duplicated (bool): If True clean text from duplicated characters.
        punc (bool): If True clean text form punctuations.
        max_size (int): Maximum vocabulary size.
        vocab_size (int): Vocabulary size.
        vocab (set(str)): Vocaburaly.
        words_freq (:obj:`Counter`): Words frequency.
        strip_stopword (bool): If True clean text from stopwords.
        stopword (:obj:`StopWord`): Stopword class.
        w2i (dict(str, int)): Dictionary of word to index.
        i2w (dict(int, str)): Dictionary of index to word.

    Warnings:
        If you set stip_stop_words True you should set stopword, if not
        an Exception raised.
    """

    def __init__(self,
                 max_vocab_size: int = None,
                 do_lower_case: bool = True,
                 strip_duplicated: bool = False,
                 strip_punc: bool = False,
                 strip_stop_words: bool = False,
                 stopword: StopWord = None) -> None:
        """Create BasicTokenizer class.

        Args:
            max_vocab_size (int, optional): Maximum vocabulary size. Defaults to None.
            do_lower_case (bool, optional): If True lower case text. Defaults to True.
            strip_duplicated (bool, optional): If True clean text from duplicated characters. Defaults to False.
            strip_punc (bool, optional): If True clean text form punctuations. Defaults to False.
            strip_stop_words (bool, optional): If True clean text from stopwords. Defaults to False.
            stopword (:obj:`StopWord`, optional): StopWord class to escape stopwords . Defaults to None.

        Warnings:
            If you set stip_stop_words True you should set stopword, if not
            an Exception raised.


        Raises:
            ValueError: When strip_stop_words is True and stopword is None.
        """
        self.lower_case = do_lower_case
        self.duplicated = strip_duplicated
        self.punc = strip_punc
        self.max_size = max_vocab_size
        self.w2i = dict()
        self.i2w = dict()
        self.vocab_size = 0
        self.vocab = set()
        self.words_freq = Counter()

        if strip_stop_words:
            if stopword is None or not isinstance(stopword, StopWord):
                raise ValueError('you must set param "stopword : StopWord" ')
            self.stopword = stopword
            self.strip_stopword = True
        else:
            self.strip_stopword = False

    def tokenize(self, text: str):
        """Tokenize text.

        Args:
            text (str): Input text line.

        Returns:
            list: Tokens.
        """
        text = convert_to_unicode(text)
        text = clean_text(text)

        if self.duplicated:
            text = strip_duplicated_letter(text)
        if self.punc:
            text = strip_punc(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.lower_case:
                token = strip_accents(token.lower())
            if self.strip_stopword:
                if self.stopword(token):
                    continue

            split_tokens.extend(self.split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def split_on_punc(self, text: str):
        """Splits punctuation on a piece of text.

        Returns:
            list: Tokens.
        """
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def save(self, id: str, override: bool = False) -> bool:
        """Save Tokenizer.

        Args:
            id (str): Unique id of tokenizer.
            override (bool, optional): If True erase old tokenizer file. Defaults to False.

        Raises:
            FileExistsError: When tokenizer file is exist and override is False.

        Returns:
            bool: Tokenizer saved or not.
        """
        tk_name = id.lower()

        folder_name = datapath(tk_name)
        tk_path = os.path.join(folder_name, 'token.pt')

        os.makedirs(folder_name, exist_ok=True)

        if override:
            if os.path.exists(tk_path):
                os.unlink(tk_path)

        if os.path.exists(tk_path):
            raise FileExistsError()
        with open(tk_path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    def load(self, id: str):
        """Load saved tokenizer.

        Args:
            id (str): Unique id of tokenizer.

        Returns:
            `Tokenizer`: Tokenizer .
        """
        return load_tokenizer(id)

    def encode(self, text: str):
        """Encode text to ids.

        Args:
            text (str): Input text line.

        Raises:
            NotTrainedError: When tokenizer not trained yet.

        Returns:
            list: ids.
        """
        if len(self.vocab) == 0:
            raise NotTrainedError(self)

        output = []
        tokens = self.tokenize(text)
        for token in tokens:
            try:
                id = self.w2i[token]
            except KeyError:
                continue
            output.append(id)
        return output

    def decode(self, ids: list):
        """Decode ids to text.

        Args:
            ids (list): Input ids.

        Returns:
            list: Output words.
        """
        output = []
        for id in ids:
            try:
                token = self.i2w[id]
            except KeyError:
                token = '[UNK]'

            output.append(token)
        return output

    def __call__(self, text: str):
        """Encode and return tensor from text data.

        Args:
            text (str): Input text line.

        Returns:
            :obj:`torch.Tensor`: Tensor of output ids.
        """
        en = self.encode(text)
        return torch.tensor(en, device=DEVICE).view(-1, 1)

    def word_frequency(self,
                       path: str,
                       word_freq: Counter = None):
        """Get word frequency from text file.

        Args:
            path (str): Path of text data.
            word_freq (:obj:`Counter`, optional): Old Counter to update it with new Counter. Defaults to None.

        Raises:
            FileExistsError: When data file not exist.

        Returns:
            :obj:`Counter`: Words frequency.
        """
        if not os.path.exists(path):
            raise FileExistsError('data file not found!')
        words = []

        with open(path, "r+b") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                for line in tqdm(iter(m.readline, b""), desc='train tokenizer ', unit=' lines'):
                    words += self.tokenize(line)

        c = Counter(words)
        if word_freq is not None and isinstance(word_freq, Counter):
            word_freq.update(c)
            return word_freq

        return c

    def train(self, path: str = None, paths: list = None):
        """Train current Tokenizer.

        Args:
            path (str, optional): Path of text data. Defaults to None.
            paths (list, optional): Paths of text data. Defaults to None.

        Notes:
            Split text into small pieces of text it's take less time.
            Big file always take more time then small pieces of file.


        Raises:
            ValueError: When path and paths is both None.
            TypeError: When path or paths is wrong type.
        """
        if path is None and paths is None:
            raise ValueError(
                "you must set 'path :str' or 'paths list[ str ]' ")

        if paths is not None and isinstance(paths, list):
            for path in paths:
                self.words_freq = self.word_frequency(path, self.words_freq)

        elif path is not None and isinstance(path, str):
            self.words_freq = self.word_frequency(path, self.words_freq)
        else:
            raise TypeError('path must be str and paths must be list[str]')

        if self.max_size is not None and isinstance(self.max_size, int):
            words = set(sorted(self.words_freq)[0:self.max_size])
        else:
            words = set(sorted(self.words_freq))

        words.add('[UNK]')
        self.vocab = words
        self.vocab_size = len(self.vocab)

        self.w2i = dict(
            [(word, i) for i, word in enumerate(self.vocab)]
        )
        self.i2w = dict(
            [(i, word) for i, word in enumerate(self.vocab)]
        )

    def from_pretrain(self, vocab_file: str, words_freq: str):
        """Load tokenizer from saved vocab file and words frequency file.

        Args:
            vocab_file (str): Path of vocabulary file.
            words_freq (str): Path of words frequency file.

        Raises:
            FileExistsError: When vocabulary file not exist.
            FileExistsError: When words frequency file not exist.
            ValueError: When words frequency file is corrupted.
        """
        if not os.path.exists(vocab_file):
            raise FileExistsError('vocab file not found!')
        if not os.path.exists(words_freq):
            raise FileExistsError('words frequency file not found!')

        with open(vocab_file, "r", encoding='utf8', errors="ignore") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                words = set(m.read().decode('utf8').splitlines())

                words.add('[UNK]')
                self.vocab = words
                self.vocab_size = len(self.vocab)

                self.w2i = dict(
                    [(word, i) for i, word in enumerate(self.vocab)]
                )
                self.i2w = dict(
                    [(i, word) for i, word in enumerate(self.vocab)]
                )

        with open(words_freq, 'rb') as f:
            words_freq = pickle.load(f)
            if isinstance(words_freq, Counter):
                self.words_freq = words_freq
            else:
                raise ValueError('words_freq path is not a counter !!')

    def save_pretrain(self, vocab_file: str, words_freq: str):
        """Save vocabulary file and words frequency file.

        Args:
            vocab_file (str): Path of vocabulary file.
            words_freq (str): Path of words frequency file.
        """
        with open(words_freq, 'wb') as f:
            pickle.dump(self.words_freq, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(vocab_file, encoding='utf8', mode='w') as f:
            for word in self.vocab:
                f.write(word + '\n')


class WordTokenizer():
    """Word Tokenizer class.

    Attributes:
        basic (:obj:`BasicTokenizer`): BasicTokenizer class.
        prefix_suffix (:obj:`PrefixSuffix`) PrefixSuffix class.

    """

    def __init__(self,
                 max_vocab_size: int = None,
                 do_lower_case: bool = True,
                 strip_duplicated: bool = False,
                 trip_punc: bool = False,
                 strip_stop_words: bool = False,
                 stopword: StopWord = None):
        """Create WordTokenizer class.

        Args:
            max_vocab_size (int, optional): Maximum vocabulary size. Defaults to None.
            do_lower_case (bool, optional): If True lower case text. Defaults to True.
            strip_duplicated (bool, optional): If True clean text from duplicated characters. Defaults to False.
            strip_punc (bool, optional): If True clean text form punctuations. Defaults to False.
            strip_stop_words (bool, optional): If True clean text from stopwords. Defaults to False.
            stopword (:obj:`StopWord`, optional): StopWord class to escape stopwords . Defaults to None.

        """
        self.basic = BasicTokenizer(max_vocab_size,
                                    do_lower_case,
                                    strip_duplicated,
                                    trip_punc,
                                    strip_stop_words,
                                    stopword)
        self.prefix_suffix = PrefixSuffix()

    def tokenize(self, text: str):
        """Tokenize text.

        Args:
            text (str): Input text line.

        Returns:
            list: Tokens.
        """
        output_tokens = []
        for token in self.basic.tokenize(text):
            sub_tokens = self.prefix_suffix(token)
            if len(sub_tokens) == 1:
                output_tokens.append(token)
            else:
                root = sub_tokens[1]
                if root in self.basic.vocab:
                    output_tokens.extend(sub_tokens)
                else:
                    output_tokens.append(token)
        return output_tokens

    def train(self, path: str = None, paths: list = None):
        """Train current Tokenizer.

        Args:
            path (str, optional): Path of text data. Defaults to None.
            paths (list, optional): Paths of text data. Defaults to None.
        """
        self.basic.train(path, paths)

    def from_pretrain(self, vocab_file: str, words_freq: str):
        """Load tokenizer from saved vocab file and words frequency file.

        Args:
            vocab_file (str): Path of vocabulary file.
            words_freq (str): Path of words frequency file.

        Raises:
            FileExistsError: When vocabulary file not exist.
            FileExistsError: When words frequency file not exist.
            ValueError: When words frequency file is corrupted.
        """
        self.basic.from_pretrain(vocab_file, words_freq)

    def save_pretrain(self, vocab_file: str, words_freq: str):
        """Save vocabulary file and words frequency file.

        Args:
            vocab_file (str): Path of vocabulary file.
            words_freq (str): Path of words frequency file.
        """
        self.basic.save_pretrain(vocab_file, words_freq)

    def encode(self, text: str):
        """Encode text to ids.

        Args:
            text (str): Input text line.

        Raises:
            NotTrainedError: When tokenizer not trained yet.

        Returns:
            list: ids.
        """
        if len(self.basic.vocab) == 0:
            raise NotTrainedError(self)

        output = []
        tokens = self.tokenize(text)
        for token in tokens:
            try:
                id = self.basic.w2i[token]
            except KeyError:
                continue
            output.append(id)
        return output

    def decode(self, ids: list):
        """Decode ids to text.

        Args:
            ids (list): Input ids.

        Returns:
            list: Output words.
        """
        output = []
        for id in ids:
            try:
                token = self.basic.i2w[id]
            except KeyError:
                token = '[UNK]'
            output.append(token)
        return ' '.join(output)

    def __call__(self, text):
        """Encode and return tensor from text data.

        Args:
            text (str): Input text line.

        Returns:
            :obj:`torch.Tensor`: Tensor of output ids.
        """
        en = self.encode(text)
        return torch.tensor(en, device=DEVICE).view(-1, 1)

    def save(self, id: str, override: bool = False):
        """Save WordTokenizer.

        Args:
            id (str): Unique id of tokenizer.
            override (bool, optional): If True erase old tokenizer file. Defaults to False.

        Raises:
            FileExistsError: When tokenizer file is exist and override is False.

        Returns:
            bool: Tokenizer saved or not.
        """
        tk_name = id.lower()

        folder_name = datapath(tk_name)
        tk_path = os.path.join(folder_name, 'token.pt')

        os.makedirs(folder_name, exist_ok=True)

        if override:
            if os.path.exists(tk_path):
                os.unlink(tk_path)

        if os.path.exists(tk_path):
            raise FileExistsError()
        with open(tk_path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, id: str):
        """Load saved word tokenizer.

        Args:
            id (str): Unique id of tokenizer.

        Returns:
            :obj:`WordTokenizer`: Word tokenizer .
        """
        return load_tokenizer(id)

    def init(self, do_lower_case: bool,
             strip_duplicated: bool,
             strip_punc: bool,
             stopword: StopWord):
        """Reinit attributes of WordTokenizer.

        Args:
            do_lower_case (bool): If True lower case text.
            strip_duplicated (bool): If True clean text from duplicated characters.
            strip_punc (bool): If True clean text form punctuations.
            stopword (:obj:`StopWord`): Stopword class.
        """
        self.lower_case = do_lower_case
        self.duplicated = strip_duplicated
        self.punc = strip_punc
        self.stopword = stopword


class PrefixSuffix():
    """Detect Prefix and Suffix .

    Attributes:
        suffix (list(str)): List of all suffix.
        prefix (list(str)): List of all prefix.

    Examples:
        These examples illustrate how to use PrefixSuffix class.

        >>> clean = PrefixSuffix()
        >>> clean('الهدف')
        ['##ال', 'هدف']
        >>> clean('نمتم')
        ['نم', 'تم##']

    """

    def __init__(self,
                 prefix=['ب',
                         'ك',
                         'س',
                         'و',
                         'ال',
                         'أ',
                         'ف',
                         'ل'],
                 suffix=['ين',
                         'ان',
                         'و',
                         'ه',
                         'ك',
                         'ا',
                         'ي',
                         'ن',
                         'ت',
                         'ات',
                         'ون',
                         'وا',
                         'تم',
                         'هم',
                         'كم']):
        """Create PrefixSuffix class.

        Args:
            prefix (list, optional): List of prefix. Defaults to ['ب', 'ك', 'س', 'و', 'ال', 'أ', 'ف', 'ل'].
            suffix (list, optional): List of suffix. Defaults to ['ين', 'ان', 'و', 'ه', 'ك', 'ا', 'ي', 'ن'
            , 'ت', 'ات', 'ون', 'وا', 'تم', 'هم', 'كم'].
        """
        self.suffix = suffix
        self.prefix = prefix

    def __call__(self, token: str):
        """Get Prefix and Suffix.

        Args:
            token (str): Input token.

        Returns:
            list(str): List with prefix (if exist) and suffix (if exist) and base.
        """
        return self.get(token)

    def get(self, token: str):
        """Get Prefix and Suffix.

        Args:
            token (str): Input token.

        Returns:
            list(str): List with prefix (if exist) and suffix (if exist) and base.
        """
        output = []

        if len(token) <= 3:
            output.append(token)
            return output

        for _p in self.prefix:
            if token.startswith(_p):
                token = token.replace(_p, '', 1)
                output.append(f'##{_p}')
                break

        for _s in self.suffix:
            if token.endswith(_s):
                token = token.replace(_s, '', 1)
                output.append(token)
                output.append(f'{_s}##')
                break

        if len(output) < 2:
            output.append(token)

        return output


class CharTokenizer():
    """Char Tokenizer class.

    Attributes:
        lower_case (bool): If True lower case text.
        duplicated (bool): If True clean text from duplicated characters.
        stopword (:obj:`StopWord`): Stopword class.
        clean (bool): If True clean data text.
        c2i (dict(str, int)): Dictionary of character to index.
        i2c (dict(int, str)): Dictionary of index to character.
        vocab_size (int): Vocabulary size.
        vocab (set(str)): Vocaburaly.

    """

    def __init__(self,
                 do_lower_case: bool = True,
                 strip_duplicated: bool = False,
                 do_clean: bool = True,
                 stopword: StopWord = None):
        """Create CharTokenizer class.

        Args:
            do_lower_case (bool, optional): If True lower case text. Defaults to True.
            strip_duplicated (bool, optional): If True clean text from duplicated characters. Defaults to False.
            do_clean (bool, optional): If True clean data text. Defaults to True.
            stopword (:obj:`StopWord`, optional): StopWord class to escape stopwords . Defaults to None.

        """
        self.lower_case = do_lower_case
        self.duplicated = strip_duplicated
        self.stopword = stopword
        self.clean = do_clean
        self.c2i = dict()
        self.i2c = dict()
        self.vocab_size = 0
        self.vocab = set()

    def tokenize(self, text: str) -> list:
        """Character tokenize text.

        Args:
            text (str): Input text.

        Returns:
            list: Tokens
        """
        text = convert_to_unicode(text)

        if self.clean:
            text = clean_text(text)

        if self.duplicated:
            text = strip_duplicated_letter(text)

        if self.lower_case:
            text = strip_accents(text.lower())

        if self.stopword is not None and isinstance(self.stopword, StopWord):
            output = []
            for word in whitespace_tokenize(text):
                if self.stopword(word):
                    continue
                output.append(word)
            text = " ".join(output)

        return [char for char in text]

    def save(self, id: str, override: bool = False):
        """Save CharTokenizer.

        Args:
            id (str): Unique id of tokenizer.
            override (bool, optional): If True erase old tokenizer file. Defaults to False.

        Raises:
            FileExistsError: When tokenizer file is exist and override is False.

        Returns:
            bool: Tokenizer saved or not.
        """
        tk_name = id.lower()

        folder_name = datapath(tk_name)
        tk_path = os.path.join(folder_name, 'token.pt')

        os.makedirs(folder_name, exist_ok=True)

        if override:
            if os.path.exists(tk_path):
                os.unlink(tk_path)

        if os.path.exists(tk_path):
            raise FileExistsError()
        with open(tk_path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    def load(self, id: str):
        """Load saved character tokenizer.

        Args:
            id (str): Unique id of tokenizer.

        Returns:
            :obj:`CharTokenizer`: Character tokenizer .
        """
        return load_tokenizer(id)

    def encode(self, text: str):
        """Encode text to ids.

        Args:
            text (str): Input text.

        Raises:
            NotTrainedError: When tokenizer not trained yet.

        Returns:
            list: ids.
        """
        if len(self.vocab) == 0:
            raise NotTrainedError(self)

        output = []
        tokens = self.tokenize(text)
        for token in tokens:
            try:
                id = self.c2i[token]
            except KeyError:
                continue
            output.append(id)
        return output

    def decode(self, ids: list):
        """Decode ids to characters.

        Args:
            ids (list): Input ids.

        Returns:
            list: Output characters.
        """
        output = []
        for id in ids:
            try:
                char = self.i2c[id]
            except KeyError:
                char = '[UNK]'

            output.append(char)
        return output

    def __call__(self, text: str):
        """Encode and return tensor from text data.

        Args:
            text (str): Input text.

        Returns:
            :obj:`torch.Tensor`: Tensor of output ids.
        """
        en = self.encode(text)
        return torch.tensor(en, device=DEVICE).view(-1, 1)

    def train(self, path: str = None, paths: list = None):
        """Train character tokenizer.

        Args:
            path (str, optional): Path of text data. Defaults to None.
            paths (list, optional): Paths of text data. Defaults to None.

        Raises:
            ValueError: When path and paths both are None.
            TypeError: When path or paths is wrong type.
        """
        if path is None and paths is None:
            raise ValueError(
                "you must set 'path :str' or 'paths list[ str ]' "
            )

        self.vocab.add('[UNK]')

        if paths is not None and isinstance(paths, list):
            for path in paths:
                self.chars(path)

        elif path is not None and isinstance(path, str):
            self.chars(path)
        else:
            raise TypeError('path must be str and paths must be list[str]')

        self._check_vocab()

    def chars(self, path: str):
        """Get unique characters from file and update current vocab.

        Args:
            path (str): Path of text data.
        """
        self.vocab.update(unique_chars(path, self.lower_case))

    def from_pretrain(self, vocab_file: str):
        """Load character tokenizer from saved vocab file.

        Args:
            vocab_file (str): Path of vocabulary file.

        Raises:
            FileExistsError: When vocabulary file not exist.
        """
        if not os.path.exists(vocab_file):
            raise FileExistsError('vocab file not found!')

        with open(vocab_file, "r", encoding='utf8', errors="ignore") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                chars = set(m.read().decode('utf8').splitlines())

                self.vocab = chars
                self._check_vocab()

    def save_pretrain(self, vocab_file: str):
        """Save vocabulary file.

        Args:
            vocab_file (str): Path of vocabulary file.
        """
        with open(vocab_file, encoding='utf8', mode='w') as f:
            f.writelines(self.vocab)

    def init(self, do_lower_case: bool,
             strip_duplicated: bool,
             do_clean: bool,
             stopword: StopWord):
        """Reinit attributes of CharTokenizer.

        Args:
            do_lower_case (bool): If True lower case text.
            strip_duplicated (bool): If True clean text from duplicated characters.
            do_clean (bool): If True clean text.
            stopword (:obj:`StopWord`): Stopword class.
        """
        self.lower_case = do_lower_case
        self.duplicated = strip_duplicated
        self.clean = do_clean
        self.stopword = stopword

    def _check_vocab(self):
        """Check vocabulary and fix error if exists."""
        if SOC_CHAR in self.vocab:
            self.vocab.remove(SOC_CHAR)

        if EOC_CHAR in self.vocab:
            self.vocab.remove(EOC_CHAR)

        vocab = list(self.vocab)

        vocab.insert(SOC_TOKEN, SOC_CHAR)
        vocab.insert(EOC_TOKEN, EOC_CHAR)

        self.c2i = dict(
            [(char, i) for i, char in enumerate(vocab)]
        )
        self.i2c = dict(
            [(i, char) for i, char in enumerate(vocab)]
        )

        self.vocab.add(SOC_CHAR)
        self.vocab.add(EOC_CHAR)

        self.vocab_size = len(self.vocab)
