#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""Preprocess.Text module
@author: Aymen Jemi (jemix) <jemiaymen@gmail.com> at SUDO-AI

Preprocessing for natural language processing, sudoai.preprocess
is a lightweight library to clean and tokenize text.


Examples:
    These examples illustrate how to use sudoai Preprocess module.

    Create and save CharTokenizer:

    >>> tokenizer = CharTokenizer(strip_duplicated=True, do_lower_case=True)
    >>> tokenizer.train(path="data.txt")
    >>> tokenizer.save('id_chartokenizer', True)
    train chartokenizer: 67534 lines [00:00, 79429.67 lines/s]
    True

    >>> tokenizer('winek')
    tensor([[ 2],
            [53],
            [20],
            [59],
            [18]], device='cuda:0')

    >>> en = tokenizer.encode('winek')
    >>> tokenizer.decode(en)
    ['w', 'i', 'n', 'e', 'k']


"""

from ..preprocess.text import (InputTypeError,
                               NotTrainedError,
                               StopWord,
                               BasicTokenizer,
                               WordTokenizer,
                               PrefixSuffix,
                               CharTokenizer,
                               convert_to_unicode,
                               clean_text,
                               whitespace_tokenize,
                               strip_duplicated_letter,
                               strip_accents,
                               strip_punc,
                               unique_words,
                               unique_words_with_pattern,
                               word_frequency,
                               unique_chars)

__all__ = ['InputTypeError',
           'NotTrainedError',
           'StopWord',
           'BasicTokenizer',
           'WordTokenizer',
           'PrefixSuffix',
           'CharTokenizer',
           'convert_to_unicode',
           'clean_text',
           'whitespace_tokenize',
           'strip_duplicated_letter',
           'strip_accents',
           'strip_punc',
           'unique_words',
           'unique_words_with_pattern',
           'word_frequency',
           'unique_chars']
