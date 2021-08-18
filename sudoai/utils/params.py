#!/usr/bin/env python

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

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


ARABIC_PATTERN = r"[\u0600-\u065F\u066A-\u06EF\u06FA-\u06FF0-9]+"
LATIN_LOWER_PATTERN = r"[a-z0-9]+"
LATIN_PATTERN = r"[a-zA-Z0-9']+"


LATIN_LETTRE = ["'",
                ' ',
                '_',
                '-',
                '0',
                '1',
                '2',
                '3',
                '4',
                '5',
                '6',
                '7',
                '8',
                '9',
                'a',
                'b',
                'c',
                'd',
                'e',
                'f',
                'g',
                'h',
                'i',
                'j',
                'k',
                'l',
                'm',
                'n',
                'o',
                'p',
                'q',
                'r',
                's',
                't',
                'u',
                'v',
                'w',
                'x',
                'y',
                'z',
                'A',
                'B',
                'C',
                'D',
                'E',
                'F',
                'G',
                'H',
                'I',
                'J',
                'K',
                'L',
                'M',
                'N',
                'O',
                'P',
                'Q',
                'R',
                'S',
                'T',
                'U',
                'V',
                'W',
                'X',
                'Y',
                'Z']

LATIN_LOWER_LETTRE = ["'",
                      ' ',
                      '_',
                      '-',
                      '0',
                      '1',
                      '2',
                      '3',
                      '4',
                      '5',
                      '6',
                      '7',
                      '8',
                      '9',
                      'a',
                      'b',
                      'c',
                      'd',
                      'e',
                      'f',
                      'g',
                      'h',
                      'i',
                      'j',
                      'k',
                      'l',
                      'm',
                      'n',
                      'o',
                      'p',
                      'q',
                      'r',
                      's',
                      't',
                      'u',
                      'v',
                      'w',
                      'x',
                      'y',
                      'z']

ARABIC_LETTRE = ["'",
                 ' ',
                 '0',
                 '1',
                 '2',
                 '3',
                 '4',
                 '5',
                 '6',
                 '7',
                 '8',
                 '9',
                 'ا',
                 'ب',
                 'ت',
                 'ة',
                 'ث',
                 'ج',
                 'ح',
                 'خ',
                 'د',
                 'ذ',
                 'ر',
                 'ز',
                 'س',
                 'ش',
                 'ص',
                 'ض',
                 'ط',
                 'ظ',
                 'ع',
                 'غ',
                 'ف',
                 'ق',
                 'ك',
                 'ل',
                 'م',
                 'ن',
                 'ه',
                 'و',
                 'ي',
                 'ء',
                 'آ',
                 'أ',
                 'ؤ',
                 'إ',
                 'ئ']


SOC_TOKEN = 0
EOC_TOKEN = 1

SOC_CHAR = '<'
EOC_CHAR = '>'

# max number of words
MAX_WORDS = 256
# max number of chars in word
MAX_LENGTH = 1000
