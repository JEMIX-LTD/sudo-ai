#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""Social module
@author: Aymen Jemi (jemix) <jemiaymen@gmail.com> at SUDO-AI

The social module is for extract data from social media.


Tip:
    If you save your authentication,for next time don't enter
    password just load_session().

Examples:
    These examples illustrate how to use :obj:`Instagram` class.

    >>> insta = Instagram(username='foulane', password='1234abcd',max=20)
    >>> comments = insta.get_comments('CRc3ZTGjxHr')
    >>> insta.username2fullname('jemiaymen')
    'أيمن الجامي'

"""

from ..social.core import Instagram

__all__ = ['Instagram']
