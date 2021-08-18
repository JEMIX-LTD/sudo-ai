#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
@author: Aymen Jemi (jemix) <jemiaymen@gmail.com>

Copyright (c) 2021 Aymen Jemi SUDO-AI
"""
import os.path
import pickle
from datetime import datetime
from time import sleep

import requests
import sudoai

from ..utils import datapath


class Instagram():
    """Extract data from instagram posts.

    Attributes:
        verbose (int): Verbose value.
        access_token (str): Auth Token for authentication.
        session (:obj:`requests.Session`): Current session.
        query_tasks (dict): Tasks like user_post,comment ...
        max (int): Maximum comment by page.
        comment (dict): All comments.
        username (str): Current username.
        password (str): Current password.
        is_auth (bool): If True current class is authenticated.
        user (dict): Current user details.
        headers (str): Current session headers.

    See Also:
        For more information check the quickstart docs http://sudoai.tech/quickstart

    """

    def __init__(self,
                 token: str = None,
                 username: str = None,
                 password: str = None,
                 max: int = 12,
                 verbose: int = 0):
        """Create Instagram class

        Args:
            token (str, optional): Auth Token for authentication. Defaults to None.
            username (str, optional): Current username. Defaults to None.
            password (str, optional): Current password. Defaults to None.
            max (int, optional): Maximum comment by page. Defaults to 12.
            verbose (int, optional): Verbose value. Defaults to 0.
        """

        self.verbose = verbose
        self.access_token = token
        self.session = requests.Session()
        self.query_tasks = {'user_post': 17888483320059182,
                            'comment': 17852405266163336,
                            'like': 17864450716183058,
                            'feed': 17842794232208280,
                            'follower': 17851374694183129}
        if max > 50:
            self.max = 50
        elif max < 12:
            self.max = 12
        else:
            self.max = max
        self.comment = dict()
        self.username = username
        self.password = password
        self.is_auth = False
        self.user = dict()
        self.user['username'] = username
        self.headers = {
            'User-Agent': """Mozilla/5.0 (Windows NT 10.0; Win64; x64)
            AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36
            Edg/90.0.818.49"""
        }

        if self.password is None:
            if self.access_token is None:
                self.load_session()

    def get_comments(self, shortcode: str, with_full_name: bool = False):
        """Get comments from instagram post with shortcode.

        Args:
            shortcode (str): Short code from instagram post.
            with_full_name (bool, optional): If True get full name for comment user (take more time). Defaults to False.

        Returns:
            dict: Dict with count of comments and data.
        """

        url = 'https://www.instagram.com/graphql/query/?query_id=%d&variables={\"shortcode\":\"%s\",\"first\":%d}' % (
            self.query_tasks['comment'], shortcode, self.max)

        nostop = True
        self.comment['data'] = []
        self.comment['shortcode'] = shortcode

        if self.verbose == 1:
            sudoai.__log__.info(url)

        while nostop:
            data = self.session.get(url)
            if self.verbose == 1:
                sudoai.__log__.info(data.json())
            if data.ok:
                data = data.json()
                data = data['data']['shortcode_media']
                self.comment['comment_count'] = data['edge_media_to_comment']['count']
                for x in data['edge_media_to_comment']['edges']:
                    if self.verbose == 1:
                        sudoai.__log__.info(x)
                    r = dict()
                    r['id'] = x['node']['id']
                    r['text'] = x['node']['text']
                    r['created_at'] = x['node']['created_at']
                    r['owner_id'] = x['node']['owner']['id']
                    r['owner_username'] = x['node']['owner']['username']
                    if with_full_name:
                        r['full_name'] = self.username2fullname(
                            r['owner_username']
                        )

                    self.comment['data'].append(r)

                nostop = data['edge_media_to_comment']['page_info']['has_next_page']
                url = '''https://www.instagram.com/graphql/query/?query_id
                       =%d&variables={\"shortcode\":\"%s\",\"first\":%d ,\"after\":\"%s\"}''' % (
                    self.query_tasks['comment'],
                    shortcode,
                    self.max,
                    data['edge_media_to_comment']['page_info']['end_cursor'])
            else:
                break
            sudoai.__log__.info(
                f"total comments is :{self.comment['comment_count']}")
        return self.comment

    def save_session(self):
        """Save current session.

        Returns:
            bool: If session saved True else False.
        """
        if self.user['username'] and self.is_auth:
            session_file_name = f'#inst_{self.username}'

            session_file_name = os.path.join(
                datapath('social'), session_file_name)

            with open(session_file_name, 'wb') as f:
                pickle.dump(self.session, f)
                sudoai.__log__.debug(f'session saved in {session_file_name}')
            return True
        sudoai.__log__.warning('you can\'t save user is not authenticated')
        return False

    def load_session(self):
        """Load saved session.

        Raises:
            Exception: When username is None.
        """
        if self.username is not None:
            session_file_name = f'#inst_{self.username}'

            session_file_name = os.path.join(
                datapath('social'), session_file_name)

            if os.path.exists(session_file_name):
                with open(session_file_name, 'rb') as f:
                    self.session = pickle.load(f)
                self.valid_session()
            else:
                sudoai.__log__.error(
                    f"[{self.username}] session not exist check if you saved")
        else:
            sudoai.__log__.error("username is not found please enter username")
            raise Exception("you must enter username to load session")

    def valid_session(self):
        """Test if saved session is a valid session. """

        url = 'https://www.instagram.com/accounts/edit/?__a=1'
        test = self.session.get(url)
        if test.ok:
            if self.verbose == 1:
                print(test.json())
            test = test.json()
            self.is_auth = test['form_data']['username'] == self.user['username']
            if self.is_auth:
                sudoai.__log__.info("is a valid session")
            else:
                sudoai.__log__.info(
                    "is not a valid session please check your username and password")
        else:
            self.is_auth = False
            sudoai.__log__.info(
                "session is not exist please check if session exist")

    def login(self):
        """Authentification logic.

        Returns:
            bool: If auth is valid True, else False.
        """

        url = 'https://www.instagram.com/accounts/login/'
        url_main = url + 'ajax/'
        self.headers['referer'] = 'https://www.instagram.com/accounts/login/'
        csrf_token = ''

        auth = {'username': self.username, 'enc_password': self.password}

        r = self.session.get(
            'https://www.instagram.com/data/shared_data/',
            headers=self.headers
        )

        if r.ok:
            r = r.json()
            csrf_token = r['config']['csrf_token']
        else:
            sudoai.__log__.error(r.text)
            self.is_auth = False
            return self.is_auth

        auth['enc_password'] = '#PWD_INSTAGRAM_BROWSER:0:{}:{}'.format(
            int(datetime.timestamp(datetime.now())), auth['enc_password'])

        self.headers['x-csrftoken'] = csrf_token
        _auth = self.session.post(url_main, data=auth, headers=self.headers)
        if _auth.ok:
            _auth = _auth.json()
            if self.verbose == 1:
                sudoai.__log__.debug(_auth)

            if _auth['authenticated'] is not True:
                sudoai.__log__.error(
                    'error auth username or password is not valid')
                self.is_auth = False
            else:
                self.is_auth = True
                sudoai.__log__.info('auth with success')
        else:
            sudoai.__log__.error(_auth)
        return self.is_auth

    def username2fullname(self, username: str):
        """Convert username to fullname.

        Args:
            username (str): username to convert.

        Returns:
            str: Fullname.
            None: If username not exist.
        """
        if self.is_auth is False:
            return None
        url = f'https://www.instagram.com/{username}/?__a=1'
        r = self.session.get(url)

        if r.ok:
            r = r.json()
            sleep(0.3)
            return r['graphql']['user']['full_name']
        return None
