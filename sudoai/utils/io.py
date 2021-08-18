#!/usr/bin/env python

# -*- coding: utf-8 -*-

import os
import shutil

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from .utils import datapath


class InputOutput():

    def __init__(self):
        self.is_auth = False
        self.setting_file = os.path.join(datapath('config'), 'settings.yaml')
        self.creds_file = os.path.join(datapath('config'), 'credentials.json')
        self.fid_file = os.path.join(datapath('config'), 'fid')
        if os.path.exists(self.fid_file):
            with open(self.fid_file, 'r') as f:
                self.folderid = f.read()
        else:
            raise FileNotFoundError('folderid file config not found !')

    def auth(self):
        self.gauth = GoogleAuth(settings_file=self.setting_file)

        if os.path.exists(self.creds_file):
            self.gauth.LoadCredentialsFile(self.creds_file)

        if self.gauth.access_token_expired:
            self.gauth.LocalWebserverAuth()
            self.gauth.SaveCredentialsFile(self.creds_file)

        self.drive = GoogleDrive(self.gauth)
        self.is_auth = True

    def search(self, id):
        if self.is_auth is False:
            self.auth()

        query = f"'{self.folderid}' in parents and trashed=false"
        file_list = self.drive.ListFile({'q': query}).GetList()

        for file in file_list:
            if file['title'] == id + '.zip':
                return file['id']
        return False

    def download_from_drive(self, id):
        file_id = self.search(id)
        if file_id:
            folder_name = datapath(id)
            os.makedirs(folder_name, exist_ok=True)

            data_file = os.path.join(folder_name, 'data.zip')

            gdrive_file = self.drive.CreateFile({'id': file_id})
            gdrive_file.GetContentFile(data_file)

            self.unzip(id)
        else:
            raise FileNotFoundError(f'model [{id}] not found in drive')

    def unzip(self, id):
        folder_name = datapath(id)
        path = os.path.join(folder_name, 'data.zip')
        shutil.unpack_archive(path, folder_name, format='zip')

    def zip(self, id):
        folder_name = datapath(id)
        if os.path.exists(folder_name):
            if not os.path.exists(os.path.join(folder_name, 'data.zip')):
                shutil.make_archive(folder_name, 'zip', folder_name)
                shutil.move(folder_name + '.zip',
                            os.path.join(folder_name, 'data.zip'))
        else:
            raise FileNotFoundError(f'model [{id}] not found')

    def upload_in_drive(self, id):
        self.zip(id)

        data_file = os.path.join(datapath(id), 'data.zip')
        if os.path.exists(data_file):

            if self.is_auth is False:
                self.auth()

            model_file = self.drive.CreateFile(
                {
                    'parents': [{'id': self.folderid}],
                    'title': id + '.zip'
                }
            )

            model_file.SetContentFile(data_file)
            model_file.Upload()
        else:
            raise FileNotFoundError(f'data model [{id}] not found')
