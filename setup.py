#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
@author: Aymen Jemi (jemix) <jemiaymen@gmail.com> at SUDO-AI
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="sudoai",
    version="0.1.2",
    author="Aymen Jemi",
    author_email="jemiaymen@gmail.com",
    description="AI Solution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/suai-tn/sudo-ai",
    license='MIT',
    project_urls={
        "Bug Tracker": "https://github.com/suai-tn/sudo-ai/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    packages=find_packages(),
    install_requires= requirements,
    extras_require={
        'reports': [
            "dash-daq==0.5.0",
            "wordcloud-fa==0.1.8",
            "dash_bootstrap_components==0.12.2"
        ],
    },


    entry_points={
        "console_scripts": [
            "sudoai-io=sudoai.cli:io",
            "sudoai-ch2m=sudoai.cli:ch2m"
        ]
    },
    python_requires=">=3.6",
)
