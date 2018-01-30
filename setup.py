#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages

VERSION = '0.0.1'

long_description = ""

setup_info = dict(
    # Metadata
    name='pytorchviz',
    version=VERSION,
    author='Sergey Zagoruyko',
    author_email='sergey.zagoruyko@enpc.fr',
    url='https://github.com/pytorch/pytorchviz',
    description='a small package to create visualizations of PyTorch execution graphs',
    long_description=long_description,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True,

    install_requires=[
        'torch',
        'graphviz'
    ]
)

setup(**setup_info)
