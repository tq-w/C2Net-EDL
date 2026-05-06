#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

setup(
    name='c2net-edl',
    version='1.0.0',
    description='Evidential Deep Learning for Long-Tail Multi-Label Medical Image Classification',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    author='Research Team',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'einops>=0.7.0',
        'timm>=0.9.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'numpy>=1.24.0',
        'Pillow>=10.0.0',
        'PyYAML>=6.0',
        'tensorboard>=2.14.0',
        'matplotlib>=3.7.0',
        'packaging>=23.0',
        'munkres>=1.1.4',
    ],
    extras_require={
        'dev': ['pytest>=7.0.0', 'black>=23.0.0', 'flake8>=6.0.0'],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)