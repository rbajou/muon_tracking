#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
# import subprocess
# import pip
# import sys
import glob 

REQUIREMENTS = [
    'pandas',
    'numpy',
    'scipy',
    'matplotlib',
    'scikit-learn',
    'scikit-image',
    'pyjson',
    'pylandau', #used in 'analysis.py'
]


setup(
    name='muon_tracking',
    version='0.1.0',
    description="Muon tracking in scintillator tracker with RANSAC",
    author="Raphaël Bajou",
    author_email='r.bajou2@gmail.com',
    url='https://github.com/rbajou/muon_tracking.git',
    packages=find_packages(),#['muon_tracking'],
    package_dir={
        'muon_tracking': 'muon_tracking',
    },  
    #data_files=glob.glob('muon_tracking/AcquisitionParams/ChannelNoXYMaps/*.json'),
    package_data={'muon_tracking':glob.glob('muon_tracking/AcquisitionParams/**')},
    include_package_data=True,
    install_requires=REQUIREMENTS,
    zip_safe=False,
    keywords='Muon tracking RANSAC',
    classifiers=[
        'Programming Language :: Python :: 3.7',
    ]
)