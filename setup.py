# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name="sparat",
    version='0.2',
    author=u"Ivana Kajić, Jan Gosmann",
    author_email="i2kajic@uwaterloo.ca",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'seaborn',
        'doit',
        'jupyter',
        'joblib',
        'sklearn',
        'nengo==2.1',
        'pytry>=0.9.1',
    ])
