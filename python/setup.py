#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from distutils.core import setup
from setuptools import setup
import como
# prevent the error when building Windows .exe
import codecs
try:
    codecs.lookup('mbcs')
except LookupError:
    ascii = codecs.lookup('ascii')
    func = lambda name, enc=ascii: {True: enc}.get(name=='mbcs')
    codecs.register(func)


try:
    with open('README.txt') as file:
        long_description = file.read()  # now assign long_description=long_description below
except IOError:  # file not found
    pass

setup(name="como",
      keywords=["optimization", "CO-CMA-ES", "multiobjective optimization"],
      packages=["como", "cma"],
      requires=["numpy", "cma"],
      package_data={'': ['LICENSE']},  # i.e. como/LICENSE
      )