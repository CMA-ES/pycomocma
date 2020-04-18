#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from distutils.core import setup

from setuptools import setup
import warnings
from comocma import __version__  # assumes that the right module is visible first in path, i.e., cma folder is in current folder
from comocma import __doc__ as long_description

# prevent the error when building Windows .exe
import codecs
try:
    codecs.lookup('mbcs')
except LookupError:
    ascii = codecs.lookup('ascii')
    func = lambda name, enc=ascii: {True: enc}.get(name=='mbcs')
    codecs.register(func)


try:
    with open('readme.md') as file_:
        long_description = file_.read()  # now assign long_description=long_description below
except IOError:  # file not found
    warnings.warn("readme.md file not found")
else:
    try:
        with open('readme.txt') as file:
            long_description = file.read()  # now assign long_description=long_description below
    except IOError:  # file not found
        pass

setup(name='comocma',
      long_description=long_description,  # __doc__
      version=__version__.split()[0],
      description= "Multiobjective framework Sofomore, instantiated with"
                    "the single-objective solver CMA-ES to obtain"
                    "the Multiobjective evolutionary algorithm COMO-CMA-ES.",
      long_description_content_type = 'text/markdown',
      author="Cheikh Toure and Nikolaus Hansen",
      author_email="first_author_firstname.first_author_lastname@polytechnique.edu",
      # " second_author_firstname.second_author_lastname@inria.fr", 
      # maintainer="Cheikh Toure and Nikolaus Hansen",
      # maintainer_email="first_author_firstname.first_author_lastname at polytechnique dot edu" +
      # " second_author_firstname.second_author_lastname at inria dot fr",
      url="https://github.com/CMA-ES/pycomocma",
      license="BSD",
      classifiers = [
          "Intended Audience :: Science/Research",
          "Intended Audience :: Education",
          "Intended Audience :: Other Audience",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          "Development Status :: 4 - Beta",
          "Environment :: Console",
          "License :: OSI Approved :: BSD License",
      ],
      keywords=["optimization", "multi-objective", "CMA-ES", "cmaes", "evolution strategy",],
      packages = ['comocma'],
      install_requires=["cma>=3", "moarchiving", "numpy"],
      package_data={'': ['LICENSE']},
      )

