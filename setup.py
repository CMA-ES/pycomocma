#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from distutils.core import setup

from setuptools import setup

#from comocma.como import __version__  # assumes that the right module is visible first in path, i.e., cma folder is in current folder
#from como import __doc__ as long_description
#from como import __author__ as authors

# prevent the error when building Windows .exe
import codecs
try:
    codecs.lookup('mbcs')
except LookupError:
    ascii = codecs.lookup('ascii')
    func = lambda name, enc=ascii: {True: enc}.get(name=='mbcs')
    codecs.register(func)

  # indicates a multi-file module and that we have a cma folder and cma/__init__.py file

try:
    with open('README.md') as file_:
        long_description = file_.read()  # now assign long_description=long_description below
except IOError:  # file not found
    warnings.warn("README.md file not found")
else:
    try:
        with open('README.txt') as file:
            long_description = file.read()  # now assign long_description=long_description below
    except IOError:  # file not found
        pass




setup(name='comocma',
   #   long_description=long_description,  # __doc__, # can be used in the cma.py file
   #   version=__version__.split()[0],
   #   author=authors,
  #    py_modules=['como', 'hv', 'moarchiving', 'nondominatedarchive', 'sofomore_logger'],
      packages = ['comocma']
      )

