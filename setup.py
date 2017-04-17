# -*- coding: utf-8 -*-

import sys

from setuptools import setup, find_packages

exec(open('fftoptionlib/version.py').read())


def check_python_version():
    if sys.version_info[:2] < (3, 4):
        print('Python 3.4 or newer is required. Python version detected: {}'.format(sys.version_info))
        sys.exit(-1)


def main():
    setup(name='fftoptionlib',
          version=__version__,
          author='ArrayStream (Yu Zheng, Ran Fan, Yongxin Yang)',
          author_email='team@arraystream.com',
          url='https://github.com/arraystream/fftoptionlib',
          description='FFT-based Option Pricing Method in Python',
          long_description='FFT-based Option Pricing Method in Python',
          classifiers=[
              'Development Status :: 4 - Beta',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.3',
              'Programming Language :: Python :: 3.4',
              'Programming Language :: Python :: 3.5',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Intended Audience :: Financial and Insurance Industry'
          ],
          license='MIT',
          packages=find_packages(include=['fftoptionlib']),
          install_requires=['numpy', 'scipy', 'pandas'],
          platforms='any')


if __name__ == '__main__':
    check_python_version()
    main()
