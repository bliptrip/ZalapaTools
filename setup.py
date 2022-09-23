#!/usr/bin/env python3

from setuptools import setup
import os, platform

version = "1.0.0"

requirements_fh = open('./requirements.txt', 'r')
requirements = [r.rstrip() for r in requirements_fh.readlines()]
requirements_fh.close()

setup(name='ztools',
      version=version,
      zip_safe=True,
      description='My Zalapa-lab tools, mostly related to digital image processing.',
      long_description='',
      url='https://github.com/bliptrip/ZalapaTools',
      author='Andrew Maule',
      author_email='developer@bliptrip.net',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: Linux',
        'Programming Language :: Python :: 3.x',
        'Topic :: Scientific/Engineering'],
      license='GPLv3',
      packages=['ztools',
                'ztools.dip',
                'ztools.dip.chaincodes',
                'ztools.cameracalibration'],
      install_requires=requirements,
      scripts=['ztools/dip/sse2masks.py',
               'ztools/dip/masks2sse.py',
               'ztools/dip/pmasks2sse.py']
    )
