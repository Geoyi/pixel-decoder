#!/usr/bin/env python
from setuptools import setup, find_packages
from imp import load_source
from os import path
import io

__version__ = load_source('pixel_decoder.version', 'pixel_decoder/version.py').__version__

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with io.open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if 'git+' not in x]

#readme
with open('README.md') as f:
    readme = f.read()

setup(
    name='pixel_decoder',
    author='Zhuangfang NaNa Yi',
    author_email='nana@developmentseed.org',
    version=__version__,
    description='A deep learning algorithm collection to train satellite segmentation',
    url='https://github.com/Geoyi/pixel-decoder',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='',
    entry_points={
        'console_scripts': ['pixel_decoder=pixel_decoder.main:cli'],
    },
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    install_requires=install_requires,
    dependency_links=dependency_links,
    long_description=readme,
    long_description_content_type="text/markdown"
)
