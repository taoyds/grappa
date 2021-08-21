import sys

from setuptools import setup, find_packages

setup(
    name='weaksp',
    version='0.1.0',
    description='learning latent alignments for weakly supervised semantic parsing',
    packages=find_packages(exclude=["*_test.py", "test_*.py"]),
    install_requires=[
        'allennlp==0.8.4',
        'tqdm~=4.28.1',
    ],
)
