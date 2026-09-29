from setuptools import find_packages
from setuptools import setup

setup(
    name='convchart_interfaces',
    version='0.0.0',
    packages=find_packages(
        include=('convchart_interfaces', 'convchart_interfaces.*')),
)
