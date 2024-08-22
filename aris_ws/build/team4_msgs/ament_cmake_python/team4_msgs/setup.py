from setuptools import find_packages
from setuptools import setup

setup(
    name='team4_msgs',
    version='0.0.0',
    packages=find_packages(
        include=('team4_msgs', 'team4_msgs.*')),
)
