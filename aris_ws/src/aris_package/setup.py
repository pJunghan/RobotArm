from setuptools import find_packages, setup
import os
import glob

package_name = 'aris_package'

setup(
    name=package_name,
    version='0.1.0',  # 적절한 버전 번호로 업데이트
    packages=find_packages(exclude=['test']),

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rds',
    maintainer_email='parkjh05109@gmail.com',
    description='ROS2 package for robotic arm and delivery system',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motion2 = aris_package.motion2:main',
        ],
    },
)