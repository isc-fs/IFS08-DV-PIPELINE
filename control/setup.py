from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jaime',
    maintainer_email='jaimeperezgil21@gmail.com',
    description='Vehicle controller: pure-pursuit lateral steering + PI longitudinal velocity tracking, with EBS / regen / autonomous-stop logic. Consumes /Path and the SLAM odometry, publishes /control_command.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Lifecycle entry. ROS node name "control_node" matches the
            # AUTONOMY_LIFECYCLE_NODES list in mode_manager.
            'control_node = control.control_node:main',
        ],
    },
)
