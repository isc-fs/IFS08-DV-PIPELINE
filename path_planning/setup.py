from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'path_planning'

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
    description='Centerline path planner: subscribes to the SLAM cone map (/Conos), runs the FaSTTUBe per-side cone-sort + cross-side matching algorithm, and publishes a smooth nav_msgs/Path on /Path for the controller.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Lifecycle entry. ROS node name "path_planning_node" matches
            # the AUTONOMY_LIFECYCLE_NODES list in mode_manager.
            'path_planning_node = path_planning.path_planning:main',
        ],
    },
)
