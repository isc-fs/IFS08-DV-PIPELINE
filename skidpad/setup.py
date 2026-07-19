from setuptools import find_packages, setup

package_name = 'skidpad'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alvaro',
    maintainer_email='agonzaleztabernero@gmail.com',
    description='Deterministic skidpad: fixed FS-Rules figure-eight reference '
                'path, arc-length progress driver, and the ROS runtime '
                'path_planning_node delegates to in skidpad mode. No perception '
                '— the track is fully rule-defined.',
    license='GPL-3.0-or-later',
    tests_require=['pytest'],
    # No console_scripts: this package is a library imported by
    # path_planning_node (and cone_slam for the pose passthrough), not a
    # standalone node. mode_manager still runs the fixed AUTONOMY_NODE_ORDER.
    entry_points={},
)
