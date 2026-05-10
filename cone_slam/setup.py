from setuptools import setup

package_name = "cone_slam"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Raul Moran",
    maintainer_email="raul@isc-fs.com",
    description="Cone-association graph SLAM (GTSAM) for IFS-08 DV pipeline.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Lifecycle SLAM node. ROS node name is "slam_node" to match
            # the AUTONOMY_LIFECYCLE_NODES list in mode_manager. Subscribes
            # /imu + /Conos_raw + /motor_rpm + /testing_only/odom (active
            # state only); publishes /tf (odom → base_link),
            # /cone_slam/state, /Conos, GT-aligned diagnostics.
            "slam_node = cone_slam.cone_graph_slam_node:main",
        ],
    },
)
