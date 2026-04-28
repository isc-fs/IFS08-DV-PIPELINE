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
            # Main SLAM node: subscribes /imu + /lidar/Lidar1 (PR A timing
            # trigger; PR B switches to /Conos_raw); publishes /tf
            # (odom → base_link) and /cone_slam/state.
            "cone_graph_slam = cone_slam.cone_graph_slam_node:main",
        ],
    },
)
