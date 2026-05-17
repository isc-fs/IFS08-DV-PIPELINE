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
            # Phase 1 of the two-phase SLAM rewrite (#496). Node name
            # is "slam_node" to match AUTONOMY_LIFECYCLE_NODES in
            # mode_manager. Subscribes /odom + /Conos_raw + /Conos_Orange
            # + /testing_only/odom (active state only); publishes
            # /slam/pose, /Conos, /slam/finished, GT-aligned diagnostics;
            # broadcasts map → odom TF.
            "slam_node = cone_slam.slam_node:main",
            # Legacy cone-graph SLAM kept as a second entry point so the
            # replay regression suite + side-by-side comparison tool can
            # still spin it up for baselines.
            "slam_node_legacy = cone_slam.cone_graph_slam_node:main",
        ],
    },
)
