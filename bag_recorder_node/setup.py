from setuptools import find_packages, setup

package_name = "bag_recorder_node"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Raul Moran",
    maintainer_email="raul@isc-fs.com",
    description=(
        "bag_recorder_node — hosts /bag_recorder/start + "
        "/bag_recorder/stop services so the MC web backend can "
        "drive ros2 bag recordings from inside dv_pipeline_stack. "
        "See #465."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "bag_recorder_node = bag_recorder_node.node:main",
        ],
    },
)
