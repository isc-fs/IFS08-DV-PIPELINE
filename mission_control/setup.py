from setuptools import find_packages, setup

package_name = "mission_control"

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
        "mission_control_node — DV pipeline lifecycle orchestrator. "
        "Drives mode_manager and forwards control commands between "
        "the autonomy stack and sim_supervisor / uDV."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mission_control_node = mission_control.mission_control_node:main",
        ],
    },
)
