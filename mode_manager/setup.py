from setuptools import find_packages, setup

package_name = "mode_manager"

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
        "mode_manager_node — fans out lifecycle change_state calls to "
        "the autonomy nodes and propagates the mission strategy flag."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mode_manager_node = mode_manager.mode_manager_node:main",
        ],
    },
)
