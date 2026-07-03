from setuptools import find_packages, setup

package_name = "node_base"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jaime",
    maintainer_email="jaimeperezgil21@gmail.com",
    description="Shared BaseLifecycleNode for Python autonomy nodes",
    license="GPL-3.0-or-later",
    tests_require=["pytest"],
    entry_points={},
)
