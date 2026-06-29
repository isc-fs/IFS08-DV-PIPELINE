from setuptools import find_packages, setup

package_name = "car_supervisor"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ISC Driverless",
    maintainer_email="driverless@isc-fs.com",
    description=(
        "On-vehicle mission/actuation adapter. Replaces sim_supervisor on "
        "the real car: translates the uDV AS state (/assi/state) + mission "
        "(/ami/mission) into mission_control_node's SetMission / "
        "RuntimeControl protocol, and relays the control output to the uDV "
        "(/steering/cmd, /force_ebs) — only while AS Driving."
    ),
    license="GPL-3.0-or-later",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "car_supervisor_node = "
            "car_supervisor.car_supervisor_node:main",
        ],
    },
)
