from setuptools import find_packages, setup

package_name = "car_sensor_bridge"

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
        "On-vehicle sensor input adapter: converts the uDV steering "
        "sensor from degrees to the radians the EKF expects, and "
        "republishes the inverter wheel speed as /motor_rpm. Bridges "
        "the unit / source mismatches a plain topic remap cannot."
    ),
    license="GPL-3.0-or-later",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "car_sensor_bridge_node = "
            "car_sensor_bridge.car_sensor_bridge_node:main",
        ],
    },
)
