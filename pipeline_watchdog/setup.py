from setuptools import setup

package_name = "pipeline_watchdog"

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
    maintainer="Alvaro Gonzalez",
    maintainer_email="agonzaleztabernero@gmail.com",
    description=(
        "pipeline_watchdog_node — independent supervisor that trips "
        "AS Emergency when the autonomy stack goes stale while running."
    ),
    license="GPL-3.0-or-later",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Plain (non-lifecycle) node: comes up with the management trio
            # and runs for the whole session, so mode_manager can never tear
            # down the supervisor.
            "pipeline_watchdog_node = "
            "pipeline_watchdog.pipeline_watchdog_node:main",
        ],
    },
)
