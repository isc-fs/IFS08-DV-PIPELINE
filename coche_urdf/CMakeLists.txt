cmake_minimum_required(VERSION 3.8)
project(coche_urdf)

find_package(ament_cmake REQUIRED)

# Install share files (URDF, launch, meshes, rviz). Pure-data package —
# no compilable targets.
install(
  DIRECTORY urdf launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
