cmake_minimum_required(VERSION 3.14)
project(swb_isaac_world)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

# `swb_isaac_world_server`: the Isaac world server as a launch node (swb_isaac_world.launch.xml,
# roadmap 043). It runs the Python package from this source tree, in Isaac's own Python
# (scripts/isaac_env.sh); SWB_ISAAC_DIR overrides the tree at run time.
set(SWB_ISAAC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
configure_file(scripts/swb_isaac_world_server.in ${CMAKE_CURRENT_BINARY_DIR}/swb_isaac_world_server @ONLY)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/swb_isaac_world_server DESTINATION lib/${PROJECT_NAME})

ament_package()
