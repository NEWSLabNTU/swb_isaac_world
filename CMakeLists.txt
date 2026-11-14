cmake_minimum_required(VERSION 3.14)
project(swb_isaac_world)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
# The examples a user copies into a map or project (the rough road's road_detail.yaml files;
# roadmap 076 A9): share/swb_isaac_world/examples/.
install(DIRECTORY swb_isaac_world/examples/ DESTINATION share/${PROJECT_NAME}/examples)

# `swb_isaac_world_server`: the Isaac world server as a launch node (swb_isaac_world.launch.xml,
# roadmap 043). It runs the Python package from this source tree, in Isaac's own Python
# (scripts/isaac_env.sh); SWB_ISAAC_DIR overrides the tree at run time.
set(SWB_ISAAC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
configure_file(scripts/swb_isaac_world_server.in ${CMAKE_CURRENT_BINARY_DIR}/swb_isaac_world_server @ONLY)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/swb_isaac_world_server DESTINATION lib/${PROJECT_NAME})

# `ros2 run swb_isaac_world <entry>`: a module of the package from the same source tree
# (scripts/swb_isaac_world_entry.in), in Isaac's Python (isaac) or the system's (system).
#   make_world   a map's Isaac world: export, physics layer, world file (`make_world`)
#   road_detail  check or picture a road_detail.yaml (`road_layer`)
#   surfaces     check or picture a surfaces.yaml (`surfaces`)
foreach(entry "make_world:swb_isaac_world.make_world:isaac"
              "road_detail:swb_isaac_world.road_layer:system"
              "surfaces:swb_isaac_world.surfaces:system")
  string(REPLACE ":" ";" parts "${entry}")
  list(GET parts 0 ENTRY_NAME)
  list(GET parts 1 ENTRY_MODULE)
  list(GET parts 2 ENTRY_PYTHON)
  configure_file(scripts/swb_isaac_world_entry.in ${CMAKE_CURRENT_BINARY_DIR}/${ENTRY_NAME} @ONLY)
  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${ENTRY_NAME} DESTINATION lib/${PROJECT_NAME})
endforeach()

ament_package()
