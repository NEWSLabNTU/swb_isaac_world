"""A course from open data (roadmap 058): a real road's section, with its terrain and aerial
photo, as an Isaac world and Autoware's maps.

    python3 -m swb_isaac_world.course <course.yaml> <map dir> [steps...]

`courses/<name>.yaml` names the road (an Overpass filter), the section's two ends, the lanes and
the map frame's origin. The data is fetched once into `<map dir>/raw/` (OpenStreetMap through
Overpass; GSI's 5 m DEM and aerial photo tiles) and reused offline after. The geometry and the
Autoware maps run under the system Python (pyproj, lanelet2); the USD world is written by
Isaac's Python from the geometry those steps leave in `<map dir>/course/`.
"""
