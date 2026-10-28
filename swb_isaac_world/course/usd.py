"""C6: the course's USD world from `<map dir>/course/meshes.npz`, and its `isaac/world.yaml`.

Run by Isaac's Python (it has `pxr`; the system Python, which made the meshes, has pyproj and
lanelet2 instead):

    $ISAAC_ROOT/venv/bin/python -m swb_isaac_world.course.usd <map dir>

One stage, metres, z up, in the map frame (`map_origin` 0): the terrain, its aerial photo as
its texture, labelled Terrain; the road, labelled Road; its lines, labelled RoadLine (CARLA's
SimReady Q-codes, so the world server's ground paths and surface defaults take them as they
do a town's). The terrain and the road carry triangle-mesh colliders; the lines are a few
millimetres above the road and are not colliders. The sun and sky are the world server's, from
the map's latitude.
"""
import argparse
import os
import shutil
import sys

import numpy as np

ROAD_COLOUR = (0.16, 0.16, 0.17)
LINE_COLOURS = {"line_centre": (0.85, 0.66, 0.08), "line_left": (0.9, 0.9, 0.9),
                "line_right": (0.9, 0.9, 0.9)}
QCODE = "semantics:labels:wikidata_qcode"
LABELS = {"Terrain": "Q186131", "Road": "Q34442", "Lines": "Q1392287"}
DEFAULT = "/Course"


def _material(stage, path, colour=None, texture=None, roughness=0.85):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, path + "/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    if texture:
        st = UsdShade.Shader.Define(stage, path + "/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, path + "/photo")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb")
    else:
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat


def _mesh(stage, path, points, faces, uv=None, normals=None, collider=False, label=None,
          material=None):
    from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = points.min(axis=0), points.max(axis=0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([tuple(lo.tolist()), tuple(hi.tolist())]))
    if normals is not None:
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if uv is not None:
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                    UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
    prim = mesh.GetPrim()
    if collider:
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
    if label:
        prim.CreateAttribute(QCODE, Sdf.ValueTypeNames.TokenArray).Set([label])
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    return prim


def _grid_normals(points, shape):
    ny, nx = shape
    z = points[:, 2].reshape(ny, nx)
    step = points[1, 0] - points[0, 0]
    gy, gx = np.gradient(z, step)
    n = np.stack([-gx, -gy, np.ones_like(z)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n.reshape(-1, 3)


def build(map_dir):
    from pxr import Sdf, Usd, UsdGeom

    work = os.path.join(map_dir, "course")
    out_dir = os.path.join(map_dir, "isaac")
    os.makedirs(out_dir, exist_ok=True)
    m = np.load(os.path.join(work, "meshes.npz"))
    photo = os.path.join(out_dir, "terrain_photo.jpg")
    shutil.copyfile(os.path.join(work, "photo.jpg"), photo)
    path = os.path.join(out_dir, "world.usdc")
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, DEFAULT)
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, DEFAULT + "/Looks")
    terrain_mat = _material(stage, DEFAULT + "/Looks/TerrainPhoto", texture="./terrain_photo.jpg",
                            roughness=0.95)
    road_mat = _material(stage, DEFAULT + "/Looks/Asphalt", colour=ROAD_COLOUR, roughness=0.8)
    shape = tuple(int(v) for v in m["terrain_shape"])
    _mesh(stage, DEFAULT + "/Terrain", m["terrain_points"], m["terrain_faces"], uv=m["terrain_uv"],
          normals=_grid_normals(m["terrain_points"], shape), collider=True, label=LABELS["Terrain"],
          material=terrain_mat)
    _mesh(stage, DEFAULT + "/Road", m["road_points"], m["road_faces"], collider=True,
          label=LABELS["Road"], material=road_mat)
    UsdGeom.Xform.Define(stage, DEFAULT + "/Lines")
    stage.GetPrimAtPath(DEFAULT + "/Lines").CreateAttribute(
        QCODE, Sdf.ValueTypeNames.TokenArray).Set([LABELS["Lines"]])
    for k, colour in LINE_COLOURS.items():
        mat = _material(stage, f"{DEFAULT}/Looks/{k}", colour=colour, roughness=0.6)
        _mesh(stage, f"{DEFAULT}/Lines/{k}", m[f"{k}_points"], m[f"{k}_faces"], material=mat)
    stage.GetRootLayer().Save()
    with open(os.path.join(out_dir, "world.yaml"), "w") as f:
        f.write("# Generated by swb_isaac_world.course.usd (roadmap 058): a course from open data,\n"
                "# in the Autoware map frame (a local transverse Mercator, map_projector_info.yaml).\n"
                f"usd: {os.path.realpath(path)}\n"
                "map_origin: [0.0, 0.0, 0.0]\n"
                "ground_paths:\n"
                '  - "/World/env/Terrain"\n'
                '  - "/World/env/Road"\n'
                '  - "/World/env/Lines"\n')
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("map_dir")
    a = ap.parse_args(argv)
    print(build(a.map_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
