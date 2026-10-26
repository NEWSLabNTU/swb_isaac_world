"""A run's pictures and soil state for a video (`--record <dir>`, off by default).

The first ego a scenario spawns gets three cameras of its own, under its body so PhysX carries
them: a dash camera where the rig's `camera6` sits (whether or not that sensor is on), a
low wheel camera behind and outside the rear-left wheel looking forward and down at it, so
the wheel's sinkage is in the picture, and a chase camera behind and above the car, which
keeps the car and each ground it crosses in view (a run over several zones). A fourth stands still beside the first soil zone ahead
of the ego at its spawn, past the zone's far end, looking back along the ego's line: the
render mesh shows a tile's ruts only once the wheels are 3 m past it (`soil_tiles`), so they
appear behind the car, which the wheel camera never sees. Every frame the world server
renders, the pictures are written as JPEGs, and one JSON line holds the ego's state and each
wheel's ground: class, sinkage, soil resistance, friction and slip, and what the car's onboard
computers estimated and did (`ecu`: speed, friction, sideslip, ABS and TC factors, ESC, the
wheel torques). Written by a thread, so
the frame does not wait for the disk.

    <dir>/<time>-<ego name>/dash/000123.jpg, wheel/..., chase/..., track/..., state.jsonl

Not for runs that are judged: more render products cost frame time, and each new one keeps a
little memory after it is gone (docs/issues/003).
"""
import json
import logging
import math
import os
import queue
import threading
import time

log = logging.getLogger("swb_isaac_world")

SIZE = (960, 540)
# The wheel camera in base_link (x forward, y left, z up; the rear axle at x 0): behind and
# outside the rear-left wheel, looking forward and down at the wheel's back and the ground it
# has just left, so the rut comes out from under the tire toward the camera.
WHEEL_EYE = (-1.5, 1.85, 0.85)
WHEEL_AT = (-0.2, 0.83, 0.0)
WHEEL_FOV = 70.0
REAR_LEFT = 2  # aib's WHEELS order
# The chase camera in base_link: behind the car, up and to its left, looking ahead at the
# ground just past its nose, so the car and every ground it crosses stay in the picture.
CHASE_EYE = (-7.5, 3.0, 3.0)
CHASE_AT = (3.0, 0.0, 0.3)
CHASE_FOV = 60.0
# The trackside camera: past the zone's far end and to the right of the ego's line, m.
TRACK_BEYOND, TRACK_SIDE, TRACK_HEIGHT, TRACK_FOV = 3.0, 2.6, 1.3, 60.0


def _look_at(eye, at):
    """The quaternion (w, x, y, z) turning a USD camera (looking down -z, +y up) at `eye` to
    look at `at`, keeping the world's z up."""
    from pxr import Gf

    f = Gf.Vec3d(*(a - e for a, e in zip(at, eye))).GetNormalized()
    right = Gf.Cross(f, Gf.Vec3d(0, 0, 1)).GetNormalized()
    up = Gf.Cross(right, f)
    m = Gf.Matrix4d(right[0], right[1], right[2], 0, up[0], up[1], up[2], 0,
                    -f[0], -f[1], -f[2], 0, 0, 0, 0, 1)
    q = m.ExtractRotationQuat()
    return (q.GetReal(), *q.GetImaginary())


def _ahead(origin, heading, zones, reach=400.0):
    """The first of `zones` (soil zones, map frame) a ray from `origin` along `heading` (unit,
    map frame) enters within `reach`: (zone, distance in, distance out), or None."""
    best = None
    for z in zones:
        x0, x1, y0, y1 = z.bbox
        lo, hi = 0.0, reach
        for o, d, a, b in ((origin[0], heading[0], x0, x1), (origin[1], heading[1], y0, y1)):
            if abs(d) < 1e-9:
                if not a <= o <= b:
                    lo, hi = 1.0, 0.0
                continue
            t0, t1 = sorted(((a - o) / d, (b - o) / d))
            lo, hi = max(lo, t0), min(hi, t1)
        if lo <= hi and (best is None or lo < best[1]):
            best = (z, lo, hi)
    return best


class Recorder:
    def __init__(self, root, stage, ego, rig, name, surface=None):
        import omni.replicator.core as rep
        from aib.isaac.sensors import _mount
        from aib.sensing import APERTURE_MM, USD_CAMERA_IN_LINK
        from pxr import Gf, UsdGeom

        self.ego = ego
        self.stage = stage
        self.dir = os.path.join(root, time.strftime("%Y-%m-%d_%H-%M-%S") + f"-{name}")
        for d in ("dash", "wheel", "chase", "track"):
            os.makedirs(os.path.join(self.dir, d), exist_ok=True)
        self.state = open(os.path.join(self.dir, "state.jsonl"), "w")
        w, h = SIZE

        def camera(path, hfov):
            cam = UsdGeom.Camera.Define(stage, path)
            cam.CreateHorizontalApertureAttr(APERTURE_MM)
            cam.CreateVerticalApertureAttr(APERTURE_MM * h / w)
            cam.CreateFocalLengthAttr(APERTURE_MM / (2 * math.tan(math.radians(hfov) / 2)))
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
            return cam

        dash_rig = rig.sensors.get("camera6")
        _mount(stage, f"{ego.path}/rec_dash_mount", dash_rig.transform if dash_rig else
               ((0.9, 0.0, 2.0), (1.0, 0.0, 0.0, 0.0)))
        dash = camera(f"{ego.path}/rec_dash_mount/camera",
                      float(dash_rig.extra.get("horizontal_fov", 74.5)) if dash_rig else 74.5)
        dash.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(*USD_CAMERA_IN_LINK))
        wheel = camera(f"{ego.path}/rec_wheel", WHEEL_FOV)
        wheel.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*WHEEL_EYE))
        wheel.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(*_look_at(WHEEL_EYE, WHEEL_AT)))
        chase = camera(f"{ego.path}/rec_chase", CHASE_FOV)
        chase.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*CHASE_EYE))
        chase.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(*_look_at(CHASE_EYE, CHASE_AT)))
        cams = [("dash", dash), ("wheel", wheel), ("chase", chase)]
        track = self._trackside(stage, ego, surface, camera)
        if track is not None:
            cams.append(("track", track))
        self.views = []
        for kind, cam in cams:
            rp = rep.create.render_product(str(cam.GetPath()), resolution=SIZE)
            ann = rep.AnnotatorRegistry.get_annotator("rgb")
            ann.attach([rp.path])
            self.views.append((kind, rp, ann))
        self.frame = 0
        self.dropped = 0
        self.queue = queue.Queue(maxsize=120)
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.writer.start()
        log.info("recording ego %r to %s", name, self.dir)

    @staticmethod
    def _trackside(stage, ego, surface, camera):
        """A still camera at the first soil zone ahead of `ego`, or None."""
        from pxr import Gf, UsdGeom

        if surface is None:
            return None
        zones = [z for z in surface.map.zones if z.soil]
        (x, y, _), (w, qx, qy, qz) = ego.state.position, ego.state.quat
        yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        h = (math.cos(yaw), math.sin(yaw))
        ox, oy, oz = surface.origin
        ahead = _ahead((x + ox, y + oy), h, zones)
        if ahead is None:
            log.info("recorder: no soil zone ahead of the ego: no trackside camera")
            return None
        zone, t_in, t_out = ahead
        gz = (zone.ground_z or 0.0) - oz
        right = (h[1], -h[0])
        d = t_out + TRACK_BEYOND
        eye = (x + h[0] * d + right[0] * TRACK_SIDE, y + h[1] * d + right[1] * TRACK_SIDE,
               gz + TRACK_HEIGHT)
        m = t_in + 0.35 * (t_out - t_in)
        at = (x + h[0] * m, y + h[1] * m, gz)
        cam = camera("/World/rec_track", TRACK_FOV)
        cam.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*eye))
        cam.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(*_look_at(eye, at)))
        log.info("recorder: trackside camera at (%.1f, %.1f, %.1f) on zone %s", *eye, zone.id)
        return cam

    def capture(self, sim_time, surface, ego_ros):
        """After a rendered frame: both pictures and the state line."""
        import numpy as np

        images = []
        for kind, _, ann in self.views:
            d = ann.get_data()
            if d is None or not hasattr(d, "shape") or d.size == 0:
                return
            images.append((kind, np.ascontiguousarray(np.asarray(d)[..., :3], dtype=np.uint8)))
        n = self.frame
        self.frame += 1
        try:
            self.queue.put_nowait((n, images, self._state(n, sim_time, surface, ego_ros)))
        except queue.Full:
            self.dropped += 1

    def _state(self, n, sim_time, surface, ego_ros):
        s, v = self.ego.state, self.ego.vehicle
        w, qx, qy, qz = s.quat
        cmd = ego_ros.cmd if ego_ros is not None else None
        row = {"frame": n, "t": round(sim_time, 3),
               "x": round(s.position[0], 3), "y": round(s.position[1], 3),
               "z": round(s.position[2], 4),
               "yaw_deg": round(math.degrees(math.atan2(2 * (w * qz + qx * qy),
                                                        1 - 2 * (qy * qy + qz * qz))), 2),
               "pitch_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * qy - qz * qx))))), 2),
               "speed": round(s.velocity[0], 3), "accel": round(s.acceleration[0], 3),
               "accel_cmd": cmd and round(cmd.acceleration, 3),
               "tire_angle": round(s.tire_angle, 4), "wheels": []}
        # The onboard computers (sim_vehicle_ecu): what they estimated and what they did
        dbg = v.ecu.debug
        est = dbg.get("estimate") or {}
        if est:
            row["ecu"] = {"speed": round(est["speed"], 3), "mu": round(est["friction"], 3),
                          "sideslip_deg": round(math.degrees(est["sideslip"]), 2),
                          "abs": [round(a, 2) for a in dbg.get("abs", ())],
                          "tc": dbg.get("tc") and round(dbg["tc"], 2),
                          "esc": dbg.get("esc")}
            if v.torques is not None:
                row["ecu"]["drive"] = [round(t) for t in v.torques.drive]
                row["ecu"]["brake"] = [round(t) for t in v.torques.brake]
        ws = surface.wheels.get(v.path) if surface is not None else None
        for i in range(len(v.wheel_paths)):
            wheel = {}
            if ws is not None and ws.surfaces[i] is not None:
                g = ws.surfaces[i]
                wheel["class"] = g.cls
                if g.zone:
                    wheel["zone"] = g.zone
            if ws is not None and ws.mu[i] is not None:
                wheel["mu"] = round(ws.mu[i], 3)
                wheel["of_peak"] = round(ws.g[i], 3)  # past the tire's peak (054 T1)
            if v.wheel_states is not None:
                wheel["slip"] = round(v.wheel_states[i].slip, 3)
                wheel["slip_angle_deg"] = round(math.degrees(v.wheel_states[i].slip_angle), 2)
            if est.get("slip"):
                wheel["slip_est"] = round(est["slip"][i], 3)
            out = surface.soil_out.get((v.path, i)) if surface is not None else None
            if out is not None:
                o, share = out
                wheel.update(sinkage_cm=round(100 * o.sinkage, 2),
                             resistance_n=round(o.resistance * share),
                             traction_limit_n=round(o.traction_limit),
                             side_n=round(surface.side_force.get((v.path, i), 0.0)))
            row["wheels"].append(wheel)
        return row

    def _write(self):
        from PIL import Image

        while True:
            item = self.queue.get()
            if item is None:
                return
            n, images, row = item
            for kind, rgb in images:
                Image.fromarray(rgb).save(os.path.join(self.dir, kind, f"{n:06d}.jpg"),
                                          quality=90)
            self.state.write(json.dumps(row) + "\n")

    def close(self):
        for kind, rp, ann in self.views:
            try:
                ann.detach([rp.path])
                rp.destroy()
            except Exception:  # noqa: BLE001
                log.exception("recorder %s camera teardown", kind)
        if self.stage.GetPrimAtPath("/World/rec_track"):
            self.stage.RemovePrim("/World/rec_track")
        self.queue.put(None)
        self.writer.join(timeout=30)
        self.state.close()
        log.info("recorded %d frames to %s (%d dropped)", self.frame, self.dir, self.dropped)
