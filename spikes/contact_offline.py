"""Roadmap 066 R5 offline: road_contact's corner under a PhysX-like body at 60 Hz, against the
4 kHz reference, on the fixture lanes. No Isaac.

The body stands in for PhysX's: a sprung mass on PhysX's spring and damper over the base road
(flat here), integrated in 8 substeps per 60 Hz step, plus road_contact's force held over the
step, as the Isaac host applies it.

    python3 spikes/contact_offline.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import fixture_course as fc  # noqa: E402
from swb_isaac_world import road_contact as rc  # noqa: E402

# aib's Model 3 front corner (066 R0)
SPRUNG, K, C = 413.6, 41.8e3, 3.74e3
TRAVEL, SAG, R, WIDTH = 0.20, 0.097, 0.37, 0.235
PX_TRAVEL = 0.30  # R2: PhysX's travel beyond the real one, its frame raised to match


def run(name, speed, rate=60.0, vsub=8, inner=8):
    _, f, _, _ = fc.LANES[name]
    y = fc.TREAD_HALF
    height = lambda x, yy: f(x, yy)
    base = lambda x, yy: np.zeros(np.broadcast(x, yy).shape)
    tc = rc.TireContact([rc.Corner(SPRUNG, K, C, TRAVEL, R, WIDTH, px_travel=PX_TRAVEL)],
                        inner=inner)
    dt = 1.0 / rate
    x = fc.X0 - 5.0
    z_att = R + PX_TRAVEL - SAG + 0.0  # attachment at rest, base at 0
    j_rest = SAG
    z_rest = z_att
    vz = 0.0
    rows, inner_rows, cost = [], [], []
    while x < fc.X0 + fc.EXTENT[name] + 15.0:
        t0 = time.perf_counter()
        o = tc.step(dt, height, base, np.array([[x, y]]), np.array([[1.0, 0.0]]),
                    np.array([speed]), np.array([z_att]), np.array([vz]))
        cost.append(time.perf_counter() - t0)
        h = dt / vsub
        for _ in range(vsub):  # PhysX's corner on the base road
            j = j_rest + (z_rest - z_att)
            fp = K * min(max(j, 0), PX_TRAVEL) - C * vz if j > 0 else 0.0
            fp = max(fp, 0.0)
            a = (fp + o["dfz"][0]) / SPRUNG - rc.G
            vz += a * h
            z_att += vz * h
        for i, ft in enumerate(o["load_inner"][0]):  # inner steps: x at each
            inner_rows.append((x + speed * dt * (i + 1) / inner, float(ft)))
        x += speed * dt
        rows.append((x, float(o["load"][0]), z_att - z_rest, float(o["airborne"][0]),
                     float(o["dfx"][0])))
    xs = np.array([r[0] for r in inner_rows]); load = np.array([r[1] for r in inner_rows])
    static = load[xs < fc.X0 - 2.0].mean()
    rise = np.nonzero(load > static * 1.05)[0]
    return {"static_N": round(static, 1), "peak_N": round(float(load.max()), 1),
            "min_N": round(float(load.min()), 1),
            "first_rise_x": None if not len(rise) else round(float(xs[rise[0]] - fc.X0), 3),
            "airborne_s": round(float(sum(r[3] for r in rows)) / rate, 3),
            "body_rise_m": round(float(max(r[2] for r in rows)), 3),
            "dfx_min_N": round(float(min(r[4] for r in rows)), 1),
            "cost_ms_p50": round(sorted(cost)[len(cost) // 2] * 1e3, 3)}


def cam_road(name):
    """road_contact's effective height along the left track, for the 4 kHz reference."""
    _, f, _, _ = fc.LANES[name]
    tc = rc.TireContact([rc.Corner(SPRUNG, K, C, TRAVEL, R, WIDTH)])
    x0 = fc.X0 - 5.0
    s, w, _ = tc.envelope(f, np.array([[x0, fc.TREAD_HALF]]), np.array([[1.0, 0.0]]),
                          fc.EXTENT[name] + 20.0)
    return x0 + s, w[0]


if __name__ == "__main__":
    out = {}
    for name, v in (("cleat_05", 10), ("cleat_10", 10), ("cleat_10", 20), ("cleat_15", 10),
                    ("pothole", 10), ("washboard", 10), ("washboard", 20), ("boulder", 10)):
        ours = run(name, v)
        ref = fc.reference(name, v, SPRUNG, K, C, travel=TRAVEL, sag=SAG, radius=R)[1]
        refc = fc.reference(name, v, SPRUNG, K, C, travel=TRAVEL, sag=SAG, radius=R,
                            road=cam_road(name), rim=rc.Corner.rim_clearance,
                            damper=rc.Corner(SPRUNG, K, C, TRAVEL, R, WIDTH).damper)[1]
        out[f"{name}@{v}"] = {"ours60": ours, "ref_cam": refc, "ref_disc": ref}
        print(name, v, json.dumps(ours), "\n   ref_cam ", json.dumps(refc),
              "\n   ref_disc", json.dumps(ref), flush=True)
    print("CONTACT-OFFLINE " + json.dumps(out))
