"""Roadmap 066 R3 offline: the generated road against its targets. No Isaac.

    python3 spikes/road_detail_offline.py

- the line PSD of a wheel track against ISO 8608's class with gravel's break (ratios per band);
- the golden-car IRI per class, and of rough_road.py's corridor's left track (its ruts, crown,
  potholes, washboard and water bars on class C);
- a tile's generation time and memory (MB per km of a 20 m corridor at 5 cm);
- the bilinear height the tire reads against the render mesh's triangles (question 3, 5 mm).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from swb_isaac_world import road_detail as rd  # noqa: E402

out = {}
plain = rd.Section(crown=0.0, ditch_depth=0.0, shoulder=3.0)
x = np.arange(0.0, 1000.0, rd.SPACING)
for cls in "BCDE":
    d = rd.Detail([rd.Corridor(np.array([[0.0, 0.0], [1100.0, 0.0]]), seed=5,
                               roughness=rd.Roughness(cls=cls), section=plain)])
    h = d.height(x, np.full_like(x, 0.83))
    n, g = rd.line_psd(h, rd.SPACING)
    target = d.corridors[0].roughness.psd(n)
    bands = {}
    for lo, hi in ((0.05, 0.1), (0.1, 0.3), (0.3, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 7.0)):
        s = (n >= lo) & (n < hi)
        bands[f"{lo}-{hi}"] = round(float(np.mean(g[s]) / np.mean(target[s])), 3)
    out[cls] = {"iri_m_per_km": round(rd.iri(h, rd.SPACING), 2),
                "rms_mm": round(float(h.std() * 1e3), 1), "psd_ratio": bands}
    print(cls, json.dumps(out[cls]), flush=True)

# rough_road.py's corridor (the same arguments)
X_START = 20.0
corridor = rd.Corridor(
    np.array([[0.0, 0.0], [X_START + 2200.0, 0.0]]), seed=66, roughness=rd.Roughness(cls="C"),
    ruts=rd.Ruts(offset=0.83), potholes=rd.Potholes(),
    washboard=(rd.Washboard(X_START + 600.0, X_START + 700.0, pitch=0.5),),
    water_bars=rd.WaterBars(at=(X_START + 1500.0, X_START + 1550.0, X_START + 1600.0)))
d = rd.Detail([corridor], max_tiles=512)
xs = np.arange(X_START, X_START + 2000.0, rd.SPACING)
t0 = time.perf_counter()
left = d.height(xs, np.full_like(xs, 0.8335))
gen_s = time.perf_counter() - t0
tm = sorted(d.made_ms)
out["corridor"] = {
    "iri_left_m_per_km": round(rd.iri(left, rd.SPACING), 2),
    "iri_left_excluding_features": None,
    "potholes": len(corridor._holes),
    "tile_ms": {"p50": round(tm[len(tm) // 2], 1), "max": round(tm[-1], 1), "n": len(tm)},
    "mb_per_km_20m_float64": round(1000 * 20 / rd.SPACING ** 2 * 8 / 2 ** 20, 1),
    "mb_per_km_20m_float32": round(1000 * 20 / rd.SPACING ** 2 * 4 / 2 ** 20, 1)}
smooth = rd.Detail([rd.Corridor(corridor.centreline, seed=66, roughness=rd.Roughness(cls="C"))])
out["corridor"]["iri_left_excluding_features"] = round(rd.iri(
    smooth.height(xs, np.full_like(xs, 0.8335)), rd.SPACING), 2)
print("corridor", json.dumps(out["corridor"]), flush=True)

# question 3: bilinear against the two triangles a renderer makes of each quad
t = d.tile_of(int(X_START // rd.TILE) + 40, 0)
rng = np.random.default_rng(0)
k = 200000
i, j = rng.integers(0, d.n, k), rng.integers(0, d.n, k)
a, b = rng.random(k), rng.random(k)
h00, h10, h01, h11 = t[j, i], t[j, i + 1], t[j + 1, i], t[j + 1, i + 1]
bil = (h00 * (1 - a) + h10 * a) * (1 - b) + (h01 * (1 - a) + h11 * a) * b
tri = np.where(a >= b, h00 + a * (h10 - h00) + b * (h11 - h10),
               h00 + b * (h01 - h00) + a * (h11 - h01))
diff = np.abs(bil - tri)
out["mesh_vs_height_mm"] = {"p99": round(float(np.percentile(diff, 99)) * 1e3, 2),
                            "max": round(float(diff.max()) * 1e3, 2)}
print("mesh", json.dumps(out["mesh_vs_height_mm"]))
print("ROAD-DETAIL-OFFLINE " + json.dumps(out))
