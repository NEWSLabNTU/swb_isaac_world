"""The open data a course is built from, fetched once and cached under `<map dir>/raw/`.

OpenStreetMap through the Overpass API (ODbL: OpenStreetMap contributors); GSI's tiles
(出典：国土地理院): the 5 m DEM as 256 x 256 text grids, metres, `e` where there is no
data, and the seamless aerial photo as 256 x 256 JPEGs. Both on the Web Mercator tile grid.
"""
import json
import math
import os
import subprocess
import time

import numpy as np

USER_AGENT = "swb-course/0.1 (scenario-world-bridge roadmap 058)"
OVERPASS = "https://overpass-api.de/api/interpreter"
GSI = "https://cyberjapandata.gsi.go.jp/xyz/{layer}/{z}/{x}/{y}.{ext}"


def _get(url, out, data=None, tries=5):
    """Download `url` to `out` (POSTing `data`), retrying with backoff; True when it exists after."""
    if os.path.exists(out):
        return True
    os.makedirs(os.path.dirname(out), exist_ok=True)
    wait = 5.0
    for _ in range(tries):
        cmd = ["curl", "-s", "-f", "-m", "120", "-A", USER_AGENT, "-o", out + ".part", url]
        if data is not None:
            cmd[1:1] = ["--data-urlencode", data]
        if subprocess.run(cmd).returncode == 0 and os.path.getsize(out + ".part") > 0:
            os.replace(out + ".part", out)
            return True
        if data is None:  # a missing tile is not retried
            break
        time.sleep(wait)
        wait *= 2
    if os.path.exists(out + ".part"):
        os.remove(out + ".part")
    return False


def osm_ways(raw, filt, box):
    """The ways matching an Overpass filter in a box (south, west, north, east), with their
    node ids, geometry and tags."""
    out = os.path.join(raw, "road.json")
    s, w, n, e = box
    q = f"[out:json][timeout:100];{filt}({s},{w},{n},{e});out body geom;"
    if not _get(OVERPASS, out, data="data=" + q):
        raise RuntimeError(f"Overpass did not answer for {filt}")
    return [el for el in json.load(open(out))["elements"] if el["type"] == "way"]


def tile_xy(lat, lon, z):
    """Fractional Web Mercator tile coordinates of (lat, lon) at zoom z."""
    n = 2 ** z
    return ((lon + 180.0) / 360.0 * n,
            (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)


def tile_latlon(x, y, z):
    """The (lat, lon) of fractional tile coordinates."""
    n = 2 ** z
    return (math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))), x / n * 360.0 - 180.0)


def tiles(raw, layer, z, ext, south, west, north, east):
    """Fetch the tiles covering the box; returns (x0, y0, nx, ny) of the tile block."""
    xa, ya = (int(v) for v in tile_xy(north, west, z))
    xb, yb = (int(v) for v in tile_xy(south, east, z))
    for x in range(xa, xb + 1):
        for y in range(ya, yb + 1):
            _get(GSI.format(layer=layer, z=z, x=x, y=y, ext=ext),
                 os.path.join(raw, layer, f"{z}_{x}_{y}.{ext}"))
    return xa, ya, xb - xa + 1, yb - ya + 1


class Dem:
    """The DEM tiles of a block as one grid, sampled bilinearly at (lat, lon)."""

    def __init__(self, raw, layer, z, block):
        x0, y0, nx, ny = block
        self.z, self.x0, self.y0 = z, x0, y0
        g = np.full((ny * 256, nx * 256), np.nan)
        for i in range(nx):
            for j in range(ny):
                path = os.path.join(raw, layer, f"{z}_{x0 + i}_{y0 + j}.txt")
                if not os.path.exists(path):
                    continue
                rows = [r.split(",") for r in open(path).read().split()]
                for r, row in enumerate(rows):
                    g[j * 256 + r, i * 256:i * 256 + len(row)] = [np.nan if v == "e" else float(v) for v in row]
        self.grid = g

    def at(self, lat, lon):
        x, y = tile_xy(lat, lon, self.z)
        # pixel centres: a tile's pixel k covers [k, k+1) / 256 of it
        px, py = (x - self.x0) * 256 - 0.5, (y - self.y0) * 256 - 0.5
        i, j = int(math.floor(px)), int(math.floor(py))
        fx, fy = px - i, py - j
        g = self.grid
        if not (0 <= i < g.shape[1] - 1 and 0 <= j < g.shape[0] - 1):
            return float("nan")
        return float((1 - fy) * ((1 - fx) * g[j, i] + fx * g[j, i + 1])
                     + fy * ((1 - fx) * g[j + 1, i] + fx * g[j + 1, i + 1]))


def mosaic(raw, layer, z, block, out):
    """The photo tiles of a block as one JPEG; returns its path and (x0, y0) in tiles."""
    from PIL import Image

    x0, y0, nx, ny = block
    m = Image.new("RGB", (nx * 256, ny * 256), (60, 80, 50))
    for i in range(nx):
        for j in range(ny):
            path = os.path.join(raw, layer, f"{z}_{x0 + i}_{y0 + j}.jpg")
            if os.path.exists(path):
                m.paste(Image.open(path).convert("RGB"), (i * 256, j * 256))
    m.save(out, quality=92)
    return out
