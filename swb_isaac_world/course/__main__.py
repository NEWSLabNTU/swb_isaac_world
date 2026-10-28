"""`python3 -m swb_isaac_world.course <course.yaml> <map dir> [steps...]` (steps: fetch,
centreline; all by default). See the package's docstring and roadmap 058."""
import argparse
import json
import os
import sys

import numpy as np
import yaml

from . import centreline as cl
from . import data

STEPS = ("fetch", "centreline")


def section_box(course, frame, pts=None):
    """(south, west, north, east) of the section plus the course's margin."""
    road = course["road"]
    m = course["margin"]
    if pts is None:
        pts = [frame.xy(*road["from"]), frame.xy(*road["to"])]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    s, w = frame.latlon(min(xs) - m, min(ys) - m)
    n, e = frame.latlon(max(xs) + m, max(ys) + m)
    return s, w, n, e


def plot(c, photo, block, z, frame, out):
    """The centreline over the photo, the hairpins marked, and the height profile below."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(photo).convert("RGB")
    x0, y0 = block[0], block[1]

    def px(x, y):
        tx, ty = data.tile_xy(*frame.latlon(x, y), z)
        return ((tx - x0) * 256, (ty - y0) * 256)

    d = ImageDraw.Draw(img)
    d.line([px(x, y) for x, y in zip(c["x"], c["y"])], fill=(255, 40, 40), width=3)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    for i, h in enumerate(c["hairpins"]):
        u, v = px(h["x"], h["y"])
        d.ellipse((u - 14, v - 14, u + 14, v + 14), outline=(255, 230, 0), width=4)
        d.text((u + 18, v - 14), f"H{i + 1} R{h['radius']:.0f}", fill=(255, 230, 0), font=font)
    sx, sy = px(c["x"][0], c["y"][0])
    d.text((sx + 10, sy + 10), "start", fill=(255, 255, 255), font=font)
    # the profile
    W, H = img.width, 360
    prof = Image.new("RGB", (W, H), (20, 20, 24))
    dp = ImageDraw.Draw(prof)
    s, zz, zd = c["s"], c["z"], c["z_dem"]
    lo, hi = float(min(zd.min(), zz.min())), float(max(zd.max(), zz.max()))
    P = lambda si, zi: (20 + si / s[-1] * (W - 40), H - 20 - (zi - lo) / (hi - lo + 1e-6) * (H - 60))
    dp.line([P(a, b) for a, b in zip(s, zd)], fill=(120, 120, 120), width=1)
    dp.line([P(a, b) for a, b in zip(s, zz)], fill=(255, 40, 40), width=3)
    for i, h in enumerate(c["hairpins"]):
        u, v = P(h["s"], h["z"])
        dp.text((u - 10, v - 34), f"H{i + 1}", fill=(255, 230, 0), font=font)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    dp.text((20, 8), f"height: road (red), DEM under it (grey); {lo + frame.alt0:.0f}-{hi + frame.alt0:.0f} m, "
                     f"{s[-1]:.0f} m of road", fill=(230, 230, 230), font=small)
    out_img = Image.new("RGB", (W, img.height + H))
    out_img.paste(img, (0, 0))
    out_img.paste(prof, (0, img.height))
    out_img.save(out, quality=88)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("course")
    ap.add_argument("map_dir")
    ap.add_argument("steps", nargs="*", default=list(STEPS))
    a = ap.parse_args(argv)
    course = yaml.safe_load(open(a.course))
    raw = os.path.join(a.map_dir, "raw")
    work = os.path.join(a.map_dir, "course")
    os.makedirs(work, exist_ok=True)
    frame = cl.Frame(course["origin"])
    road = course["road"]

    ways = data.osm_ways(raw, road["overpass"], road["box"])
    box = section_box(course, frame)
    dz, pz = course["dem"]["zoom"], course["photo"]["zoom"]
    dem_block = data.tiles(raw, course["dem"]["layer"], dz, "txt", *box)
    photo_block = data.tiles(raw, course["photo"]["layer"], pz, "jpg", *box)
    if "fetch" in a.steps:
        print(f"fetched: {len(ways)} ways, DEM {dem_block[2]}x{dem_block[3]} tiles, "
              f"photo {photo_block[2]}x{photo_block[3]} tiles")
    if "centreline" not in a.steps:
        return 0
    dem = data.Dem(raw, course["dem"]["layer"], dz, dem_block)
    c = cl.build(course, ways, dem, frame)
    # the box again, from the centreline itself (the section may bulge past its ends)
    box = section_box(course, frame, list(zip(c["x"], c["y"])))
    photo_block = data.tiles(raw, course["photo"]["layer"], pz, "jpg", *box)
    dem_block = data.tiles(raw, course["dem"]["layer"], dz, "txt", *box)
    np.savez(os.path.join(work, "centreline.npz"), **{k: v for k, v in c.items() if k != "hairpins"})
    summary = {
        "length_m": round(float(c["s"][-1]), 1),
        "z_from_m": round(float(c["z"][0] + frame.alt0), 1), "z_to_m": round(float(c["z"][-1] + frame.alt0), 1),
        "max_grade_pct": round(float(np.abs(c["grade"]).max() * 100), 1),
        "min_radius_m": round(float(1.0 / np.abs(c["curvature"]).max()), 1),
        "hairpins": c["hairpins"], "box": [round(v, 6) for v in box],
        "dem_block": dem_block, "photo_block": photo_block}
    json.dump(summary, open(os.path.join(work, "centreline.json"), "w"), indent=1)
    photo = data.mosaic(raw, course["photo"]["layer"], pz, photo_block, os.path.join(work, "photo.jpg"))
    plot(c, photo, photo_block, pz, frame, os.path.join(work, "centreline.jpg"))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("hairpins",)}))
    for i, h in enumerate(c["hairpins"]):
        print(f"H{i + 1}: s {h['s']:7.1f} m  ({h['x']:7.1f}, {h['y']:7.1f})  R {h['radius']:5.1f} m  "
              f"turn {h['turn_deg']:+6.1f} deg  {h['direction']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
