#!/usr/bin/env python3
"""The road-physics demo (roadmap 057) from a `spikes/materials_demo.py` run: per pad a title
card, then its start, brake and turn tests filmed by the course camera with the dash camera
inset and a HUD, and a closing chart of every material's numbers.

    scripts/materials_video.py <materials_demo out dir> [out.mp4] [--fast 2] [--size 1920x1080]

Each test is cut from its recording (`tests.jsonl`: `record`, `t_from`, `t_to`) with LEAD s
before it. Quiet stretches -- the car steady, nothing sliding -- play at `--fast` x (3). The HUD
shows speed, longitudinal acceleration, sideslip (from the car's track against its heading,
the ground truth), the tire panel (each tire coloured from grip to slide, the velocity against
the nose), the rear-left wheel's camera with its slip ratio (framed green to red as it spins
or locks) and, on soil pads, the deepest wheel's sinkage. Frames are the recorder's (20 a
second of simulated time), so the video runs at the physics' own speed. System Python with
Pillow and numpy; ffmpeg on PATH.
"""
import argparse
import json
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 20
LEAD = 0.6          # s of each test shown before its window
TAIL = 0.4          # s after it
TITLE_S = 1.2       # s of a pad's title card
CHART_S = 7.0       # s of the closing chart
# s of a test at real speed; after it the rest plays at --fast x (the start's speed has
# levelled off or the car is stuck, the turn has shown its slide or spin)
FULL_S = {"start": 3.0, "turn": 3.5}
DASH_CROP = 0.35
WHEEL_CROP = (0.22, 0.0, 0.72, 0.62)  # the wheel camera's frame around the tire (x0, y0, x1, y1)    # the dash camera's top fraction cut off (sky)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG, FG, DIM, ACCENT, BAD, SOIL = (16, 16, 18), (240, 240, 240), (150, 150, 155), \
    (255, 196, 64), (255, 80, 70), (205, 140, 75)
TESTS = {"start": "full throttle from standstill", "brake": "emergency brake from 50 km/h",
         "turn": "25 deg steering step at 50 km/h"}
# drift_mustang's wheels in base_link (the rear axle at x 0), aib's order: FL, FR, RL, RR
WHEEL_BASE, TRACK = 2.886, 1.586
WHEEL_XY = ((WHEEL_BASE, TRACK / 2), (WHEEL_BASE, -TRACK / 2), (0.0, TRACK / 2),
            (0.0, -TRACK / 2))
SLIP_FULL, ANGLE_FULL = 0.15, 8.0  # a tire slides past roughly its peak's slip / slip angle
_fonts = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(BOLD if bold else FONT, size)
    return _fonts[key]


def slide(w):
    """0 for a tire well inside its grip, 1 for one at or past its peak."""
    return min(1.0, math.hypot((w.get("slip") or 0.0) / SLIP_FULL,
                               (w.get("slip_angle_deg") or 0.0) / ANGLE_FULL))


def slide_color(k):
    """Green to yellow to red."""
    if k < 0.5:
        return (int(80 + 350 * k), 210, 80)
    return (255, int(210 - 340 * (k - 0.5)), 60)


def text_box(d, xy, s, size, fill=FG, bold=True, pad=8, bg=(0, 0, 0)):
    f = font(size, bold)
    x, y = xy
    l, t, r, b = d.textbbox((x + pad, y + pad), s, font=f)
    d.rectangle((x, y, r + pad, b + pad), fill=bg)
    d.text((x + pad, y + pad), s, fill=fill, font=f)
    return r + pad, b + pad


def tire_panel(d, x0, y0, r, beta_deg, s=1.0):
    """The car from above, nose up, its tires coloured by how much they slide, and its
    velocity's direction against its heading (after autoware-drift's lap_video.slide_hud)."""
    w, h = int(250 * s), int(270 * s)
    d.rectangle((x0, y0, x0 + w, y0 + h), fill=(0, 0, 0))
    d.text((x0 + 10, y0 + 8), "tires: grip → slide", fill=FG, font=font(int(17 * s)))
    cx, cy, px = x0 + w / 2, y0 + 150 * s, 28.0 * s
    mid = WHEEL_BASE / 2
    P = lambda lx, ly: (cx - ly * px, cy - (lx - mid) * px)  # noqa: E731
    d.polygon([P(3.72, 0.95), P(3.72, -0.95), P(-1.0, -0.95), P(-1.0, 0.95)], outline=FG,
              width=2)
    for (lx, ly), wh in zip(WHEEL_XY, r["wheels"]):
        steer = (r.get("tire_angle") or 0.0) if lx > 0 else 0.0
        a, b = math.cos(steer), math.sin(steer)
        pts = [P(lx + dx * a - dy * b, ly + dx * b + dy * a)
               for dx, dy in ((0.33, 0.12), (0.33, -0.12), (-0.33, -0.12), (-0.33, 0.12))]
        d.polygon(pts, fill=slide_color(slide(wh)))
    if abs(r["speed"]) > 0.5:
        beta = math.radians(beta_deg)
        tip = P(mid + 3.2 * math.cos(beta), 3.2 * math.sin(beta))
        d.line([P(mid, 0), P(mid + 3.2, 0)], fill=DIM, width=1)
        d.line([P(mid, 0), tip], fill=ACCENT, width=int(4 * s))
        d.ellipse((tip[0] - 5, tip[1] - 5, tip[0] + 5, tip[1] + 5), fill=ACCENT)
    d.text((x0 + 10, y0 + h - 34 * s), f"sideslip {beta_deg:+6.1f}°",
           fill=BAD if abs(beta_deg) > 4 else FG, font=font(int(21 * s), True))


def sideslip(rows):
    """Per row: the angle (deg) between the car's track (its position's change over +-1 frame)
    and its heading; 0 below 1 m/s."""
    out = []
    for i, r in enumerate(rows):
        a, b = rows[max(0, i - 1)], rows[min(len(rows) - 1, i + 1)]
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        if abs(r["speed"]) < 1.0 or math.hypot(dx, dy) < 1e-3:
            out.append(0.0)
            continue
        d = math.degrees(math.atan2(dy, dx)) - r["yaw_deg"]
        out.append((d + 180.0) % 360.0 - 180.0)
    return out


def quiet(r, beta):
    """Nothing to watch: steady, nothing sliding, not sinking."""
    return (abs(r["accel"]) < 1.0 and abs(beta) < 2.0 and max(slide(w) for w in r["wheels"]) < 0.4
            and abs(r.get("tire_angle") or 0.0) < 0.05) or r["speed"] < 0.05


class Writer:
    def __init__(self, out, W, H):
        self.W, self.H, self.n = W, H, 0
        self.ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
             "-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart", out],
            stdin=subprocess.PIPE)

    def put(self, img):
        self.ff.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
        self.n += 1

    def close(self):
        self.ff.stdin.close()
        self.ff.wait()


def title_card(wr, k, n, title, cls, soil):
    W, H = wr.W, wr.H
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    s = H / 1080
    d.text((W / 2, H * 0.42), title, fill=FG, font=font(int(96 * s), True), anchor="mm")
    sub = f"surface class {cls}" + ("  ·  deformable soil" if soil else "")
    d.text((W / 2, H * 0.53), sub, fill=SOIL if soil else DIM, font=font(int(38 * s)),
           anchor="mm")
    d.text((W / 2, H * 0.62), f"{k} / {n}   ·   start · brake · turn", fill=DIM,
           font=font(int(30 * s)), anchor="mm")
    for _ in range(int(TITLE_S * FPS)):
        wr.put(img)


def test_frames(wr, t, fast, k, n):
    rec = t["record"]
    rows = [json.loads(line) for line in open(os.path.join(rec, "state.jsonl"))]
    beta = sideslip(rows)
    W, H = wr.W, wr.H
    s = H / 1080
    t0, t1 = t["t_from"] - LEAD, t["t_to"] + TAIL
    skip = 0
    for r, b in zip(rows, beta):
        if not t0 <= r["t"] <= t1:
            continue
        course = os.path.join(rec, "course", f"{r['frame']:06d}.jpg")
        dash = os.path.join(rec, "dash", f"{r['frame']:06d}.jpg")
        if not (os.path.exists(course) and os.path.exists(dash)):
            continue
        sped = fast > 1 and r["t"] > t["t_from"] + 0.5 and (
            quiet(r, b) or r["t"] > t["t_from"] + FULL_S.get(t["test"], 1e9))
        if sped:
            skip += 1
            if skip % fast:
                continue
        img = Image.open(course).convert("RGB").resize((W, H), Image.BILINEAR)
        d = ImageDraw.Draw(img)
        iw = int(W * 0.25)
        ix, iy = W - iw - int(24 * s), int(24 * s)
        dimg = Image.open(dash).convert("RGB")
        dw, dh = dimg.size  # its upper part is sky: show the hood and the ground ahead
        dimg = dimg.crop((0, int(dh * DASH_CROP), dw, dh))
        ih = int(iw * dimg.size[1] / dimg.size[0])
        img.paste(dimg.resize((iw, ih), Image.BILINEAR), (ix, iy))
        d.rectangle((ix - 2, iy - 2, ix + iw + 1, iy + ih + 1), outline=FG, width=2)
        text_box(d, (ix, iy), "dash camera", int(18 * s), pad=5)
        # what and where
        x, y = int(24 * s), int(24 * s)
        _, y = text_box(d, (x, y), t["title"], int(44 * s), fill=ACCENT)
        text_box(d, (x, y + 6), f"{t['test']}: {TESTS.get(t['test'], '')}", int(26 * s))
        if sped:
            text_box(d, (W // 2 - int(40 * s), int(24 * s)), f"{fast}×", int(30 * s),
                     fill=DIM)
        # numbers, bottom left
        lines = [(f"{3.6 * r['speed']:5.1f} km/h", FG, 54),
                 (f"accel {r['accel']:+5.1f} m/s²", BAD if r["accel"] < -6 else FG, 30)]
        sink = max((w.get("sinkage_cm") or 0.0) for w in r["wheels"])
        if any(w.get("sinkage_cm") is not None for w in r["wheels"]):
            lines.append((f"sinkage {sink:4.1f} cm", SOIL, 30))
        mu = r["wheels"][0].get("mu")
        if mu is not None:
            lines.append((f"friction μ {mu:.2f}", DIM, 26))
        y = H - int(24 * s) - sum(int(sz * s * 1.35) for _, _, sz in lines) - int(16 * s)
        d.rectangle((int(24 * s), y, int(24 * s) + int(400 * s), H - int(24 * s)), fill=(0, 0, 0))
        y += int(10 * s)
        for txt, col, sz in lines:
            d.text((int(40 * s), y), txt, fill=col, font=font(int(sz * s), True))
            y += int(sz * s * 1.35)
        tire_panel(d, W - int(274 * s), H - int(294 * s), r, b, s)
        wheel = os.path.join(rec, "wheel", f"{r['frame']:06d}.jpg")
        if os.path.exists(wheel):  # the rear-left (driven) wheel: does it spin or lock?
            ww = int(W * 0.16)
            wimg = Image.open(wheel).convert("RGB")
            iw0, ih0 = wimg.size
            wimg = wimg.crop((int(iw0 * WHEEL_CROP[0]), int(ih0 * WHEEL_CROP[1]),
                              int(iw0 * WHEEL_CROP[2]), int(ih0 * WHEEL_CROP[3])))
            wh = int(ww * wimg.size[1] / wimg.size[0])
            wx, wy = W - int(274 * s) - ww - int(16 * s), H - int(58 * s) - wh
            img.paste(wimg.resize((ww, wh), Image.BILINEAR), (wx, wy))
            rl = r["wheels"][2]
            k = slide(rl)
            d.rectangle((wx - 2, wy - 2, wx + ww + 1, wy + wh + 1), outline=slide_color(k),
                        width=3)
            sl = rl.get("slip")
            text_box(d, (wx - 2, wy + wh + 3), "rear-left wheel" + ("" if sl is None else
                                                                 f"  slip {sl:+.2f}"),
                     int(18 * s), fill=slide_color(k) if k > 0.5 else FG, pad=5)
        wr.put(img)


def chart(wr, tests):
    """Per material: speed after the start, stopping distance from 50 km/h, peak sideslip in
    the turn."""
    W, H = wr.W, wr.H
    s = H / 1080
    pads, by = [], {}
    for t in tests:
        if t["pad"] not in by:
            pads.append((t["pad"], t["title"]))
            by[t["pad"]] = {}
        by[t["pad"]][t["test"]] = t
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W / 2, int(50 * s)), "every ground, the same car (drift_mustang, no ABS)", fill=FG,
           font=font(int(44 * s), True), anchor="mm")
    cols = [("start", "speed after 6 s, km/h", lambda t: t.get("speed_end_kmh"), "{:.0f}"),
            ("brake", "stopping distance from 50 km/h, m", lambda t: t.get("stopping_distance_m"),
             "{:.1f}"),
            ("turn", "peak sideslip in the turn, deg", lambda t: abs(t.get("peak_sideslip_deg")
                                                                      or 0.0), "{:.0f}")]
    label_w = int(260 * s)
    cw = (W - label_w - int(80 * s)) // 3
    top, row_h = int(150 * s), int((H - int(190 * s)) / max(1, len(pads)))
    for j, (test, head, val, fmt) in enumerate(cols):
        x0 = label_w + int(40 * s) + j * cw
        d.text((x0, top - int(50 * s)), head, fill=ACCENT, font=font(int(24 * s), True))
        vals = [val(by[p][test]) for p, _ in pads if test in by[p] and val(by[p][test]) is not None]
        hi = max(vals) if vals else 1.0
        for i, (p, _) in enumerate(pads):
            y = top + i * row_h
            t = by[p].get(test)
            if t is None or val(t) is None:
                d.text((x0, y), "-", fill=DIM, font=font(int(22 * s)))
                continue
            v = val(t)
            bw = int((cw - int(130 * s)) * v / hi) if hi > 0 else 0
            col = (90, 170, 230)
            note = ""
            if test == "brake" and not t.get("stopped", True):
                col, note = BAD, "+"
            d.rectangle((x0, y + int(4 * s), x0 + max(2, bw), y + row_h - int(10 * s)), fill=col)
            d.text((x0 + bw + int(8 * s), y + int(2 * s)), fmt.format(v) + note, fill=FG,
                   font=font(int(22 * s)))
    d.text((W / 2, H - int(30 * s)), "red: still moving when the test ended (7 s)", fill=BAD,
           font=font(int(22 * s)), anchor="mm")
    for i, (_, title) in enumerate(pads):
        d.text((int(40 * s), top + i * row_h + int(2 * s)), title, fill=FG,
               font=font(int(24 * s), True))
    for _ in range(int(CHART_S * FPS)):
        wr.put(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="a materials_demo.py out dir (tests.jsonl, record/)")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--fast", type=int, default=3, help="speed-up of quiet stretches")
    ap.add_argument("--size", default="1920x1080")
    a = ap.parse_args()
    W, H = (int(v) for v in a.size.split("x"))
    tests = [json.loads(line) for line in open(os.path.join(a.run, "tests.jsonl"))]
    soil = {}
    for t in tests:  # a pad is soil when its recording has a sinkage under any wheel
        if not soil.get(t["pad"]):
            with open(os.path.join(t["record"], "state.jsonl")) as f:
                soil[t["pad"]] = any(w.get("sinkage_cm") is not None
                                     for line in f for w in json.loads(line)["wheels"])
    out = a.out or os.path.join(a.run, "materials_demo.mp4")
    wr = Writer(out, W, H)
    pads = []
    for t in tests:
        if t["pad"] not in pads:
            pads.append(t["pad"])
    for k, pad in enumerate(pads, 1):
        mine = [t for t in tests if t["pad"] == pad]
        title_card(wr, k, len(pads), mine[0]["title"], pad.replace("_wet", " (wet)"), soil[pad])
        for t in mine:
            test_frames(wr, t, a.fast, k, len(pads))
        print(f"{pad}: {wr.n / FPS:.1f} s so far", flush=True)
    chart(wr, tests)
    wr.close()
    print(f"{out}: {wr.n / FPS:.1f} s, {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
