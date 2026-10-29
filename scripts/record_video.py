#!/usr/bin/env python3
"""A recorded run (`--record`, `recorder.py`) as one video: the dash camera and, beside it,
the trackside camera at the soil zone or the chase camera (`--right`), the wheel camera inset
in it (or beside the dash camera when there is neither), the ego's state, what its onboard computers
estimated (speed, friction, each wheel's slip) and did (ABS, TC), and each wheel's ground below.

    scripts/record_video.py <record dir>/<time>-<ego> [out.mp4] [--from T] [--to T] [--title S]
        [--right track|chase|wheel]

Frames are the world server's, one per tick of simulated time. The frame rate is read from
state.jsonl (the median step of "t", rounded to a whole number: 20 for a 50 ms tick, 50 for
a 20 ms one), so the video runs at the scenario's own speed whatever the tick or the run's
real-time factor was. --from and --to are times, not frame numbers. System Python with Pillow and
numpy; ffmpeg on PATH.
"""
import argparse
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 960, 540
STRIP = 260
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
WHEELS = ("front left", "front right", "rear left", "rear right")
BG, FG, DIM, ACCENT, SOIL = (18, 18, 20), (235, 235, 235), (140, 140, 145), (255, 196, 64), \
    (190, 120, 60)


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def plot(draw, box, ts, ys, now, lo, hi, color, label):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(60, 60, 64))
    draw.text((x0 + 6, y0 + 4), label, fill=DIM, font=font(15))
    if len(ts) < 2 or hi <= lo:
        return
    t0, t1 = ts[0], max(ts[-1], ts[0] + 1e-6)
    pts = [(x0 + (t - t0) / (t1 - t0) * (x1 - x0),
            y1 - (min(max(y, lo), hi) - lo) / (hi - lo) * (y1 - y0)) for t, y in zip(ts, ys)
           if t <= now]
    if len(pts) > 1:
        draw.line(pts, fill=color, width=2)
    xn = x0 + (now - t0) / (t1 - t0) * (x1 - x0)
    draw.line([(xn, y0), (xn, y1)], fill=(90, 90, 96))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--from", dest="t_from", type=float, default=None)
    ap.add_argument("--to", dest="t_to", type=float, default=None)
    ap.add_argument("--title", default="")
    ap.add_argument("--right", choices=("track", "chase", "wheel"), default=None,
                    help="the right-hand picture: the trackside camera (the default when the "
                         "run has one), the chase camera (a run over several zones) or the "
                         "wheel camera")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(os.path.join(a.run, "state.jsonl"))]
    # The frame rate is the run's own: the median step of t over the whole run (before the
    # window cuts it), rounded to a whole number of frames a second.
    steps = np.diff([r["t"] for r in rows])
    fps = int(round(1.0 / float(np.median(steps)))) if len(steps) and np.median(steps) > 0 else 20
    rows = [r for r in rows if (a.t_from is None or r["t"] >= a.t_from)
            and (a.t_to is None or r["t"] <= a.t_to)
            and os.path.exists(os.path.join(a.run, "dash", f"{r['frame']:06d}.jpg"))]
    if not rows:
        raise SystemExit("no frames in that window")
    out = a.out or os.path.join(a.run, "run.mp4")
    def has(kind):
        return os.path.isdir(os.path.join(a.run, kind)) and bool(os.listdir(os.path.join(a.run, kind)))
    right = a.right or ("track" if has("track") else "wheel")
    if not has(right):
        raise SystemExit(f"the run has no {right} pictures")
    ts = [r["t"] for r in rows]
    speed = [3.6 * r["speed"] for r in rows]
    sink = [max((w.get("sinkage_cm") or 0.0) for w in r["wheels"]) for r in rows]
    resist = [sum((w.get("resistance_n") or 0) for w in r["wheels"]) for r in rows]
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                           "-pix_fmt", "rgb24", "-s", f"{2 * W}x{H + STRIP}", "-r", str(fps),
                           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                           "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    for k, r in enumerate(rows):
        frame = Image.new("RGB", (2 * W, H + STRIP), BG)
        pic = lambda kind: Image.open(os.path.join(a.run, kind, f"{r['frame']:06d}.jpg"))  # noqa: E731
        frame.paste(pic("dash"), (0, 0))
        d = ImageDraw.Draw(frame)
        labels = [(10, 10, "dash camera")]
        if right != "wheel":
            frame.paste(pic(right), (W, 0))
            iw, ih = W * 3 // 8, H * 3 // 8
            ix, iy = W + 12, H - ih - 12  # over the sidewalk, clear of the car
            frame.paste(pic("wheel").resize((iw, ih)), (ix, iy))
            d.rectangle((ix - 2, iy - 2, ix + iw + 1, iy + ih + 1), outline=FG, width=2)
            labels += [(W + 10, 10, "trackside, soil zone" if right == "track" else "chase camera"),
                       (ix + 6, iy + 6, "rear-left wheel")]
        else:
            frame.paste(pic("wheel"), (W, 0))
            labels.append((W + 10, 10, "rear-left wheel"))
        for x, y0, label in labels:
            d.rectangle((x, y0, x + 12 * len(label) + 16, y0 + 30), fill=(0, 0, 0))
            d.text((x + 8, y0 + 3), label, fill=FG, font=font(20, True))
        if a.title:
            d.text((2 * W - 12 * len(a.title) - 20, 13), a.title, fill=ACCENT, font=font(20, True))
        y = H + 12
        ground = next((w for w in r["wheels"] if w.get("zone")), None)
        where = f"{ground['class']} (zone {ground['zone']})" if ground else \
            (r["wheels"][0].get("class") or "-")
        cmd = "-" if r["accel_cmd"] is None else f"{r['accel_cmd']:+.2f}"
        d.text((16, y), f"t {r['t']:7.2f} s  speed {3.6 * r['speed']:5.1f} km/h  "
               f"accel {r['accel']:+.2f} m/s2 (Autoware asks {cmd})  "
               f"pitch {r['pitch_deg']:+.1f} deg", fill=FG, font=font(18, True))
        d.text((16, y + 30), f"ground: {where}", fill=SOIL if ground else DIM, font=font(19, True))
        ecu = r.get("ecu")
        if ecu:  # the onboard computers (sim_vehicle_ecu): what they estimated, ABS and TC
            abs_min = min(ecu.get("abs") or [1.0])
            tc = 1.0 if ecu.get("tc") is None else ecu["tc"]
            d.text((392, y + 33), f"ECU estimates {3.6 * ecu['speed']:5.1f} km/h, friction "
                   f"{ecu['mu']:.2f} | ABS {abs_min:.2f}  TC {tc:.2f}",
                   fill=ACCENT if abs_min < 1.0 or tc < 1.0 else FG, font=font(16, True))
        y += 66
        d.text((16, y), "wheel          ground      sinkage   resistance   friction   slip  est.",
               fill=DIM, font=font(16))
        for i, w in enumerate(r["wheels"]):
            s = w.get("sinkage_cm")
            res = w.get("resistance_n")
            line = (f"{WHEELS[i]:<14} {(w.get('class') or '-'):<11} "
                    f"{'' if s is None else f'{s:5.1f} cm':>8}   "
                    f"{'' if res is None else f'{res:6d} N':>10}   "
                    f"{'' if w.get('mu') is None else format(w['mu'], '.2f'):>8}   "
                    f"{'' if w.get('slip') is None else format(w['slip'], '+.2f'):>5} "
                    f"{'' if w.get('slip_est') is None else format(w['slip_est'], '+.2f'):>5}")
            d.text((16, y + 22 * (i + 1)), line, fill=SOIL if s is not None else FG, font=font(16))
        x0 = 1000
        plot(d, (x0, H + 12, x0 + 430, H + STRIP - 12), ts, speed, r["t"], 0,
             max(speed) * 1.1 + 1, (120, 200, 255), "speed, km/h")
        plot(d, (x0 + 450, H + 12, x0 + 905, H + 125), ts, sink, r["t"], 0,
             max(sink) * 1.15 + 0.5, SOIL, f"deepest wheel sinkage, cm (max {max(sink):.1f})")
        plot(d, (x0 + 450, H + 137, x0 + 905, H + STRIP - 12), ts, resist, r["t"], 0,
             max(resist) * 1.1 + 10, (230, 110, 90), "soil resistance, N (4 wheels)")
        ff.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        if k % 200 == 0:
            print(f"{k}/{len(rows)}", flush=True)
    ff.stdin.close()
    ff.wait()
    print(out)


if __name__ == "__main__":
    main()
