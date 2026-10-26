"""Calibration (roadmap 053, ice): the tire's longitudinal friction curve on ice from Virginia
Tech's tire tests on ice (Zenodo 1009017, 1009023, 1009025; CC BY 4.0), the zips unpacked in
<data dir> (`Tyre_ice_tests_Part{1,2,3}/*.mat`).

    python3 src/swb_isaac_world/spikes/ice_curve.py <data dir>

Each test runs the wheel through four programmed slips in turn (its name: P_<slips>_<bar>p
<tenths>_<load N>_EVE2[_WF_t<cond>][_brk]_<repeat>), braking (`brk`) or driving. The files
hold no slip channel: the wheel's speed is the one controlled, the carriage's constant, so a
run is cut into four equal segments after the wheel is up to speed and each segment's
nominal slip is its own (the wheel's speed checks it: it falls by the slip braking and rises
by 1 / (1 - slip) driving). Per segment, the mean of Fx / Fz over its last 60 %. Prints the
curve of bare ice (2.5 and 3.5 bar, 5 kN) and of ice under a water film (`WF`, all five
conditions, 2.5 bar, 5 kN), braking, driving and their mean.
"""
import collections
import glob
import os
import re
import sys

import numpy as np
import scipy.io as sio

NAME = re.compile(r"P_([\d_]+?)_(\dp\d)_(\d+)_EVE2_(?:(WF_t\w+?)_)?(brk_)?(\d)$")


def segments(root):
    for f in sorted(glob.glob(os.path.join(root, "Tyre_ice_tests_Part*", "*.mat"))):
        r = sio.loadmat(f, squeeze_me=True, struct_as_record=False)["res"]
        m = NAME.match(r.Name)
        if not m:
            continue
        slips = [int(x) for x in m.group(1).split("_")]
        t = r.Time
        dt = t[1] - t[0]
        w = np.gradient(np.unwrap(np.radians(r.WhlPos)), t)
        up = np.nonzero(w > np.radians(5))[0]
        edges = np.linspace(up[0] + int(3 / dt), up[-1] - int(1 / dt), 5).astype(int)
        for k in range(4):
            a, b = edges[k], edges[k + 1]
            a += int(0.4 * (b - a))
            yield {"film": m.group(4), "brake": bool(m.group(5)),
                   "bar": float(m.group(2).replace("p", ".")), "load": int(m.group(3)),
                   "slip": slips[k],
                   "mu": float(np.mean(r.WhlLongFrcFilt[a:b] / r.WhlVertFrcFilt[a:b]))}


def curve(segs, keep):
    g = collections.defaultdict(lambda: {True: [], False: []})
    for s in segs:
        if keep(s):
            g[s["slip"]][s["brake"]].append(abs(s["mu"]) if s["slip"] else s["mu"])
    for slip in sorted(g):
        b, d = g[slip][True], g[slip][False]
        print(f"  {slip:2d} %  braking {np.mean(b):.3f} (sd {np.std(b):.3f}, {len(b)})  "
              f"driving {np.mean(d):.3f} (sd {np.std(d):.3f}, {len(d)})  mean {np.mean(b + d):.3f}")


def main():
    segs = list(segments(sys.argv[1]))
    print("bare ice, 2.5 and 3.5 bar, 5 kN")
    curve(segs, lambda s: s["film"] is None and s["load"] == 5000 and s["bar"] in (2.5, 3.5))
    print("ice under a water film, all five conditions, 2.5 bar, 5 kN")
    curve(segs, lambda s: s["film"] is not None)


if __name__ == "__main__":
    main()
