"""Calibration (roadmap 053 step 3): our soil grid's single wheel against a measured tire
test (`spikes/references/<name>.yaml`), at the reference's tire size and load, the slips it
measured. Prints measured against ours per slip; with --fit, fits the named soil keys first
(Nelder-Mead on their logarithms) and prints the class's `soil:` row as fitted.

    python3 src/swb_isaac_world/spikes/fit_reference.py he2020_srtt_medium --soil sand \\
        [--fit n,k_phi,friction_angle,janosi_k,slip_sinkage,rolling_resistance] \\
        [--set cohesion=0,k_c=0] [--start janosi_k=0.01,...]

The grid wheel is rigid, as in the world server: a fit gives an equivalent rigid wheel for
the tire measured, not the soil's plate row. The rig is crm_calibrate.grid_run's: the
carrier at 1 m/s, slip pinned, the steady second half of a 6 m run.

What it matches, and its tolerance (053 step 4): the drawbar pull coefficient within +-20 %
from the first measured point at or past the peak's slip to slip 0.5 (the rise before the
peak is too steep to hold to a share), towed (slip 0) within 20 %; the sinkage within +-25 %
of the reference's `sinkage:`, or of its `rut:` depth up to slip 0.27, where He (2020) says
the rut is the sinkage, and not shallower than the rut above it (a lower bound there).
"""
import argparse
import math
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from swb_isaac_world import soil, surfaces  # noqa: E402

SPEED = 1.0
RUT_IS_SINKAGE = 0.27  # slip up to which the reference's rut depth stands for the sinkage
SLIP_MAX = 0.55  # the tolerance's range ends at 0.5; the point just past it is kept


def load_reference(name):
    with open(os.path.join(HERE, "references", name + ".yaml")) as f:
        return yaml.safe_load(f)


def grid_run(params, tire, slip, dt=1 / 60, length=6.0):
    g = soil.SoilGrid(params)
    out, x = [], 0.0
    while x < length:
        g.begin_step()
        out.append(g.wheel(0, x, 0.013, 0.0, tire["load"], tire["radius"], tire["width"],
                           SPEED / (1 - slip), SPEED, dt, slip))
        x += SPEED * dt
    half = out[len(out) // 2:]
    mean = {f: sum(getattr(o, f) for o in half) / len(half)
            for f in ("sinkage", "resistance", "thrust")}
    return mean["sinkage"], (mean["thrust"] - mean["resistance"]) / tire["load"]


def compare(params, ref):
    """Per measured slip: (slip, measured DPC, ours, measured depth, our sinkage, the slip
    above which the measured depth is only a lower bound)."""
    measured = ref.get("sinkage") or ref["rut"]
    exact = RUT_IS_SINKAGE if "sinkage" not in ref else math.inf
    depth = dict((s, z) for s, z in measured)
    rows = []
    for s, dpc in ref["dpc"]:
        if s > SLIP_MAX:
            continue
        z, d = grid_run(params, ref["tire"], s)
        rows.append((s, dpc, d, depth.get(s), z, exact))
    for s, r in measured:  # slips the pull was not measured at
        if s <= SLIP_MAX and s not in dict(ref["dpc"]):
            z, d = grid_run(params, ref["tire"], s)
            rows.append((s, None, d, r, z, exact))
    return sorted(rows)


def judged(rows):
    """The rows the tolerance holds, and whether each passes: (kind, slip, error, ok, whether
    the measurement is a lower bound)."""
    peak = max((r for r in rows if r[1] is not None), key=lambda r: r[1])[0]
    out = []
    for s, dpc, d, depth, z, exact in rows:
        if dpc is not None and (s == 0.0 or s >= peak):
            e = (d - dpc) / abs(dpc)
            out.append(("pull", s, e, abs(e) <= 0.2, False))
        if depth is not None:
            e = (z - depth) / depth
            out.append(("sinkage", s, e, abs(e) <= 0.25 if s <= exact else e >= -0.25,
                        s > exact))
    return out


def cost(rows):
    """The judged errors squared, ten times over beyond the tolerance, and the rise before
    the peak, which the tolerance leaves out, against half the peak (it fixes Janosi K)."""
    c = 0.0
    for kind, s, e, _, bound in judged(rows):
        if bound:
            e = min(e, 0.0)
        tol = 0.2 if kind == "pull" else 0.25
        c += e * e + 10.0 * max(0.0, abs(e) - tol) ** 2  # outside the tolerance costs more
    peak = max(r[1] for r in rows if r[1] is not None)
    top = max((r for r in rows if r[1] is not None), key=lambda r: r[1])[0]
    for s, dpc, d, *_ in rows:
        if dpc is not None and 0.0 < s < top:
            c += ((d - dpc) / (0.5 * peak)) ** 2
    return c


def fixed_values(text):
    for kv in filter(None, text.split(",")):
        k, v = kv.split("=")
        yield k, math.radians(float(v)) if k == "friction_angle" else float(v)


def with_values(base, values):
    d = dict(base.__dict__)
    d.update(values)
    return soil.SoilParams(**d)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reference")
    ap.add_argument("--soil", default="sand", help="the class whose soil row starts the fit")
    ap.add_argument("--fit", default="", help="soil keys to fit, comma-separated")
    ap.add_argument("--set", default="", help="key=value,... set before fitting (degrees)")
    ap.add_argument("--start", default="", help="key=value,... the fit's start (degrees)")
    args = ap.parse_args(argv)
    ref = load_reference(args.reference)
    params = surfaces.load_classes()[args.soil].soil
    fixed = dict(fixed_values(args.set))
    params = with_values(params, fixed)
    print(f"reference: {ref['name']}")
    show(params, ref, f"{args.soil} as given" + (f", {args.set}" if args.set else ""))
    keys = [k for k in args.fit.split(",") if k]
    if keys:
        from scipy.optimize import minimize

        start = dict(fixed_values(args.start))
        x0 = np.log([max(start.get(k, getattr(params, k)), 1e-6) for k in keys])

        def f(x):
            return cost(compare(with_values(params, dict(zip(keys, np.exp(x)))), ref))

        r = minimize(f, x0, method="Nelder-Mead",
                     options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 600})
        params = with_values(params, dict(zip(keys, np.exp(r.x))))
        show(params, ref, f"fitted ({', '.join(keys)}; cost {r.fun:.4f}, {r.nfev} runs)")


def show(params, ref, title):
    rows = compare(params, ref)
    verdict = {(k, s): (e, ok) for k, s, e, ok, _ in judged(rows)}
    print(f"\n{title}")
    print(f"  soil: n {params.n:.4g}, k_c {params.k_c:.4g}, k_phi {params.k_phi:.4g}, "
          f"cohesion {params.cohesion:.4g}, friction_angle "
          f"{math.degrees(params.friction_angle):.3g}, janosi_k {params.janosi_k:.4g}, "
          f"slip_sinkage {params.slip_sinkage:.3g}, "
          f"rolling_resistance {params.rolling_resistance:.3g}")
    print(f"  slip   DPC meas  ours   err    {'sink' if 'sinkage' in ref else ' rut'} mm  "
          "ours mm  err")
    for s, dpc, d, rut, z, _ in rows:
        pe = verdict.get(("pull", s))
        se = verdict.get(("sinkage", s))
        print(f"  {s:4.3f}  " + (f"{dpc:+.3f}" if dpc is not None else "      ")
              + f"  {d:+.3f}  " + (f"{pe[0]:+5.0%}{'' if pe[1] else '!'}" if pe else "      ")
              + ("    " + f"{rut * 1000:5.1f}" if rut is not None else "         ")
              + f"   {z * 1000:5.1f}  "
              + (f"{se[0]:+5.0%}{'' if se[1] else '!'}" if se else ""))
    bad = [(k, s) for k, s, _, ok, _ in judged(rows) if not ok]
    print(f"  within tolerance: {'yes' if not bad else 'no, ' + str(bad)}")


if __name__ == "__main__":
    main()
