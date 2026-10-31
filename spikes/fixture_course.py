"""The fixture course of roadmap 066 R0, and the reference corner its traces are judged by.

Pure numpy; no Isaac. `road_fixtures.py` builds the lanes from `LANES` in Isaac and drives
aib's Model 3 over them; `reference()` drives a quarter car with what PhysX's corner lacks
(a wheel mass on a tire spring, the tire's geometric envelope, a bump stop) over the same
profile, so the two traces can be laid side by side.

Lanes run along +x, centred on their `y`; a fixture starts at x = X0. The car's left wheels
run at lane y + tread / 2. Heights in m above the lane's flat ground.

    python3 spikes/fixture_course.py      # the reference over every fixture, as JSON
"""
import json
import math

import numpy as np

X0 = 100.0           # where each fixture starts
LANE_X = (20.0, 180.0)
LANE_HALF = 2.0      # lane half width, m
TREAD_HALF = 0.80    # the left wheels' offset from the lane centre (the spec's tread / 2)


def _box_x(x, x0, length, h):
    """A step up of h at x0 and down at x0 + length: a cleat or a kerb-like rock."""
    return np.where((x >= x0) & (x <= x0 + length), h, 0.0)


def cleat(h, length=0.20):
    return lambda x, y: _box_x(x, X0, length, h)


def pothole(depth=0.08, length=0.50, width=0.80):
    """Under the left wheels only, vertical edges."""
    def f(x, y):
        inside = (x >= X0) & (x <= X0 + length) & (np.abs(y - TREAD_HALF) <= width / 2)
        return np.where(inside, -depth, 0.0)
    return f


def washboard(length=30.0, pitch=0.70, amplitude=0.010):
    """Corrugation: a sinusoid of 2 x amplitude peak to peak across the whole lane (TRH 20:
    pitch in cm ~ modal speed in km/h; 20 mm peak to peak measured on a gravel road)."""
    def f(x, y):
        inside = (x >= X0) & (x <= X0 + length)
        return np.where(inside, amplitude * (1 - np.cos(2 * math.pi * (x - X0) / pitch)), 0.0)
    return f


def kicker(height=0.50, ramp=5.0, top=3.0):
    """A crest that throws the car: up `height` over `ramp`, flat for `top`, then a vertical
    drop back to the lane."""
    def f(x, y):
        up = np.clip((x - X0) / ramp, 0.0, 1.0) * height
        return np.where(x <= X0 + ramp + top, up, 0.0) * (x >= X0)
    return f


def boulder(height=0.20, radius=0.30):
    """A static rounded boulder (a cosine dome) under the left wheels."""
    def f(x, y):
        r = np.hypot(x - (X0 + radius), y - TREAD_HALF)
        return np.where(r < radius, height * 0.5 * (1 + np.cos(math.pi * r / radius)), 0.0)
    return f


def flat():
    return lambda x, y: np.zeros(np.broadcast(x, y).shape)


# name: (lane centre y, profile, x samples in the fixture window, whether y varies, window)
LANES = {
    "flat": (0.0, flat(), None, False),
    "cleat_05": (6.0, cleat(0.05), None, False),
    "cleat_10": (12.0, cleat(0.10), None, False),
    "cleat_15": (18.0, cleat(0.15), None, False),
    "pothole": (24.0, pothole(), 0.02, True),
    "washboard": (30.0, washboard(), 0.02, False),
    "kicker": (36.0, kicker(), None, False),
    "boulder": (42.0, boulder(), 0.02, True),
    "rock": (48.0, flat(), None, False),  # a loose rock on flat ground (road_fixtures.py)
}
# The fixtures' x extents beyond X0, for sampling and for the run's window
EXTENT = {"flat": 1.0, "cleat_05": 0.2, "cleat_10": 0.2, "cleat_15": 0.2, "pothole": 0.5,
          "washboard": 30.0, "kicker": 8.0, "boulder": 0.6, "rock": 0.3}
# Edges that are vertical: x positions where the profile jumps, sampled 1 mm either side
EDGES = {"cleat_05": (0.0, 0.2), "cleat_10": (0.0, 0.2), "cleat_15": (0.0, 0.2),
         "pothole": (0.0, 0.5), "kicker": (8.0,)}


def lane_grid(name):
    """The lane's mesh grid: x samples, y samples, heights (len(y), len(x)), in lane frame
    (y relative to the lane centre)."""
    _, f, dx, ydep = LANES[name]
    ext = EXTENT[name]
    xs = [LANE_X[0], X0 - 1.0, LANE_X[1]]
    if dx is not None:
        xs += list(np.arange(X0 - 0.5, X0 + ext + 0.5, dx))
    else:
        xs += list(np.linspace(X0, X0 + ext, 2 if name.startswith("cleat") else 41))
        xs += [X0 + ext + 1.0]
    for e in EDGES.get(name, ()):
        xs += [X0 + e - 0.001, X0 + e + 0.001]
    xs = np.unique(np.round(np.array(xs), 4))
    ys = np.arange(-LANE_HALF, LANE_HALF + 1e-9, 0.05) if ydep else np.array([-LANE_HALF,
                                                                               LANE_HALF])
    X, Y = np.meshgrid(xs, ys)
    return xs, ys, f(X, Y)


def track(name, y=TREAD_HALF, dx=0.002, ext=None):
    """The left wheels' line through the fixture: x samples and heights."""
    _, f, _, _ = LANES[name]
    ext = EXTENT[name] if ext is None else ext
    x = np.arange(X0 - 5.0, X0 + ext + 15.0, dx)
    return x, f(x, np.full_like(x, y))


def envelope(x, h, r):
    """The rigid disc's lowest centre height over the profile, minus r: where a round wheel of
    radius r rests on (x, h) -- the tire's geometric envelope of a step."""
    dx = x[1] - x[0]
    n = int(r / dx)
    out = np.full_like(h, -np.inf)
    for k in range(-n, n + 1):
        u = k * dx
        lift = math.sqrt(max(r * r - u * u, 0.0)) - r
        out = np.maximum(out, np.roll(h, -k) + lift)
    return out


def reference(name, speed, sprung, k, c, travel=0.20, sag=None, wheel_mass=42.0,
              k_tire=230e3, c_tire=150.0, radius=0.370, bump=0.03, k_bump=None, rate=4000.0):
    """A quarter car over the left track of `name` at `speed`: sprung mass on k, c; a wheel of
    `wheel_mass` on a tire spring k_tire that lifts off; the tire's geometric envelope; a
    progressive bump stop over the last `bump` m of compression travel (stiffness rising to
    k_bump, default 20 k). Returns the time series and the summary the Isaac traces are
    compared with."""
    g = 9.81
    if sag is None:
        sag = sprung * g / k
    k_bump = 20 * k if k_bump is None else k_bump
    x, h = track(name)
    env = envelope(x, h, radius)
    dt = 1.0 / rate
    t_end = (x[-1] - x[0]) / speed
    n = int(t_end / dt)
    # states: body z (zs), wheel z (zu), relative to their static positions
    zs, vs, zu, vu = 0.0, 0.0, 0.0, 0.0
    tire_static = (sprung + wheel_mass) * g
    comp_room = travel - sag  # compression left at rest
    out = {"t": [], "x": [], "road": [], "load": [], "body_acc": [], "susp": [], "zs": []}
    for i in range(n):
        xi = x[0] + speed * i * dt
        road = float(np.interp(xi, x, env))
        defl = zu - zs  # + : compression beyond static
        f_spring = k * defl + c * (vu - vs)
        over = defl - (comp_room - bump)
        if over > 0:
            f_spring += k_bump * over ** 2 / bump  # progressive: 0 at contact, k_bump at the end
        if defl < -sag:  # full droop: the spring cannot pull
            f_spring = max(f_spring, k * (-sag) + c * (vu - vs))
        tire_defl = road - zu
        f_tire = max(tire_static + k_tire * tire_defl + c_tire * (0.0 - vu), 0.0)
        a_s = (f_spring) / sprung
        a_u = (f_tire - tire_static - f_spring) / wheel_mass
        vs += a_s * dt
        zs += vs * dt
        vu += a_u * dt
        zu += vu * dt
        if i % int(rate / 500) == 0:
            out["t"].append(i * dt)
            out["x"].append(xi)
            out["road"].append(road)
            out["load"].append(f_tire)
            out["body_acc"].append(a_s)
            out["susp"].append(defl)
            out["zs"].append(zs)
    load = np.array(out["load"])
    xs = np.array(out["x"])
    rise = np.nonzero(load > tire_static * 1.05)[0]
    summary = {"static_N": round(tire_static, 1), "peak_N": round(float(load.max()), 1),
               "min_N": round(float(load.min()), 1),
               "first_rise_x": None if not len(rise) else round(float(xs[rise[0]] - X0), 3),
               "body_acc_peak": round(float(np.abs(out["body_acc"]).max()), 2),
               "airborne_s": round(float((load <= 1.0).sum()) / 500.0, 3),
               "body_rise_m": round(float(np.max(out["zs"])), 3)}
    return out, summary


if __name__ == "__main__":
    # The spec's front corner (tesla_model3: 1.6 Hz, damping ratio 0.45); the Isaac run prints
    # the rig's own values, which replace these when comparing.
    m = 1760 * 0.5 / 2
    w = 2 * math.pi * 1.6
    k = m * w * w
    c = 2 * 0.45 * math.sqrt(k * m)
    res = {}
    for name, v in (("cleat_05", 10), ("cleat_10", 10), ("cleat_10", 20), ("cleat_15", 10),
                    ("pothole", 10), ("washboard", 10), ("washboard", 20), ("kicker", 15),
                    ("boulder", 10)):
        res[f"{name}@{v}"] = reference(name, v, m, k, c)[1]
    print(json.dumps(res, indent=1))
