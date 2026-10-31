"""The tire's vertical contact with the road's small-scale detail (roadmap 066 R5). Pure numpy.

PhysX's wheel query sees only the smooth base road: once per PhysX step, one ray, a rigid wheel
on a spring (docs/issues/029: it misses short obstacles or throws the car off them). This model
gives each wheel what PhysX's lacks, against the one true height function:

- the tire's envelope of the road: a tandem of `CAMS_ALONG` x `CAMS_ACROSS` cams over the
  contact patch (Zegelaar, Schmeitz; MF-Swift's road feeler); each cam is a rigid disc of the
  tire's radius, and the effective height is the cams' mean, the effective slope the front
  cams' height less the rear's over their spacing;
- the tire's radial spring and damper, which lifts off;
- the wheel's vertical mass (the unsprung corner);
- the suspension's spring, damper, a progressive bump stop over its last `bump` m and a rebound
  stop at full droop.

It integrates `inner` times per PhysX step (480 Hz at 60 Hz), with the body's attachment point
moving on at its velocity from the step's start. What it returns per wheel is what PhysX's own
corner does not already apply: the suspension force on the body less the force PhysX's spring
and damper give on the base road at the same body motion (`dfz`, at the wheel's centre, along
world z), and the effective plane's push back along the wheel's heading (`dfx`, at the
contact). PhysX's spring sees the body's motion as ours does, so in the linear range the
difference does not depend on the body at all: it is the spring and damper across the wheel's
offset from where PhysX puts it. The tire's load (`load`) goes to PhysX's tire through its
friction (`friction_scale`): PhysX's tire saturates at mu times its own load.

The height functions take arrays of world x, y and return heights (m), broadcasting.
"""
from dataclasses import dataclass

import numpy as np

G = 9.81
CAMS_ALONG = 5
CAMS_ACROSS = 6
# The tandem's spacing as a share of the contact length (Stallmann et al. 2014's p_ls, 0.8-0.9).
P_LS = 0.8
# Half the contact length of a passenger tire under its static load, m (illustrative; fitted
# cams replace it).
HALF_CONTACT = 0.07
DU = 0.01  # the road's sample spacing under a cam, m
# A rebound stop at full droop this many times the spring's rate: the strut carries the wheel.
REBOUND_RATE = 20.0
FRICTION_SCALE_MAX = 3.0


@dataclass
class Corner:
    """One wheel's corner: the suspension as aib's rig sets it, and what PhysX does not know."""

    sprung: float  # the corner's sprung mass, kg (its static load is sprung g)
    k: float  # spring rate, N/m
    c: float  # damper rate, N s/m
    travel: float  # suspension travel, m
    radius: float  # tire radius, m
    width: float  # tire width, m
    wheel_mass: float = 42.0  # unsprung mass per corner, kg (wheel, tire, brake, hub, half arms)
    k_tire: float = 230e3  # tire radial rate, N/m
    c_tire: float = 150.0  # tire radial damping, N s/m
    bump: float = 0.03  # the bump stop's engagement before full compression, m
    k_bump: float = 0.0  # the bump stop's rate at full compression, N/m (0: 20 k)
    # The tire's deflection at which the rim meets the road through it, m: three quarters of a
    # 235/45 R18's 106 mm section. Beyond it the rim's own stiffness adds (k_rim, 0: 10 k_tire).
    rim_clearance: float = 0.08
    k_rim: float = 0.0


class TireContact:
    """The corners of one vehicle (or several: wheels are independent)."""

    def __init__(self, corners, inner=8):
        self.n = n = len(corners)
        self.inner = inner
        f = lambda name: np.array([getattr(cn, name) for cn in corners], dtype=float)
        self.k, self.c, self.travel = f("k"), f("c"), f("travel")
        self.sprung = f("sprung")
        self.r, self.width, self.m = f("radius"), f("width"), f("wheel_mass")
        self.kt, self.ct = f("k_tire"), f("c_tire")
        self.bump = f("bump")
        self.rim = f("rim_clearance")
        kr = f("k_rim")
        self.krim = np.where(kr > 0, kr, 10.0 * self.kt)
        kb = f("k_bump")
        self.kb = np.where(kb > 0, kb, 20.0 * self.k)
        self.zu = np.zeros(n)  # wheel centre, world z
        self.vu = np.zeros(n)
        # The tire's static deflection: our wheel's centre sits this far below PhysX's rigid
        # wheel on the same road, so that at rest the two springs carry the same load.
        self.defl0 = (self.sprung + self.m) * G / self.kt
        self.w_prev = None
        self.ready = False
        rmax = float(self.r.max())
        self.lift_u = np.arange(-rmax, rmax + 1e-9, DU)
        self.ls = 2 * P_LS * HALF_CONTACT
        self.along = np.linspace(-self.ls / 2, self.ls / 2, CAMS_ALONG)
        self.across = (np.arange(CAMS_ACROSS) + 0.5) / CAMS_ACROSS - 0.5  # shares of width

    def reset(self):
        self.ready = False
        self.w_prev = None

    def _lift(self):
        """Each wheel's disc below its lowest point at the sample offsets lift_u (<= 0)."""
        r = self.r[:, None]
        u = self.lift_u[None, :]
        return np.where(np.abs(u) <= r, np.sqrt(np.maximum(r * r - u * u, 0.0)) - r, -np.inf)

    def envelope(self, height, xy, heading, span):
        """The effective road along each wheel's path over the step: s (m along the heading
        from the contact) on a DU grid from 0 to `span`, the effective height (n, len(s)) and
        slope (n, len(s))."""
        s = np.arange(0.0, max(span, 0.0) + 2 * DU, DU)
        pad = float(self.lift_u[-1]) + self.ls / 2
        q = np.arange(-pad, s[-1] + pad + 1e-9, DU)  # sample stations along the heading
        hx, hy = heading[:, 0], heading[:, 1]
        lat = self.across[None, :] * self.width[:, None]  # (n, across)
        # (n, across, stations)
        px = xy[:, 0, None, None] + hx[:, None, None] * q[None, None, :] \
            - hy[:, None, None] * lat[:, :, None]
        py = xy[:, 1, None, None] + hy[:, None, None] * q[None, None, :] \
            + hx[:, None, None] * lat[:, :, None]
        h = np.asarray(height(px, py), dtype=float)
        m = len(self.lift_u)
        win = np.lib.stride_tricks.sliding_window_view(h, m, axis=2)  # (n, across, nq-m+1, m)
        disc = np.max(win + self._lift()[:, None, None, :], axis=3)  # disc bottom at centres
        # centre stations of `disc`: q[(m-1)/2 ...]
        c0 = q[(m - 1) // 2]
        cams = []
        for a in self.along:
            idx = np.clip(np.rint((s + a - c0) / DU).astype(int), 0, disc.shape[2] - 1)
            cams.append(disc[:, :, idx].mean(axis=1))  # (n, len(s)), mean across
        cams = np.stack(cams)  # (along, n, len(s))
        w = cams.mean(axis=0)
        slope = (cams[-1] - cams[0]) / self.ls
        return s, w, slope

    def step(self, dt, height, base, xy, heading, speed, z_att, v_att):
        """One PhysX step for every wheel: `xy` (n, 2) the contact's world position, `heading`
        (n, 2) the wheel's unit heading, `speed` (n,) its ground speed along it, `z_att` and
        `v_att` (n,) the suspension attachment's (the suspension frame's) world height and
        vertical velocity. `height` is the true road, `base` what PhysX's query sees. A jounce
        is PhysX's: travel + wheel centre z - attachment z (the suspension along world z), so
        PhysX's own is known from the body and the base road alone. Returns a dict of per-wheel arrays: dfz, dfx (N, the step's mean), load
        (the tire's mean radial force, N), px_load (PhysX's spring and damper, N), airborne
        (share of the step with no tire force), w (effective height at the step's start),
        load_inner (n, inner) the tire's force at each inner step, rim_strike (m, > 0: the rim
        struck the road through the tire)."""
        n, N = self.n, self.inner
        h = dt / N
        span = float(np.max(np.abs(speed))) * dt
        sgn = np.where(speed < 0, -1.0, 1.0)
        hd = heading * sgn[:, None]
        s, w, slope = self.envelope(height, xy, hd, span)
        zb = np.asarray(base(xy[:, 0], xy[:, 1]), dtype=float) * np.ones(n)
        # the base road under each inner step (PhysX's spring follows it)
        d_in = np.abs(speed)[:, None] * (np.arange(1, N + 1) * h)[None, :]
        zb_in = np.asarray(base(xy[:, 0, None] + hd[:, 0, None] * d_in,
                                xy[:, 1, None] + hd[:, 1, None] * d_in), dtype=float) \
            * np.ones((n, N))
        if not self.ready:  # the wheel at rest on the road
            self.zu = w[:, 0] + self.r - self.defl0
            self.vu = np.zeros(n)
            self.ready = True
            self.w_prev = w[:, 0].copy()
        # What does not depend on our wheel, for every inner step at once (n, N): the road
        # under the tire, the attachment's height, and PhysX's spring and damper on the base
        # road (its wheel rigid on the base; no force once its wheel leaves it).
        W, S = _at(w, d_in), _at(slope, d_in)
        Wdot = np.diff(np.concatenate([self.w_prev[:, None], W], axis=1), axis=1) / h
        ZA = z_att[:, None] + v_att[:, None] * (np.arange(1, N + 1) * h)[None, :]
        JP = self.travel[:, None] + zb_in + self.r[:, None] - ZA
        VB = np.diff(np.concatenate([zb[:, None], zb_in], axis=1), axis=1) / h
        FP = np.where(JP > 0, self.k[:, None] * np.clip(JP, 0, self.travel[:, None])
                      + self.c[:, None] * (VB - v_att[:, None]), 0.0)
        FP = np.maximum(FP, 0.0)
        FS = np.empty((n, N))
        FT = np.empty((n, N))
        COMP = np.empty((n, N))
        low = self.travel - self.bump
        kb_b = self.kb / self.bump
        for i in range(N):  # our wheel: sequential
            j = self.travel + self.zu + self.defl0 - ZA[:, i]
            fs = self.k * j + self.c * (self.vu - v_att)
            over = np.maximum(j - low, 0.0)
            fs += kb_b * over * over + REBOUND_RATE * self.k * np.minimum(j, 0.0)
            comp = W[:, i] + self.r - self.zu
            rim = np.maximum(comp - self.rim, 0.0)
            ft = self.kt * comp - self.ct * (self.vu - Wdot[:, i]) + self.krim * rim
            ft = np.where(comp > 0, np.maximum(ft, 0.0), 0.0)
            self.vu = self.vu + (ft - fs - self.m * G) / self.m * h
            self.zu = self.zu + self.vu * h
            FS[:, i], FT[:, i], COMP[:, i] = fs, ft, comp
        self.w_prev = W[:, -1].copy()
        out = {"dfz": (FS - FP).mean(axis=1), "dfx": -(FT * S).mean(axis=1) * sgn,
               "load": FT.mean(axis=1), "px_load": FP.mean(axis=1),
               "airborne": (FT <= 0).mean(axis=1),
               # the tire's deflection beyond the rim's clearance at its worst, m (> 0: the
               # rim struck the road: a pinch, a bent rim)
               "rim_strike": np.maximum(COMP.max(axis=1) - self.rim, 0.0)}
        out["w"] = w[:, 0]
        out["load_inner"] = FT  # the tire's force at each inner step, for diagnostics
        return out

    def friction_scale(self, load, px_load):
        """The share of the rig's friction each wheel gets so that PhysX's tire, saturating at
        mu times its own load (the sprung share), grips as the tire's real load allows."""
        return np.clip(load / np.maximum(px_load + self.m * G, 1.0), 0.0, FRICTION_SCALE_MAX)


def _at(a, d):
    """Each row of `a` (sampled every DU from 0) at its own distances d (n,) or (n, k) (m),
    linearly."""
    d = np.asarray(d, dtype=float)
    f = np.clip(d / DU, 0.0, a.shape[1] - 1.000001)
    i = f.astype(int)
    rows = np.arange(a.shape[0]).reshape((-1,) + (1,) * (d.ndim - 1))
    return a[rows, i] + (a[rows, i + 1] - a[rows, i]) * (f - i)
