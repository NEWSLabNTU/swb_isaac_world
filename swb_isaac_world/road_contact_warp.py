"""`road_contact.TireContact`'s envelope and inner loop as one compiled warp kernel on the CPU
(066 R4): numpy spends most of a step in its per-call overhead on arrays of four wheels; the
same arithmetic compiled costs microseconds. The road's heights are still sampled by numpy (the
height function is the detail's tiles); the kernel takes them and does the rest: each cam's
disc over the samples, the tandem's effective height and slope, their values under each inner
step, and the wheel's integration. float64, as numpy: `test_road_contact.py` checks the two
agree where warp imports (Isaac's Python).

Imported by `road_contact` only when a contact asks for it; importing it compiles the kernel
once (warp caches it on disk).
"""
import warp as wp

import numpy as np

wp.init()

f64 = wp.float64
NEG = -1.0e30  # the disc below a sample it does not reach


@wp.func
def _eff(disc: wp.array2d(dtype=f64), along: wp.array(dtype=f64), i: int, k: int,
         c0: f64, du: f64, ls: f64, nd: int, want_slope: int):
    """The effective height (want_slope 0) or slope (1) at grid station k of wheel i, from the
    cams' across-mean disc at each station (its index clipped as numpy's is)."""
    na = along.shape[0]
    s = f64(float(k)) * du
    total = f64(0.0)
    first = f64(0.0)
    last = f64(0.0)
    for a in range(na):
        t = wp.clamp(int(wp.round((s + along[a] - c0) / du)), 0, nd - 1)
        v = disc[i, t]
        total += v
        if a == 0:
            first = v
        if a == na - 1:
            last = v
    if want_slope == 1:
        return (last - first) / ls
    return total / f64(float(na))


@wp.func
def _at(w: wp.array2d(dtype=f64), i: int, d: f64, du: f64, ns: int):
    """Row i of `w` (sampled every du from 0) at distance d, linearly (road_contact._at)."""
    f = wp.clamp(d / du, f64(0.0), f64(float(ns)) - f64(1.000001))
    j = int(f)
    return w[i, j] + (w[i, j + 1] - w[i, j]) * (f - f64(float(j)))


@wp.func
def _damper(v: f64, cb: f64, cr: f64, knee: f64, high: f64):
    c = cr
    if v > f64(0.0):
        c = cb
    a = wp.abs(v)
    x = a
    if a > knee:
        x = knee + high * (a - knee)
    return wp.sign(v) * c * x


@wp.kernel
def _contact(h: wp.array3d(dtype=f64), lift: wp.array2d(dtype=f64),
             along: wp.array(dtype=f64), c0: f64, du: f64, ls: f64, m: int,
             nq: int, ns: int, d_in: wp.array2d(dtype=f64), za: wp.array2d(dtype=f64),
             k: wp.array(dtype=f64), cb: wp.array(dtype=f64), cr: wp.array(dtype=f64),
             knee: wp.array(dtype=f64), high: wp.array(dtype=f64), low: wp.array(dtype=f64),
             kb_b: wp.array(dtype=f64), k_reb: wp.array(dtype=f64),
             bars: wp.array2d(dtype=f64), px_travel: wp.array(dtype=f64),
             defl0: wp.array(dtype=f64), r: wp.array(dtype=f64), kt: wp.array(dtype=f64),
             ct: wp.array(dtype=f64), rim: wp.array(dtype=f64), krim: wp.array(dtype=f64),
             mass: wp.array(dtype=f64), g: f64, hstep: f64, v_att: wp.array(dtype=f64),
             init: int,
             w: wp.array2d(dtype=f64), slope: wp.array2d(dtype=f64),
             w_prev: wp.array(dtype=f64), zu: wp.array(dtype=f64), vu: wp.array(dtype=f64),
             fs_out: wp.array2d(dtype=f64), ft_out: wp.array2d(dtype=f64),
             comp_out: wp.array2d(dtype=f64), j_out: wp.array2d(dtype=f64),
             s_out: wp.array2d(dtype=f64), disc: wp.array2d(dtype=f64)):
    n = h.shape[0]
    across = h.shape[1]
    nd = nq - m + 1
    big_n = d_in.shape[1]
    # each disc's bottom at each station, the mean across the tire: the max over the samples
    # under it of sample + lift, searched outward from the disc's centre (lift falls with the
    # distance on both sides), stopping where even the row's highest sample could not beat it:
    # the same max, most of the samples skipped
    c = (m - 1) // 2
    reach = wp.max(c, m - 1 - c)
    for i in range(n):
        for t in range(nd):
            disc[i, t] = f64(0.0)
        for a in range(across):
            top = f64(NEG)
            for q in range(nq):
                top = wp.max(top, h[i, a, q])
            for t in range(nd):
                best = h[i, a, t + c] + lift[i, c]
                for d in range(1, reach + 1):
                    lo = c - d
                    hi = c + d
                    done = 1
                    if lo >= 0:
                        if lift[i, lo] + top > best:
                            done = 0
                            best = wp.max(best, h[i, a, t + lo] + lift[i, lo])
                    if hi < m:
                        if lift[i, hi] + top > best:
                            done = 0
                            best = wp.max(best, h[i, a, t + hi] + lift[i, hi])
                    if done == 1:
                        break
                disc[i, t] += best
        for t in range(nd):
            disc[i, t] = disc[i, t] / f64(float(across))
    # the effective road on the step's grid
    for i in range(n):
        for kk in range(ns):
            w[i, kk] = _eff(disc, along, i, kk, c0, du, ls, nd, 0)
            slope[i, kk] = _eff(disc, along, i, kk, c0, du, ls, nd, 1)
    if init == 1:  # the wheel at rest on the road
        for i in range(n):
            zu[i] = w[i, 0] + r[i] - defl0[i]
            vu[i] = f64(0.0)
            w_prev[i] = w[i, 0]
    for it in range(big_n):  # our wheel: sequential
        for i in range(n):
            j_out[i, it] = px_travel[i] + zu[i] + defl0[i] - za[i, it]
        for i in range(n):
            j = j_out[i, it]
            fs = k[i] * j + _damper(vu[i] - v_att[i], cb[i], cr[i], knee[i], high[i])
            over = wp.max(j - low[i], f64(0.0))
            fs += kb_b[i] * over * over + k_reb[i] * wp.min(j, f64(0.0))
            for q in range(n):
                fs += bars[i, q] * j_out[q, it]
            wi = _at(w, i, d_in[i, it], du, ns)
            wl = w_prev[i]
            if it > 0:
                wl = _at(w, i, d_in[i, it - 1], du, ns)
            wdot = (wi - wl) / hstep
            comp = wi + r[i] - zu[i]
            strike = wp.max(comp - rim[i], f64(0.0))
            ft = kt[i] * comp - ct[i] * (vu[i] - wdot) + krim[i] * strike
            ft = wp.max(ft, f64(0.0))
            if comp <= f64(0.0):
                ft = f64(0.0)
            fs_out[i, it] = fs
            ft_out[i, it] = ft
            comp_out[i, it] = comp
            s_out[i, it] = _at(slope, i, d_in[i, it], du, ns)
        for i in range(n):
            vu[i] = vu[i] + (ft_out[i, it] - fs_out[i, it] - mass[i] * g) / mass[i] * hstep
            zu[i] = zu[i] + vu[i] * hstep
    for i in range(n):
        w_prev[i] = _at(w, i, d_in[i, big_n - 1], du, ns)


def _arr(a):
    a = np.ascontiguousarray(a, np.float64)
    return wp.array(a, dtype=f64, device="cpu", copy=False)


class Kernel:
    """One TireContact's kernel: its parameters and buffers made once, the launch recorded once
    (warp's per-launch argument packing costs more than the kernel), `run` per step. The
    buffers grow, and the launch is recorded again, when a step needs more stations."""

    def __init__(self, tc, g):
        self.tc = tc
        n = tc.n
        r = tc.r[:, None]
        u = tc.lift_u[None, :]
        lift = np.where(np.abs(u) <= r, np.sqrt(np.maximum(r * r - u * u, 0.0)) - r, NEG)
        self.m = lift.shape[1]
        self.np = {}  # numpy views of every array the kernel takes
        self.wp = {}
        for name, v in (("k", tc.k), ("cb", tc.c_bump), ("cr", tc.c_reb), ("knee", tc.knee),
                        ("high", tc.high), ("low", tc.travel - tc.bump),
                        ("kb_b", tc.kb / tc.bump), ("k_reb", tc.rebound_rate * tc.k),
                        ("bars", tc.bars), ("px_travel", tc.px_travel), ("defl0", tc.defl0),
                        ("r", tc.r), ("kt", tc.kt), ("ct", tc.ct), ("rim", tc.rim),
                        ("krim", tc.krim), ("mass", tc.m), ("along", tc.along),
                        ("lift", lift), ("zu", np.zeros(n)), ("vu", np.zeros(n)),
                        ("w_prev", np.zeros(n)), ("v_att", np.zeros(n))):
            self._put(name, np.array(v, dtype=np.float64))
        self.g = float(g)
        self.big_n = 0
        self.cap_ns = self.cap_nq = 0
        self.cmd = None

    def _put(self, name, a):
        a = np.ascontiguousarray(a, np.float64)
        self.np[name] = a
        self.wp[name] = wp.array(a, dtype=f64, device="cpu", copy=False)

    def _grow(self, ns, nq, big_n):
        n, across = self.tc.n, len(self.tc.across)
        self.cap_ns, self.cap_nq, self.big_n = max(ns, 2 * self.cap_ns, 16), \
            max(nq, 2 * self.cap_nq, 128), big_n
        self._put("h", np.zeros((n, across, self.cap_nq)))
        self._put("disc", np.zeros((n, self.cap_nq)))
        for name in ("w", "slope"):
            self._put(name, np.zeros((n, self.cap_ns)))
        for name in ("d_in", "za", "FS", "FT", "COMP", "J", "S"):
            self._put(name, np.zeros((n, big_n)))
        a = self.wp
        self.cmd = wp.launch(_contact, dim=1, device="cpu", record_cmd=True, inputs=[
            a["h"], a["lift"], a["along"], f64(0.0), f64(self.tc.du), f64(self.tc.ls), self.m,
            0, 0, a["d_in"], a["za"], a["k"], a["cb"], a["cr"], a["knee"], a["high"],
            a["low"], a["kb_b"], a["k_reb"], a["bars"], a["px_travel"], a["defl0"], a["r"],
            a["kt"], a["ct"], a["rim"], a["krim"], a["mass"], f64(self.g), f64(0.0),
            a["v_att"], 0, a["w"], a["slope"], a["w_prev"], a["zu"], a["vu"], a["FS"],
            a["FT"], a["COMP"], a["J"], a["S"], a["disc"]])

    def run(self, h, c0, ns, d_in, za, v_att, hstep, init):
        """`h` (n, across, stations) the road's samples; returns {w, FS, FT, COMP, J, S} and
        leaves zu, vu, w_prev in the contact."""
        tc, a = self.tc, self.np
        nq = h.shape[2]
        if ns > self.cap_ns or nq > self.cap_nq or d_in.shape[1] != self.big_n:
            self._grow(ns, nq, d_in.shape[1])
        a["h"][:, :, :nq] = h
        a["d_in"][...] = d_in
        a["za"][...] = za
        a["v_att"][...] = v_att
        a["zu"][...] = tc.zu
        a["vu"][...] = tc.vu
        if tc.w_prev is not None:
            a["w_prev"][...] = tc.w_prev
        cmd = self.cmd
        cmd.set_param_by_name("c0", f64(c0))
        cmd.set_param_by_name("nq", int(nq))
        cmd.set_param_by_name("ns", int(ns))
        cmd.set_param_by_name("hstep", f64(hstep))
        cmd.set_param_by_name("init", int(init))
        cmd.launch()
        tc.zu, tc.vu, tc.w_prev = a["zu"].copy(), a["vu"].copy(), a["w_prev"].copy()
        out = {name: a[name].copy() for name in ("FS", "FT", "COMP", "J", "S")}
        out["w"] = a["w"][:, :ns].copy()
        return out
