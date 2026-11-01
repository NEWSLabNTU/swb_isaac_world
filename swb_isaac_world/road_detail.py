"""A road's small-scale surface: one seeded height function per road corridor (roadmap 066 R3).
Pure numpy.

The world's own ground is the base road, smooth below a few metres, which PhysX's wheel query
sees. This adds what lies on it, as a height above the base, m:

- the cross-section, as a function of the distance v from the centreline: a straight "A"
  crown (FHWA: 4 %, never over 6 %) or a level section (Japanese forest roads, 林業専用道), the
  shoulder break, a 1V:3H foreslope, a flat-bottom ditch and a 1V:2H backslope;
- an isotropic 2D roughness field with ISO 8608's spectrum, Gd(n) = Gd(n0) (n / n0)^-w with
  n0 = 0.1 cycles/m, broken at 2 cycles/m as measured gravel is (Zuraulis et al. 2021);
- two ruts at half the track width with a windrow between them;
- potholes, Poisson-placed along the road, weighted to sags;
- washboard in zones (braking, acceleration, climbs): a sinusoid of wavelength about the
  modal speed in km/h / 100 m (TRH 20), slightly asymmetric;
- water bars, oblique ridges across the road, spaced by grade.

Everything is drawn from the corridor's seed per tile or per segment, so a tile generated again
(streamed back in) is identical. The detail is sampled into world-aligned tiles of TILE m at
SPACING m; `height` reads them bilinearly, and the render mesh and the lidar are built from
the same samples (question 3: one road).

The centreline is a polyline in the world frame (x, y); u is the distance along it, v the
signed distance to its left.
"""
from dataclasses import dataclass, field
import math

import numpy as np

SPACING = 0.05  # m between samples: six or more on a 0.3 m washboard wave, ISO 8608's 10 c/m
TILE = 16.0  # m, a tile's side; the roughness field's longest wave is 2 TILE
N0 = 0.1  # ISO 8608's reference spatial frequency, cycles/m
# ISO 8608 classes: the geometric mean of Gd(n0) per class, m^3
ISO_CLASS = {c: 16e-6 * 4 ** i for i, c in enumerate("ABCDEFGH")}
N_MAX = 10.0  # cycles/m: the shortest wave a tire feels (ISO 8608's upper limit)


@dataclass
class Section:
    """The cross-section, m. `crown`: the cross slope each way from the centreline (0: level).
    The carriageway is 2 `half_width` wide; then a shoulder, a foreslope down to the ditch,
    its flat bottom and the backslope up to the base level again."""

    half_width: float = 2.5
    crown: float = 0.04
    shoulder: float = 0.5
    foreslope: float = 1.0 / 3.0  # V:H
    ditch_depth: float = 0.30  # below the shoulder's edge
    ditch_bottom: float = 0.6
    backslope: float = 0.5

    def height(self, v):
        a = np.abs(v)
        w = self.half_width
        h = -self.crown * np.minimum(a, w + self.shoulder)
        edge = -self.crown * (w + self.shoulder)
        x = a - (w + self.shoulder)
        fore = self.ditch_depth / self.foreslope
        bottom = edge - self.ditch_depth
        back = (0.0 - bottom) / self.backslope  # up to the base (0) again
        h = np.where(x > 0, edge - self.foreslope * x, h)
        h = np.where(x > fore, bottom, h)
        x2 = x - fore - self.ditch_bottom
        h = np.where(x2 > 0, np.minimum(bottom + self.backslope * x2, 0.0), h)
        return np.where(x2 > back, 0.0, h)

    def extent(self):
        """The half width beyond which the section is the base's own, m."""
        edge = self.crown * (self.half_width + self.shoulder) + self.ditch_depth
        return (self.half_width + self.shoulder + self.ditch_depth / self.foreslope
                + self.ditch_bottom + edge / self.backslope)


@dataclass
class Roughness:
    """ISO 8608 class `cls` (or `gd0`, m^3), waviness `w` below the break at `n_break`
    cycles/m and `w_high` above it. Zuraulis et al. measured the break on gravel but not the
    slope beyond it; w_high = 1.5 (more short-wave energy, from corrugation and stones) is
    assumed."""

    cls: str = "D"
    gd0: float = 0.0
    w: float = 2.0
    n_break: float = 2.0
    w_high: float = 1.5

    def psd(self, n):
        """The displacement PSD along a line, m^3 (m^2 per cycle/m)."""
        g0 = self.gd0 or ISO_CLASS[self.cls]
        n = np.asarray(n, dtype=float)
        low = g0 * (np.maximum(n, 1e-9) / N0) ** -self.w
        gb = g0 * (self.n_break / N0) ** -self.w
        return np.where(n <= self.n_break, low, gb * (np.maximum(n, 1e-9) / self.n_break)
                        ** -self.w_high)

    def psd2(self, kappa):
        """The isotropic 2D PSD at radial frequency kappa (cycles/m) whose line sections have
        `psd`: for G(n) ~ n^-w, G2 = G Gamma((w+1)/2) / (2 sqrt(pi) Gamma(w/2)) / kappa (G is
        one-sided, G2 over the whole plane), applied per band."""
        k = np.maximum(np.asarray(kappa, dtype=float), 1e-9)
        c = lambda x: math.gamma((x + 1) / 2) / (2 * math.sqrt(math.pi) * math.gamma(x / 2))
        return self.psd(k) * np.where(k <= self.n_break, c(self.w), c(self.w_high)) / k


@dataclass
class Ruts:
    """Two ruts at +-`offset` of the centreline (half a track) and a windrow between them."""

    depth: float = 0.03
    width: float = 0.30  # 2 sigma of each rut's Gaussian
    offset: float = 0.80
    windrow: float = 0.02  # the centre windrow's height
    windrow_width: float = 0.5

    def height(self, v):
        s = self.width / 2
        h = -self.depth * (np.exp(-0.5 * ((v - self.offset) / s) ** 2)
                           + np.exp(-0.5 * ((v + self.offset) / s) ** 2))
        return h + self.windrow * np.exp(-0.5 * (v / (self.windrow_width / 2)) ** 2)


@dataclass
class Potholes:
    """Poisson along the road at `per_100m`, `sag_weight` times as many in a sag (the base's
    grade changing from down to up); depth and diameter uniform in the ranges (USACE TM 5-626:
    1.5-10 cm deep, 0.3-1 m across), within the carriageway; edges 3 cm wide."""

    per_100m: float = 2.0
    depth: tuple = (0.015, 0.08)
    diameter: tuple = (0.3, 1.0)
    edge: float = 0.03
    sag_weight: float = 3.0


@dataclass
class Washboard:
    """A corrugation zone from u0 to u1 m along the road: peak-to-peak `height`, wavelength
    `pitch` (m; TRH 20: cm about the modal speed in km/h), `skew` > 0 steepening each ridge's
    far face. Ramps in and out over two waves."""

    u0: float
    u1: float
    height: float = 0.02
    pitch: float = 0.7
    skew: float = 0.3

    def h(self, u):
        ph = 2 * math.pi * (u - self.u0) / self.pitch
        wave = np.cos(ph - self.skew * np.sin(ph))  # skewed: the ridges lean
        ramp = np.clip(np.minimum(u - self.u0, self.u1 - u) / (2 * self.pitch), 0.0, 1.0)
        return np.where((u >= self.u0) & (u <= self.u1), self.height / 2 * (1 - wave) * ramp,
                        0.0)


@dataclass
class WaterBars:
    """Oblique ridges across the carriageway at `angle` deg from square, `height` and
    `width` (a rubber bar's: 92 % of a Tokyo survey's cross drains; their height is not
    published, assumed). Spaced by the base's grade: `spacing(grade %)` m."""

    height: float = 0.06
    width: float = 0.25
    angle: float = 35.0
    at: tuple = ()  # u positions; empty: from the grade (`spacing`)

    @staticmethod
    def spacing(grade):
        """USFS-style cross-drain spacing on erodible soil, m: closer on steeper grades
        (assumed fit to the guides' tables: 120 m at 2 %, 40 m at 10 %)."""
        return float(np.clip(240.0 / max(grade, 1.0) ** 0.68, 20.0, 150.0))


@dataclass
class Corridor:
    """One road's detail: a centreline polyline (k, 2) world m, or (k, 3) with the base road's
    height along it (the world's road surface is built from the same line; `base_z`), and its
    features. `grade`:
    the base road's grade along u, a function u -> % (positive uphill), for the potholes'
    sags and the water bars' spacing; None: level."""

    centreline: np.ndarray
    seed: int = 0
    section: Section = field(default_factory=Section)
    roughness: Roughness = field(default_factory=Roughness)
    ruts: Ruts = None
    potholes: Potholes = None
    washboard: tuple = ()
    water_bars: WaterBars = None
    grade: object = None
    rocks: object = None  # rocks.Zones: the rock cover by zone (R4); None: no rocks

    def __post_init__(self):
        line = np.asarray(self.centreline, dtype=float)
        self.z = line[:, 2] if line.shape[1] > 2 else np.zeros(len(line))
        self.centreline = line[:, :2]
        d = np.diff(self.centreline, axis=0)
        self.seg_len = np.hypot(d[:, 0], d[:, 1])
        self.seg_dir = d / self.seg_len[:, None]
        self.seg_u = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.seg_u[-1])
        self.reach = self.section.extent() + 1.0
        self._holes = self._place_potholes()
        self._bars = self._place_bars()

    def base_z(self, u):
        """The base road's height at u along the centreline, m (level across it)."""
        return np.interp(u, self.seg_u, self.z)

    def uv(self, x, y):
        """Each point's (u, v) on the centreline: its nearest segment's (P,), and whether it lies
        within the corridor's reach."""
        p = np.stack([np.ravel(x), np.ravel(y)], axis=1)
        a = self.centreline[:-1]
        best_d = np.full(len(p), np.inf)
        u = np.zeros(len(p))
        v = np.zeros(len(p))
        for s in range(len(a)):  # segments: few near any tile
            rel = p - a[s]
            t = np.clip(rel @ self.seg_dir[s], 0.0, self.seg_len[s])
            foot = a[s] + t[:, None] * self.seg_dir[s]
            d = np.hypot(*(p - foot).T)
            better = d < best_d
            best_d = np.where(better, d, best_d)
            u = np.where(better, self.seg_u[s] + t, u)
            cross = self.seg_dir[s, 0] * rel[:, 1] - self.seg_dir[s, 1] * rel[:, 0]
            v = np.where(better, cross, v)
        return u, v, best_d <= self.reach

    def _rng(self, *key):
        return np.random.default_rng([self.seed & 0xFFFFFFFF, *[int(k) & 0xFFFFFFFF
                                                                for k in key]])

    def _place_potholes(self):
        """(u, v, depth, radius) per pothole, per 100 m segment of the road."""
        if self.potholes is None:
            return np.zeros((0, 4))
        ph = self.potholes
        out = []
        for seg in range(int(math.ceil(self.length / 100.0))):
            u0 = seg * 100.0
            rate = ph.per_100m
            if self.grade is not None:  # a sag: the grade rises through it
                g0, g1 = self.grade(u0), self.grade(min(u0 + 100.0, self.length))
                if g0 < 0 < g1 or (g1 - g0) > 2.0:
                    rate *= ph.sag_weight
            rng = self._rng(1, seg)
            for _ in range(rng.poisson(rate)):
                r = rng.uniform(*ph.diameter) / 2
                out.append((u0 + rng.uniform(0, 100.0),
                            rng.uniform(-self.section.half_width + r,
                                        self.section.half_width - r),
                            rng.uniform(*ph.depth), r))
        return np.array(out).reshape(-1, 4)

    def _place_bars(self):
        if self.water_bars is None:
            return np.zeros(0)
        if self.water_bars.at:
            return np.array(self.water_bars.at, dtype=float)
        out, u = [], 0.0
        while True:
            g = abs(self.grade(u)) if self.grade is not None else 0.0
            u += WaterBars.spacing(g)
            if u >= self.length:
                return np.array(out)
            out.append(u)

    def features(self, u, v):
        """Everything but the roughness at road coordinates (u, v), m."""
        h = self.section.height(v)
        inside = np.abs(v) <= self.section.half_width
        if self.ruts is not None:
            h = h + np.where(inside, self.ruts.height(v), 0.0)
        for (pu, pv, depth, r) in self._holes:
            d = np.hypot(u - pu, v - pv)
            e = self.potholes.edge
            h = h - depth * np.clip((r - d) / e, 0.0, 1.0)
        for zone in self.washboard:
            h = h + np.where(inside, zone.h(u), 0.0)
        if len(self._bars):
            wb = self.water_bars
            t = math.tan(math.radians(wb.angle))
            s = u - v * t  # along the road, the bar's oblique line through v
            for ub in self._bars:
                d = (s - ub) * math.cos(math.radians(wb.angle))
                bar = np.where(np.abs(d) < wb.width / 2,
                               wb.height * np.cos(math.pi * d / wb.width) ** 2, 0.0)
                h = h + np.where(inside, bar, 0.0)
        return h


class Detail:
    """The corridors' detail as tiles: `height(x, y)` m above the base road, any array shape.
    Tiles are made on first use and kept (`max_tiles` most recently used). With `workers`,
    `prefetch` makes the tiles about points in a worker thread, so that a tile takes no frame's
    time when the car reaches it (one takes about 40 ms)."""

    def __init__(self, corridors, spacing=SPACING, tile=TILE, max_tiles=64, workers=0,
                 processes=False, extras=None):
        """`extras`: "module:function", called as function(detail, i, j) after each tile is
        made, where the tile is made (the worker), its result kept per tile (`extra`); rocks'
        placement (`rocks:place`) is one."""
        import threading
        self.lock = threading.RLock()
        self.corridors = list(corridors)
        self.spacing, self.tile = spacing, tile
        self.extras_name, self._extras_fn = extras, None
        self.extras = {}  # (i, j) -> extras' result, kept for the session
        self.pool = None
        if workers and processes:
            self.pool = TileProcess(self.corridors, spacing, tile, extras)
        elif workers:
            import concurrent.futures
            self.pool = concurrent.futures.ThreadPoolExecutor(workers)
        self.pending = {}  # tile -> its worker's future
        self.n = int(round(tile / spacing))
        self.tiles = {}  # (i, j) -> (n + 1, n + 1) heights, rows along y
        self.max_tiles = max_tiles
        self.fields = {}  # (corridor, i, j) -> its roughness field on 2 tile
        self.amps = {}  # corridor -> its spectrum's amplitudes on the field's grid
        self.touched = {}  # (i, j) -> whether a corridor reaches the tile
        self.made_ms = []  # each tile's generation time
        self.stall_ms = []  # tile_of's waits for a tile not made ahead (the caller's thread)

    def _field(self, ci, i, j):
        """The roughness of corridor `ci` over the window of lattice cell (i, j), which spans
        2 tiles from (i, j) tile's origin: a periodic FFT field with the 2D spectrum."""
        key = (ci, i, j)
        with self.lock:
            f = self.fields.get(key)
        if f is not None:
            return f
        m = 2 * self.n
        amp = self.amps.get(ci)
        if amp is None:  # the spectrum's amplitudes, the same for every cell of a corridor
            k = np.fft.fftfreq(m, d=self.spacing)
            kx, ky = np.meshgrid(k, k)
            kappa = np.hypot(kx, ky)
            band = (kappa >= 1.0 / (2 * self.tile)) & (kappa <= N_MAX)
            dk = 1.0 / (m * self.spacing)
            psd2 = self.corridors[ci].roughness.psd2(kappa)
            amp = self.amps[ci] = np.where(band, np.sqrt(psd2 * dk * dk), 0.0)
        rng = self.corridors[ci]._rng(2, i, j)
        z = (rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))) / math.sqrt(2)
        # the real part of the inverse transform of a complex Gaussian spectrum: each mode's
        # variance amp^2, the 2D PSD's integral over the band
        f = np.real(np.fft.ifft2(amp * z)) * m * m * math.sqrt(2)
        with self.lock:
            if len(self.fields) > 2 * self.max_tiles:
                self.fields.pop(next(iter(self.fields)))
            self.fields[key] = f
        return f

    def _make(self, i, j):
        n, d = self.n, self.spacing
        xs = i * self.tile + np.arange(n + 1) * d
        ys = j * self.tile + np.arange(n + 1) * d
        X, Y = np.meshgrid(xs, ys)
        out = np.zeros(X.shape)
        for ci, c in enumerate(self.corridors):
            # a corridor that reaches nowhere near this tile adds nothing
            cx, cy = xs[n // 2], ys[n // 2]
            u, v, near = c.uv(np.array([cx]), np.array([cy]))
            if v.size and abs(v[0]) > c.reach + self.tile:
                continue
            u, v, near = c.uv(X, Y)
            if not near.any():
                continue
            u, v, near = u.reshape(X.shape), v.reshape(X.shape), near.reshape(X.shape)
            h = c.features(u, v)
            # the roughness: four overlapping windows (cells (i-1..i, j-1..j)), sin^2 + cos^2
            # = 1 in each axis so the variance is the field's everywhere
            r = np.zeros(X.shape)
            for di in (-1, 0):
                for dj in (-1, 0):
                    f = self._field(ci, i + di, j + dj)
                    ox = (np.arange(n + 1) - di * n)  # this tile's samples in the window
                    oy = (np.arange(n + 1) - dj * n)
                    wx = np.sin(math.pi * ox / (2 * n))
                    wy = np.sin(math.pi * oy / (2 * n))
                    r += f[np.ix_(oy % (2 * n), ox % (2 * n))] * np.outer(wy, wx)
            # the roughness is the carriageway's and its shoulders', fading out over the ditch
            fade = np.clip((c.section.half_width + c.section.shoulder + 1.0 - np.abs(v)), 0, 1)
            out += np.where(near, h + r * fade, 0.0)
        return out

    def _keep(self, key, t):
        with self.lock:
            self.tiles[key] = t
            while len(self.tiles) > self.max_tiles:
                self.tiles.pop(next(iter(self.tiles)))

    def surface_tile(self, i, j):
        """The tile's samples with the base road under them: what the render mesh shows (the
        corridor's base_z where it reaches; 0 elsewhere, the world's own ground there)."""
        n, d = self.n, self.spacing
        X, Y = np.meshgrid(i * self.tile + np.arange(n + 1) * d,
                           j * self.tile + np.arange(n + 1) * d)
        base = np.zeros(X.shape)
        for c in self.corridors:
            if not c.z.any():
                continue
            u, _, near = c.uv(X, Y)
            base = np.where(near.reshape(X.shape), c.base_z(u).reshape(X.shape), base)
        return base + self.tile_of(i, j)

    def tile_of(self, i, j):
        with self.lock:
            t = self.tiles.pop((i, j), None)
            if t is not None:  # most recently used last
                self.tiles[(i, j)] = t
                return t
            job = self.pending.get((i, j))
        t0 = _now()
        t = None
        if job is not None:
            try:
                t, _, extra = job.result()  # being made: wait for it
            except Exception:  # the worker failed or is gone: make it here
                t = None
        if t is None:
            t, ms, extra = self._timed(i, j)
            self.made_ms.append(ms)
        self._keep((i, j), t)
        if self.extras_name:
            with self.lock:
                self.extras.setdefault((i, j), extra)
        self.stall_ms.append((_now() - t0) * 1e3)
        return t

    def tiles_near(self, x, y, reach):
        """The tiles within `reach` m of the points (x, y), as keys."""
        x, y = np.ravel(x), np.ravel(y)
        keys = set()
        for i0, i1, j0, j1 in zip(np.floor((x - reach) / self.tile).astype(int),
                                  np.floor((x + reach) / self.tile).astype(int),
                                  np.floor((y - reach) / self.tile).astype(int),
                                  np.floor((y + reach) / self.tile).astype(int)):
            keys.update((int(i), int(j)) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1))
        return keys

    def touches(self, i, j):
        """Whether a corridor reaches the tile: one that does not is all zeros."""
        hit = self.touched.get((i, j))
        if hit is None:
            cx, cy = (i + 0.5) * self.tile, (j + 0.5) * self.tile
            hit = False
            for c in self.corridors:
                _, v, _ = c.uv(np.array([cx]), np.array([cy]))
                if not v.size or abs(v[0]) <= c.reach + self.tile:
                    hit = True
                    break
            self.touched[(i, j)] = hit  # a bool per tile: kept for the session
        return hit

    def ready(self, key):
        with self.lock:
            return key in self.tiles

    def prefetch(self, keys):
        """Make these tiles in the worker (at once, without one)."""
        for key in keys:
            with self.lock:
                if key in self.tiles or key in self.pending:
                    continue
                if self.pool is None:
                    job = None
                else:
                    job = self.pending[key] = self.pool.submit(self._timed, *key)
                    job.add_done_callback(lambda f, key=key: self._made(key, f))
            if job is None:
                self.tile_of(*key)

    def _timed(self, i, j):
        t0 = _now()
        t = self._make(i, j)
        extra = None
        if self.extras_name:
            if self._extras_fn is None:
                import importlib

                mod, fn = self.extras_name.split(":")
                self._extras_fn = getattr(importlib.import_module(mod), fn)
            self._keep((i, j), t)  # the extras read the tile's heights
            extra = self._extras_fn(self, i, j)
        return t, (_now() - t0) * 1e3, extra

    def extra(self, i, j):
        """The tile's extras, once made (None before)."""
        with self.lock:
            return self.extras.get((i, j))

    def _made(self, key, job):
        try:
            if not job.cancelled() and job.exception() is None:
                t, ms, extra = job.result()
                self.made_ms.append(ms)
                self._keep(key, t)
                if self.extras_name:
                    with self.lock:
                        self.extras.setdefault(key, extra)
        finally:
            with self.lock:
                self.pending.pop(key, None)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)

    def height(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        fx, fy = x / self.spacing, y / self.spacing
        gx, gy = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
        ti, tj = gx // self.n, gy // self.n
        out = np.empty(x.shape)
        ti, tj, gx, gy = ti.ravel(), tj.ravel(), gx.ravel(), gy.ravel()
        fx, fy = fx.ravel(), fy.ravel()
        flat_out = out.reshape(-1)
        if not ti.size:
            return out
        # a car's samples lie on one to four neighbouring tiles: loop over their box, no sort
        for i in range(int(ti.min()), int(ti.max()) + 1):
            on_i = ti == i
            for j in range(int(tj.min()), int(tj.max()) + 1):
                sel = on_i & (tj == j)
                if not sel.any():
                    continue
                t = self.tile_of(i, j)
                lx, ly = gx[sel] - i * self.n, gy[sel] - j * self.n
                ax, ay = fx[sel] - gx[sel], fy[sel] - gy[sel]
                h00, h10 = t[ly, lx], t[ly, lx + 1]
                h01, h11 = t[ly + 1, lx], t[ly + 1, lx + 1]
                flat_out[sel] = ((h00 * (1 - ax) + h10 * ax) * (1 - ay)
                                 + (h01 * (1 - ax) + h11 * ax) * ay)
        return out


def _now():
    import time
    return time.perf_counter()


class TileProcess:
    """Makes tiles in a child process: a thread in the caller competes for the GIL with the
    simulation loop (in Isaac it made 2 of the 3 tiles a second 80 km/h needs). The child is
    `python -m swb_isaac_world.road_detail --worker`, numpy only, on pipes, not
    multiprocessing: spawn would import the caller's __main__ again (a second SimulationApp),
    fork a process holding CUDA. The executor's `submit` and `shutdown`, for `Detail`."""

    def __init__(self, corridors, spacing, tile, extras=None):
        import os
        import pickle
        import subprocess
        import sys
        import threading

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            [root] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
        self.proc = subprocess.Popen([sys.executable, "-m", "swb_isaac_world.road_detail",
                                      "--worker"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, env=env, cwd=root)
        self.lock = threading.Lock()
        self.jobs = {}  # (i, j) -> future, in the order asked
        pickle.dump((corridors, spacing, tile, extras), self.proc.stdin)
        self.proc.stdin.flush()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def submit(self, fn, i, j):
        import concurrent.futures
        import pickle

        job = concurrent.futures.Future()
        job.set_running_or_notify_cancel()
        with self.lock:
            self.jobs[(i, j)] = job
            try:
                pickle.dump((i, j), self.proc.stdin)
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError) as e:
                self.jobs.pop((i, j))
                job.set_exception(e)
        return job

    def _read(self):
        import pickle

        while True:
            try:
                key, t, ms, extra = pickle.load(self.proc.stdout)
            except Exception as e:  # the child is gone: every waiting tile fails
                with self.lock:
                    jobs, self.jobs = list(self.jobs.values()), {}
                for job in jobs:
                    job.set_exception(RuntimeError(f"road tile process ended: {e!r}"))
                return
            with self.lock:
                job = self.jobs.pop(key, None)
            if job is not None:
                job.set_result((t, ms, extra))

    def shutdown(self, wait=False, cancel_futures=True):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        if wait:
            self.proc.wait()
        else:
            self.proc.terminate()


def _worker():
    """The child of TileProcess: (corridors, spacing, tile, extras), then (i, j) per tile, on
    stdin; (key, heights, ms, extra) per tile on stdout. Ends at stdin's end."""
    import os
    import pickle
    import sys

    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # anything printed goes to stderr, not into the tiles
    inp = sys.stdin.buffer
    corridors, spacing, tile, extras = pickle.load(inp)
    d = Detail(corridors, spacing, tile, max_tiles=16, extras=extras)
    while True:
        try:
            key = pickle.load(inp)
        except EOFError:
            return
        t, ms, extra = d._timed(*key)
        pickle.dump((key, t, ms, extra), out, protocol=pickle.HIGHEST_PROTOCOL)
        out.flush()


# --- measures (R3's acceptance) ---------------------------------------------------------------

def line_psd(h, dx, segment=4096):
    """Welch's displacement PSD of a profile h sampled every dx m: (n cycles/m, G m^3)."""
    seg = min(segment, len(h))
    win = np.hanning(seg)
    step = seg // 2
    acc, k = 0.0, 0
    for s in range(0, len(h) - seg + 1, step):
        x = h[s:s + seg] - h[s:s + seg].mean()
        acc = acc + np.abs(np.fft.rfft(x * win)) ** 2
        k += 1
    n = np.fft.rfftfreq(seg, d=dx)
    return n, 2.0 * acc / k * dx / (win ** 2).sum()


def iri(h, dx, speed=80.0 / 3.6):
    """The International Roughness Index of profile h (m, every dx m), m/km: the golden quarter
    car (k1 653, k2 63.3, c 6.0, mu 0.15 per unit sprung mass) at 80 km/h on the profile's 250 mm
    moving average, its suspension's mean absolute rate over the speed (Sayers 1995)."""
    n = max(1, int(round(0.25 / dx)))
    p = np.convolve(h, np.ones(n) / n, mode="valid")
    k1, k2, c, mu = 653.0, 63.3, 6.0, 0.15
    a = np.array([[0, 1, 0, 0], [-k2, -c, k2, c], [0, 0, 0, 1],
                  [k2 / mu, c / mu, -(k1 + k2) / mu, -c / mu]])
    b = np.array([0, 0, 0, k1 / mu])
    dt = dx / speed
    # exact discretisation (zero-order hold on the profile)
    from scipy.linalg import expm
    m = np.zeros((5, 5))
    m[:4, :4], m[:4, 4] = a * dt, b * dt
    e = expm(m)
    ad, bd = e[:4, :4], e[:4, 4]
    # start at rest on the first 11 m's mean slope (the standard's initialisation)
    j = min(len(p) - 1, int(round(11.0 / dx)))
    slope = (p[j] - p[0]) / (j * dx) if j else 0.0
    x = np.array([p[0], slope * speed, p[0], slope * speed])
    acc = 0.0
    for z in p:
        x = ad @ x + bd * z
        acc += abs(x[1] - x[3])
    return acc * dt / (len(p) * dx) * 1000.0  # (m/s * s) / m -> m/m -> m/km


def iso2631_wk(acc, rate):
    """ISO 2631-1's Wk weighting (vertical, seated) applied to an acceleration signal (m/s^2)
    sampled at `rate` Hz, in the frequency domain; returns the weighted signal."""
    f = np.fft.rfftfreq(len(acc), d=1.0 / rate)
    s = 2j * math.pi * np.maximum(f, 1e-9)
    w = lambda hz: 2 * math.pi * hz
    f1, f2, f3, f4, q4, f5, q5, f6, q6 = 0.4, 100.0, 12.5, 12.5, 0.63, 2.37, 0.91, 3.35, 0.91
    hh = s * s / (s * s + w(f1) / (1 / math.sqrt(2)) * s + w(f1) ** 2)
    hl = w(f2) ** 2 / (s * s + w(f2) / (1 / math.sqrt(2)) * s + w(f2) ** 2)
    ht = (1 + s / w(f3)) / (1 + s / (q4 * w(f4)) + (s / w(f4)) ** 2)
    hs = ((1 + s / (q5 * w(f5)) + (s / w(f5)) ** 2) / (1 + s / (q6 * w(f6)) + (s / w(f6)) ** 2)
          * (w(f5) / w(f6)) ** 2)
    hw = hh * hl * ht * hs
    hw[0] = 0.0
    return np.fft.irfft(np.fft.rfft(acc) * hw, n=len(acc))


if __name__ == "__main__" and "--worker" in __import__("sys").argv:
    _worker()
