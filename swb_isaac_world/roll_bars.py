"""The car's anti-roll bars on every world (roadmap 066 R5): Isaac's PhysX vehicle has none.

A real Model 3 has a bar on each axle; the road layer's corner (`road_contact`, R2) has them,
so without these a world with a `road_detail.yaml` and one without drove two different cars
at the limit (055's suite: the dry sine with dwell spun on one and not on the other). On a
world without the road layer, `RollBars` adds them to PhysX's car, before each substep:

- each axle's differential travel from PhysX's own geometry: a wheel on the ground is
  compressed by the frame's height over its ground hit (`road_layer._Car` reads the same), a
  wheel off it is at full droop;
- the bar's force at each wheel (R2's rates, `road_contact.anti_roll_bars`), on the body at
  the wheel's attachment along the body's up, for the next step (`apply_force`), as the
  road layer's corner force goes;
- each wheel's friction scaled by its load with the bar over PhysX's own (`scale`), since
  PhysX's tire saturates at mu times the load its spring gives it: the bar's load transfer
  reaches the tires as a real bar's does.

A few vector operations per car and substep. Pure.
"""
import numpy as np

from . import road_contact as rc

G = 9.81
SCALE_STEP = 0.01  # the friction scale's step: a USD write per wheel only when it moves
# aib.isaac.vehicle.WHEELS' order (front?, side): front left, front right, rear left, rear right
ORDER = ((True, 1), (True, -1), (False, 1), (False, -1))


def qrot(q, v):
    """Vector `v` rotated by the unit quaternion q = (w, x, y, z), as aib.frames.qrot."""
    w, x, y, z = q
    vx, vy, vz = v
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (vx + w * tx + y * tz - z * ty, vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx)


class _Bars:
    def __init__(self, p):
        (front, rear), sag = p.suspension()
        self.frame_z = p.wheel_radius + p.suspension_travel - sag
        self.radius, self.travel, self.wheel_weight = p.wheel_radius, p.suspension_travel, \
            p.wheel_mass * G
        self.attach = [((p.wheel_base if f else 0.0), side * p.wheel_tread / 2)
                       for f, side in ORDER]
        sides = {k: i for i, k in enumerate(ORDER)}
        n = len(ORDER)
        self.k = np.zeros((n, n))  # force at each wheel = k @ jounce
        for i, j, rate in rc.anti_roll_bars((sides[(True, 1)], sides[(True, -1)]),
                                            (sides[(False, 1)], sides[(False, -1)]),
                                            front[1], rear[1]):
            self.k[i, i] += rate
            self.k[j, j] += rate
            self.k[i, j] -= rate
            self.k[j, i] -= rate
        self.scale = np.ones(n)


def jounce(attach_z, hit_z, on_ground, radius, travel):
    """PhysX's compression per wheel, m (0 at full droop): the wheel's centre stands `radius`
    over its ground hit, and the frame is `travel` over the centre at full droop."""
    d = np.asarray(attach_z) - (np.asarray(hit_z) + radius)
    return np.where(on_ground, np.clip(travel - d, 0.0, travel), 0.0)


def forces(k, x):
    """Each wheel's bar force, N, up on the body at its attachment (k @ jounce)."""
    return k @ x


def friction_scale(px_load, bar, wheel_weight, most=rc.FRICTION_SCALE_MAX):
    """The share of the rig's friction each wheel gets so that PhysX's tire, saturating at mu
    times its own load, grips as the load with the bar allows."""
    own = np.asarray(px_load) + wheel_weight
    return np.clip((own + np.asarray(bar)) / np.maximum(own, 1.0), 0.0, most)


class RollBars:
    def __init__(self, apply_force=None):
        self.apply_force = apply_force
        self.cars = {}  # vehicle path -> _Bars

    def before_substep(self, vehicles):
        for v in vehicles:
            if v.last is None or not v.contacts:
                continue
            car = self.cars.get(v.path)
            if car is None:
                car = self.cars[v.path] = _Bars(v.p)
            (px, py, pz), q, _ = v.last
            att = [qrot(q, (x, y, car.frame_z)) for x, y in car.attach]
            on = np.array([c.on_ground for c in v.contacts])
            hit = np.array([c.position[2] for c in v.contacts])
            x = jounce([pz + a[2] for a in att], hit, on, car.radius, car.travel)
            f = forces(car.k, x)
            # in steps of SCALE_STEP: aib writes a wheel's friction to USD only when it moves
            car.scale = np.where(on, np.round(friction_scale([c.load for c in v.contacts], f,
                                                             car.wheel_weight) / SCALE_STEP)
                                 * SCALE_STEP, 1.0)
            if self.apply_force is not None:
                up = qrot(q, (0.0, 0.0, 1.0))
                for i, a in enumerate(att):
                    if f[i] != 0.0:
                        self.apply_force(v.path, tuple(float(f[i]) * u for u in up),
                                         (px + a[0], py + a[1], pz + a[2]))

    def scale(self, path, i):
        car = self.cars.get(path)
        return 1.0 if car is None else float(car.scale[i])

    def forget(self, path):
        self.cars.pop(path, None)
