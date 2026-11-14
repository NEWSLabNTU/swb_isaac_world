"""`/swb/wheels`: each ego wheel's ground and tire state, as a ROS topic (roadmap 076 A9).

The world server publishes it on the ego's ROS domain, from the ego's node (aib's `EgoRos`),
when `--wheels-topic` is on (the default). The wheels are sampled after a physics substep, at
most every `PERIOD` s of simulation time (20 Hz; at 60 Hz physics, every third substep), and
each sample is one message, stamped at its simulation time (the ego's clock: sim time plus
BASE_NS, as `/clock`).

The message is a standard `diagnostic_msgs/DiagnosticArray` (no interface package to build),
one `DiagnosticStatus` per wheel:

| field | value |
|---|---|
| `name` | `<ego name>/<wheel>`, the wheel one of `front_left`, `front_right`, `rear_left`, `rear_right` |
| `hardware_id` | the ego's name (the scenario's entity) |
| `level` | OK (0); WARN (1) when the wheel is off the ground |
| `message` | the surface class under the wheel, `none` off the ground |
| `values` | the key/values below, every value a decimal string (`nan` when not known) |

| key | unit | what |
|---|---|---|
| `class` | | surface class (`surface_classes.yaml`), as `message` |
| `zone` | | the surface map's zone under the wheel, empty outside one |
| `friction` | 1 | the peak friction the ground gives the wheel (the recorder's `mu`) |
| `slip_ratio` | 1 | longitudinal slip, PhysX's |
| `slip_angle` | rad | lateral slip angle, PhysX's |
| `sinkage` | m | in soil, how deep the wheel sinks; 0 elsewhere |
| `normal_load` | N | what the ground carries: the suspension's force |

```bash
ros2 topic echo /swb/wheels --field status   # in the ego's domain
```
"""
import math

TOPIC = "/swb/wheels"
PERIOD = 0.05  # s of simulation time between samples (20 Hz)
WHEEL_NAMES = ("front_left", "front_right", "rear_left", "rear_right")  # aib's WHEELS order
KEYS = ("class", "zone", "friction", "slip_ratio", "slip_angle", "sinkage", "normal_load")
OK, WARN = 0, 1  # DiagnosticStatus levels


class Decimator:
    """`due(t)` is true at the first time and then once `period` s have passed since the last
    true one; a time that goes back (a new episode) starts it again."""

    def __init__(self, period=PERIOD):
        self.period = period
        self.last = None

    def due(self, t):
        if self.last is None or t < self.last or t - self.last >= self.period - 1e-9:
            self.last = t
            return True
        return False

    def reset(self):
        self.last = None


def _num(v):
    return float("nan") if v is None else float(v)


def wheel_rows(vehicle, surface=None):
    """One dict per wheel of aib `vehicle` (its `path`, `wheel_paths`, `wheel_states`,
    `contacts`) on `surface` (`SurfaceLayer`, or None): the module docstring's keys, numbers
    as floats (nan when not known), and `on_ground`."""
    ws = surface.wheels.get(vehicle.path) if surface is not None else None
    states = getattr(vehicle, "wheel_states", None)
    contacts = getattr(vehicle, "contacts", None)
    rows = []
    for i in range(len(vehicle.wheel_paths)):
        g = ws.surfaces[i] if ws is not None else None
        out = surface.soil_out.get((vehicle.path, i)) if surface is not None else None
        on_ground = bool(contacts[i].on_ground) if contacts is not None else None
        rows.append({
            "class": g.cls if g is not None else "",
            "zone": (g.zone or "") if g is not None else "",
            "friction": _num(ws.mu[i] if ws is not None else None),
            "slip_ratio": _num(states[i].slip if states is not None else None),
            "slip_angle": _num(states[i].slip_angle if states is not None else None),
            "sinkage": float(out[0].sinkage) if out is not None else 0.0,
            "normal_load": _num(contacts[i].load if contacts is not None else None),
            "on_ground": on_ground,
        })
    return rows


def _text(v):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.6g}"
    return str(v)


def to_message(ego_name, rows, stamp, msgs):
    """A DiagnosticArray of `rows` (`wheel_rows`) at `stamp`; `msgs` gives the classes
    (`DiagnosticArray`, `DiagnosticStatus`, `KeyValue`), so this runs without ROS."""
    arr = msgs.DiagnosticArray()
    arr.header.stamp = stamp
    for i, row in enumerate(rows):
        wheel = WHEEL_NAMES[i] if i < len(WHEEL_NAMES) else f"wheel{i}"
        st = msgs.DiagnosticStatus()
        st.name = f"{ego_name}/{wheel}"
        st.hardware_id = str(ego_name)
        st.level = bytes([WARN if row.get("on_ground") is False else OK])
        st.message = row["class"] or "none"
        st.values = [msgs.KeyValue(key=k, value=_text(row[k])) for k in KEYS]
        arr.status.append(st)
    return arr


class Messages:
    """diagnostic_msgs' classes, imported on first use (Isaac's bundled ROS 2 has them)."""

    def __init__(self):
        from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

        self.DiagnosticArray = DiagnosticArray
        self.DiagnosticStatus = DiagnosticStatus
        self.KeyValue = KeyValue


class Publisher:
    """`/swb/wheels` on an rclpy node (the ego's)."""

    def __init__(self, node, msgs=None):
        from rclpy.qos import QoSProfile, ReliabilityPolicy

        self.msgs = msgs or Messages()
        self.pub = node.create_publisher(
            self.msgs.DiagnosticArray, TOPIC,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))

    def publish(self, ego_name, rows, stamp):
        self.pub.publish(to_message(ego_name, rows, stamp, self.msgs))
