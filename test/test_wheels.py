"""`/swb/wheels` (roadmap 076 A9): the rows, the decimation and the message; no ROS."""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swb_isaac_world import wheels  # noqa: E402


class FakeMsg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class DiagnosticArray(FakeMsg):
    def __init__(self):
        super().__init__(header=NS(stamp=None), status=[])


MSGS = NS(DiagnosticArray=DiagnosticArray, DiagnosticStatus=FakeMsg, KeyValue=FakeMsg)


def vehicle():
    return NS(path="/World/entities/ego",
              wheel_paths=[f"/w{i}" for i in range(4)],
              wheel_states=[NS(slip=0.1 * i, slip_angle=0.01 * i) for i in range(4)],
              contacts=[NS(load=4000.0 + i, on_ground=i != 3) for i in range(4)])


def surface(v):
    ws = NS(surfaces=[NS(cls="mud_wet", zone="mud_road"), NS(cls="asphalt", zone=None),
                      NS(cls="asphalt", zone=None), None],
            mu=[0.29, 1.12, 1.12, None])
    return NS(wheels={v.path: ws},
              soil_out={(v.path, 0): (NS(sinkage=0.042, resistance=300.0), 1.0)})


class Rows(unittest.TestCase):
    def test_each_wheels_ground_and_tire(self):
        v = vehicle()
        rows = wheels.wheel_rows(v, surface(v))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["class"], "mud_wet")
        self.assertEqual(rows[0]["zone"], "mud_road")
        self.assertAlmostEqual(rows[0]["friction"], 0.29)
        self.assertAlmostEqual(rows[0]["sinkage"], 0.042)
        self.assertEqual(rows[1]["sinkage"], 0.0)
        self.assertAlmostEqual(rows[2]["slip_ratio"], 0.2)
        self.assertAlmostEqual(rows[2]["slip_angle"], 0.02)
        self.assertEqual(rows[2]["normal_load"], 4002.0)
        self.assertEqual(rows[3]["class"], "")
        self.assertTrue(math.isnan(rows[3]["friction"]))
        self.assertFalse(rows[3]["on_ground"])

    def test_no_surface_layer_and_no_step_yet(self):
        v = vehicle()
        v.wheel_states = v.contacts = None
        rows = wheels.wheel_rows(v, None)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(math.isnan(r["slip_ratio"]) and math.isnan(r["normal_load"])
                            for r in rows))
        self.assertIsNone(rows[0]["on_ground"])


class Decimation(unittest.TestCase):
    def test_20_hz_from_60_hz_physics(self):
        d = wheels.Decimator()
        dt = 1 / 60
        due = [n for n in range(1, 61) if d.due(n * dt)]
        self.assertEqual(len(due), 20)
        self.assertEqual(due[:3], [1, 4, 7])

    def test_one_per_frame_when_the_step_is_longer(self):
        d = wheels.Decimator()
        self.assertEqual([d.due(t) for t in (0.1, 0.2, 0.3)], [True, True, True])

    def test_time_going_back_starts_again(self):
        d = wheels.Decimator()
        self.assertTrue(d.due(5.0))
        self.assertFalse(d.due(5.01))
        self.assertTrue(d.due(0.0))


class Message(unittest.TestCase):
    def test_one_status_per_wheel_with_the_documented_keys(self):
        v = vehicle()
        msg = wheels.to_message("ego", wheels.wheel_rows(v, surface(v)), "STAMP", MSGS)
        self.assertEqual(msg.header.stamp, "STAMP")
        self.assertEqual([s.name for s in msg.status],
                         ["ego/front_left", "ego/front_right", "ego/rear_left", "ego/rear_right"])
        fl = msg.status[0]
        self.assertEqual(fl.hardware_id, "ego")
        self.assertEqual(fl.message, "mud_wet")
        self.assertEqual(fl.level, bytes([wheels.OK]))
        values = {kv.key: kv.value for kv in fl.values}
        self.assertEqual(list(values), list(wheels.KEYS))
        self.assertEqual(values["friction"], "0.29")
        self.assertEqual(values["sinkage"], "0.042")
        self.assertEqual(values["normal_load"], "4000")
        rr = msg.status[3]
        self.assertEqual(rr.message, "none")
        self.assertEqual(rr.level, bytes([wheels.WARN]))
        self.assertEqual({kv.key: kv.value for kv in rr.values}["friction"], "nan")

    def test_every_key_is_documented(self):
        for k in wheels.KEYS:
            self.assertIn(f"`{k}`", wheels.__doc__)


if __name__ == "__main__":
    unittest.main()
