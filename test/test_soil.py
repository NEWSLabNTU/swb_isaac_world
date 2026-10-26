"""The soil layer's wheel models (047 S3): the survey's trends, on the shipped class table's
sandy loam and on Wong's dry sand (deep: the mechanics show), for aib's default wheel (0.37 m
radius, 0.235 m wide)."""
import math
import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swb_isaac_world import soil, surfaces  # noqa: E402
from swb_isaac_world.soil import SoilError, SoilGrid, tire_slip, wong_reece  # noqa: E402

TABLE = surfaces.load_classes()
SAND = TABLE["sand"].soil
# Wong's dry sand with 047 S5's fit, the table's `sand` until 053 fitted it to a measured tire:
# a wheel sinks 8 cm in it, so the grid's mechanics show plainly.
DEEP = soil.parse_params({"n": 1.1, "k_c": 990.0, "k_phi": 1528430.0, "cohesion": 1040.0,
                          "friction_angle": 28.0, "janosi_k": 0.015, "slip_sinkage": 0.5})
LOAM = TABLE["dirt"].soil
MUD = TABLE["mud_wet"].soil  # unvalidated, a wet clay row (047 D6)
R, B = 0.37, 0.235


def drive(grid, load=4500.0, slip=0.2, speed=5.0, dt=1 / 60, length=6.0, y=0.013, key=0):
    """One wheel straight along +x; the steady second half's mean WheelSoil fields."""
    out, x = [], 0.0
    while x < length:
        grid.begin_step()
        out.append(grid.wheel(key, x, y, 0.0, load, R, B, speed / (1 - slip), speed, dt, slip))
        x += speed * dt
    half = out[len(out) // 2:]
    return {f: sum(getattr(o, f) for o in half) / len(half)
            for f in ("sinkage", "resistance", "thrust", "traction_limit", "load")}


def wr(p, load=4500.0, slip=0.2, speed=5.0):
    return wong_reece(p, load, R, B, speed / (1 - slip), speed, slip=slip)


class ParamsTest(unittest.TestCase):
    def test_the_table_has_soil_for_sand_dirt_snow_and_unvalidated_mud(self):
        self.assertAlmostEqual(SAND.friction_angle, math.radians(16.4))  # 053, the tire's
        self.assertIsNotNone(TABLE["sand_loose"].soil)
        self.assertEqual(LOAM.n, 0.7)
        self.assertIsNotNone(TABLE["snow_loose"].soil)
        self.assertIsNotNone(MUD)
        self.assertIsNotNone(TABLE["mud_wet"].unvalidated)
        self.assertIsNone(TABLE["mud_soft"].unvalidated)
        self.assertIsNone(TABLE["sand"].unvalidated)
        self.assertIsNone(TABLE["asphalt"].soil)
        # 053's fits to He 2020 (sand) and Wismer 1966 (mud), 047 S5's (dirt)
        self.assertEqual((SAND.slip_sinkage, LOAM.slip_sinkage, MUD.slip_sinkage), (2.5, 0.65, 1.41))
        for name, c in TABLE.classes.items():  # every soil class says what backs its row
            self.assertEqual(c.soil is not None, c.calibration is not None, name)
        self.assertIn("he2020_srtt_medium", TABLE["sand"].calibration)

    def test_the_wetter_mud_sinks_deeper_and_holds_a_rear_drive_car(self):
        """Wet clay (053, Wismer 1966): its grip is cohesion's, so it does not fall with the
        load as sand's friction does; the wetter clay sinks deeper and resists more, so its
        driven wheels at 20 % slip pull less than the undriven ones resist (a rear-drive car
        stalls, as on sand_loose), while on the firmer one they pull more. Both sink deeper
        with slip."""
        soft_p = TABLE["mud_soft"].soil
        firm, soft = drive(SoilGrid(MUD)), drive(SoilGrid(soft_p))
        self.assertGreater(soft["sinkage"], 1.5 * firm["sinkage"])
        for p, stalls in ((MUD, False), (soft_p, True)):
            towed = drive(SoilGrid(p), slip=0.0)
            pull = (firm if p is MUD else soft)
            net = (pull["thrust"] - pull["resistance"]) - towed["resistance"]
            self.assertEqual(net < 0, stalls)
            self.assertGreater(drive(SoilGrid(p), slip=0.5)["sinkage"],
                               drive(SoilGrid(p), slip=0.1)["sinkage"])

    def test_rolling_resistance_adds_its_share_of_the_load(self):
        # 053: what a rigid wheel lacks, a tire flexing in soft ground
        extra = soil.SoilParams(**{**SAND.__dict__, "rolling_resistance": 0.05})
        none = soil.SoilParams(**{**SAND.__dict__, "rolling_resistance": 0.0})
        a, b = drive(SoilGrid(extra), slip=0.0), drive(SoilGrid(none), slip=0.0)
        self.assertAlmostEqual(a["resistance"] - b["resistance"], 0.05 * a["load"], delta=1.0)
        self.assertAlmostEqual(wr(extra).resistance - wr(none).resistance,
                               0.05 * wr(extra).load, delta=1.0)

    def test_slip_sinkage_stops_growing_where_its_measurement_ended(self):
        self.assertEqual(soil.slip_factor(0.8, 2.5, 0.3), soil.slip_factor(0.3, 2.5, 0.3))
        self.assertLess(soil.slip_factor(0.2, 2.5, 0.3), soil.slip_factor(0.3, 2.5, 0.3))
        self.assertEqual(soil.slip_factor(0.99), soil.slip_factor(0.95))  # the default
        self.assertEqual(SAND.slip_sinkage_until, 0.3)

    def test_every_problem_is_listed(self):
        with self.assertRaises(SoilError) as e:
            soil.parse_params({"n": 0, "k_c": -1, "k_phi": "x", "friction_angle": 95,
                               "janosi_k": 0.01, "colour": 1}, "sand: soil")
        text = str(e.exception)
        for needle in ("n 0 is not a number > 0", "k_c -1", "k_phi 'x'", "needs cohesion",
                       "friction_angle 95", "unknown key(s) ['colour']", "sand: soil"):
            self.assertIn(needle, text)


class ClosedFormTest(unittest.TestCase):
    """C, Wong-Reece on fresh flat soil."""

    def test_the_soil_carries_the_load(self):
        for p in (DEEP, LOAM):
            self.assertAlmostEqual(wr(p).load, 4500.0, delta=20.0)

    def test_sinkage_and_resistance_grow_with_load_and_slip(self):
        for p in (DEEP, LOAM):
            loads = [wr(p, load=w) for w in (3000.0, 4500.0, 6000.0)]
            self.assertEqual(loads, sorted(loads, key=lambda o: o.sinkage))
            self.assertEqual(loads, sorted(loads, key=lambda o: o.resistance))
            slips = [wr(p, slip=i).sinkage for i in (0.0, 0.2, 0.5)]
            self.assertEqual(slips, sorted(slips))
            self.assertLess(slips[0] * 1.3, slips[2])

    def test_resistance_outgrows_thrust_as_the_wheel_sinks(self):
        ratios = [wr(DEEP, load=w).resistance / wr(DEEP, load=w).thrust
                  for w in (3000.0, 4500.0, 6000.0)]
        self.assertEqual(ratios, sorted(ratios))

    def test_soft_soil_grips_less_than_asphalt(self):
        for p in (SAND, TABLE["sand_loose"].soil, LOAM, DEEP):
            mu = wr(p).mu(4500.0)
            self.assertLess(mu, TABLE["asphalt"].mu)
            self.assertGreater(mu, 0.2)  # sand's measured on a road tire: about 0.25-0.3

    def test_a_locked_wheel_pulls_back_and_reversing_mirrors(self):
        locked = wong_reece(DEEP, 4500.0, R, B, 0.0, 5.0, slip=-1.0)
        self.assertLess(locked.thrust, -0.9 * locked.traction_limit)
        fwd, back = wr(DEEP, speed=3.0), wr(DEEP, speed=-3.0)
        self.assertAlmostEqual(fwd.sinkage, back.sinkage)
        self.assertAlmostEqual(fwd.thrust, -back.thrust)

    def test_no_load_no_soil(self):
        self.assertEqual(wr(DEEP, load=0.0).sinkage, 0.0)


class FeatherTest(unittest.TestCase):
    def test_edge_weights_match_the_pointwise_weight_and_feather_the_edge(self):
        poly = [[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [0.0, 4.0]]
        xs, ys = np.linspace(0.0, 1.6, 33), np.linspace(1.0, 2.6, 33)
        w = soil.edge_weights(poly, xs, ys, 1.0)
        self.assertEqual(w.shape, (33, 33))
        for i, j in ((0, 0), (5, 7), (20, 32), (32, 16)):
            self.assertAlmostEqual(w[i, j], soil.edge_weight(poly, xs[i], ys[j], 1.0))
        self.assertEqual(w[0, 10], 0.0)  # on the edge
        self.assertAlmostEqual(w[10, 20], 0.5)  # 0.5 m in
        self.assertEqual(w[32, 20], 1.0)  # 1.6 m in


class GridTest(unittest.TestCase):
    """A, the sparse grid."""

    def test_sinkage_grows_with_load_and_slip(self):
        z = [drive(SoilGrid(DEEP), load=w)["sinkage"] for w in (3000.0, 4500.0, 6000.0)]
        self.assertEqual(z, sorted(z))
        z = [drive(SoilGrid(DEEP), slip=i)["sinkage"] for i in (0.0, 0.2, 0.5)]
        self.assertEqual(z, sorted(z))
        # 047 S5: Chrono's SCM (treaded wheel) and CRM sink 1.26-1.41 times deeper at 50 %
        # slip than at none; Lyasko's factor as published gave 1.7
        self.assertTrue(1.2 < z[2] / z[0] < 1.45, z[2] / z[0])

    def test_a_creeping_wheel_digs_no_deeper_than_a_fast_one_at_the_same_tire_slip(self):
        # docs/issues/010: PhysX's tire pulls at slip 0.17 = (r omega - v) / (v + 4), 0.7 m/s
        # of overspeed at a creep, which as real slip (0.87) dug the wheel in
        z = {}
        for v in (0.1, 7.5):
            g, x, out, spin = SoilGrid(MUD), 0.0, [], v + 0.17 * (v + 4.0)
            while x < 4.0:
                g.begin_step()
                out.append(g.wheel(0, x, 0.013, 0.0, 4300.0, R, B, spin, v, 1 / 60,
                                   tire_slip(spin, v, 4.0)).sinkage)
                x += v / 60
            z[v] = sum(out[len(out) // 2:]) / (len(out) - len(out) // 2)
        self.assertLess(z[0.1], 1.02 * z[7.5], z)

    def test_the_tire_slip_is_the_real_slip_above_the_floor(self):
        self.assertAlmostEqual(tire_slip(10.0, 8.0, 4.0), 0.2)
        self.assertAlmostEqual(tire_slip(6.0, 8.0, 4.0), -0.25)
        self.assertAlmostEqual(tire_slip(0.8, 0.1, 4.0), 0.175)
        self.assertAlmostEqual(tire_slip(4.0, 0.0, 4.0), 1.0)  # spinning in place

    def test_the_rut_stays_and_a_second_pass_sinks_less_and_rolls_easier(self):
        for p in (DEEP, LOAM):
            g = SoilGrid(p)
            first = drive(g)
            rut = g.depth(3.0, 0.0)
            self.assertGreater(rut, 0.8 * first["sinkage"])
            second = drive(g)
            # sinkage below the undisturbed surface: the second pass adds less than the first
            self.assertLess(second["sinkage"] - rut, 0.5 * first["sinkage"])
            self.assertLess(second["resistance"], 0.8 * first["resistance"])

    def test_resistance_outgrows_thrust_as_the_wheel_sinks(self):
        r = [drive(SoilGrid(DEEP), load=w) for w in (3000.0, 6000.0)]
        self.assertLess(r[0]["resistance"] / r[0]["traction_limit"],
                        r[1]["resistance"] / r[1]["traction_limit"])

    def test_first_pass_is_near_the_closed_form(self):
        # Not the same stress field: Wong-Reece's normal stress falls to 0 at the bottom of the
        # wheel (exit angle 0) while the grid's nodes there carry the most, so the grid sinks
        # less and pushes less soil (sand 0.72 and 0.53 of it, loam 0.60 and 0.40); S5's
        # calibration against Chrono decides which one to trust. The grip agrees.
        for p in (DEEP, LOAM):
            g, c = drive(SoilGrid(p)), wr(p)
            self.assertAlmostEqual(g["load"], 4500.0, delta=45.0)
            for f, lo in (("sinkage", 0.5), ("resistance", 0.35)):
                self.assertTrue(lo < g[f] / getattr(c, f) < 1.0, (f, g[f], getattr(c, f)))
            self.assertAlmostEqual(g["traction_limit"] / c.traction_limit, 1.0, delta=0.3)

    def test_the_physics_rate_does_not_matter(self):
        # D4: the sweep integrates the patch along the path, so 60 Hz reads as 600 Hz.
        for speed in (5.0, 25.0):
            slow = drive(SoilGrid(DEEP), slip=0.1, speed=speed, dt=1 / 60, length=12.0)
            fast = drive(SoilGrid(DEEP), slip=0.1, speed=speed, dt=1 / 600, length=12.0)
            self.assertAlmostEqual(slow["sinkage"] / fast["sinkage"], 1.0, delta=0.03)
            self.assertAlmostEqual(slow["resistance"] / fast["resistance"], 1.0, delta=0.08)

    def test_it_repeats(self):
        a, b = SoilGrid(DEEP), SoilGrid(DEEP)
        self.assertEqual(drive(a), drive(b))
        for k in a.blocks:
            np.testing.assert_array_equal(a.blocks[k].sinkage, b.blocks[k].sinkage)

    def test_blocks_are_allocated_only_where_wheels_went(self):
        g = SoilGrid(DEEP)
        drive(g, length=40.0, speed=10.0)
        # a 0.235 m track along y = 0.013 touches two block rows (block = 0.8 m)
        rows = {by for _, by in g.blocks}
        self.assertLessEqual(len(rows), 2)
        self.assertLess(g.nbytes() / 40.0 * 1000.0, 12e6)  # per km of one wheel's track
        self.assertEqual(g.depth(10.0, 5.0), 0.0)

    def test_dirty_blocks_are_handed_out_once(self):
        g = SoilGrid(DEEP)
        drive(g, length=3.0)
        dirty = g.dirty()
        self.assertTrue(dirty)
        self.assertGreater(max(float(v.max()) for v in dirty.values()), 0.05)
        self.assertEqual(g.dirty(), {})

    def test_tile_heights_span_blocks_and_share_edges(self):
        g = SoilGrid(DEEP)
        drive(g, length=3.0, y=0.4)
        n = 32  # a 1.6 m tile
        left, right = g.heights(0, 0, n), g.heights(n, 0, n)
        self.assertEqual(left.shape, (n + 1, n + 1))
        np.testing.assert_array_equal(left[n, :], right[0, :])
        self.assertAlmostEqual(float(left[20, 8]), g.depth(20 * 0.05, 8 * 0.05), places=6)
        self.assertEqual(float(g.heights(0, 64, n).max()), 0.0)
        keys = g.dirty_blocks()
        self.assertTrue(keys)
        g.handed_out(keys[:1])
        self.assertEqual(len(g.dirty_blocks()), len(keys) - 1)

    def test_a_teleport_is_not_swept(self):
        g = SoilGrid(DEEP)
        g.begin_step()
        g.wheel(0, 0.0, 0.0, 0.0, 4500.0, R, B, 5.0, 5.0, 1 / 60, 0.0)
        g.begin_step()
        g.wheel(0, 20.0, 0.0, 0.0, 4500.0, R, B, 5.0, 5.0, 1 / 60, 0.0)
        self.assertEqual(g.depth(10.0, 0.0), 0.0)

    def test_saved_ruts_load_into_a_fresh_grid(self):
        g = SoilGrid(DEEP)
        first = drive(g)
        fresh = SoilGrid(DEEP)
        fresh.load(g.save())
        self.assertEqual(fresh.dirty(), {})  # loaded ruts are already in the collision
        self.assertLess(drive(fresh)["resistance"], 0.8 * first["resistance"])

    def test_a_dry_run_changes_nothing(self):
        g = SoilGrid(DEEP)
        g.begin_step()
        out = g.wheel(0, 0.0, 0.0, 0.0, 4500.0, R, B, 5.0, 5.0, 1 / 60, 0.0, commit=False)
        self.assertGreater(out.sinkage, 0.0)
        self.assertEqual(g.blocks, {})

    def test_too_heavy_bottoms_out(self):
        g = SoilGrid(DEEP)
        g.begin_step()
        out = g.wheel(0, 0.0, 0.0, 0.0, 2e6, R, B, 0.0, 0.0, 1 / 60, 0.0)
        self.assertAlmostEqual(out.sinkage, soil.MAX_SINKAGE, places=6)
        self.assertLess(out.load, 2e6)

    def test_grip_is_held_at_the_measured_tires_and_a_layers_base_grips(self):
        # issue 012: a tire does not shear the soil to its strength; on a layer's base the
        # base's share of the load grips at the class's mu too
        out = soil.WheelSoil(0.02, 100.0, 0.0, 2000.0, 4000.0)
        self.assertAlmostEqual(soil.grip(out, 4000.0, 0.3, DEEP), 0.3)
        self.assertAlmostEqual(soil.grip(out, 4000.0, 0.6, DEEP), 0.5)
        layer = soil.SoilParams(**{**DEEP.__dict__, "sinkage_limit": 0.05})
        bottomed = soil.WheelSoil(0.05, 100.0, 0.0, 50.0, 1000.0)
        self.assertAlmostEqual(soil.grip(bottomed, 4000.0, 0.25, layer),
                               (50.0 + 3000.0 * 0.25) / 4000.0)
        self.assertAlmostEqual(soil.grip(bottomed, 4000.0, 0.25, DEEP), 50.0 / 4000.0)
        self.assertEqual(soil.grip(bottomed, 0.0, 0.25, layer), 0.0)

    def test_a_layer_on_a_hard_base_sinks_no_deeper_than_its_limit(self):
        # 053: fresh snow compacts to its final density and the base carries the rest
        layer = soil.SoilParams(**{**DEEP.__dict__, "sinkage_limit": 0.05})
        g = SoilGrid(layer)
        g.begin_step()
        out = g.wheel(0, 0.0, 0.0, 0.0, 2e6, R, B, 0.0, 0.0, 1 / 60, 0.0)
        self.assertAlmostEqual(out.sinkage, 0.05, places=6)
        self.assertLess(out.load, 2e6)
        self.assertAlmostEqual(wr(layer, load=2e6).sinkage, 0.05, places=3)
        self.assertEqual(soil.SoilParams(**{**DEEP.__dict__, "sinkage_limit": 1.0}).deepest(),
                         soil.MAX_SINKAGE)


try:
    importlib.import_module("swb_isaac_world.soil_warp")  # Isaac's Python has warp
    HAVE_WARP = True
except ImportError:
    HAVE_WARP = False


@unittest.skipUnless(HAVE_WARP, "warp: run under Isaac's Python")
class WarpTest(unittest.TestCase):
    """The compiled sweep is the numpy one: float32 against float64, and the wheel bottom by
    bisection against interpolated candidates (nodes within 0.2 mm, measured)."""

    def test_the_kernels_agree(self):
        for p in (DEEP, LOAM):
            for slip, speed in ((0.2, 5.0), (0.1, 25.0), (-0.3, 8.0)):
                a, b = SoilGrid(p, kernel="numpy"), SoilGrid(p, kernel="warp")
                ra = drive(a, slip=slip, speed=speed, length=8.0)
                rb = drive(b, slip=slip, speed=speed, length=8.0)
                for f in ra:
                    self.assertAlmostEqual(rb[f] / ra[f], 1.0, delta=2e-3, msg=(f, slip, speed))
                self.assertEqual(set(a.blocks), set(b.blocks))
                for k in a.blocks:
                    np.testing.assert_allclose(a.blocks[k].sinkage, b.blocks[k].sinkage,
                                               atol=5e-4)

    def test_compiled_is_fast(self):
        import time
        g = SoilGrid(DEEP, kernel="warp")
        drive(g, speed=25.0, length=2.0)  # compiled and warm
        t0 = time.perf_counter()
        n = 0
        x = 2.0
        while x < 20.0:
            g.begin_step()
            g.wheel(0, x, 0.013, 0.0, 4500.0, R, B, 27.7, 25.0, 1 / 60, 1 - 25.0 / 27.7)
            x += 25.0 / 60
            n += 1
        self.assertLess((time.perf_counter() - t0) / n, 400e-6)  # 141 us measured, loaded host



class SidewallTest(unittest.TestCase):
    """054 T2: the soil against a sunk tire's side."""

    def test_bearing_factors_are_terzaghis(self):
        nc, ng = soil.bearing_factors(0.0)
        self.assertAlmostEqual(nc, 2.0 + math.pi, places=6)  # Prandtl's 5.14
        self.assertEqual(ng, 0.0)
        nc, ng = soil.bearing_factors(math.radians(30.0))
        self.assertAlmostEqual(nc, 30.14, delta=0.05)  # the tables' N_c, N_gamma (Vesic)
        self.assertAlmostEqual(ng, 22.40, delta=0.05)

    def test_it_grows_with_the_height_and_nothing_without(self):
        length = soil.contact_length(R, 0.04)
        self.assertAlmostEqual(length, 2 * math.sqrt(2 * R * 0.04 - 0.04 ** 2))
        self.assertEqual(soil.sidewall(MUD, 0.0, length), 0.0)
        f2, f4 = (soil.sidewall(MUD, z, length) for z in (0.02, 0.04))
        self.assertGreater(f4, 2 * f2)  # cohesion linear, weight quadratic in the height
        self.assertAlmostEqual(f4, 1464.0, delta=5.0)  # mud at 4 cm: worked by hand
        self.assertGreater(soil.sidewall(DEEP, 0.04, length), 0.0)

    def test_fresh_soil_stands_at_the_sinkage_both_sides(self):
        o = wr(MUD)
        self.assertEqual(o.side, (o.sinkage, o.sinkage))
        g = SoilGrid(MUD, kernel="numpy")
        out = None
        for n in range(30):
            g.begin_step()
            out = g.wheel(0, n * 5.0 / 60, 0.013, 0.0, 4500.0, R, B, 5.0 / 0.8, 5.0, 1 / 60, 0.2)
        self.assertGreater(out.sinkage, 0.01)
        for h in out.side:
            self.assertAlmostEqual(h, out.sinkage, delta=1e-6)

    def test_an_old_rut_beside_the_tire_lowers_that_side(self):
        g = SoilGrid(MUD, kernel="numpy")
        drive(g, y=0.013 - 0.3, key=1)  # a rut 30 cm to the right
        out = None
        for n in range(30):  # a second wheel beside it, its right side over the old rut's
            g.begin_step()
            out = g.wheel(2, n * 5.0 / 60, 0.013 - 0.3 + B, 0.0, 4500.0, R, B, 5.0 / 0.8,
                          5.0, 1 / 60, 0.2)
        left, right = out.side
        self.assertAlmostEqual(left, out.sinkage, delta=1e-6)
        self.assertLess(right, left)

class EdgeTest(unittest.TestCase):
    def test_the_soil_fades_in_from_the_zone_edge(self):
        sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
        self.assertAlmostEqual(soil.edge_distance(sq, 5, 1), 1.0)
        self.assertAlmostEqual(soil.edge_weight(sq, 0.0, 5.0, 0.74), 0.0)
        self.assertAlmostEqual(soil.edge_weight(sq, 0.37, 5.0, 0.74), 0.5)
        self.assertEqual(soil.edge_weight(sq, 5.0, 5.0, 0.74), 1.0)
        self.assertEqual(soil.edge_weight(sq, 0.1, 5.0, 0.0), 1.0)


if __name__ == "__main__":
    unittest.main()
