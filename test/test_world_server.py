"""The world server's guards against a wedged Kit (docs/issues/011), without Isaac."""
import enum
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swb_isaac_world import world_server  # noqa: E402


class Level(enum.Enum):  # omni.log.Level's names
    INFO = -1
    ERROR = 1
    FATAL = 2


class GuardTest(unittest.TestCase):
    def test_only_a_fatal_message_calls_back(self):
        seen = []
        consume = world_server.fatal_consumer(lambda channel, msg: seen.append((channel, msg)))
        args = ("module", "file.cpp", "func", 1)
        consume("omni.rtx", Level.ERROR, *args, "an error", 0, 0, 0)
        consume("omni.rtx", Level.INFO, *args, "info", 0, 0, 0)
        consume("omni.rtx", Level.FATAL, *args, "Out of resource descriptors!", 0, 0, 0)
        self.assertEqual(seen, [("omni.rtx", "Out of resource descriptors!")])

    def test_the_watchdog_fires_unless_cancelled(self):
        fired = threading.Event()
        world_server.startup_watchdog(0.05, fired.set)
        self.assertTrue(fired.wait(2.0))
        quiet = threading.Event()
        world_server.startup_watchdog(0.2, quiet.set).cancel()
        self.assertFalse(quiet.wait(0.4))


if __name__ == "__main__":
    unittest.main()
