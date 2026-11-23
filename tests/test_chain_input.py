"""Game-free HOSCHAIN1 input contract checks: python -I -B tests/test_chain_input.py."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import chain


class CaptureInput(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "synthetic.chain"

    def capture(self, data):
        self.path.write_bytes(data)
        return self.path

    def main_capture(self, records):
        path = self.capture(b"HOSCHAIN1\nthread 0 Presenter\nrecords %d\n" % len(records) + b"".join(records))
        old_argv = sys.argv
        self.addCleanup(setattr, sys, "argv", old_argv)
        sys.argv = ["chain.py", str(path)]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            chain.main()
        return output.getvalue()

    def test_single_present_has_no_frame(self):
        with self.assertRaisesRegex(SystemExit, "at least two presents"):
            self.main_capture([chain.REC.pack(100, 0, 0, 0, chain.PRESENT)])

    def test_invalid_header_has_diagnostic(self):
        for header in (b"", b"\n", b"thread\n", b"thread nope Name\n", b"thread 0\n",
                       b"records\n", b"records nope\n", b"records -1\n"):
            with self.subTest(header=header):
                path = self.capture(b"HOSCHAIN1\n" + header)
                with self.assertRaisesRegex(SystemExit, "invalid HOS_CHAIN header"):
                    chain.load(path)

    def test_payload_missing_whole_record(self):
        record = chain.REC.pack(100, 0, 0, 0, chain.PRESENT)
        for payload in (b"", record):
            with self.subTest(size=len(payload)):
                path = self.capture(b"HOSCHAIN1\nrecords 2\n" + payload)
                with self.assertRaisesRegex(SystemExit, "truncated.*records"):
                    chain.load(path)

    def test_zero_presents_keeps_existing_diagnostic(self):
        with self.assertRaisesRegex(SystemExit, "no presents recorded"):
            self.main_capture([])

    def test_one_present_per_thread_has_no_frame(self):
        records = [chain.REC.pack(t, 0, 0, th, chain.PRESENT) for t, th in ((100, 0), (200, 1))]
        with self.assertRaisesRegex(SystemExit, "at least two presents"):
            self.main_capture(records)

    def test_valid_frames_keep_report(self):
        for count in (2, 3):
            with self.subTest(count=count):
                records = [chain.REC.pack(t * 1000000, 0, 0, 0, chain.PRESENT) for t in range(count)]
                output = self.main_capture(records)
                self.assertIn("%d frames presented by Presenter, 1.00 ms a frame (1000.0 fps)" % (count - 1), output)
                self.assertIn("-- the path, a frame (ms): 1.00 in all", output)

    def test_valid_records_keep_values_and_stable_time_order(self):
        records = [(30, 21, 31, 1, chain.WAIT), (10, 22, 32, 0, chain.PRESENT),
                   (30, 23, 33, 1, chain.WOKE), (20, 24, 34, 1, chain.SIGNAL),
                   (40, 25, 35, 0, chain.SLEEP), (50, 26, 36, 0, chain.AWAKE)]
        payload = b"".join(chain.REC.pack(*record) for record in records)
        path = self.capture(b"HOSCHAIN1\nthread 0 Presenter\nthread 1 Worker\nrecords 6\n" + payload)
        names, loaded = chain.load(path)
        self.assertEqual(names, {0: "Presenter", 1: "Worker"})
        self.assertEqual(loaded, sorted(records, key=lambda record: record[0]))

    def test_trailing_bytes_keep_declared_record_boundary(self):
        record = chain.REC.pack(100, 0, 0, 0, chain.PRESENT)
        path = self.capture(b"HOSCHAIN1\nrecords 1\n" + record + b"unused trailing bytes")
        self.assertEqual(chain.load(path), ({}, [chain.REC.unpack(record)]))

    def test_bad_magic_keeps_existing_diagnostic(self):
        for magic in (b"", b"HOSCHAIN2\n"):
            with self.subTest(magic=magic):
                with self.assertRaisesRegex(SystemExit, "not a HOS_CHAIN recording"):
                    chain.load(self.capture(magic))

    def test_truncated_declared_payload(self):
        record = chain.REC.pack(100, 0, 0, 0, chain.PRESENT)
        path = self.capture(b"HOSCHAIN1\nthread 0 Presenter\nrecords 2\n" + record + record[:-1])
        with self.assertRaisesRegex(SystemExit, "truncated.*records"):
            chain.load(path)


if __name__ == "__main__":
    unittest.main()
