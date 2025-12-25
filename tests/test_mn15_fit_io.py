import pathlib
import sys
import tempfile
import unittest

import numpy as np


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))


from xclib.tools.mn15_fit_io import read_mn15_fit_file  # noqa: E402


def _line(name: str, values) -> str:
    return name + " " + " ".join(str(v) for v in values)


class TestReadMN15FitFile(unittest.TestCase):
    def test_reads_dict(self):
        values0 = list(range(78))
        values1 = [0.5] * 78

        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "NC15"
            fp.write_text(
                "\n".join(
                    [
                        "# comment",
                        _line("sys0", values0),
                        "",
                        _line("sys1", values1),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            out = read_mn15_fit_file(fp)
            self.assertEqual(set(out.keys()), {"sys0", "sys1"})
            self.assertEqual(out["sys0"].shape, (78,))
            self.assertTrue(np.allclose(out["sys0"][:3], np.asarray([0, 1, 2], dtype=np.float64)))

    def test_raises_on_wrong_value_count(self):
        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "NC15"
            fp.write_text(_line("bad", [0.0] * 10) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_mn15_fit_file(fp)


if __name__ == "__main__":
    unittest.main()

