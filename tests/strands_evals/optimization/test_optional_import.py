"""numpy and scipy are `optimization` extras: core imports must not require them.

Runs in a subprocess with numpy/scipy imports blocked and asserts that
`strands_evals` still imports. Only `strands_evals.optimization.bayesbench`
may depend on the extra.
"""

import subprocess
import sys

BLOCKED_IMPORT_SCRIPT = """
import sys


class BlockScientificStack:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("numpy", "scipy"):
            raise ImportError(f"{name} must only be imported behind the optimization extra")


sys.meta_path.insert(0, BlockScientificStack())

import strands_evals  # noqa: F401  core package must stay numpy-free
"""


def test_core_import_does_not_require_numpy_or_scipy():
    result = subprocess.run(
        [sys.executable, "-c", BLOCKED_IMPORT_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
