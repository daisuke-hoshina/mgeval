"""Entry point for ``python -m mgeval``.

This shim ensures the repository root ``__main__.py`` is executed when the
package is run as a module, matching the original CLI behavior.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    runpy.run_path(str(repo_root / "__main__.py"), run_name="__main__")


if __name__ == "__main__":
    main()
