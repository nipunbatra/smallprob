"""Rebuild the static showcase from the library and executed notebooks.

Run: python -m examples.build_site
No browser Python runtime or web build dependencies are required.
"""

from pathlib import Path
import shutil
import subprocess
import sys


def build():
    from .build_site_data import build as build_cases

    root = Path(__file__).resolve().parents[1]
    site = root / "website"
    (site / "notebooks").mkdir(parents=True, exist_ok=True)
    (site / "downloads").mkdir(parents=True, exist_ok=True)
    build_cases()
    for name, preview in [
        ("smallprob_examples", "intro"),
        ("smallprob_pyro_gallery", "pyro-gallery"),
    ]:
        notebook = root / "notebooks" / f"{name}.ipynb"
        shutil.copy2(notebook, site / "downloads" / notebook.name)
        subprocess.run(
            [sys.executable, "-m", "examples.export_notebook", str(notebook),
             "--output", str(site / "notebooks" / f"{preview}.html")],
            cwd=root, check=True,
        )
    (site / ".nojekyll").touch()
    print(f"Showcase ready: {site / 'index.html'}")


if __name__ == "__main__":
    build()
