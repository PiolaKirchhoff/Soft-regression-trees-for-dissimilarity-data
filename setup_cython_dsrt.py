"""Build the three Cython extensions used by the DSRT experiments.

Place this file alongside SMO.pyx, util.pyx, and function.pyx, then run:

    python setup_cython_dsrt.py build_ext --inplace

The compiled modules retain their original import names.
"""

from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


SOURCE_DIR = Path(__file__).resolve().parent
MODULES = ("util", "function", "SMO")

# Require all three Cython sources before starting the build.
missing_sources = [
    f"{module}.pyx"
    for module in MODULES
    if not (SOURCE_DIR / f"{module}.pyx").is_file()
]
if missing_sources:
    raise FileNotFoundError(
        "Missing Cython sources in the setup directory: "
        + ", ".join(missing_sources)
    )

extensions = [
    Extension(
        name=module,
        sources=[str(SOURCE_DIR / f"{module}.pyx")],
        include_dirs=[np.get_include()],
    )
    for module in MODULES
]

setup(
    name="dsrt-cython-extensions",
    ext_modules=cythonize(
        extensions,
        annotate=True,
        compiler_directives={
            "linetrace": True,
            "binding": True,
        },
    ),
)
