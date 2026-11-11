"""Build the compiled kernels in place: ``pixi run build-ext`` (cpu environment)."""

import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

ext = Extension(
    "kmb._emeans_kernel",
    ["kmb/_emeans_kernel.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "build_ext", "--inplace"]
    setup(name="kmb-kernels", ext_modules=cythonize([ext], quiet=True), script_args=sys.argv[1:])
