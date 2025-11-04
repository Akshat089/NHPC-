from setuptools import setup, Extension
import pybind11
from pybind11.setup_helpers import build_ext

ext_modules = [
    Extension(
        "mcrdl",  # module name (import mcrdl)
        ["basic.cpp"],  # source files
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3", "-Wall"],
    ),
]

setup(
    name="mcrdl",
    version="0.1.0",
    author="Akshat Betala",
    description="Minimal PyBind11 example for C++ Comm backend",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
