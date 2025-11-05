from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="mcrdl",
    version="0.1.0",
    author="Akshat Betala",
    description="Minimal PyTorch + PyBind11 distributed backend example",
    ext_modules=[
        CppExtension(
            name="mcrdl",
            sources=["basic.cpp"],
            extra_compile_args=["-O3", "-std=c++17", "-Wall"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
