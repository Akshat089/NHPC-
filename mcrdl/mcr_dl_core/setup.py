from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(
    name="mcrdl",
    ext_modules=[
        CppExtension(
            name="mcrdl",
            sources=["basic.cpp"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-Wall"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
