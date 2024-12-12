from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "byte_tokenizer",
        ["byte_tokenizer.cpp"],
    ),
]

setup(
    name="byte_tokenizer",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=["pybind11>=2.6.2"],
)
