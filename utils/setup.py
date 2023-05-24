from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("nw_align.pyx", 
    annotate=True)
)