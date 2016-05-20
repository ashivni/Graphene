from distutils.core import setup, Extension
import numpy.distutils.misc_util

c_ext = Extension("_cGBUtils", ["_cGBUtils.c"])

setup(
    ext_modules=[c_ext],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)

c_ext = Extension("_cPolyUtils", ["_cPolyUtils.c"])

setup(
    ext_modules=[c_ext],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)


# Build using: python setup.py build_ext --inplace
