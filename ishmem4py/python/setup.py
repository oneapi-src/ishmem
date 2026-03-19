# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

from setuptools import setup
from setuptools.dist import Distribution

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    _bdist_wheel = None


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


cmdclass = {}

if _bdist_wheel is not None:
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False


    cmdclass["bdist_wheel"] = bdist_wheel


setup(distclass=BinaryDistribution, cmdclass=cmdclass)
