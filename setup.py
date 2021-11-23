
import glob
import os
import sys
from distutils.core import setup

sys.path.insert(
    0, os.path.realpath(os.path.join(os.path.dirname(__file__), "python"))
)

setup(
    name="dynamics",
    description="Differentiable protein dynamics",
    author="Charles Harris",
    version="0.1.0",
    packages=["dynamics"],
    scripts=glob.glob("bin/*"),
    license="For academic usage only",
    test_suite="tests",
)