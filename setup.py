from setuptools import setup
import io

# Read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="marc",
    packages=["navicat_marc"],
    version="0.1.0",
    description="Modular Analysis of Representative Conformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="rlaplaza, lcmd-epfl",
    author_email="laplazasolanas@gmail.com",
    url="https://github.com/lcmd-epfl/marc/",
    keywords=["compchem"],
    classifiers=["Programming Language :: Python :: 3"],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "networkx",
        "scikit-learn",
        "setuptools",
    ],
    include_package_data=True,
)
