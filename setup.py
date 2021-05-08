#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

PACKAGE_NAME = "aicsmlsegment"
MODULE_VERSION = ""


setup_requirements = [
    "pytest-runner>=5.2",
]

test_requirements = [
    "black>=19.10b0",
    "codecov>=2.1.4",
    "flake8>=3.8.3",
    "flake8-debugger>=3.2.1",
    "pytest>=5.4.3",
    "pytest-cov>=2.9.0",
    "pytest-raises>=0.11",
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "bump2version>=1.0.1",
    "coverage>=5.1",
    "ipython>=7.15.0",
    "m2r>=0.2.1",
    "pytest-runner>=5.2",
    "Sphinx>=2.0.0b1,<3",
    "sphinx_rtd_theme>=0.4.3",
    "tox>=3.15.2",
    "twine>=3.1.1",
    "wheel>=0.34.2",
]

requirements = [
    "numpy>=1.15.1",
    "scipy>=1.1.0",
    "scikit-image",
    "pandas>=0.23.4",
    "aicsimageio>3.3.0",
    "tqdm",
    "pyyaml",
    "monai>=0.4.0",
    "pytorch-lightning",
]

extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "all": [
        *requirements,
        *dev_requirements,
    ],
}

setup(
    author="AICS",
    author_email="jianxuc@alleninstitute.org",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: Free for non-commercial use",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="Scripts for ML structure segmentation.",
    entry_points={
        "console_scripts": [
            "dl_train={}.bin.train:main".format(PACKAGE_NAME),
            "dl_predict={}.bin.predict:main".format(PACKAGE_NAME),
            "curator_merging={}.bin.curator.curator_merging:main".format(PACKAGE_NAME),
            "curator_sorting={}.bin.curator.curator_sorting:main".format(PACKAGE_NAME),
            "curator_takeall={}.bin.curator.curator_takeall:main".format(PACKAGE_NAME),
        ]
    },
    install_requires=requirements,
    license="Allen Institute Software License",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="aicsmlsegment",
    name=PACKAGE_NAME,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.7",
    setup_requires=setup_requirements,
    test_suite="aicsmlsegment/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/AllenCell/aics-ml-segmentation",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version=MODULE_VERSION,
    zip_safe=False,
)
