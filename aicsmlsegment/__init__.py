# -*- coding: utf-8 -*-

"""Top-level package for aicsmlsegment."""

__author__ = "Jianxu Chen"
__email__ = "jianxuc@alleninstitute.org"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.0.9"


def get_module_version():
    return __version__


from .example import Example  # noqa: F401
