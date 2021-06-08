#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple example of a test file using a function.
NOTE: All test file names must have one of the two forms.
- `test_<XYY>.py`
- '<XYZ>_test.py'

Docs: https://docs.pytest.org/en/latest/
      https://docs.pytest.org/en/latest/goodpractices.html#conventions-for-python-test-discovery
"""
import numpy as np


# This test just checks to see if the raw step instantiates and runs
def test_dummy(n=3):
    arr = np.ones((3, 3), dtype=np.uint8)
    assert arr.shape[0] == n
