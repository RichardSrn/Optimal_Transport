#! /usr/bin/env python3

"""
An example algorithm to test.
"""

import ot


def ot_emd(hist1, hist2, c):
    coupling = ot.emd(hist1, hist2, M=c)

    return coupling
