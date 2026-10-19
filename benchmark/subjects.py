"""The synthetic subjects both benchmarks score. Keep in step with subjects.mjs.

Subject s has an age inside every voxel's training range (no clamping, so
PCNtoolkit and the browser kernel compute the same thing), alternating sex,
and at voxel v a response 1.5 sd-ish away from that voxel's training mean.
"""

import math


def age(s):
    return 50.0 + (s * 7.919) % 25.0


def sex_level(s):
    """PCNtoolkit's Sex level: "0" (female) or "1" (male)."""
    return "0" if s % 2 == 0 else "1"


def response(v, s, out_m, out_s):
    return out_m + out_s * 1.5 * math.sin(0.618 * v + 1.3 * s + 0.1)
