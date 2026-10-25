"""OKLCH colour maths (Björn Ottosson, 2020) used for every derived tone in the brand kit.

Shades and tints are made by moving L/C/h in OKLCH, so they stay perceptually even and keep
their hue instead of drifting toward grey or toward a neighbouring colour.
"""
from __future__ import annotations

import math


def _s2l(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _l2s(c):
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def hex_to_oklch(h):
    r, g, b = (_s2l(v / 255) for v in hex_to_rgb(h))
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = (math.copysign(abs(v) ** (1 / 3), v) for v in (l, m, s))
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return L, math.hypot(a, bb), math.degrees(math.atan2(bb, a)) % 360


def _oklch_to_lin(L, C, H):
    a = C * math.cos(math.radians(H))
    b = C * math.sin(math.radians(H))
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
            -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
            -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_)


def in_gamut(L, C, H):
    return all(-1e-5 <= v <= 1 + 1e-5 for v in _oklch_to_lin(L, C, H))


def oklch_to_hex(L, C, H):
    """OKLCH -> sRGB hex. Out-of-gamut colours keep L and h and lose only chroma."""
    L = max(0.0, min(1.0, L))
    C = max(0.0, C)
    if not in_gamut(L, C, H):
        lo, hi = 0.0, C
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if in_gamut(L, mid, H) else (lo, mid)
        C = lo
    rgb = (_l2s(min(1.0, max(0.0, v))) for v in _oklch_to_lin(L, C, H))
    return "#%02X%02X%02X" % tuple(round(min(1.0, max(0.0, v)) * 255) for v in rgb)


def tone(hexc, dL=0.0, dC=0.0, dh=0.0):
    L, C, H = hex_to_oklch(hexc)
    return oklch_to_hex(L + dL, C + dC, H + dh)


def mix(a, b, t):
    """Linear-light mix of two hex colours (t=0 -> a, t=1 -> b): an opaque pre-blend."""
    ca = [_s2l(v / 255) for v in hex_to_rgb(a)]
    cb = [_s2l(v / 255) for v in hex_to_rgb(b)]
    return "#%02X%02X%02X" % tuple(round(_l2s(x + (y - x) * t) * 255) for x, y in zip(ca, cb))


def luminance(h):
    r, g, b = (_s2l(v / 255) for v in hex_to_rgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    la, lb = sorted((luminance(a), luminance(b)))
    return (lb + 0.05) / (la + 0.05)
