#!/usr/bin/env python3
"""Vector exports of the brand artwork for Adobe Illustrator and print.

API
---
    export_rgb(svg_text, out_basename, **opts)
        -> <base>.pdf, <base>.ai, <base>.eps            RGB, vector, gradients as smooth shadings
    export_cmyk(shapes, size, out_basename, title, **opts)
        -> <base>_CMYK.pdf, <base>_CMYK.ai, <base>_CMYK.eps
           shapes = [(name, svg_path_d, (c, m, y, k)), ...]  floats 0..1, painted in order;
           an optional 4th item {"fill_rule": "evenodd", "clip": d | [d, ...]} per shape.
    srgb_to_cmyk(hex)  -> (c, m, y, k)   ICC: sRGB -> Ghostscript default_cmyk.icc
                                          ("Artifex CMYK SWOP Profile"), relative colorimetric + BPC
    svg_to_cmyk_shapes(svg_text) -> (shapes, size) ready for export_cmyk (flat solid-fill art only)

CLI:  python3 vector_export.py rgb   in.svg OUT_BASE [--title T] [--scale S] [--flat-groups] [--ppi N]
      python3 vector_export.py cmyk  in.svg OUT_BASE [--title T] [--scale S]
      python3 vector_export.py color '#EE3F2A' '#3B81F1' ...
      python3 vector_export.py trace OUT_DIR          (exports the logo trace, RGB + CMYK)

Input: "layered SVG"
--------------------
Root <svg viewBox="0 0 W H">; <g> groups (nested) are named layers -- the name is inkscape:label, else
data-name, else id. Leaves: path (M/L/H/V/C/S/Q/T/A/Z, abs + rel), polygon, polyline, rect (rx/ry),
circle, ellipse, line. Paint: hex/rgb()/rgba()/hsl()/named colours, currentColor, url(#gradient)
(linear/radial, stops + stop-opacity, gradientUnits, gradientTransform, spreadMethod, href
inheritance). opacity, fill-opacity, fill-rule, stroke*, transform (translate/scale/rotate/skew/
matrix), clip-path (+clipPathUnits, clip-rule), mask (luminance/alpha, maskUnits,
maskContentUnits), filter, <use>, nested <svg>, presentation attributes, style="" and simple
<style> CSS (tag/.class/#id compounds, descendant/child combinators).
Not supported (skipped with a warning): <text> (outline it first), <image>, pattern fills, markers.

Output conventions
------------------
* Page = artboard: MediaBox = CropBox = BleedBox = TrimBox = ArtBox = [0 0 W*scale H*scale].
  scale defaults to 1 pt per SVG user unit (1024 px -> 1024 pt; Illustrator's own px = pt rule).
* .ai is a byte-identical copy of the .pdf: a PDF-compatible Illustrator file, i.e. exactly what
  Inkscape writes for .ai (Illustrator opens it through its PDF importer). It carries no
  AIPrivateData and no OCG layers (Illustrator ignores OCGs). Info + XMP name this tool, not Adobe.
* Every SVG leaf becomes its own path object with its own paint operator. Fills are written in
  absolute page coordinates with 4 decimals, so vertices shared by adjacent facets are
  byte-identical (no geometric hairline gaps).
* Group structure: each *named* SVG group becomes a Form XObject (PDF's "group of graphics"
  object) invoked inside marked content  /SVGGroup <</Name (layer name)>> BDC .. EMC ; leaves are
  tagged /SVGShape <</Name (..)>>. A Form is a transparency group only where SVG compositing needs
  one (group opacity, mask, filter), so flat art stays transparency-free. group_mode="flat"
  inlines everything instead.
* Gradients -> shading patterns (ShadingType 2 axial / 3 radial, stitched Type 2 functions,
  Extend = pad; repeat/reflect are tiled into the function). Non-uniform stop-opacity -> a
  luminosity soft mask holding a DeviceGray shading of the same geometry. Everything stays vector.
* clip-path -> W / W* clipping paths (several clip children are unioned with the nonzero rule
  after orientation normalisation). mask -> soft mask whose content is vector.
* filter -> the ONLY rasterised element type in the PDF: the filtered element is rendered by
  librsvg (rsvg-convert, the reference renderer) at `raster_ppi` (default 300 ppi at artwork size)
  into an RGB image + alpha SMask placed exactly over the filter region. If the filter output is
  "effect underneath SourceGraphic" (an feMerge whose last node is SourceGraphic, or a single
  feDropShadow) only the effect is rasterised and the element itself is drawn on top as vectors.
* EPS (PostScript has no transparency): opaque art is pure vector (fills, clips, shfill shadings),
  %%BoundingBox / %%HiResBoundingBox = the artboard. Transparent units (opacity < 1, alpha
  gradients, masks, filters) are flattened against white paper the way Illustrator's flattener
  does: a solid/gradient element with alpha that overlaps nothing painted before it stays vector
  (colours pre-blended with white); anything else is rendered by librsvg at `flatten_ppi` and placed
  under a vector clip of the element's own outline (single leaves) or a 1-bit coverage mask
  (groups, masks, filters). Every flattened unit is listed in the returned report.
"""
from __future__ import annotations

import base64
import copy
import datetime as _dt
import hashlib
import io
import math
import os
import re
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
import zlib

__version__ = "1.0"

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)
ET.register_namespace("inkscape", INKSCAPE_NS)

GS_ICC_DIR = "/usr/share/color/icc/ghostscript"
CMYK_PROFILE = os.path.join(GS_ICC_DIR, "default_cmyk.icc")   # "Artifex CMYK SWOP Profile"
DEFAULT_CREATOR = "K brand kit build - vector_export.py"
DEFAULT_PRODUCER = f"vector_export.py {__version__} (hand-written PDF/PostScript, Python)"

WARNINGS: list = []


def _warn(msg):
    if msg not in WARNINGS:
        WARNINGS.append(msg)


# =========================================================================== numbers / matrices
def _n(v, nd=4):
    """Compact fixed-point number (PDF/PS reals must not use exponents)."""
    if v != v or abs(v) == float("inf"):
        raise ValueError("non-finite coordinate")
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "", "-") else s


I = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def mmul(m, n):
    """m o n : apply n first, then m.  Matrices are (a b c d e f) as in SVG/PDF."""
    a, b, c, d, e, f = m
    A, B, C, D, E, F = n
    return (a * A + c * B, b * A + d * B, a * C + c * D, b * C + d * D,
            a * E + c * F + e, b * E + d * F + f)


def mapply(m, x, y):
    return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])


def minv(m):
    a, b, c, d, e, f = m
    det = a * d - b * c
    if abs(det) < 1e-12:
        return None
    return (d / det, -b / det, -c / det, a / det, (c * f - d * e) / det, (b * e - a * f) / det)


def mtranslate(x, y):
    return (1.0, 0.0, 0.0, 1.0, x, y)


def mscale(sx, sy=None):
    return (sx, 0.0, 0.0, sx if sy is None else sy, 0.0, 0.0)


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_NUM_RE = re.compile(_NUM)
_TF_RE = re.compile(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)")


def parse_transform(s):
    m = I
    for name, args in _TF_RE.findall(s or ""):
        v = [float(x) for x in _NUM_RE.findall(args)]
        if name == "matrix" and len(v) >= 6:
            t = tuple(v[:6])
        elif name == "translate" and v:
            t = mtranslate(v[0], v[1] if len(v) > 1 else 0.0)
        elif name == "scale" and v:
            t = mscale(v[0], v[1] if len(v) > 1 else v[0])
        elif name == "rotate" and v:
            a = math.radians(v[0])
            t = (math.cos(a), math.sin(a), -math.sin(a), math.cos(a), 0.0, 0.0)
            if len(v) >= 3:
                t = mmul(mtranslate(v[1], v[2]), mmul(t, mtranslate(-v[1], -v[2])))
        elif name == "skewX" and v:
            t = (1.0, 0.0, math.tan(math.radians(v[0])), 1.0, 0.0, 0.0)
        elif name == "skewY" and v:
            t = (1.0, math.tan(math.radians(v[0])), 0.0, 1.0, 0.0, 0.0)
        else:
            continue
        m = mmul(m, t)
    return m


# =========================================================================== lengths / colours
_UNITS = {"": 1.0, "px": 1.0, "pt": 4 / 3, "pc": 16.0, "mm": 96 / 25.4, "cm": 96 / 2.54,
          "in": 96.0, "em": 16.0, "ex": 8.0, "q": 96 / 101.6}
_LEN_RE = re.compile(r"\s*(" + _NUM + r")\s*(%|[a-zA-Z]*)\s*$")


def parse_length(s, ref=1.0, default=0.0):
    """Length in user units; percentages are relative to `ref`."""
    if s is None:
        return default
    m = _LEN_RE.match(str(s))
    if not m:
        return default
    v, u = float(m.group(1)), m.group(2).lower()
    if u == "%":
        return v / 100.0 * ref
    return v * _UNITS.get(u, 1.0)


def parse_fraction(s, default):
    """objectBoundingBox coordinate: '50%' -> 0.5, '0.5' -> 0.5."""
    if s is None:
        return default
    return parse_length(s, ref=1.0, default=default)


_RGB_FN = re.compile(r"rgba?\(\s*([^)]*)\)\s*$", re.I)


def parse_color(s, current=(0.0, 0.0, 0.0)):
    """-> ((r, g, b) floats 0..1, alpha) or None."""
    s = (s or "").strip()
    if not s or s == "none":
        return None
    low = s.lower()
    if low == "currentcolor":
        return tuple(current), 1.0
    if low == "transparent":
        return (0.0, 0.0, 0.0), 0.0
    m = _RGB_FN.match(s)
    if m:
        parts = [p for p in re.split(r"[\s,/]+", m.group(1).strip()) if p]
        if len(parts) >= 3:
            ch = []
            for p in parts[:3]:
                ch.append(float(p[:-1]) / 100.0 if p.endswith("%") else float(p) / 255.0)
            a = 1.0
            if len(parts) >= 4:
                p = parts[3]
                a = float(p[:-1]) / 100.0 if p.endswith("%") else float(p)
            return tuple(min(1.0, max(0.0, v)) for v in ch), min(1.0, max(0.0, a))
    try:
        from PIL import ImageColor
        v = ImageColor.getrgb(s)
    except Exception:
        _warn(f"unparsed colour {s!r} (painted black)")
        return (0.0, 0.0, 0.0), 1.0
    return (v[0] / 255.0, v[1] / 255.0, v[2] / 255.0), (v[3] / 255.0 if len(v) > 3 else 1.0)


def hex_to_rgb(h):
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def parse_opacity(s, default=1.0):
    if s is None:
        return default
    s = str(s).strip()
    try:
        v = float(s[:-1]) / 100.0 if s.endswith("%") else float(s)
    except ValueError:
        return default
    return min(1.0, max(0.0, v))


_URL_RE = re.compile(r"url\(\s*['\"]?#([^)'\"]+)['\"]?\s*\)\s*(.*)$")


def parse_url(s):
    m = _URL_RE.match((s or "").strip())
    return (m.group(1), m.group(2).strip()) if m else (None, None)


# =========================================================================== path geometry
# A path is a list of ops: ('M', x, y) ('L', x, y) ('C', x1, y1, x2, y2, x, y) ('Z',)

def _arc_to_cubics(x1, y1, rx, ry, phi_deg, fa, fs, x2, y2):
    if x1 == x2 and y1 == y2:
        return []
    rx, ry = abs(rx), abs(ry)
    if rx == 0 or ry == 0:
        return [("L", x2, y2)]
    phi = math.radians(phi_deg % 360.0)
    cp, sp = math.cos(phi), math.sin(phi)
    dx2, dy2 = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    x1p = cp * dx2 + sp * dy2
    y1p = -sp * dx2 + cp * dy2
    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1:
        s = math.sqrt(lam)
        rx *= s
        ry *= s
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = math.sqrt(max(0.0, num / den)) if den else 0.0
    if fa == fs:
        coef = -coef
    cxp = coef * rx * y1p / ry
    cyp = -coef * ry * x1p / rx
    cx = cp * cxp - sp * cyp + (x1 + x2) / 2.0
    cy = sp * cxp + cp * cyp + (y1 + y2) / 2.0

    def ang(ux, uy, vx, vy):
        return math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)

    ux, uy = (x1p - cxp) / rx, (y1p - cyp) / ry
    vx, vy = (-x1p - cxp) / rx, (-y1p - cyp) / ry
    th1 = ang(1.0, 0.0, ux, uy)
    dth = ang(ux, uy, vx, vy)
    if not fs and dth > 0:
        dth -= 2 * math.pi
    elif fs and dth < 0:
        dth += 2 * math.pi
    n = max(1, int(math.ceil(abs(dth) / (math.pi / 2) - 1e-9)))
    delta = dth / n
    t = 4.0 / 3.0 * math.tan(delta / 4.0)

    def mp(u, v):
        return (cx + rx * cp * u - ry * sp * v, cy + rx * sp * u + ry * cp * v)

    out = []
    for i in range(n):
        a1 = th1 + i * delta
        a2 = a1 + delta
        c1, s1, c2, s2 = math.cos(a1), math.sin(a1), math.cos(a2), math.sin(a2)
        p1 = mp(c1 - t * s1, s1 + t * c1)
        p2 = mp(c2 + t * s2, s2 - t * c2)
        p3 = (x2, y2) if i == n - 1 else mp(c2, s2)
        out.append(("C", p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]))
    return out


_PATH_TOKEN = re.compile(r"\s*,?\s*([MmZzLlHhVvCcSsQqTtAa])|\s*,?\s*(" + _NUM + ")")


def parse_path_d(d):
    """SVG path data -> list of ops (absolute; quadratics/arcs as cubics). Stops at the first error."""
    d = d or ""
    pos, n = 0, len(d)
    out = []
    cmd = None
    cur = (0.0, 0.0)
    start = (0.0, 0.0)
    last_c2 = None   # reflection point for S
    last_q = None    # reflection point for T
    open_sub = False

    def skip():
        nonlocal pos
        while pos < n and d[pos] in " \t\r\n,":
            pos += 1

    def number():
        nonlocal pos
        skip()
        m = _NUM_RE.match(d, pos)
        if not m:
            raise ValueError(f"number expected at {pos}")
        pos = m.end()
        return float(m.group())

    def flag():
        nonlocal pos
        skip()
        if pos < n and d[pos] in "01":
            pos += 1
            return d[pos - 1] == "1"
        raise ValueError(f"arc flag expected at {pos}")

    def ensure_start():
        nonlocal open_sub
        if not open_sub:
            out.append(("M", cur[0], cur[1]))
            open_sub = True

    try:
        while True:
            skip()
            if pos >= n:
                break
            ch = d[pos]
            if ch.isalpha():
                cmd = ch
                pos += 1
                if cmd in "Zz":
                    if open_sub:
                        out.append(("Z",))
                        open_sub = False
                    cur = start
                    last_c2 = last_q = None
                    continue
            elif cmd is None or cmd in "Zz":
                raise ValueError(f"command expected at {pos}")
            rel = cmd.islower()
            C = cmd.upper()
            ox, oy = cur if rel else (0.0, 0.0)
            if C == "M":
                x, y = number() + ox, number() + oy
                out.append(("M", x, y))
                open_sub = True
                cur = start = (x, y)
                cmd = "l" if rel else "L"
                last_c2 = last_q = None
            elif C == "L":
                ensure_start()
                x, y = number() + ox, number() + oy
                out.append(("L", x, y))
                cur = (x, y)
                last_c2 = last_q = None
            elif C == "H":
                ensure_start()
                x = number() + (cur[0] if rel else 0.0)
                out.append(("L", x, cur[1]))
                cur = (x, cur[1])
                last_c2 = last_q = None
            elif C == "V":
                ensure_start()
                y = number() + (cur[1] if rel else 0.0)
                out.append(("L", cur[0], y))
                cur = (cur[0], y)
                last_c2 = last_q = None
            elif C in "CS":
                ensure_start()
                if C == "C":
                    x1, y1 = number() + ox, number() + oy
                else:
                    x1, y1 = (2 * cur[0] - last_c2[0], 2 * cur[1] - last_c2[1]) if last_c2 else cur
                x2, y2 = number() + ox, number() + oy
                x, y = number() + ox, number() + oy
                out.append(("C", x1, y1, x2, y2, x, y))
                last_c2 = (x2, y2)
                last_q = None
                cur = (x, y)
            elif C in "QT":
                ensure_start()
                if C == "Q":
                    qx, qy = number() + ox, number() + oy
                else:
                    qx, qy = (2 * cur[0] - last_q[0], 2 * cur[1] - last_q[1]) if last_q else cur
                x, y = number() + ox, number() + oy
                out.append(("C", cur[0] + 2 / 3 * (qx - cur[0]), cur[1] + 2 / 3 * (qy - cur[1]),
                            x + 2 / 3 * (qx - x), y + 2 / 3 * (qy - y), x, y))
                last_q = (qx, qy)
                last_c2 = None
                cur = (x, y)
            elif C == "A":
                ensure_start()
                rx, ry, phi = number(), number(), number()
                fa, fs = flag(), flag()
                x, y = number() + ox, number() + oy
                out.extend(_arc_to_cubics(cur[0], cur[1], rx, ry, phi, fa, fs, x, y))
                cur = (x, y)
                last_c2 = last_q = None
            else:
                raise ValueError(f"unknown command {cmd}")
    except (ValueError, TypeError) as e:
        _warn(f"path data error ({e}); rendered up to the error like SVG renderers do")
    return out


def transform_path(path, m):
    out = []
    for op in path:
        k = op[0]
        if k == "Z":
            out.append(op)
        elif k == "C":
            a = mapply(m, op[1], op[2])
            b = mapply(m, op[3], op[4])
            c = mapply(m, op[5], op[6])
            out.append(("C", a[0], a[1], b[0], b[1], c[0], c[1]))
        else:
            p = mapply(m, op[1], op[2])
            out.append((k, p[0], p[1]))
    return out


def path_ops(path, nd=4):
    """PDF/PS path construction operators (shared syntax, see the PS prolog)."""
    parts = []
    for op in path:
        k = op[0]
        if k == "M":
            parts.append(f"{_n(op[1], nd)} {_n(op[2], nd)} m")
        elif k == "L":
            parts.append(f"{_n(op[1], nd)} {_n(op[2], nd)} l")
        elif k == "C":
            parts.append(" ".join(_n(v, nd) for v in op[1:]) + " c")
        else:
            parts.append("h")
    return "\n".join(parts)


def path_to_d(path, nd=4):
    parts = []
    for op in path:
        k = op[0]
        if k in "ML":
            parts.append(f"{k}{_n(op[1], nd)} {_n(op[2], nd)}")
        elif k == "C":
            parts.append("C" + " ".join(_n(v, nd) for v in op[1:]))
        else:
            parts.append("Z")
    return " ".join(parts)


def _cubic_extrema(p0, p1, p2, p3):
    a = -p0 + 3 * p1 - 3 * p2 + p3
    b = 2 * (p0 - 2 * p1 + p2)
    c = p1 - p0
    ts = []
    if abs(a) < 1e-12:
        if abs(b) > 1e-12:
            ts.append(-c / b)
    else:
        disc = b * b - 4 * a * c
        if disc >= 0:
            r = math.sqrt(disc)
            ts += [(-b + r) / (2 * a), (-b - r) / (2 * a)]
    vals = []
    for t in ts:
        if 0 < t < 1:
            mt = 1 - t
            vals.append(mt ** 3 * p0 + 3 * mt * mt * t * p1 + 3 * mt * t * t * p2 + t ** 3 * p3)
    return vals


def path_bbox(path):
    """Tight geometric bbox (x0, y0, x1, y1) or None."""
    xs, ys = [], []
    cur = (0.0, 0.0)
    for op in path:
        k = op[0]
        if k in "ML":
            cur = (op[1], op[2])
            xs.append(cur[0])
            ys.append(cur[1])
        elif k == "C":
            xs += [op[5]] + _cubic_extrema(cur[0], op[1], op[3], op[5])
            ys += [op[6]] + _cubic_extrema(cur[1], op[2], op[4], op[6])
            cur = (op[5], op[6])
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def bbox_transform(bb, m):
    if bb is None:
        return None
    pts = [mapply(m, x, y) for x in (bb[0], bb[2]) for y in (bb[1], bb[3])]
    return (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))


def subpaths(path):
    subs, cur = [], None
    for op in path:
        if op[0] == "M":
            cur = [op]
            subs.append(cur)
        elif cur is not None:
            cur.append(op)
    return subs


def flatten_subpath(sub, seg=10):
    pts = []
    cur = None
    for op in sub:
        k = op[0]
        if k in "ML":
            cur = (op[1], op[2])
            pts.append(cur)
        elif k == "C":
            x0, y0 = cur
            for i in range(1, seg + 1):
                t = i / seg
                mt = 1 - t
                pts.append((mt ** 3 * x0 + 3 * mt * mt * t * op[1] + 3 * mt * t * t * op[3] + t ** 3 * op[5],
                            mt ** 3 * y0 + 3 * mt * mt * t * op[2] + 3 * mt * t * t * op[4] + t ** 3 * op[6]))
            cur = (op[5], op[6])
    return pts


def poly_area(pts):
    a = 0.0
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        a += x0 * y1 - x1 * y0
    return a / 2.0


def reverse_subpath(sub):
    """Reverse one subpath (starts with M, may end with Z)."""
    closed = sub[-1][0] == "Z"
    segs = [op for op in sub if op[0] != "Z"]
    cur = (segs[0][1], segs[0][2])
    rev = []
    for op in segs[1:]:
        if op[0] == "L":
            rev.append(("L", cur[0], cur[1]))
            cur = (op[1], op[2])
        elif op[0] == "C":
            rev.append(("C", op[3], op[4], op[1], op[2], cur[0], cur[1]))
            cur = (op[5], op[6])
    out = [("M", cur[0], cur[1])] + rev[::-1]
    if closed:
        out.append(("Z",))
    return out


def path_geom(path, seg=10):
    """Shapely geometry (union of subpath polygons, holes ignored -> conservative superset)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    polys = []
    for sub in subpaths(path):
        pts = flatten_subpath(sub, seg)
        if len(pts) >= 3:
            p = Polygon(pts)
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty:
                polys.append(p)
    return unary_union(polys) if polys else None


def union_for_nonzero(items):
    """[(path, rule)] -> one path whose nonzero fill is (approximately exactly) the union."""
    from shapely.geometry import Point, Polygon
    out = []
    for path, rule in items:
        subs = subpaths(path)
        flats = [flatten_subpath(s) for s in subs]
        areas = [poly_area(f) if len(f) >= 3 else 0.0 for f in flats]
        if rule == "evenodd" and len(subs) > 1:
            polys = [Polygon(f).buffer(0) if len(f) >= 3 else None for f in flats]
            for i, s in enumerate(subs):
                if polys[i] is None or polys[i].is_empty:
                    continue
                rp = polys[i].representative_point()
                depth = sum(1 for j, q in enumerate(polys) if j != i and q is not None and q.contains(Point(rp)))
                want = 1 if depth % 2 == 0 else -1
                out += s if areas[i] * want >= 0 else reverse_subpath(s)
        else:
            total = sum(areas)
            for s in subs:
                out += s if total >= 0 else reverse_subpath(s)
    return out


# --------------------------------------------------------------------------- basic shapes
K_CIRCLE = 0.5522847498307936


def _ellipse_path(cx, cy, rx, ry):
    k = K_CIRCLE
    return [("M", cx + rx, cy),
            ("C", cx + rx, cy + k * ry, cx + k * rx, cy + ry, cx, cy + ry),
            ("C", cx - k * rx, cy + ry, cx - rx, cy + k * ry, cx - rx, cy),
            ("C", cx - rx, cy - k * ry, cx - k * rx, cy - ry, cx, cy - ry),
            ("C", cx + k * rx, cy - ry, cx + rx, cy - k * ry, cx + rx, cy),
            ("Z",)]


def _rect_path(x, y, w, h, rx, ry):
    if rx <= 0 or ry <= 0:
        return [("M", x, y), ("L", x + w, y), ("L", x + w, y + h), ("L", x, y + h), ("Z",)]
    k = K_CIRCLE
    kx, ky = k * rx, k * ry
    return [("M", x + rx, y), ("L", x + w - rx, y),
            ("C", x + w - rx + kx, y, x + w, y + ry - ky, x + w, y + ry),
            ("L", x + w, y + h - ry),
            ("C", x + w, y + h - ry + ky, x + w - rx + kx, y + h, x + w - rx, y + h),
            ("L", x + rx, y + h),
            ("C", x + rx - kx, y + h, x, y + h - ry + ky, x, y + h - ry),
            ("L", x, y + ry),
            ("C", x, y + ry - ky, x + rx - kx, y, x + rx, y),
            ("Z",)]


def _points(s):
    v = [float(x) for x in _NUM_RE.findall(s or "")]
    return list(zip(v[0::2], v[1::2]))


def shape_path(el, vw, vh):
    """Path (local user space) of a basic shape element, or None if it renders nothing."""
    tag = _local(el.tag)
    diag = math.sqrt((vw * vw + vh * vh) / 2.0)
    L = parse_length
    if tag == "path":
        p = parse_path_d(el.get("d"))
        return p or None
    if tag == "rect":
        x, y = L(el.get("x"), vw), L(el.get("y"), vh)
        w, h = L(el.get("width"), vw), L(el.get("height"), vh)
        if w <= 0 or h <= 0:
            return None
        rxs, rys = el.get("rx"), el.get("ry")
        rx = L(rxs, vw) if rxs not in (None, "auto") else None
        ry = L(rys, vh) if rys not in (None, "auto") else None
        if rx is None and ry is None:
            rx = ry = 0.0
        elif rx is None:
            rx = ry
        elif ry is None:
            ry = rx
        rx, ry = min(max(rx, 0.0), w / 2.0), min(max(ry, 0.0), h / 2.0)
        return _rect_path(x, y, w, h, rx, ry)
    if tag == "circle":
        r = L(el.get("r"), diag)
        if r <= 0:
            return None
        return _ellipse_path(L(el.get("cx"), vw), L(el.get("cy"), vh), r, r)
    if tag == "ellipse":
        rxs, rys = el.get("rx"), el.get("ry")
        rx = L(rxs, vw) if rxs not in (None, "auto") else None
        ry = L(rys, vh) if rys not in (None, "auto") else None
        rx = ry if rx is None else rx
        ry = rx if ry is None else ry
        if not rx or not ry or rx <= 0 or ry <= 0:
            return None
        return _ellipse_path(L(el.get("cx"), vw), L(el.get("cy"), vh), rx, ry)
    if tag == "line":
        return [("M", L(el.get("x1"), vw), L(el.get("y1"), vh)), ("L", L(el.get("x2"), vw), L(el.get("y2"), vh))]
    if tag in ("polyline", "polygon"):
        pts = _points(el.get("points"))
        if len(pts) < 2:
            return None
        p = [("M",) + pts[0]] + [("L",) + q for q in pts[1:]]
        if tag == "polygon":
            p.append(("Z",))
        return p
    return None


# =========================================================================== SVG document model
def _local(tag):
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


SHAPES = {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon"}
CONTAINERS = {"g", "a", "switch", "svg", "use"}
NONRENDER = {"defs", "clipPath", "mask", "filter", "linearGradient", "radialGradient", "pattern",
             "symbol", "marker", "style", "title", "desc", "metadata", "script", "namedview",
             "font", "font-face", "cursor", "view", "stop"}
INHERITED = {"fill": "black", "fill-opacity": "1", "fill-rule": "nonzero", "stroke": "none",
             "stroke-width": "1", "stroke-opacity": "1", "stroke-linecap": "butt",
             "stroke-linejoin": "miter", "stroke-miterlimit": "4", "stroke-dasharray": "none",
             "stroke-dashoffset": "0", "clip-rule": "nonzero", "color": "black",
             "visibility": "visible", "paint-order": "normal"}
NONINHERITED = ("opacity", "clip-path", "mask", "filter", "display", "stop-color", "stop-opacity",
                "mask-type", "mix-blend-mode")
PROPS = set(INHERITED) | set(NONINHERITED)


def _decls(text):
    out = {}
    for part in (text or "").split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            v = v.replace("!important", "").strip()
            out[k.strip().lower()] = v
    return out


_COMPOUND = re.compile(r"^(\*|[A-Za-z][\w-]*)?((?:[.#][\w-]+)*)$")


def _parse_selector(sel):
    toks = re.findall(r">|[^\s>]+", sel.strip())
    parts, comb = [], " "
    for t in toks:
        if t == ">":
            comb = ">"
            continue
        m = _COMPOUND.match(t)
        if not m or t == "":
            return None
        tag = m.group(1) if m.group(1) not in (None, "*") else None
        ids = re.findall(r"#([\w-]+)", m.group(2))
        cls = re.findall(r"\.([\w-]+)", m.group(2))
        parts.append((comb, tag, ids, cls))
        comb = " "
    if not parts:
        return None
    spec = (sum(len(p[2]) for p in parts), sum(len(p[3]) for p in parts), sum(1 for p in parts if p[1]))
    return parts, spec


def _parse_css(text):
    text = re.sub(r"/\*.*?\*/", "", text or "", flags=re.S)
    text = re.sub(r"@[^{;]*;", "", text)
    rules = []
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", text):
        decls = _decls(m.group(2))
        for sel in m.group(1).split(","):
            if sel.strip().startswith("@"):
                continue
            p = _parse_selector(sel)
            if p is None:
                _warn(f"CSS selector not supported, ignored: {sel.strip()!r}")
                continue
            rules.append((p[1], len(rules), p[0], decls))
    rules.sort(key=lambda r: (r[0], r[1]))
    return rules


class Node:
    __slots__ = ("kind", "el", "ctm", "name", "opacity", "clip", "mask", "filter", "children",
                 "path", "style", "vp_clip", "in_res")

    def __init__(self, kind, el, ctm, name, style):
        self.kind = kind            # 'group' | 'shape'
        self.el = el                # document element (raster subsets are cut from it)
        self.ctm = ctm              # local user space -> root user space
        self.name = name
        self.style = style          # computed style dict
        self.opacity = parse_opacity(style.get("opacity"))
        self.clip = parse_url(style.get("clip-path"))[0]
        self.mask = parse_url(style.get("mask"))[0]
        self.filter = parse_url(style.get("filter"))[0]
        self.children = []
        self.path = None
        self.vp_clip = None         # nested <svg> viewport clip: path in root user space
        self.in_res = False         # content of a <mask>/<clipPath> (not in document paint order)


class SvgDoc:
    def __init__(self, svg_text, bbox_mode="rsvg"):
        if isinstance(svg_text, bytes):
            svg_text = svg_text.decode("utf-8")
        if bbox_mode not in ("rsvg", "tight"):
            raise ValueError("bbox_mode must be 'rsvg' or 'tight'")
        self.bbox_mode = bbox_mode
        self.root = ET.fromstring(svg_text)
        if _local(self.root.tag) != "svg":
            raise ValueError("root element is not <svg>")
        self.parent = {c: p for p in self.root.iter() for c in p}
        self.index = {el: i for i, el in enumerate(self.root.iter())}
        self.ids = {}
        for el in self.root.iter():
            i = el.get("id")
            if i and i not in self.ids:
                self.ids[i] = el
        css = "".join((el.text or "") for el in self.root.iter() if _local(el.tag) == "style")
        self.css = _parse_css(css)
        vb = [float(v) for v in _NUM_RE.findall(self.root.get("viewBox") or "")]
        if len(vb) == 4 and vb[2] > 0 and vb[3] > 0:
            self.viewbox = tuple(vb)
        else:
            w = parse_length(self.root.get("width"), 100.0, 0.0) or 1024.0
            h = parse_length(self.root.get("height"), 100.0, 0.0) or w
            self.viewbox = (0.0, 0.0, w, h)
        self.vw, self.vh = self.viewbox[2], self.viewbox[3]
        self.title = None
        for el in self.root:
            if _local(el.tag) == "title" and (el.text or "").strip():
                self.title = el.text.strip()
                break
        self._spec_cache = {}
        self._bbox_cache = {}
        root_style = self.computed(dict(INHERITED), self.spec(self.root))
        self.nodes = []
        for ch in self.root:
            nd = self.build(ch, root_style, I)
            if nd is not None:
                self.nodes.append(nd)

    # ------------------------------------------------------------------ styles
    def _match(self, el, parts):
        def match_compound(e, tag, ids, cls):
            if tag and _local(e.tag) != tag:
                return False
            if ids and e.get("id") not in ids:
                return False
            if cls:
                have = set((e.get("class") or "").split())
                if not set(cls) <= have:
                    return False
            return True

        i = len(parts) - 1
        comb, tag, ids, cls = parts[i]
        if not match_compound(el, tag, ids, cls):
            return False
        cur = el
        while i > 0:
            need_comb = parts[i][0]
            i -= 1
            _, tag, ids, cls = parts[i]
            if need_comb == ">":
                cur = self.parent.get(cur)
                if cur is None or not match_compound(cur, tag, ids, cls):
                    return False
            else:
                cur = self.parent.get(cur)
                while cur is not None and not match_compound(cur, tag, ids, cls):
                    cur = self.parent.get(cur)
                if cur is None:
                    return False
        return True

    def spec(self, el):
        """Specified properties: presentation attributes < CSS rules < style attribute."""
        r = self._spec_cache.get(el)
        if r is not None:
            return r
        s = {}
        for k in PROPS:
            v = el.get(k)
            if v is not None:
                s[k] = v.strip()
        if self.css:
            for _, _, parts, decls in self.css:
                if self._match(el, parts):
                    s.update({k: v for k, v in decls.items() if k in PROPS})
        st = el.get("style")
        if st:
            s.update({k: v for k, v in _decls(st).items() if k in PROPS})
        self._spec_cache[el] = s
        return s

    @staticmethod
    def computed(parent, spec):
        c = {k: parent.get(k, INHERITED[k]) for k in INHERITED}
        for k in INHERITED:
            v = spec.get(k)
            if v is not None and v != "inherit":
                c[k] = v
        for k in NONINHERITED:
            v = spec.get(k)
            if v is not None and v != "inherit":
                c[k] = v
        return c

    def style_at(self, el):
        """Computed style of an element that lives outside the render tree (mask, clipPath...)."""
        chain = []
        e = el
        while e is not None:
            chain.append(e)
            e = self.parent.get(e)
        st = dict(INHERITED)
        for e in reversed(chain):
            st = self.computed(st, self.spec(e))
        return st

    @staticmethod
    def layer_name(el):
        return el.get(f"{{{INKSCAPE_NS}}}label") or el.get("data-name") or el.get("id")

    def href(self, el):
        h = el.get(f"{{{XLINK_NS}}}href") or el.get("href")
        if h and h.startswith("#"):
            return self.ids.get(h[1:])
        return None

    # ------------------------------------------------------------------ render tree
    def build(self, el, parent_style, ctm, use_stack=(), in_res=False):
        tag = _local(el.tag)
        if tag in NONRENDER or not tag:
            return None
        if tag not in SHAPES and tag not in CONTAINERS:
            _warn(f"<{tag}> is not supported and was skipped"
                  + (" (convert text to outlines)" if tag == "text" else ""))
            return None
        spec = self.spec(el)
        if spec.get("display", "").strip() == "none":
            return None
        st = self.computed(parent_style, spec)
        m = mmul(ctm, parse_transform(el.get("transform")))
        node = Node("shape" if tag in SHAPES else "group", el, m, self.layer_name(el), st)
        node.in_res = in_res
        if in_res and node.filter:
            _warn("filter inside mask/clip content ignored")
            node.filter = None
        if tag in SHAPES:
            node.path = shape_path(el, self.vw, self.vh)
            if node.path is None:
                return None
            return node
        if tag == "use":
            ref = self.href(el)
            if ref is None or ref in use_stack:
                return None
            t = mtranslate(parse_length(el.get("x"), self.vw), parse_length(el.get("y"), self.vh))
            inner_ctm = mmul(m, t)
            if _local(ref.tag) == "symbol":
                kids = [self.build(c, st, inner_ctm, use_stack + (ref,), in_res) for c in ref]
            else:
                kids = [self.build(ref, st, inner_ctm, use_stack + (ref,), in_res)]
            node.children = [k for k in kids if k is not None]
            # descendants of a <use> instance are rasterised (if ever) through the <use> element
            for k in node.children:
                for d in _walk(k):
                    d.el = el
            return node
        if tag == "svg":
            x, y = parse_length(el.get("x"), self.vw), parse_length(el.get("y"), self.vh)
            w = parse_length(el.get("width"), self.vw, self.vw)
            h = parse_length(el.get("height"), self.vh, self.vh)
            vb = [float(v) for v in _NUM_RE.findall(el.get("viewBox") or "")]
            inner = mmul(m, mtranslate(x, y))
            if len(vb) == 4 and vb[2] > 0 and vb[3] > 0:
                par = (el.get("preserveAspectRatio") or "xMidYMid meet").split()
                sx, sy = w / vb[2], h / vb[3]
                if par[0] != "none":
                    s = min(sx, sy) if (len(par) < 2 or par[1] == "meet") else max(sx, sy)
                    al = par[0]
                    tx = {"xMin": 0, "xMid": 0.5, "xMax": 1}.get(al[:4], 0.5) * (w - vb[2] * s)
                    ty = {"YMin": 0, "YMid": 0.5, "YMax": 1}.get(al[4:], 0.5) * (h - vb[3] * s)
                    vbm = mmul(mtranslate(tx, ty), mmul(mscale(s), mtranslate(-vb[0], -vb[1])))
                else:
                    vbm = mmul(mscale(sx, sy), mtranslate(-vb[0], -vb[1]))
                inner = mmul(inner, vbm)
            if (el.get("overflow") or spec.get("overflow") or "hidden") not in ("visible", "auto"):
                node.vp_clip = transform_path(_rect_path(x, y, w, h, 0, 0), m)
            node.children = [k for k in (self.build(c, st, inner, use_stack, in_res) for c in el) if k is not None]
            return node
        kids = []
        for c in el:
            k = self.build(c, st, m, use_stack, in_res)
            if k is not None:
                kids.append(k)
                if tag == "switch":
                    break
        node.children = kids
        return node

    # ------------------------------------------------------------------ geometry queries
    def bbox(self, node, M):
        """Tight bbox of node geometry mapped by M o node.ctm (fills; strokes ignored per SVG bbox)."""
        if node.kind == "shape":
            return path_bbox(transform_path(node.path, mmul(M, node.ctm)))
        bb = None
        for c in node.children:
            bb = bbox_union(bb, self.bbox(c, M))
        return bb

    def local_bbox(self, node):
        """objectBoundingBox of a node in its own user space.
        bbox_mode 'rsvg' reproduces librsvg/cairo (the reference renderer): a leaf's box is its tight
        device-space box mapped back through the inverse CTM (bbox of a bbox), and each group level
        maps child boxes as rectangles. Identical to 'tight' (SVG2) for axis-aligned transforms,
        looser for rotated/skewed content."""
        key = id(node)
        if key in self._bbox_cache:
            return self._bbox_cache[key]
        inv = minv(node.ctm)
        if inv is None:
            bb = None
        elif self.bbox_mode == "tight":
            bb = self.bbox(node, inv)
        else:
            bb = self._rsvg_bbox(node)
        self._bbox_cache[key] = bb
        return bb

    def _rsvg_bbox(self, node):
        inv = minv(node.ctm)
        if inv is None:
            return None
        if node.kind == "shape":
            return bbox_transform(path_bbox(transform_path(node.path, node.ctm)), inv)
        bb = None
        for c in node.children:
            cb = self._rsvg_bbox(c)
            if cb is not None:
                bb = bbox_union(bb, bbox_transform(cb, mmul(inv, c.ctm)))
        return bb

    def root_bbox(self, node):
        return self.bbox(node, I)

    @staticmethod
    def bb_matrix(bb):
        return (bb[2] - bb[0], 0.0, 0.0, bb[3] - bb[1], bb[0], bb[1])

    # ------------------------------------------------------------------ clip paths
    def clip_levels(self, node, cid=None, depth=0):
        """[[(path_root, rule), ...], ...] (levels are intersected, items in a level unioned);
        [] = no clipping, None = everything clipped away."""
        cid = cid or node.clip
        el = self.ids.get(cid)
        if el is None or _local(el.tag) != "clipPath" or depth > 8:
            _warn(f"clip-path reference #{cid} not found; ignored")
            return []
        base = node.ctm
        if (el.get("clipPathUnits") or "userSpaceOnUse") == "objectBoundingBox":
            bb = self.local_bbox(node)
            if bb is None or bb[2] <= bb[0] or bb[3] <= bb[1]:
                return None
            base = mmul(base, self.bb_matrix(bb))
        base = mmul(base, parse_transform(el.get("transform")))
        st0 = self.style_at(el)
        items = []
        for ch in el:
            tag = _local(ch.tag)
            m = mmul(base, parse_transform(ch.get("transform")))
            src = ch
            if tag == "use":
                ref = self.href(ch)
                if ref is None:
                    continue
                m = mmul(m, mtranslate(parse_length(ch.get("x"), self.vw), parse_length(ch.get("y"), self.vh)))
                m = mmul(m, parse_transform(ref.get("transform")))
                src, tag = ref, _local(ref.tag)
            if tag not in SHAPES:
                if tag not in NONRENDER:
                    _warn(f"<{tag}> inside clipPath not supported")
                continue
            sp = self.spec(src)
            if sp.get("display") == "none":
                continue
            st = self.computed(st0, sp)
            if st.get("visibility") in ("hidden", "collapse"):
                continue
            p = shape_path(src, self.vw, self.vh)
            if p:
                if sp.get("clip-path"):
                    _warn("clip-path on a clipPath child ignored")
                items.append((transform_path(p, m), "evenodd" if st["clip-rule"] == "evenodd" else "nonzero"))
        if not items:
            return None
        levels = [items]
        nested = parse_url(self.spec(el).get("clip-path"))[0]
        if nested:
            sub = self.clip_levels(node, nested, depth + 1)
            if sub is None:
                return None
            levels += sub
        return levels

    # ------------------------------------------------------------------ masks
    def mask_info(self, node):
        el = self.ids.get(node.mask)
        if el is None or _local(el.tag) != "mask":
            _warn(f"mask reference #{node.mask} not found; ignored")
            return None
        units = el.get("maskUnits") or "objectBoundingBox"
        cunits = el.get("maskContentUnits") or "userSpaceOnUse"
        bb = self.local_bbox(node)
        if units == "objectBoundingBox" or cunits == "objectBoundingBox":
            if bb is None or bb[2] <= bb[0] or bb[3] <= bb[1]:
                return "empty"
        if units == "objectBoundingBox":
            fx = parse_fraction(el.get("x"), -0.1)
            fy = parse_fraction(el.get("y"), -0.1)
            fw = parse_fraction(el.get("width"), 1.2)
            fh = parse_fraction(el.get("height"), 1.2)
            bw, bh = bb[2] - bb[0], bb[3] - bb[1]
            rect = (bb[0] + fx * bw, bb[1] + fy * bh, fw * bw, fh * bh)
        else:
            rect = (parse_length(el.get("x"), self.vw, -0.1 * self.vw),
                    parse_length(el.get("y"), self.vh, -0.1 * self.vh),
                    parse_length(el.get("width"), self.vw, 1.2 * self.vw),
                    parse_length(el.get("height"), self.vh, 1.2 * self.vh))
        if rect[2] <= 0 or rect[3] <= 0:
            return "empty"
        cctm = node.ctm if cunits != "objectBoundingBox" else mmul(node.ctm, self.bb_matrix(bb))
        st = self.computed(self.style_at(self.parent.get(el, el)), self.spec(el))
        kids = [self.build(c, st, cctm, in_res=True) for c in el]
        kind = (el.get("mask-type") or self.spec(el).get("mask-type") or "luminance").strip()
        rect_path = transform_path(_rect_path(*rect, 0, 0), node.ctm)
        return {"rect": rect_path, "nodes": [k for k in kids if k is not None], "kind": kind}

    # ------------------------------------------------------------------ filters
    def filter_info(self, node, ppu):
        """Region (root user units, pixel aligned at `ppu` px/unit) + split mode, or None."""
        el = self.ids.get(node.filter)
        if el is None or _local(el.tag) != "filter":
            _warn(f"filter reference #{node.filter} not found; ignored")
            return None
        bb = self.local_bbox(node)
        if (el.get("filterUnits") or "objectBoundingBox") == "objectBoundingBox":
            if bb is None or bb[2] <= bb[0] or bb[3] <= bb[1]:
                return "empty"
            fx = parse_fraction(el.get("x"), -0.1)
            fy = parse_fraction(el.get("y"), -0.1)
            fw = parse_fraction(el.get("width"), 1.2)
            fh = parse_fraction(el.get("height"), 1.2)
            bw, bh = bb[2] - bb[0], bb[3] - bb[1]
            rect = (bb[0] + fx * bw, bb[1] + fy * bh, fw * bw, fh * bh)
        else:
            rect = (parse_length(el.get("x"), self.vw, -0.1 * self.vw),
                    parse_length(el.get("y"), self.vh, -0.1 * self.vh),
                    parse_length(el.get("width"), self.vw, 1.2 * self.vw),
                    parse_length(el.get("height"), self.vh, 1.2 * self.vh))
        if rect[2] <= 0 or rect[3] <= 0:
            return "empty"
        rp = transform_path(_rect_path(*rect, 0, 0), node.ctm)
        reg = path_bbox(rp)
        vx, vy, vw, vh = self.viewbox
        reg = (max(reg[0], vx), max(reg[1], vy), min(reg[2], vx + vw), min(reg[3], vy + vh))
        if reg[2] <= reg[0] or reg[3] <= reg[1]:
            return "empty"
        reg = _pixel_align(reg, ppu, self.viewbox)
        prims = [c for c in el if _local(c.tag).startswith("fe")]
        split = None
        if prims:
            last = prims[-1]
            if _local(last.tag) == "feMerge":
                nodes = [c for c in last if _local(c.tag) == "feMergeNode"]
                if len(nodes) >= 2 and nodes[-1].get("in") == "SourceGraphic":
                    split = "merge"
            elif len(prims) == 1 and _local(last.tag) == "feDropShadow":
                split = "dropshadow"
        return {"el": el, "region": reg, "split": split, "rect_path": rp}

    # ------------------------------------------------------------------ gradients
    def gradient(self, gid, node, fill_box_root=None):
        """Resolved gradient: dict(kind, coords, matrix (gradient space -> root user space),
        stops [(t, rgb, a)], spread)  |  ('solid', rgb, a)  |  None (paint nothing)."""
        el = self.ids.get(gid)
        chain, seen = [], set()
        while el is not None and _local(el.tag) in ("linearGradient", "radialGradient") and el not in seen:
            chain.append(el)
            seen.add(el)
            el = self.href(el)
        if not chain:
            return "missing"
        kind = "linear" if _local(chain[0].tag) == "linearGradient" else "radial"

        def attr(name, default=None):
            for e in chain:
                if e.get(name) is not None and (name not in ("x1", "y1", "x2", "y2", "cx", "cy", "r", "fx", "fy", "fr")
                                                or _local(e.tag) == _local(chain[0].tag)):
                    return e.get(name)
            return default

        stops = []
        for e in chain:
            st_els = [s for s in e if _local(s.tag) == "stop"]
            if st_els:
                last = 0.0
                for s in st_els:
                    sp = self.spec(s)
                    o = parse_fraction((s.get("offset") or "0").strip(), 0.0)
                    o = min(1.0, max(last, o))
                    last = o
                    col = sp.get("stop-color", "black")
                    cur = parse_color(self.style_at(s).get("color", "black"))
                    c = parse_color(col, cur[0] if cur else (0, 0, 0)) or ((0.0, 0.0, 0.0), 0.0)
                    stops.append((o, c[0], c[1] * parse_opacity(sp.get("stop-opacity"))))
                break
        if not stops:
            return None
        if len(stops) == 1:
            return ("solid", stops[0][1], stops[0][2])
        units = attr("gradientUnits", "objectBoundingBox")
        gt = parse_transform(attr("gradientTransform"))
        spread = attr("spreadMethod", "pad")
        if units == "objectBoundingBox":
            bb = self.local_bbox(node)
            if bb is None or bb[2] <= bb[0] or bb[3] <= bb[1]:
                return None
            M = mmul(node.ctm, mmul(self.bb_matrix(bb), gt))
            X = Y = R = lambda s, d: parse_fraction(s, d)
            dflt = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 0.0, "cx": 0.5, "cy": 0.5, "r": 0.5, "fr": 0.0}
        else:
            M = mmul(node.ctm, gt)
            diag = math.sqrt((self.vw ** 2 + self.vh ** 2) / 2)
            X = lambda s, d: parse_length(s, self.vw, d)
            Y = lambda s, d: parse_length(s, self.vh, d)
            R = lambda s, d: parse_length(s, diag, d)
            dflt = {"x1": 0.0, "y1": 0.0, "x2": self.vw, "y2": 0.0, "cx": self.vw / 2, "cy": self.vh / 2,
                    "r": diag / 2, "fr": 0.0}
        if kind == "linear":
            x1, y1 = X(attr("x1"), dflt["x1"]), Y(attr("y1"), dflt["y1"])
            x2, y2 = X(attr("x2"), dflt["x2"]), Y(attr("y2"), dflt["y2"])
            if abs(x2 - x1) < 1e-12 and abs(y2 - y1) < 1e-12:
                return ("solid", stops[-1][1], stops[-1][2])
            coords = (x1, y1, x2, y2)
        else:
            cx, cy = X(attr("cx"), dflt["cx"]), Y(attr("cy"), dflt["cy"])
            r = R(attr("r"), dflt["r"])
            fx, fy = X(attr("fx"), cx), Y(attr("fy"), cy)
            fr = R(attr("fr"), dflt["fr"])
            if r <= 0:
                return ("solid", stops[-1][1], stops[-1][2])
            dd = math.hypot(fx - cx, fy - cy)
            if dd > r:   # SVG 1.1 / librsvg: move the focus onto the circle
                fx, fy = cx + (fx - cx) * r / dd, cy + (fy - cy) * r / dd
            coords = (fx, fy, max(0.0, fr), cx, cy, r)
        if minv(M) is None:
            return None
        return {"kind": kind, "coords": coords, "matrix": M, "stops": stops, "spread": spread}

    # ------------------------------------------------------------------ raster subsets
    def subset_svg(self, node, mode, crop, px, *, strip_target=True, filter_split=None, background=None):
        """Standalone SVG cut from the document and cropped to `crop` (root user units).
        mode 'alone': only `node` (its ancestors keep their transforms/styles but lose
        opacity/clip/mask/filter); mode 'upto': everything painted up to and including `node`."""
        root2 = copy.deepcopy(self.root)
        o2c = {}
        for o, c in zip(self.root.iter(), root2.iter()):
            o2c[o] = c
        c2i = {c: self.index[o] for o, c in o2c.items()}
        target = o2c[node.el]
        parent2 = {c: p for p in root2.iter() for c in p}
        anc = []
        x = target
        while x in parent2:
            x = parent2[x]
            anc.append(x)
        anc_set = set(anc)
        end = max(c2i[e] for e in target.iter())
        hoist = ET.Element(f"{{{SVG_NS}}}defs")

        def renderable(e):
            t = _local(e.tag)
            return t not in NONRENDER and bool(t)

        def walk(parent):
            for ch in list(parent):
                if ch is target:
                    continue
                if ch in anc_set:
                    walk(ch)
                    continue
                if not renderable(ch):
                    continue
                drop = (mode == "alone") or (c2i[ch] > end)
                if drop:
                    parent.remove(ch)
                    hoist.append(ch)

        walk(root2)
        override = "opacity:1;clip-path:none;mask:none;filter:none"
        if mode == "alone":
            for a in anc:
                if a is not root2:
                    a.set("style", (a.get("style") or "") + ";" + override)
            if strip_target:
                target.set("style", (target.get("style") or "") + ";opacity:1;clip-path:none;mask:none")
        root2.insert(0, hoist)
        if filter_split and node.filter:
            fel = next((e for e in root2.iter() if e.get("id") == node.filter and _local(e.tag) == "filter"), None)
            if fel is not None:
                _patch_filter(fel, filter_split)
        vx, vy, vw, vh = self.viewbox
        root2.set("x", repr(vx))
        root2.set("y", repr(vy))
        root2.set("width", repr(vw))
        root2.set("height", repr(vh))
        root2.set("viewBox", f"{vx!r} {vy!r} {vw!r} {vh!r}")
        outer = ET.Element(f"{{{SVG_NS}}}svg", {
            "viewBox": f"{crop[0]!r} {crop[1]!r} {crop[2] - crop[0]!r} {crop[3] - crop[1]!r}",
            "width": str(px[0]), "height": str(px[1]), "preserveAspectRatio": "none"})
        if background:
            ET.SubElement(outer, f"{{{SVG_NS}}}rect", {
                "x": repr(crop[0]), "y": repr(crop[1]), "width": repr(crop[2] - crop[0]),
                "height": repr(crop[3] - crop[1]), "fill": background})
        outer.append(root2)
        return ET.tostring(outer, encoding="utf-8")


def _walk(node):
    yield node
    for c in node.children:
        yield from _walk(c)


def _patch_filter(fel, split):
    """Strip the SourceGraphic layer from a filter so that only the effect is rasterised."""
    prims = [c for c in fel if _local(c.tag).startswith("fe")]
    if split == "merge":
        mg = prims[-1]
        nodes = [c for c in mg if _local(c.tag) == "feMergeNode"]
        mg.remove(nodes[-1])
    elif split == "dropshadow":
        ds = prims[0]
        g = lambda k, d: ds.get(k) or d
        sd = g("stdDeviation", "2")
        dx, dy = g("dx", "2"), g("dy", "2")
        fc = ds.get("flood-color") or _decls(ds.get("style")).get("flood-color") or "black"
        fo = ds.get("flood-opacity") or _decls(ds.get("style")).get("flood-opacity") or "1"
        fel.remove(ds)
        ns = f"{{{SVG_NS}}}"
        ET.SubElement(fel, ns + "feGaussianBlur", {"in": "SourceAlpha", "stdDeviation": sd, "result": "kvxBlur"})
        ET.SubElement(fel, ns + "feOffset", {"in": "kvxBlur", "dx": dx, "dy": dy, "result": "kvxOff"})
        ET.SubElement(fel, ns + "feFlood", {"flood-color": fc, "flood-opacity": fo, "result": "kvxFlood"})
        ET.SubElement(fel, ns + "feComposite", {"in": "kvxFlood", "in2": "kvxOff", "operator": "in"})


def _pixel_align(reg, ppu, viewbox):
    """Expand a root-space rect to whole pixels of a grid anchored at the viewBox origin."""
    vx, vy = viewbox[0], viewbox[1]
    x0 = vx + math.floor((reg[0] - vx) * ppu + 1e-6) / ppu
    y0 = vy + math.floor((reg[1] - vy) * ppu + 1e-6) / ppu
    x1 = vx + math.ceil((reg[2] - vx) * ppu - 1e-6) / ppu
    y1 = vy + math.ceil((reg[3] - vy) * ppu - 1e-6) / ppu
    return (x0, y0, max(x1, x0 + 1 / ppu), max(y1, y0 + 1 / ppu))


def rsvg_render(svg_bytes, w, h, background=None):
    """librsvg (rsvg-convert) render -> numpy uint8 RGBA array (h, w, 4)."""
    import numpy as np
    from PIL import Image
    cmd = ["rsvg-convert", "-w", str(w), "-h", str(h), "-f", "png"]
    if background:
        cmd += ["-b", background]
    out = subprocess.run(cmd, input=svg_bytes, capture_output=True, check=True).stdout
    im = Image.open(io.BytesIO(out)).convert("RGBA")
    if im.size != (w, h):
        im = im.resize((w, h), Image.LANCZOS)
    return np.asarray(im).copy()


# =========================================================================== gradient functions
def _norm_stops(stops):
    st = list(stops)
    if st[0][0] > 0:
        st.insert(0, (0.0,) + tuple(st[0][1:]))
    if st[-1][0] < 1:
        st.append((1.0,) + tuple(st[-1][1:]))
    segs = []
    for a, b in zip(st, st[1:]):
        if b[0] > a[0]:
            segs.append((a[0], b[0], a[1], b[1]))
    if not segs:
        segs.append((0.0, 1.0, st[-1][1], st[-1][1]))
    return segs


def _tiled_segments(segs, spread, n0, n1):
    if spread not in ("repeat", "reflect"):
        return segs, 0.0, 1.0
    out = []
    for k in range(n0, n1):
        if spread == "reflect" and k % 2:
            for a, b, ca, cb in reversed(segs):
                out.append((k + 1 - b, k + 1 - a, cb, ca))
        else:
            for a, b, ca, cb in segs:
                out.append((k + a, k + b, ca, cb))
    return out, float(n0), float(n1)


def _function(segs, t0, t1):
    def f2(ca, cb):
        return {"FunctionType": 2, "Domain": [0, 1], "C0": list(ca), "C1": list(cb), "N": 1}
    if len(segs) == 1 and segs[0][0] == t0 and segs[0][1] == t1 and (t0, t1) == (0.0, 1.0):
        return f2(segs[0][2], segs[0][3])
    return {"FunctionType": 3, "Domain": [t0, t1],
            "Functions": [f2(s[2], s[3]) for s in segs],
            "Bounds": [s[1] for s in segs[:-1]],
            "Encode": [v for _ in segs for v in (0, 1)]}


def _tile_range(g, box_root):
    """Gradient-parameter range covering box_root (for repeat/reflect)."""
    if g["spread"] not in ("repeat", "reflect") or box_root is None:
        return 0, 1
    inv = minv(g["matrix"])
    pts = [mapply(inv, x, y) for x in (box_root[0], box_root[2]) for y in (box_root[1], box_root[3])]
    if g["kind"] == "linear":
        x1, y1, x2, y2 = g["coords"]
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        ts = [((p[0] - x1) * dx + (p[1] - y1) * dy) / L2 for p in pts]
        lo, hi = math.floor(min(ts)), math.ceil(max(ts))
    else:
        fx, fy, fr, cx, cy, r = g["coords"]
        den = r - fr - math.hypot(cx - fx, cy - fy)
        if den <= 1e-9:
            return 0, 1
        hi = math.ceil(max((math.hypot(p[0] - fx, p[1] - fy) - fr) / den for p in pts))
        lo = 0
    lo, hi = int(lo), int(max(hi, lo + 1))
    if hi - lo > 256:
        _warn("repeat/reflect gradient tiled to 256 periods")
        hi = lo + 256
    return lo, hi


def shading_dict(g, which, box_root=None, colorspace="DeviceRGB", mapper=None, backdrop=(1.0, 1.0, 1.0)):
    """Shading dictionary in gradient space.
    which: 'color' | 'alpha' (DeviceGray stop alphas) | 'color_on' (stops composited over the solid
    `backdrop` colour; alpha*colour is quadratic between stops, so 16 sub-stops per varying interval
    keep the error < 0.1/255)."""
    stops = g["stops"]
    if which == "alpha":
        vals = [(s[0], (s[2],)) for s in stops]
        colorspace = "DeviceGray"
    elif which == "color_on":
        bd = tuple(backdrop)

        def over(col, al):
            return tuple(al * c + (1 - al) * b for c, b in zip(col, bd))
        vals = []
        for a, b in zip(stops, stops[1:]):
            n = 16 if (a[2] != b[2] and b[0] > a[0]) else 1
            for i in range(n):
                t = i / n
                al = a[2] + (b[2] - a[2]) * t
                col = [ca + (cb - ca) * t for ca, cb in zip(a[1], b[1])]
                vals.append((a[0] + (b[0] - a[0]) * t, over(col, al)))
        last = stops[-1]
        vals.append((last[0], over(last[1], last[2])))
    else:
        vals = [(s[0], tuple(s[1])) for s in stops]
    if mapper:
        vals = [(o, mapper(v)) for o, v in vals]
    segs = _norm_stops(vals)
    n0, n1 = _tile_range(g, box_root)
    segs, t0, t1 = _tiled_segments(segs, g["spread"], n0, n1)
    func = _function(segs, t0, t1)
    if g["kind"] == "linear":
        x1, y1, x2, y2 = g["coords"]
        if g["spread"] in ("repeat", "reflect"):
            dx, dy = x2 - x1, y2 - y1
            coords = [x1 + t0 * dx, y1 + t0 * dy, x1 + t1 * dx, y1 + t1 * dy]
        else:
            coords = [x1, y1, x2, y2]
        sh = {"ShadingType": 2, "ColorSpace": _Name(colorspace), "Coords": coords}
    else:
        fx, fy, fr, cx, cy, r = g["coords"]
        if g["spread"] in ("repeat", "reflect"):
            coords = [fx, fy, fr, fx + t1 * (cx - fx), fy + t1 * (cy - fy), fr + t1 * (r - fr)]
        else:
            coords = [fx, fy, fr, cx, cy, r]
        sh = {"ShadingType": 3, "ColorSpace": _Name(colorspace), "Coords": coords}
    sh["Domain"] = [t0, t1]
    sh["Function"] = func
    sh["Extend"] = [True, True]
    return sh


# =========================================================================== PDF object writer
class _Name(str):
    pass


class _Ref:
    __slots__ = ("num",)

    def __init__(self, num):
        self.num = num

    def __hash__(self):
        return hash(("ref", self.num))

    def __eq__(self, o):
        return isinstance(o, _Ref) and o.num == self.num


class _Stream:
    def __init__(self, d, data, compress=True):
        self.d = dict(d)
        self.data = data if isinstance(data, bytes) else data.encode("latin-1")
        self.compress = compress


def _pdf_string(s):
    try:
        s.encode("ascii")
        if all(32 <= ord(ch) < 127 for ch in s):
            return "(" + s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ")"
    except UnicodeEncodeError:
        pass
    return "<FEFF" + s.encode("utf-16-be").hex().upper() + ">"


def _ser(v):
    if isinstance(v, _Ref):
        return f"{v.num} 0 R"
    if isinstance(v, _Name):
        return "/" + "".join(ch if (33 <= ord(ch) <= 126 and ch not in "()<>[]{}/%#") else f"#{ord(ch):02X}" for ch in v)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return _n(v, 6)
    if isinstance(v, str):
        return _pdf_string(v)
    if isinstance(v, bytes):
        return "<" + v.hex().upper() + ">"
    if isinstance(v, (list, tuple)):
        return "[" + " ".join(_ser(x) for x in v) + "]"
    if isinstance(v, dict):
        return "<<" + " ".join(f"/{k} {_ser(x)}" for k, x in v.items()) + ">>"
    if v is None:
        return "null"
    raise TypeError(type(v))


class _PdfWriter:
    def __init__(self):
        self.objs = [None]

    def add(self, obj):
        self.objs.append(obj)
        return _Ref(len(self.objs) - 1)

    def reserve(self):
        return self.add(None)

    def set(self, ref, obj):
        self.objs[ref.num] = obj

    def tobytes(self, root, info, version="1.6"):
        out = io.BytesIO()
        out.write(f"%PDF-{version}\n%\xe2\xe3\xcf\xd3\n".encode("latin-1"))
        offsets = [0]
        h = hashlib.md5()
        for i, obj in enumerate(self.objs[1:], 1):
            offsets.append(out.tell())
            out.write(f"{i} 0 obj\n".encode())
            if isinstance(obj, _Stream):
                data = obj.data
                d = dict(obj.d)
                if obj.compress:
                    data = zlib.compress(data, 9)
                    d["Filter"] = _Name("FlateDecode")
                d["Length"] = len(data)
                out.write(_ser(d).encode("latin-1") + b"\nstream\n" + data + b"\nendstream")
                h.update(data)
            else:
                s = _ser(obj).encode("latin-1")
                out.write(s)
                h.update(s)
            out.write(b"\nendobj\n")
        xref = out.tell()
        n = len(self.objs)
        out.write(f"xref\n0 {n}\n0000000000 65535 f \n".encode())
        for off in offsets[1:]:
            out.write(f"{off:010d} 00000 n \n".encode())
        fid = h.hexdigest().upper()
        trailer = {"Size": n, "Root": root, "Info": info, "ID": [bytes.fromhex(fid), bytes.fromhex(fid)]}
        out.write(b"trailer\n" + _ser(trailer).encode() + f"\nstartxref\n{xref}\n%%EOF\n".encode())
        return out.getvalue()


class _Res:
    """Resource dictionary of one content stream. Names are document-unique (see _Names)."""

    def __init__(self):
        self.d = {"ExtGState": {}, "Pattern": {}, "XObject": {}, "Shading": {}}

    def add(self, kind, name, ref):
        self.d[kind][name] = ref
        return name

    def merge(self, other):
        for k, v in other.d.items():
            self.d[k].update(v)

    def pdf(self):
        return {k: {kk: vv for kk, vv in v.items()} for k, v in self.d.items() if v}


def _date(dt):
    return dt.strftime("D:%Y%m%d%H%M%S+00'00'")


def _now(date=None):
    if date is not None:
        return date
    sde = os.environ.get("SOURCE_DATE_EPOCH")
    if sde:
        return _dt.datetime.fromtimestamp(int(sde), _dt.timezone.utc)
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)


def _xmp(title, creator, producer, dt, doc_id):
    iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about=""\n'
        '    xmlns:dc="http://purl.org/dc/elements/1.1/"\n'
        '    xmlns:xmp="http://ns.adobe.com/xap/1.0/"\n'
        '    xmlns:pdf="http://ns.adobe.com/pdf/1.3/"\n'
        '    xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/">\n'
        '   <dc:format>application/pdf</dc:format>\n'
        f'   <dc:title><rdf:Alt><rdf:li xml:lang="x-default">{esc(title)}</rdf:li></rdf:Alt></dc:title>\n'
        f'   <xmp:CreatorTool>{esc(creator)}</xmp:CreatorTool>\n'
        f'   <xmp:CreateDate>{iso}</xmp:CreateDate>\n'
        f'   <xmp:ModifyDate>{iso}</xmp:ModifyDate>\n'
        f'   <xmp:MetadataDate>{iso}</xmp:MetadataDate>\n'
        f'   <pdf:Producer>{esc(producer)}</pdf:Producer>\n'
        f'   <xmpMM:DocumentID>uuid:{doc_id}</xmpMM:DocumentID>\n'
        f'   <xmpMM:InstanceID>uuid:{doc_id}</xmpMM:InstanceID>\n'
        '  </rdf:Description>\n'
        ' </rdf:RDF>\n'
        '</x:xmpmeta>\n'
        + (" " * 99 + "\n") * 20 +
        '<?xpacket end="w"?>').encode("utf-8")


def _finish_pdf(w, content, res, page_w, page_h, title, creator, producer, date,
                page_group=None, output_intent=None):
    """Page, Pages, Catalog, Info, XMP -> PDF bytes."""
    dt = _now(date)
    box = [0, 0, float(page_w), float(page_h)]
    cref = w.add(_Stream({}, content))
    pages = w.reserve()
    page = {"Type": _Name("Page"), "Parent": pages, "MediaBox": box, "CropBox": box,
            "BleedBox": box, "TrimBox": box, "ArtBox": box, "Resources": res.pdf(), "Contents": cref}
    if page_group:
        page["Group"] = page_group
    pref = w.add(page)
    w.set(pages, {"Type": _Name("Pages"), "Kids": [pref], "Count": 1})
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, hashlib.md5(content.encode("latin-1") + title.encode()).hexdigest()))
    meta = w.add(_Stream({"Type": _Name("Metadata"), "Subtype": _Name("XML")},
                         _xmp(title, creator, producer, dt, doc_id), compress=False))
    cat = {"Type": _Name("Catalog"), "Pages": pages, "Metadata": meta}
    if output_intent:
        cat["OutputIntents"] = [output_intent]
    root = w.add(cat)
    info = w.add({"Title": title, "Creator": creator, "Producer": producer,
                  "CreationDate": _date(dt), "ModDate": _date(dt), "Trapped": _Name("False")})
    return w.tobytes(root, info)


def _png_streams(rgba):
    """numpy RGBA -> (RGB stream dict/data, alpha stream or None)."""
    import numpy as np
    h, w = rgba.shape[:2]
    rgb = np.ascontiguousarray(rgba[:, :, :3]).tobytes()
    a = rgba[:, :, 3]
    alpha = None
    if (a < 255).any():
        alpha = np.ascontiguousarray(a).tobytes()
    return w, h, rgb, alpha


# =========================================================================== PDF emitter (RGB)
class _PdfEmitter:
    def __init__(self, doc, scale=1.0, group_mode="xobject", raster_ppi=300):
        self.doc = doc
        self.w = _PdfWriter()
        self.scale = scale
        self.B = (scale, 0.0, 0.0, -scale, -doc.viewbox[0] * scale, (doc.viewbox[1] + doc.viewbox[3]) * scale)
        self.page_w = doc.viewbox[2] * scale
        self.page_h = doc.viewbox[3] * scale
        self.group_mode = group_mode
        self.ppu = raster_ppi / 72.0 * scale       # raster pixels per user unit
        self.counter = 0
        self.transparency = False
        self.stats = {"paths": 0, "forms": 0, "transparency_groups": 0, "shadings": 0, "smasks": 0,
                      "images": 0, "image_pixels": 0, "rasterised": []}
        self._gs_cache = {}

    def _name(self, prefix):
        self.counter += 1
        return f"{prefix}{self.counter}"

    def _ext(self, res, d):
        key = repr(sorted((k, (v.num if isinstance(v, _Ref) else v)) for k, v in d.items()))
        if key not in self._gs_cache:
            dd = {"Type": _Name("ExtGState")}
            dd.update(d)
            self._gs_cache[key] = (self._name("GS"), self.w.add(dd))
        name, ref = self._gs_cache[key]
        res.add("ExtGState", name, ref)
        if "SMask" in d or d.get("ca", 1) < 1 or d.get("CA", 1) < 1:
            self.transparency = True
        return name

    def _form(self, ops, res, group=None):
        d = {"Type": _Name("XObject"), "Subtype": _Name("Form"), "BBox": [0, 0, self.page_w, self.page_h],
             "Resources": res.pdf()}
        if group:
            d["Group"] = {"Type": _Name("Group"), "S": _Name("Transparency"), "CS": _Name(group), "I": True}
            self.stats["transparency_groups"] += 1
            self.transparency = True
        self.stats["forms"] += 1
        return self.w.add(_Stream(d, "\n".join(ops)))

    def tp(self, path_root):
        return transform_path(path_root, self.B)

    # ------------------------------------------------------------------ nodes
    def emit(self, nodes, ops, res):
        for n in nodes:
            self.node(n, ops, res)

    @staticmethod
    def _has_paint(n):
        st = n.style
        if st.get("visibility") in ("hidden", "collapse"):
            return False
        return st.get("fill", "black") != "none" or st.get("stroke", "none") != "none"

    def node(self, n, ops, res):
        doc = self.doc
        if n.opacity <= 0:
            return
        if n.kind == "shape" and not self._has_paint(n):
            return
        clips = doc.clip_levels(n) if n.clip else []
        if clips is None:
            return
        if n.vp_clip:
            clips = clips + [[(n.vp_clip, "nonzero")]]
        mask = doc.mask_info(n) if n.mask else None
        if mask == "empty":
            return
        finfo = doc.filter_info(n, self.ppu) if (n.filter and not n.in_res) else None
        if finfo == "empty":
            return
        alpha = n.opacity
        composite = alpha < 1 or mask is not None
        tag = "SVGGroup" if n.kind == "group" else "SVGShape"
        if n.name:
            ops.append(f"/{tag} <</Name {_pdf_string(n.name)}>> BDC")
        wrap = bool(clips) or composite
        if wrap:
            ops.append("q")
        for level in clips:
            ops.append(self._clip_ops(level))

        def body(o, r, inline_alpha=1.0):
            if finfo is not None:
                self._filtered(n, finfo, o, r)
            elif n.kind == "group":
                self.emit(n.children, o, r)
            else:
                self._shape(n, o, r, inline_alpha)

        grouplike = n.kind == "group" or finfo is not None or mask is not None or self._fill_and_stroke(n)
        if composite and grouplike:
            o2, r2 = [], _Res()
            body(o2, r2)
            form = self._form(o2, r2, group="DeviceRGB")
            gsd = {}
            if alpha < 1:
                gsd["ca"] = alpha
                gsd["CA"] = alpha
            if mask is not None:
                gsd["SMask"] = self._smask(mask)
            gsn = self._ext(res, gsd)
            fname = self._name("Fm")
            res.add("XObject", fname, form)
            ops.append(f"/{gsn} gs /{fname} Do")
        elif n.kind == "group" and finfo is None and self.group_mode == "xobject" and n.name:
            o2, r2 = [], _Res()
            body(o2, r2)
            form = self._form(o2, r2)
            fname = self._name("Fm")
            res.add("XObject", fname, form)
            ops.append(f"/{fname} Do")
        else:
            body(ops, res, alpha)
        if wrap:
            ops.append("Q")
        if n.name:
            ops.append("EMC")

    @staticmethod
    def _fill_and_stroke(n):
        return n.kind == "shape" and n.style.get("fill", "black") != "none" and n.style.get("stroke", "none") != "none"

    def _clip_ops(self, level):
        if len(level) == 1:
            p, rule = level[0]
            return path_ops(self.tp(p)) + ("\nW* n" if rule == "evenodd" else "\nW n")
        return path_ops(self.tp(union_for_nonzero(level))) + "\nW n"

    def _smask(self, mask):
        o2, r2 = ["q", path_ops(self.tp(mask["rect"])) + "\nW n"], _Res()
        self.emit(mask["nodes"], o2, r2)
        o2.append("Q")
        form = self._form(o2, r2, group="DeviceRGB")
        self.stats["smasks"] += 1
        return {"Type": _Name("Mask"), "S": _Name("Alpha" if mask["kind"] == "alpha" else "Luminosity"), "G": form}

    # ------------------------------------------------------------------ paint
    def _paint(self, n, which):
        """-> None | ('solid', rgb, a) | ('grad', g)"""
        st = n.style
        val = st.get(which, "black" if which == "fill" else "none")
        if val == "none":
            return None
        cur = parse_color(st.get("color", "black"))
        cur = cur[0] if cur else (0.0, 0.0, 0.0)
        gid, fb = parse_url(val)
        if gid:
            g = self.doc.gradient(gid, n)
            if g == "missing":
                if not fb or fb == "none":
                    if not fb:
                        _warn(f"paint server #{gid} not found/supported; painted as none")
                    return None
                c = parse_color(fb, cur)
                return ("solid", c[0], c[1]) if c else None
            if g is None:
                return None
            if isinstance(g, tuple):
                return g
            return ("grad", g)
        c = parse_color(val, cur)
        return ("solid", c[0], c[1]) if c else None

    def _pattern(self, g, which, res, box_root):
        sh = shading_dict(g, which, box_root)
        sref = self.w.add(sh)
        self.stats["shadings"] += 1
        pat = self.w.add({"Type": _Name("Pattern"), "PatternType": 2, "Shading": sref,
                          "Matrix": list(mmul(self.B, g["matrix"]))})
        name = self._name("P")
        res.add("Pattern", name, pat)
        return name

    def _shape(self, n, ops, res, alpha):
        st = n.style
        self.stats["paths"] += 1
        order = ["fill", "stroke"]
        if st.get("paint-order", "normal").strip().startswith("stroke"):
            order = ["stroke", "fill"]
        for which in order:
            p = self._paint(n, which)
            if p is None:
                continue
            if which == "fill":
                self._fill(n, p, alpha * parse_opacity(st.get("fill-opacity")), ops, res)
            else:
                self._stroke(n, p, alpha * parse_opacity(st.get("stroke-opacity")), ops, res)

    def _fill(self, n, p, fa, ops, res):
        path_root = transform_path(n.path, n.ctm)
        pops = path_ops(self.tp(path_root))
        op = "f*" if n.style.get("fill-rule") == "evenodd" else "f"
        if p[0] == "solid":
            a = fa * p[2]
            if a <= 0:
                return
            col = " ".join(_n(v) for v in p[1]) + " rg"
            if a < 1:
                ops.append(f"q /{self._ext(res, {'ca': a})} gs {col}\n{pops}\n{op} Q")
            else:
                ops.append(f"{col}\n{pops}\n{op}")
            return
        g = p[1]
        box = path_bbox(path_root)
        alphas = [s[2] for s in g["stops"]]
        pname = self._pattern(g, "color", res, box)
        gsd = {}
        if max(alphas) - min(alphas) < 1e-6:
            fa *= alphas[0]
        else:
            # stop-opacity -> luminosity soft mask: DeviceGray shading of the same geometry
            r2 = _Res()
            an = self._pattern(g, "alpha", r2, box)
            form = self._form([f"/Pattern cs /{an} scn\n{pops}\n{op}"], r2, group="DeviceGray")
            gsd["SMask"] = {"Type": _Name("Mask"), "S": _Name("Luminosity"), "G": form}
            self.stats["smasks"] += 1
        if fa <= 0:
            return
        if fa < 1:
            gsd["ca"] = fa
        if gsd:
            ops.append(f"q /{self._ext(res, gsd)} gs /Pattern cs /{pname} scn\n{pops}\n{op} Q")
        else:
            ops.append(f"/Pattern cs /{pname} scn\n{pops}\n{op}")

    def _stroke_setup(self, n):
        st = n.style
        diag = math.sqrt((self.doc.vw ** 2 + self.doc.vh ** 2) / 2)
        sw = parse_length(st.get("stroke-width"), diag, 1.0)
        if sw <= 0:
            return None
        cap = {"butt": 0, "round": 1, "square": 2}.get(st.get("stroke-linecap"), 0)
        join = {"miter": 0, "miter-clip": 0, "round": 1, "bevel": 2, "arcs": 0}.get(st.get("stroke-linejoin"), 0)
        mm = _NUM_RE.match(st.get("stroke-miterlimit", "4") or "4")
        ml = max(1.0, float(mm.group()) if mm else 4.0)
        da = st.get("stroke-dasharray", "none")
        dash = "[] 0 d "            # always explicit: stroke clips are not wrapped in q/Q
        dashed = False
        if da and da != "none":
            vals = [parse_length(v, diag) for v in re.split(r"[\s,]+", da.strip()) if v]
            if vals and all(v >= 0 for v in vals) and sum(vals) > 0:
                if len(vals) % 2:
                    vals = vals * 2
                off = parse_length(st.get("stroke-dashoffset"), diag)
                dash = "[" + " ".join(_n(v) for v in vals) + f"] {_n(off)} d "
                dashed = True
        cm = " ".join(_n(v, 6) for v in mmul(self.B, n.ctm)) + " cm"
        return f"{cm} {_n(sw)} w {cap} J {join} j {_n(ml)} M {dash}", sw, {"cap": cap, "join": join, "ml": ml,
                                                                         "dashed": dashed}

    def _stroke(self, n, p, sa, ops, res):
        setup = self._stroke_setup(n)
        if setup is None:
            return
        setup, sw, _ = setup
        lops = path_ops(n.path, 6)
        if p[0] == "solid":
            a = sa * p[2]
            if a <= 0:
                return
            gs = f"/{self._ext(res, {'CA': a})} gs " if a < 1 else ""
            ops.append(f"q {gs}{setup}{' '.join(_n(v) for v in p[1])} RG\n{lops}\nS Q")
            return
        g = p[1]
        alphas = [s[2] for s in g["stops"]]
        box = path_bbox(transform_path(n.path, n.ctm))
        if box:
            box = (box[0] - sw, box[1] - sw, box[2] + sw, box[3] + sw)
        pname = self._pattern(g, "color", res, box)
        gsd = {}
        if max(alphas) - min(alphas) < 1e-6:
            sa *= alphas[0]
        else:
            r2 = _Res()
            an = self._pattern(g, "alpha", r2, box)
            form = self._form([f"q {setup}/Pattern CS /{an} SCN\n{lops}\nS Q"], r2, group="DeviceGray")
            gsd["SMask"] = {"Type": _Name("Mask"), "S": _Name("Luminosity"), "G": form}
            self.stats["smasks"] += 1
        if sa <= 0:
            return
        if sa < 1:
            gsd["CA"] = sa
        gs = f"/{self._ext(res, gsd)} gs " if gsd else ""
        ops.append(f"q {gs}{setup}/Pattern CS /{pname} SCN\n{lops}\nS Q")

    # ------------------------------------------------------------------ filters (raster)
    def _filtered(self, n, finfo, ops, res):
        reg = finfo["region"]
        pw = int(round((reg[2] - reg[0]) * self.ppu))
        ph = int(round((reg[3] - reg[1]) * self.ppu))
        svg = self.doc.subset_svg(n, "alone", reg, (pw, ph), strip_target=True, filter_split=finfo["split"])
        rgba = rsvg_render(svg, pw, ph)
        if rgba[:, :, 3].any():
            self._image(rgba, reg, ops, res)
            self.stats["rasterised"].append(f"{n.name or n.el.get('id') or _local(n.el.tag)} (filter #{n.filter}, "
                                            f"{pw}x{ph} px{', effect only + vector source' if finfo['split'] else ''})")
        if finfo["split"]:
            ops.append("q")
            ops.append(path_ops(self.tp(finfo["rect_path"])) + "\nW n")
            saved = n.filter
            n.filter = None
            try:
                if n.kind == "group":
                    self.emit(n.children, ops, res)
                else:
                    self._shape(n, ops, res, 1.0)
            finally:
                n.filter = saved
            ops.append("Q")

    def _image(self, rgba, reg, ops, res):
        w, h, rgb, alpha = _png_streams(rgba)
        d = {"Type": _Name("XObject"), "Subtype": _Name("Image"), "Width": w, "Height": h,
             "ColorSpace": _Name("DeviceRGB"), "BitsPerComponent": 8, "Interpolate": True}
        if alpha is not None:
            d["SMask"] = self.w.add(_Stream({"Type": _Name("XObject"), "Subtype": _Name("Image"), "Width": w,
                                             "Height": h, "ColorSpace": _Name("DeviceGray"),
                                             "BitsPerComponent": 8, "Interpolate": True}, alpha))
            self.transparency = True
        ref = self.w.add(_Stream(d, rgb))
        name = self._name("Im")
        res.add("XObject", name, ref)
        self.stats["images"] += 1
        self.stats["image_pixels"] += w * h
        x0, y0 = mapply(self.B, reg[0], reg[3])
        x1, y1 = mapply(self.B, reg[2], reg[1])
        ops.append(f"q {_n(x1 - x0, 6)} 0 0 {_n(y1 - y0, 6)} {_n(x0, 6)} {_n(y0, 6)} cm /{name} Do Q")

    def build(self, title, creator, producer, date=None):
        ops, res = [], _Res()
        self.emit(self.doc.nodes, ops, res)
        group = None
        if self.transparency:
            group = {"Type": _Name("Group"), "S": _Name("Transparency"), "CS": _Name("DeviceRGB")}
        return _finish_pdf(self.w, "\n".join(ops) + "\n", res, self.page_w, self.page_h, title, creator,
                           producer, date, page_group=group)


# =========================================================================== PostScript / EPS
_PS_PROLOG = """/KVXdict 40 dict def
KVXdict begin
/q {gsave} bind def /Q {grestore} bind def
/m {moveto} bind def /l {lineto} bind def /c {curveto} bind def /h {closepath} bind def
/f {fill} bind def /f* {eofill} bind def /W {clip newpath} bind def /W* {eoclip newpath} bind def
/S {stroke} bind def /w {setlinewidth} bind def /J {setlinecap} bind def /j {setlinejoin} bind def
/M {setmiterlimit} bind def /d {setdash} bind def /rg {setrgbcolor} bind def /RG {setrgbcolor} bind def
/k {setcmykcolor} bind def /cm {6 array astore concat} bind def
end"""


def _a85_lines(data):
    enc = base64.a85encode(data).decode("ascii") + "~>"
    # leading space: no data line can start with '%' (keeps DSC parsers happy)
    return "\n".join(" " + enc[i:i + 78] for i in range(0, len(enc), 78))


def _eps(body, page_w, page_h, title, creator, date, level, process_colors=None):
    dt = _now(date)
    bb = f"0 0 {int(math.ceil(page_w - 1e-9))} {int(math.ceil(page_h - 1e-9))}"
    hdr = ["%!PS-Adobe-3.0 EPSF-3.0",
           f"%%BoundingBox: {bb}",
           f"%%HiResBoundingBox: 0 0 {page_w:.4f} {page_h:.4f}",
           f"%%Title: ({title})",
           f"%%Creator: ({creator})",
           f"%%CreationDate: ({dt.strftime('%Y-%m-%d %H:%M:%S UTC')})",
           f"%%LanguageLevel: {level}",
           "%%DocumentData: Clean7Bit",
           "%%Pages: 1"]
    if process_colors:
        hdr.append("%%DocumentProcessColors: " + " ".join(process_colors))
    hdr += ["%%EndComments", "%%BeginProlog", _PS_PROLOG, "%%EndProlog",
            "%%Page: 1 1", "save", "KVXdict begin", body, "end", "restore", "showpage",
            "%%Trailer", "%%EOF", ""]
    return "\n".join(hdr).encode("latin-1")


def _region(path, rule, seg=24):
    """Shapely region of a page-space path under a fill rule -> (geometry | None, exact)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    polys, signs, exact = [], [], True
    for sub in subpaths(path):
        pts = flatten_subpath(sub, seg)
        if len(pts) < 3:
            continue
        a = poly_area(pts)
        if abs(a) < 1e-9:
            continue
        p = Polygon(pts)
        if not p.is_valid:
            p = p.buffer(0)
            exact = False
        if p.is_empty:
            continue
        polys.append(p)
        signs.append(a > 0)
    if not polys:
        return None, True
    if rule == "evenodd" or len(set(signs)) > 1:
        disjoint = all(polys[i].intersection(polys[j]).area < 1e-6
                       for i in range(len(polys)) for j in range(i + 1, len(polys)))
        if disjoint:
            return unary_union(polys), exact
        g = polys[0]
        for p in polys[1:]:
            g = g.symmetric_difference(p)
        return g, exact and rule == "evenodd"
    return unary_union(polys), exact


class _PsEmitter(_PdfEmitter):
    """RGB EPS writer. Shares the PDF emitter's paint resolution. PostScript has no transparency:
    every painted object is recorded (region, solid colour or not, exact clip replay), and paint
    with alpha is composited exactly over solid backdrops by painting it once per backdrop piece,
    clipped by that backdrop's own path, with the pre-blended colour (paper = white). Only paint
    over non-solid backdrops (gradients/images), group opacity over overlapping children, masks and
    filters are rasterised (composite rendered by librsvg, clipped to the paint's vector outline or
    a 1-bit coverage mask)."""

    def __init__(self, doc, scale=1.0, flatten_ppi=300):
        super().__init__(doc, scale=scale, group_mode="flat", raster_ppi=flatten_ppi)
        self.level = 2
        self.items = []          # painted records, bottom -> top
        self.clip_ops = []       # active clip operator sequences (page space)
        self.clip_geoms = []     # their regions
        self.flattened = []
        self.vec_blended = []
        self.n_res = 0
        self._comp = {}

    # ---------------------------------------------------------------- bookkeeping
    def _clipped(self, g):
        if g is None:
            return None
        for cg in self.clip_geoms:
            if cg is not None:
                g = g.intersection(cg)
        return g

    def _rec(self, geom, kind, color, tail, exact=True):
        geom = self._clipped(geom)
        if geom is None or geom.is_empty or geom.area < 1e-6:
            return
        self.items.append({"geom": geom, "box": geom.bounds, "kind": kind, "color": color,
                           "clip": list(self.clip_ops) + list(tail), "exact": exact})

    def _backdrop(self, G):
        """Solid records under region G (paint order) and whether they cover G completely;
        None when a non-solid paint (gradient, image, raster) shows through."""
        Gb = G.buffer(0.05)
        b = Gb.bounds
        rel = [it for it in self.items
               if it["box"][0] <= b[2] and b[0] <= it["box"][2] and it["box"][1] <= b[3] and b[1] <= it["box"][3]
               and it["geom"].intersects(Gb)]
        remaining = G
        for it in reversed(rel):
            if remaining.is_empty or remaining.area < 1e-4:
                break
            if it["kind"] != "solid":
                if remaining.intersection(it["geom"]).area > 1e-3:
                    return None
                continue
            if it["exact"]:
                remaining = remaining.difference(it["geom"])
        return [it for it in rel if it["kind"] == "solid"], (remaining.is_empty or remaining.area < 1e-4)

    # ---------------------------------------------------------------- tree
    def node(self, n, ops, res):
        doc = self.doc
        if n.opacity <= 0:
            return
        if n.kind == "shape" and not self._has_paint(n):
            return
        clips = doc.clip_levels(n) if n.clip else []
        if clips is None:
            return
        mask = doc.mask_info(n) if n.mask else None
        finfo = doc.filter_info(n, self.ppu) if (n.filter and not n.in_res) else None
        if mask == "empty" or finfo == "empty":
            return
        if n.vp_clip:
            clips = clips + [[(n.vp_clip, "nonzero")]]
        if n.name:
            ops.append(f"% {'group' if n.kind == 'group' else 'shape'}: {n.name.encode('ascii', 'replace').decode()}")
        if clips:
            ops.append("q")
        pushed = 0
        try:
            for level in clips:
                op = self._clip_ops(level).replace("\nW n", "\nW").replace("\nW* n", "\nW*")
                if len(level) == 1:
                    geom = _region(self.tp(level[0][0]), level[0][1])[0]
                else:
                    geom = _region(self.tp(union_for_nonzero(level)), "nonzero")[0]
                ops.append(op)
                self.clip_ops.append(op)
                self.clip_geoms.append(geom)
                pushed += 1
            if self._needs_flatten(n, mask, finfo):
                self._transparent(n, ops, mask, finfo)
            elif n.kind == "group":
                self.emit(n.children, ops, res)
            else:
                self._shape(n, ops, res, 1.0)
        finally:
            if pushed:
                del self.clip_ops[-pushed:]
                del self.clip_geoms[-pushed:]
        if clips:
            ops.append("Q")

    def _needs_flatten(self, n, mask, finfo):
        if n.opacity < 1 or mask is not None or finfo is not None:
            return True
        if n.kind != "shape":
            return False
        for which in ("fill", "stroke"):
            p = self._paint(n, which)
            if p is not None and not self._opaque(p, parse_opacity(n.style.get(which + "-opacity"))):
                return True
        return False

    @staticmethod
    def _opaque(p, alpha):
        if p[0] == "solid":
            return alpha * p[2] >= 1 - 1e-9
        return alpha >= 1 - 1e-9 and min(s[2] for s in p[1]["stops"]) >= 1 - 1e-9

    # ---------------------------------------------------------------- paint operators
    def _shfill(self, g, box_root, which, backdrop=(1.0, 1.0, 1.0)):
        self.level = 3
        sh = shading_dict(g, which, box_root, backdrop=backdrop)
        return " ".join(_n(v, 6) for v in mmul(self.B, g["matrix"])) + " cm\n" + _ser(sh) + " shfill"

    def _geo(self, n, which):
        """(region, exact, clip-operator sequence reproducing the paint's coverage) or None."""
        if which == "fill":
            ppage = self.tp(transform_path(n.path, n.ctm))
            rule = "evenodd" if n.style.get("fill-rule") == "evenodd" else "nonzero"
            g, exact = _region(ppage, rule)
            return g, exact, path_ops(ppage) + ("\nW*" if rule == "evenodd" else "\nW")
        setup = self._stroke_setup(n)
        inv = minv(mmul(self.B, n.ctm))
        if setup is None or inv is None:
            return None
        setup, sw, info = setup
        from shapely.geometry import LineString, LinearRing
        from shapely.ops import unary_union
        m = mmul(self.B, n.ctm)
        a, b, c, d = m[:4]
        s1 = math.hypot(a, b)
        s2 = math.hypot(c, d)
        k = math.sqrt(abs(a * d - b * c))
        uniform = abs(s1 - s2) < 1e-3 * max(s1, s2) and abs(a * c + b * d) < 1e-6 * max(s1, s2) ** 2
        parts = []
        for sub in subpaths(transform_path(n.path, m)):
            pts = flatten_subpath(sub, 24)
            if len(pts) < 2:
                continue
            closed = sub[-1][0] == "Z" and len(pts) >= 3
            geom = LinearRing(pts) if closed else LineString(pts)
            parts.append(geom.buffer(sw * k / 2, cap_style=["flat", "round", "square"][info["cap"]],
                                     join_style=["mitre", "round", "bevel"][info["join"]],
                                     mitre_limit=info["ml"]))
        g = unary_union(parts) if parts else None
        clipop = f"{setup}\n{path_ops(n.path, 6)}\nstrokepath W\n" + " ".join(_n(v, 9) for v in inv) + " cm"
        return g, uniform and not info["dashed"], clipop

    def _paint_op(self, n, which, p, alpha=1.0, backdrop=None):
        """PostScript painting `which` of n; with a backdrop colour, alpha is composited onto it."""
        if which == "fill":
            ppage = self.tp(transform_path(n.path, n.ctm))
            evenodd = n.style.get("fill-rule") == "evenodd"
            pops = path_ops(ppage)
            if p[0] == "solid":
                col = p[1] if backdrop is None else _over(p[1], alpha * p[2], backdrop)
                return " ".join(_n(v) for v in col) + f" rg\n{pops}\n{'f*' if evenodd else 'f'}"
            g = p[1]
            box = path_bbox(transform_path(n.path, n.ctm))
            if backdrop is None:
                return f"q\n{pops}\n{'W*' if evenodd else 'W'}\n{self._shfill(g, box, 'color')}\nQ"
            g = dict(g, stops=[(o, col, a * alpha) for o, col, a in g["stops"]])
            return f"q\n{pops}\n{'W*' if evenodd else 'W'}\n{self._shfill(g, box, 'color_on', backdrop)}\nQ"
        setup, sw, _ = self._stroke_setup(n)
        lops = path_ops(n.path, 6)
        if p[0] == "solid":
            col = p[1] if backdrop is None else _over(p[1], alpha * p[2], backdrop)
            return f"q {setup}{' '.join(_n(v) for v in col)} RG\n{lops}\nS Q"
        g = p[1]
        box = path_bbox(transform_path(n.path, n.ctm))
        if box:
            box = (box[0] - sw, box[1] - sw, box[2] + sw, box[3] + sw)
        which_sh, bd = "color", (1.0, 1.0, 1.0)
        if backdrop is not None:
            g = dict(g, stops=[(o, col, a * alpha) for o, col, a in g["stops"]])
            which_sh, bd = "color_on", backdrop
        inv = minv(mmul(self.B, n.ctm))
        return (f"q {setup}\n{lops}\nstrokepath W\n" + " ".join(_n(v, 9) for v in inv) + " cm\n"
                + self._shfill(g, box, which_sh, bd) + "\nQ")

    # opaque paint (called through _PdfEmitter._shape)
    def _fill(self, n, p, fa, ops, res):
        self._opaque_paint(n, "fill", p, ops)

    def _stroke(self, n, p, sa, ops, res):
        if self._stroke_setup(n) is not None:
            self._opaque_paint(n, "stroke", p, ops)

    def _opaque_paint(self, n, which, p, ops):
        ops.append(self._paint_op(n, which, p))
        geo = self._geo(n, which)
        if geo and geo[0] is not None:
            self._rec(geo[0], "solid" if p[0] == "solid" else "other", tuple(p[1]) if p[0] == "solid" else None,
                      [geo[2]], geo[1])

    # ---------------------------------------------------------------- transparency
    def _alpha_paint(self, n, which, p, alpha, ops):
        """Exact vector compositing of a paint with alpha over solid backdrops. False -> rasterise."""
        geo = self._geo(n, which)
        if geo is None or geo[0] is None:
            return True
        region, exact, clipop = geo
        G = self._clipped(region)
        if G is None or G.is_empty or G.area < 1e-6:
            return True
        bd = self._backdrop(G)
        if bd is None:
            return False
        backs, covered = bd
        solid = p[0] == "solid"
        under = [it for it in backs if G.intersection(it["geom"]).area > 1e-4]
        colours = {tuple(round(v, 6) for v in it["color"]) for it in under}
        if not covered:
            colours.add((1.0, 1.0, 1.0))
        if len(colours) == 1:                  # uniform backdrop: one unclipped paint
            bc = next(iter(colours))
            ops.append(self._paint_op(n, which, p, alpha, bc))
            self._rec(region, "solid" if solid else "other",
                      _over(p[1], alpha * p[2], bc) if solid else None, [clipop], exact)
            return True
        if not covered:                        # paper (white) first, then every backdrop piece
            ops.append(self._paint_op(n, which, p, alpha, (1.0, 1.0, 1.0)))
            self._rec(region, "solid" if solid else "other",
                      _over(p[1], alpha * p[2], (1.0, 1.0, 1.0)) if solid else None, [clipop], exact)
        for it in under:
            ops.append("q\n" + "\n".join(it["clip"]) + "\n" + self._paint_op(n, which, p, alpha, it["color"]) + "\nQ")
            self._rec(region.intersection(it["geom"]), "solid" if solid else "other",
                      _over(p[1], alpha * p[2], it["color"]) if solid else None, it["clip"] + [clipop], exact)
        return True

    def _transparent(self, n, ops, mask, finfo):
        label = n.name or n.el.get("id") or _local(n.el.tag)
        if mask is None and finfo is None:
            if n.kind == "shape" and not (n.opacity < 1 and self._fill_and_stroke(n)):
                self._transparent_leaf(n, n.opacity, ops, label)
                return
            if n.kind == "group" and self._disjoint_fill_leaves(n):
                # group opacity over non-overlapping single-paint leaves == per-leaf alpha
                for leaf in self._leaves(n):
                    self._transparent_leaf(leaf, n.opacity, ops, label, nested=True)
                return
        self._raster(n, ops, finfo, None, label)

    def _transparent_leaf(self, n, opacity, ops, label, nested=False):
        st = n.style
        order = ["stroke", "fill"] if st.get("paint-order", "normal").strip().startswith("stroke") else ["fill", "stroke"]
        all_vec = True
        for which in order:
            p = self._paint(n, which)
            if p is None or (which == "stroke" and self._stroke_setup(n) is None):
                continue
            alpha = opacity * parse_opacity(st.get(which + "-opacity"))
            if self._opaque(p, alpha):
                self._opaque_paint(n, which, p, ops)
                continue
            if alpha <= 0 or (p[0] == "solid" and alpha * p[2] <= 0):
                continue
            if not self._alpha_paint(n, which, p, alpha, ops):
                all_vec = False
                geo = self._geo(n, which)
                if geo and geo[0] is not None:
                    self._raster(n, ops, None, [geo], f"{label} ({which})")
        if all_vec:
            self.vec_blended.append(label if not nested else f"{label}/{n.name or _local(n.el.tag)}")

    def _leaves(self, n):
        return [d for d in _walk(n) if d.kind == "shape"]

    def _disjoint_fill_leaves(self, g):
        leaves = []
        for d in _walk(g):
            if d is g:
                continue
            if d.opacity < 1 or d.mask or d.filter or d.clip or d.vp_clip:
                return False
            if d.kind == "shape":
                if not self._has_paint(d):
                    continue
                if self._paint(d, "stroke") is not None:
                    return False
                leaves.append(d)
        regs = []
        for d in leaves:
            r = _region(self.tp(transform_path(d.path, d.ctm)), "nonzero")[0]
            if r is None:
                continue
            for q in regs:
                if r.intersection(q).area > 1e-3:
                    return False
            regs.append(r)
        return True

    def _raster(self, n, ops, finfo, regions, label):
        """Flatten n by rasterising the composite of everything painted up to and including n
        (librsvg, white paper), drawn under the vector clip(s) `regions` or a coverage mask."""
        import numpy as np
        doc = self.doc
        key = id(n)
        if key not in self._comp:
            if finfo is not None:
                reg = finfo["region"]
            else:
                bb = None
                for d in _walk(n):
                    if d.kind == "shape":
                        b = path_bbox(transform_path(d.path, d.ctm))
                        if b and d.style.get("stroke", "none") != "none":
                            su = self._stroke_setup(d)
                            if su:
                                e = su[1] * max(1.0, su[2]["ml"] / 2) * math.sqrt(abs(d.ctm[0] * d.ctm[3] - d.ctm[1] * d.ctm[2]))
                                b = (b[0] - e, b[1] - e, b[2] + e, b[3] + e)
                        bb = bbox_union(bb, b)
                if bb is None:
                    return
                vx, vy, vw, vh = doc.viewbox
                bb = (max(bb[0], vx), max(bb[1], vy), min(bb[2], vx + vw), min(bb[3], vy + vh))
                if bb[2] <= bb[0] or bb[3] <= bb[1]:
                    return
                reg = _pixel_align(bb, self.ppu, doc.viewbox)
            pw = int(round((reg[2] - reg[0]) * self.ppu))
            ph = int(round((reg[3] - reg[1]) * self.ppu))
            comp = rsvg_render(doc.subset_svg(n, "upto", reg, (pw, ph), background="#ffffff"), pw, ph)
            self.n_res += 1
            iname = f"kvxI{self.n_res}"
            ops.append(f"/{iname} currentfile /ASCII85Decode filter /ReusableStreamDecode filter\n"
                       f"{_a85_lines(zlib.compress(np.ascontiguousarray(comp[:, :, :3]).tobytes(), 9))}\ndef")
            self._comp[key] = (reg, pw, ph, iname)
        reg, pw, ph, iname = self._comp[key]
        self.level = 3
        x0, y0 = mapply(self.B, reg[0], reg[3])
        x1, y1 = mapply(self.B, reg[2], reg[1])
        place = f"{_n(x0, 6)} {_n(y0, 6)} translate {_n(x1 - x0, 6)} {_n(y1 - y0, 6)} scale"
        img = (f"<< /ImageType 1 /Width {pw} /Height {ph} /BitsPerComponent 8 /Decode [0 1 0 1 0 1] "
               f"/ImageMatrix [{pw} 0 0 -{ph} 0 {ph}] /Interpolate true /DataSource {iname} /FlateDecode filter >>")
        if regions is not None:
            for geom, exact, clipop in regions:
                ops.append(f"q\n{clipop}\n{place}\n/DeviceRGB setcolorspace {iname} 0 setfileposition\n{img} image\nQ")
                self._rec(geom, "other", None, [clipop], exact)
            how = "vector outline clip"
        else:
            cov = rsvg_render(doc.subset_svg(n, "alone", reg, (pw, ph), strip_target=False), pw, ph)
            covered = cov[:, :, 3] > 0
            bits = np.packbits(covered, axis=1).tobytes()
            mname = f"kvxM{self.n_res}"
            ops.append(f"/{mname} currentfile /ASCII85Decode filter /ReusableStreamDecode filter\n"
                       f"{_a85_lines(zlib.compress(bits, 9))}\ndef")
            ops.append(f"q\n{place}\n/DeviceRGB setcolorspace {iname} 0 setfileposition {mname} 0 setfileposition\n"
                       f"<< /ImageType 3 /InterleaveType 3\n/DataDict {img.replace('/Interpolate true ', '')}\n"
                       f"/MaskDict << /ImageType 1 /Width {pw} /Height {ph} /BitsPerComponent 1 /Decode [1 0] "
                       f"/ImageMatrix [{pw} 0 0 -{ph} 0 {ph}] /DataSource {mname} /FlateDecode filter >>\n>> image\nQ")
            ops.append(f"/{mname} null def")
            from shapely.geometry import box as sbox
            self._rec(sbox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), "other", None, [], False)
            how = "1-bit coverage mask"
        self.flattened.append(f"{label}: raster {pw}x{ph} px @ {self.ppu * 72 / self.scale:.0f} ppi, {how}")

    def build_eps(self, title, creator, date=None):
        ops = []
        self.emit(self.doc.nodes, ops, _Res())
        for _, _, _, iname in self._comp.values():
            ops.append(f"/{iname} null def")
        return _eps("\n".join(ops), self.page_w, self.page_h, _ps_text(title), _ps_text(creator), date, self.level)


def _over(col, a, backdrop):
    a = min(1.0, max(0.0, a))
    return tuple(a * c + (1 - a) * b for c, b in zip(col, backdrop))


def _ps_text(s):
    s = s.encode("ascii", "replace").decode()
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


# =========================================================================== public: RGB
def export_rgb(svg_text, out_basename, *, title=None, scale=1.0, group_mode="xobject", raster_ppi=300,
               flatten_ppi=300, bbox_mode="rsvg", creator=DEFAULT_CREATOR, producer=DEFAULT_PRODUCER, date=None):
    """Write <base>.pdf, <base>.ai (identical bytes, PDF-compatible Illustrator file) and <base>.eps.
    Returns a report dict (paths, object counts, rasterised/flattened units, warnings).
      scale       pt per SVG user unit (page = viewBox * scale)
      group_mode  'xobject' (named groups -> Form XObjects) | 'flat'
      raster_ppi  resolution of <filter> rasters in the PDF; flatten_ppi: EPS transparency rasters
      bbox_mode   'rsvg' (objectBoundingBox exactly as librsvg/cairo computes it) | 'tight' (SVG2)"""
    WARNINGS.clear()
    doc = SvgDoc(svg_text, bbox_mode=bbox_mode)
    title = title or doc.title or os.path.basename(out_basename)
    em = _PdfEmitter(doc, scale=scale, group_mode=group_mode, raster_ppi=raster_ppi)
    pdf = em.build(title, creator, producer, date)
    d = os.path.dirname(os.path.abspath(out_basename))
    os.makedirs(d, exist_ok=True)
    paths = {"pdf": out_basename + ".pdf", "ai": out_basename + ".ai", "eps": out_basename + ".eps"}
    for k in ("pdf", "ai"):
        with open(paths[k], "wb") as fh:
            fh.write(pdf)
    ps = _PsEmitter(doc, scale=scale, flatten_ppi=flatten_ppi)
    with open(paths["eps"], "wb") as fh:
        fh.write(ps.build_eps(title, creator, date))
    rep = dict(paths)
    rep.update({"page_pt": (em.page_w, em.page_h), "pdf_stats": em.stats, "eps_level": ps.level,
                "eps_flattened": ps.flattened, "eps_vector_blended": ps.vec_blended,
                "warnings": list(WARNINGS)})
    return rep


# =========================================================================== colour management
_ICC = {}


def _icc_transforms(profile=CMYK_PROFILE):
    if profile not in _ICC:
        from PIL import ImageCms
        srgb = ImageCms.createProfile("sRGB")
        cmyk = ImageCms.getOpenProfile(profile)
        intent = ImageCms.Intent.RELATIVE_COLORIMETRIC
        flags = ImageCms.Flags.BLACKPOINTCOMPENSATION
        _ICC[profile] = {
            "fwd": ImageCms.buildTransform(srgb, cmyk, "RGB", "CMYK", renderingIntent=intent, flags=flags),
            "back": ImageCms.buildTransform(cmyk, srgb, "CMYK", "RGB", renderingIntent=intent, flags=flags),
            "lab_cmyk": ImageCms.buildTransform(cmyk, ImageCms.createProfile("LAB"), "CMYK", "LAB",
                                                renderingIntent=intent, flags=flags),
            "lab_rgb": ImageCms.buildTransform(srgb, ImageCms.createProfile("LAB"), "RGB", "LAB",
                                               renderingIntent=intent),
            "name": ImageCms.getProfileDescription(cmyk).strip(),
        }
    return _ICC[profile]


def icc_profile_name(profile=CMYK_PROFILE):
    return _icc_transforms(profile)["name"]


def srgb_to_cmyk(hex_color, *, profile=CMYK_PROFILE, snap_pct=1.0, neutrals="k"):
    """sRGB hex -> (c, m, y, k) floats 0..1 through a real ICC transform: LittleCMS sRGB ->
    Ghostscript default_cmyk.icc ("Artifex CMYK SWOP Profile"), relative colorimetric + black point
    compensation (Pillow ImageCms). snap_pct rounds to whole percents (brand-guide values; None =
    raw 8-bit ICC result). neutrals='k': exact greys (r=g=b) become K-only with the same L* in the
    profile (no rich black for neutral art); 'icc' keeps the profile's rich black."""
    from PIL import Image, ImageCms
    t = _icc_transforms(profile)
    rgb = hex_to_rgb(hex_color)
    if neutrals == "k" and rgb[0] == rgb[1] == rgb[2]:
        target = ImageCms.applyTransform(Image.new("RGB", (1, 1), rgb), t["lab_rgb"]).getpixel((0, 0))[0]
        best, bk = None, 0
        for k in range(256):
            L = ImageCms.applyTransform(Image.new("CMYK", (1, 1), (0, 0, 0, k)), t["lab_cmyk"]).getpixel((0, 0))[0]
            if best is None or abs(L - target) < best:
                best, bk = abs(L - target), k
        vals = (0.0, 0.0, 0.0, bk / 255.0)
    else:
        px = ImageCms.applyTransform(Image.new("RGB", (1, 1), rgb), t["fwd"]).getpixel((0, 0))
        vals = tuple(v / 255.0 for v in px)
    if snap_pct:
        vals = tuple(round(v * 100 / snap_pct) * snap_pct / 100 for v in vals)
    return tuple(min(1.0, max(0.0, v)) for v in vals)


def cmyk_to_srgb(cmyk, profile=CMYK_PROFILE):
    """Soft proof of a CMYK value back to sRGB (same ICC, relative colorimetric + BPC) -> (r, g, b) 0..255."""
    from PIL import Image, ImageCms
    t = _icc_transforms(profile)
    px = tuple(int(round(v * 255)) for v in cmyk)
    return ImageCms.applyTransform(Image.new("CMYK", (1, 1), px), t["back"]).getpixel((0, 0))


def delta_e76(hex_color, cmyk, profile=CMYK_PROFILE):
    """CIE76 dE between the sRGB colour and its CMYK print simulation (Lab, D50)."""
    from PIL import Image, ImageCms
    t = _icc_transforms(profile)
    a = ImageCms.applyTransform(Image.new("RGB", (1, 1), hex_to_rgb(hex_color)), t["lab_rgb"]).getpixel((0, 0))
    b = ImageCms.applyTransform(Image.new("CMYK", (1, 1), tuple(int(round(v * 255)) for v in cmyk)),
                                t["lab_cmyk"]).getpixel((0, 0))

    def lab(p):
        return (p[0] * 100 / 255, (p[1] if p[1] < 128 else p[1] - 256), (p[2] if p[2] < 128 else p[2] - 256))
    la, lb = lab(a), lab(b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(la, lb)))


def ink_total(cmyk):
    return sum(cmyk)


# =========================================================================== public: CMYK
def export_cmyk(shapes, size, out_basename, title, *, scale=1.0, clip_d=None, output_intent=True,
                creator=DEFAULT_CREATOR, producer=DEFAULT_PRODUCER, date=None, max_tac=3.0):
    """Device-CMYK flat artwork: <base>_CMYK.pdf, <base>_CMYK.ai (identical bytes), <base>_CMYK.eps.
    shapes: [(name, svg_path_d, (c, m, y, k)) | (..., {"fill_rule": "evenodd", "clip": d|[d..]})]
    in SVG user units (y down) on a `size` (or (w, h)) artboard; painted in order. clip_d clips
    everything (e.g. the filleted silhouette of a rounded variant)."""
    WARNINGS.clear()
    W, H = (float(size), float(size)) if isinstance(size, (int, float)) else (float(size[0]), float(size[1]))
    pw, ph = W * scale, H * scale
    B = (scale, 0.0, 0.0, -scale, 0.0, H * scale)
    ops = []
    used = [False] * 4
    tac = {}

    def clip_ops(d):
        levels = d if isinstance(d, (list, tuple)) else [d]
        out = []
        for lv in levels:
            dd, rule = (lv, "nonzero") if isinstance(lv, str) else lv
            out.append(path_ops(transform_path(parse_path_d(dd), B)) + ("\nW* n" if rule == "evenodd" else "\nW n"))
        return out

    if clip_d:
        ops.append("q")
        ops += clip_ops(clip_d)
    cur_clip = None
    for item in shapes:
        name, d, cmyk = item[0], item[1], tuple(float(v) for v in item[2])
        opt = item[3] if len(item) > 3 else {}
        if len(cmyk) != 4 or any(v < 0 or v > 1 for v in cmyk):
            raise ValueError(f"{name}: CMYK values must be 4 floats in 0..1")
        tac[name] = round(sum(cmyk) * 100, 1)
        if sum(cmyk) > max_tac + 1e-9:
            _warn(f"{name}: total ink {sum(cmyk) * 100:.0f}% exceeds {max_tac * 100:.0f}%")
        for i, v in enumerate(cmyk):
            used[i] = used[i] or v > 0
        clip = opt.get("clip")
        if clip != cur_clip:
            if cur_clip is not None:
                ops.append("Q")
            if clip is not None:
                ops.append("q")
                ops += clip_ops(clip)
            cur_clip = clip
        path = transform_path(parse_path_d(d), B)
        if not path:
            continue
        op = "f*" if opt.get("fill_rule") == "evenodd" else "f"
        ops.append(f"/SVGShape <</Name {_pdf_string(str(name))}>> BDC\n"
                   + " ".join(_n(v) for v in cmyk) + f" k\n{path_ops(path)}\n{op}\nEMC")
    if cur_clip is not None:
        ops.append("Q")
    if clip_d:
        ops.append("Q")
    content = "\n".join(ops) + "\n"
    w = _PdfWriter()
    oi = None
    if output_intent:
        prof = open(CMYK_PROFILE, "rb").read()
        pname = icc_profile_name()
        oi = {"Type": _Name("OutputIntent"), "S": _Name("GTS_PDFX"), "OutputConditionIdentifier": "Custom",
              "OutputCondition": pname, "Info": f"{pname} (Ghostscript default_cmyk.icc)",
              "RegistryName": "http://www.color.org",
              "DestOutputProfile": w.add(_Stream({"N": 4, "Alternate": _Name("DeviceCMYK")}, prof))}
    pdf = _finish_pdf(w, content, _Res(), pw, ph, title, creator, producer, date, output_intent=oi)
    os.makedirs(os.path.dirname(os.path.abspath(out_basename)) or ".", exist_ok=True)
    paths = {"pdf": out_basename + "_CMYK.pdf", "ai": out_basename + "_CMYK.ai", "eps": out_basename + "_CMYK.eps"}
    for k in ("pdf", "ai"):
        with open(paths[k], "wb") as fh:
            fh.write(pdf)
    body = []
    for line in ops:
        line = re.sub(r"^/SVGShape <</Name (.*)>> BDC$", lambda m: "% shape: " + m.group(1), line, flags=re.M)
        line = line.replace("\nEMC", "").replace("W n", "W").replace("W* n", "W*")
        body.append(line)
    inks = [nm for nm, u in zip(("Cyan", "Magenta", "Yellow", "Black"), used) if u]
    with open(paths["eps"], "wb") as fh:
        fh.write(_eps("\n".join(body), pw, ph, _ps_text(title), _ps_text(creator), date, 2, inks))
    rep = dict(paths)
    rep.update({"page_pt": (pw, ph), "ink_total_pct": tac, "process_colors": inks,
                "profile": icc_profile_name() if output_intent else None, "warnings": list(WARNINGS)})
    return rep


def svg_to_cmyk_shapes(svg_text, convert=srgb_to_cmyk):
    """Flat solid-fill SVG -> (shapes, (W, H)) for export_cmyk. Transforms are baked into the path
    data; clip-paths become per-shape clips. Raises ValueError for gradients/opacity/masks/filters."""
    doc = SvgDoc(svg_text)
    shapes, bad = [], []
    vx, vy = doc.viewbox[0], doc.viewbox[1]
    to_art = mtranslate(-vx, -vy)

    def walk(n, clips):
        if n.opacity < 1 or n.mask or n.filter:
            bad.append(n.name or _local(n.el.tag))
            return
        cl = list(clips)
        if n.clip:
            lv = doc.clip_levels(n)
            if lv is None:
                return
            for level in lv:
                p = level[0][0] if len(level) == 1 else union_for_nonzero(level)
                rule = level[0][1] if len(level) == 1 else "nonzero"
                cl.append((path_to_d(transform_path(p, to_art)), rule))
        if n.vp_clip:
            cl.append((path_to_d(transform_path(n.vp_clip, to_art)), "nonzero"))
        if n.kind == "group":
            for c in n.children:
                walk(c, cl)
            return
        st = n.style
        if st.get("visibility") in ("hidden", "collapse"):
            return
        if st.get("stroke", "none") != "none":
            bad.append((n.name or "path") + " (stroke)")
        fill = st.get("fill", "black")
        if fill == "none":
            return
        c = parse_color(fill, (0, 0, 0))
        if parse_url(fill)[0] or c is None or c[1] < 1 or parse_opacity(st.get("fill-opacity")) < 1:
            bad.append(n.name or "path")
            return
        hx = "#%02X%02X%02X" % tuple(int(round(v * 255)) for v in c[0])
        opt = {"fill_rule": st.get("fill-rule", "nonzero")}
        if cl:
            opt["clip"] = cl
        shapes.append((n.name or f"shape{len(shapes) + 1}", path_to_d(transform_path(n.path, mmul(to_art, n.ctm))),
                       convert(hx), opt))

    for n in doc.nodes:
        walk(n, [])
    if bad:
        raise ValueError("not flat solid-fill art (gradient/opacity/mask/filter/stroke): " + ", ".join(map(str, bad)))
    return shapes, (doc.viewbox[2], doc.viewbox[3])


# =========================================================================== CLI
def _trace_exports(out_dir):
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import json
    from kmark import KParams, FACETS, flat_group, svg_doc, facets, poly_d
    k = KParams.from_dict(json.load(open(os.path.join(here, "trace_params.json"))))
    svg = svg_doc(flat_group(k, trap=False), size=k.size, title="K logo - trace")
    os.makedirs(out_dir, exist_ok=True)
    r1 = export_rgb(svg, os.path.join(out_dir, "K-Logo_trace"))
    shapes = [(n, poly_d(facets(k)[n]), srgb_to_cmyk(k.palette[n])) for n in FACETS]
    r2 = export_cmyk(shapes, k.size, os.path.join(out_dir, "K-Logo_trace"), "K logo - trace (CMYK)")
    return r1, r2


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("rgb", help="layered SVG -> RGB .pdf/.ai/.eps")
    a.add_argument("svg")
    a.add_argument("out_base")
    a.add_argument("--title")
    a.add_argument("--scale", type=float, default=1.0, help="pt per SVG user unit (default 1)")
    a.add_argument("--flat-groups", action="store_true", help="inline groups instead of Form XObjects")
    a.add_argument("--ppi", type=int, default=300, help="raster resolution for filters / EPS flattening")
    b = sub.add_parser("cmyk", help="flat solid-fill SVG -> device-CMYK .pdf/.ai/.eps (ICC-converted)")
    b.add_argument("svg")
    b.add_argument("out_base")
    b.add_argument("--title")
    b.add_argument("--scale", type=float, default=1.0)
    c = sub.add_parser("color", help="print sRGB -> CMYK conversions")
    c.add_argument("hex", nargs="+")
    c.add_argument("--raw", action="store_true", help="no whole-percent snapping")
    t = sub.add_parser("trace", help="export the logo trace (RGB + CMYK)")
    t.add_argument("out_dir")
    args = ap.parse_args(argv)
    if args.cmd == "rgb":
        rep = export_rgb(open(args.svg, encoding="utf-8").read(), args.out_base, title=args.title, scale=args.scale,
                         group_mode="flat" if args.flat_groups else "xobject", raster_ppi=args.ppi, flatten_ppi=args.ppi)
        _print_report(rep)
    elif args.cmd == "cmyk":
        svg = open(args.svg, encoding="utf-8").read()
        shapes, size = svg_to_cmyk_shapes(svg)
        title = args.title or SvgDoc(svg).title or os.path.basename(args.out_base)
        _print_report(export_cmyk(shapes, size, args.out_base, title, scale=args.scale))
    elif args.cmd == "color":
        print(f"profile: {icc_profile_name()} ({CMYK_PROFILE}); sRGB (LittleCMS built-in) -> CMYK, "
              f"relative colorimetric + BPC")
        for hx in args.hex:
            raw = srgb_to_cmyk(hx, snap_pct=None)
            v = raw if args.raw else srgb_to_cmyk(hx)
            print(f"{hx.upper():8s} C {v[0] * 100:5.1f}  M {v[1] * 100:5.1f}  Y {v[2] * 100:5.1f}  K {v[3] * 100:5.1f}"
                  f"   ink {sum(v) * 100:5.1f}%   raw {tuple(round(x * 100, 1) for x in raw)}"
                  f"   proof sRGB {cmyk_to_srgb(v)}   dE76 {delta_e76(hx, v):.1f}")
    elif args.cmd == "trace":
        r1, r2 = _trace_exports(args.out_dir)
        _print_report(r1)
        _print_report(r2)
    return 0


def _print_report(rep):
    for k, v in rep.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    sys.exit(main())
