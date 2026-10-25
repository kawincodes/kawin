#!/usr/bin/env python3
"""ExtendScript (.jsx) generators that rebuild the K brand artwork NATIVELY in Adobe Illustrator
and Adobe Photoshop.

Illustrator ignores layers in PDF-based files, so the brand kit also ships scripts the client
runs inside the Adobe apps. The app itself then builds the artwork with real layers, artboards,
global swatches and live shapes, and saves the files.

API
---
    illustrator_script(variants, palette, out_path, **opts)       -> report dict
        Writes an Illustrator .jsx file. The script creates one RGB document with one ARTBOARD and
        one LAYER per variant (named). Each top-level item gets a sub-layer. Nested items become
        named groups and every facet is a pathItem whose pathPoints (anchor, leftDirection,
        rightDirection, pointType) are the exact SVG Beziers. Fills use GLOBAL process swatches
        ("K Red", ...) in the swatch group "K Brand". Opacity, clip groups, opacity masks,
        gradients (stops reference the swatches) and filters (as Live Effects) are rebuilt too.
        The script checks its own geometry by reading it back, saves <base>.pdf, <base>_*.svg and
        <base>.ai next to itself, writes <base>_illustrator-build.txt and shows a summary alert.
        opts: base (output file stem, default = script stem), title, scale (pt per artwork unit,
              default 1), gap (between artboards, default 1/8 of the largest artboard), columns,
              export_svg / export_pdf (default True), arc_tol, swatch_group ("K Brand").
    photoshop_script(variants, palette, out_path, px=4096, **opts) -> report dict
        Writes a Photoshop .jsx file. For each variant the script creates a px x px RGB document
        (8 bit, sRGB, transparent) with a layer group named after the variant. Every facet is a
        SOLID-COLOUR SHAPE LAYER, built from a path (PathPointInfo / SubPathInfo) with the Action
        Manager "make contentLayer / solidColorLayer" call that ScriptingListener records. The
        circle and rect backgrounds are live shapes. Groups become layer groups with vector masks
        for clips. Gradients become gradient fill layers, filters become layer styles or smart
        filters, and the palette is added to the Swatches panel. The script then reads colours at
        precomputed points (colour samplers) to check where the shapes landed, saves
        <base>_<variant>.psd and writes <base>_photoshop-build.txt.
        opts: base, title, ppi (stored resolution, default 300; the art is built at 72 ppi so
              path units are pixels), sample_tol (default 2 levels), arc_tol.
    svg_path_to_mlcz(d, transform=None, arc_tol=0.01, nd=4)       -> str
        Any SVG path data (M/L/H/V/C/S/Q/T/A/Z, absolute or relative, implicit repeats, compact
        arc flags) -> absolute "M/L/C/Z" path data. Arcs become cubics with a max radial error
        of arc_tol, measured in output units after `transform`.
    parse_path(d, arc_tol=0.01) -> [Subpath]   subpaths_to_d(subs)   transform_subpaths(subs, m)
    subpath_points(sp) -> [[ax, ay, lx, ly, rx, ry, smooth]]   (Illustrator's pathPoint model)
    svg_to_variant(svg_text, name=None, background="auto")        -> variant dict
        Reads a layered SVG (see INPUT below) into the variant schema.
    trace_variants()   -> (variants, palette)   the logo trace: "K Mark" and the "K Badge" circle
    es3_lint(js_text)  -> [problem strings]     ExtendScript (ES3) conformance check

Variant schema (what both generators take)
-----------------------------------------
    {name, artboard_size: S | (W, H),
     background: None | {type: 'circle'|'rect'|'squircle'|'path', fill: hex, d: optional,
                         [cx, cy, r] | [x, y, w, h, rx, ry]},
     facets: [item, ...]}                          (paint order: first = bottom)
    item (leaf)  = {name, d, fill, [fill_rule, opacity, fill_opacity, clip, mask, effects]}
    item (group) = {name, children: [item, ...], [opacity, clip, mask, effects]}
    fill   = hex | None | {type: 'linear', x1, y1, x2, y2, stops: [(offset, hex, opacity)], [id]}
                        | {type: 'radial', cx, cy, r, [r2, angle, fx, fy], stops, [id]}
             (artwork coordinates, y down; r2/angle = minor radius / major-axis angle in degrees)
    clip   = {d, rule}      mask = {clip: {d, rule}} | {soft: [items], type: 'luminance'|'alpha'}
    effects = [{type: 'drop_shadow'|'outer_glow', dx, dy, blur (Gaussian sigma), color, opacity}
               | {type: 'blur', blur}]
    The spec form (facets of {name, d, fill: hex}) is the common case. `d` may be any SVG path
    data; it is normalised with svg_path_to_mlcz.
    palette = {key: hex}. The swatch name is "K " + Key ("red" -> "K Red"). A fill colour that is
    not in the palette gets its own global swatch "K RRGGBB" and a warning.

INPUT: "layered SVG" (svg_to_variant)
-------------------------------------
Root <svg viewBox>. Named groups: <g> with inkscape:label > data-name > id. Other elements with
ids are named the same way. Leaves: path (all commands), polygon, polyline, rect (rx/ry), circle,
ellipse, plus <use> of these. Paint: hex / rgb() / rgba() / basic named colours / currentColor /
url(#linearGradient|radialGradient) with stops, stop-opacity, gradientUnits, gradientTransform and
href inheritance. Also opacity, fill-opacity, fill-rule, transform (all forms, baked into the
coordinates), clip-path (userSpaceOnUse / objectBoundingBox), mask (a white or opaque mask becomes
a clip, anything else becomes an opacity mask) and filter. Filters are recognised as drop shadow,
glow or Gaussian blur: feGaussianBlur / feOffset / feFlood / feComposite / feMerge / feBlend /
feDropShadow / Figma-style feColorMatrix chains. Any other filter is skipped with a warning.
The bottom-most top-level leaf named disc / background / bg / badge / backdrop / tile becomes
the variant background. Not supported (warned): strokes, text, images, patterns, markers,
spreadMethod other than pad, CSS <style> sheets.

Verification: test_illustrator_jsx.py (fixture + trace, node --check, ES3 lint, acorn ES3 parse,
recording mocks of the Illustrator / Photoshop DOM in jsx_mocks/, pixel comparison of the
mock-recorded documents against librsvg renders of the input).
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

__version__ = "1.0"

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"

DEFAULT_ARC_TOL = 0.01
SWATCH_GROUP = "K Brand"
BACKGROUND_NAMES = ("disc", "background", "bg", "badge", "backdrop", "tile")
HERE = os.path.dirname(os.path.abspath(__file__))


class Warnings(list):
    """Ordered, de-duplicated warning list."""

    def add(self, msg):
        if msg not in self:
            self.append(msg)


# ============================================================================ matrices
IDENT = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def mat_mul(m, n):
    """m . n (n is applied first). Matrices are SVG (a, b, c, d, e, f)."""
    a, b, c, d, e, f = m
    A, B, C, D, E, F = n
    return (a * A + c * B, b * A + d * B, a * C + c * D, b * C + d * D,
            a * E + c * F + e, b * E + d * F + f)


def mat_apply(m, x, y):
    return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])


def mat_inv(m):
    a, b, c, d, e, f = m
    det = a * d - b * c
    if abs(det) < 1e-15:
        raise ValueError("singular transform")
    ia, ib, ic, idd = d / det, -b / det, -c / det, a / det
    return (ia, ib, ic, idd, -(ia * e + ic * f), -(ib * e + idd * f))


def mat_max_sv(m):
    """Largest singular value of the linear part (max stretch)."""
    a, b, c, d = m[:4]
    s = a * a + b * b + c * c + d * d
    det = a * d - b * c
    return math.sqrt(max(0.0, (s + math.sqrt(max(0.0, s * s - 4 * det * det))) / 2))


def parse_transform(s):
    m = IDENT
    if not s:
        return m
    for name, args in re.findall(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)", s):
        v = [float(t) for t in _NUM.findall(args)]
        if name == "matrix" and len(v) == 6:
            t = tuple(v)
        elif name == "translate" and v:
            t = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0.0)
        elif name == "scale" and v:
            t = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif name == "rotate" and v:
            a = math.radians(v[0])
            ca, sa = math.cos(a), math.sin(a)
            t = (ca, sa, -sa, ca, 0, 0)
            if len(v) >= 3:
                t = mat_mul(mat_mul((1, 0, 0, 1, v[1], v[2]), t), (1, 0, 0, 1, -v[1], -v[2]))
        elif name == "skewX" and v:
            t = (1, 0, math.tan(math.radians(v[0])), 1, 0, 0)
        elif name == "skewY" and v:
            t = (1, math.tan(math.radians(v[0])), 0, 1, 0, 0)
        else:
            continue
        m = mat_mul(m, tuple(float(x) for x in t))
    return m


# ============================================================================ numbers / colours
def fmt(v, nd=4):
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "", "-") else s


def _rnd(v, nd=4):
    r = round(float(v), nd)
    return 0.0 if r == 0 else (int(r) if r == int(r) else r)


_NAMED = {
    "black": "#000000", "white": "#FFFFFF", "red": "#FF0000", "green": "#008000",
    "lime": "#00FF00", "blue": "#0000FF", "yellow": "#FFFF00", "cyan": "#00FFFF",
    "aqua": "#00FFFF", "magenta": "#FF00FF", "fuchsia": "#FF00FF", "gray": "#808080",
    "grey": "#808080", "silver": "#C0C0C0", "maroon": "#800000", "olive": "#808000",
    "navy": "#000080", "purple": "#800080", "teal": "#008080", "orange": "#FFA500",
}


def parse_color(s):
    """CSS colour -> (HEX, alpha) or None for none/transparent. Raises ValueError if unknown."""
    if s is None:
        return None
    t = s.strip()
    tl = t.lower()
    if tl in ("none", "transparent", ""):
        return None
    if tl in _NAMED:
        return _NAMED[tl], 1.0
    if t.startswith("#"):
        h = t[1:]
        if len(h) in (3, 4):
            h = "".join(c * 2 for c in h)
        if len(h) == 6:
            return "#" + h.upper(), 1.0
        if len(h) == 8:
            return "#" + h[:6].upper(), int(h[6:], 16) / 255.0
        raise ValueError(f"bad colour {s!r}")
    m = re.match(r"rgba?\(([^)]*)\)", tl)
    if m:
        parts = [p.strip() for p in re.split(r"[,\s/]+", m.group(1)) if p.strip()]
        ch = []
        for p in parts[:3]:
            v = float(p[:-1]) * 2.55 if p.endswith("%") else float(p)
            ch.append(max(0, min(255, int(round(v)))))
        a = 1.0
        if len(parts) > 3:
            a = float(parts[3][:-1]) / 100 if parts[3].endswith("%") else float(parts[3])
        return "#%02X%02X%02X" % tuple(ch), max(0.0, min(1.0, a))
    raise ValueError(f"unsupported colour {s!r}")


def hex_rgb(h):
    h = parse_color(h)[0]
    return [int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]


def parse_opacity(s, default=1.0):
    if s is None or str(s).strip() in ("", "inherit"):
        return default
    t = str(s).strip()
    v = float(t[:-1]) / 100 if t.endswith("%") else float(t)
    return max(0.0, min(1.0, v))


def _fraction(s, default):
    if s is None:
        return default
    t = str(s).strip()
    return float(t[:-1]) / 100 if t.endswith("%") else float(t)


_UNITS = {"px": 1.0, "pt": 4 / 3, "pc": 16.0, "mm": 96 / 25.4, "cm": 96 / 2.54, "in": 96.0}


def parse_length(s, ref=1.0, default=0.0):
    if s is None or str(s).strip() == "":
        return default
    t = str(s).strip()
    if t.endswith("%"):
        return float(t[:-1]) / 100 * ref
    m = re.match(r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([a-z]*)$", t)
    if not m:
        raise ValueError(f"bad length {s!r}")
    return float(m.group(1)) * _UNITS.get(m.group(2), 1.0)


def _url(s):
    """'url(#id) fallback' -> (id, fallback or None)."""
    if not s:
        return None, None
    m = re.match(r"\s*url\(\s*['\"]?#([^)'\"]+)['\"]?\s*\)\s*(.*)$", s)
    if not m:
        return None, None
    return m.group(1), (m.group(2).strip() or None)


# ============================================================================ path data
class Subpath:
    """start point + segments ('L', p) | ('C', c1, c2, p); closed flag."""
    __slots__ = ("start", "segs", "closed")

    def __init__(self, start, segs=None, closed=False):
        self.start = (float(start[0]), float(start[1]))
        self.segs = segs if segs is not None else []
        self.closed = closed

    def __repr__(self):
        return f"Subpath({self.start}, {len(self.segs)} segs, closed={self.closed})"


class _PathReader:
    def __init__(self, d):
        self.s = d
        self.i = 0
        self.n = len(d)

    def skip(self):
        while self.i < self.n and self.s[self.i] in " \t\r\n\f,":
            self.i += 1

    def eof(self):
        self.skip()
        return self.i >= self.n

    def cmd(self):
        self.skip()
        c = self.s[self.i]
        if c in "MmLlHhVvCcSsQqTtAaZz":
            self.i += 1
            return c
        return None

    def num(self):
        self.skip()
        m = _NUM.match(self.s, self.i)
        if not m:
            raise ValueError(f"path data: number expected at {self.i}: {self.s[self.i:self.i + 24]!r}")
        self.i = m.end()
        return float(m.group())

    def flag(self):
        self.skip()
        c = self.s[self.i:self.i + 1]
        if c not in ("0", "1"):
            raise ValueError(f"path data: arc flag expected at {self.i}: {self.s[self.i:self.i + 24]!r}")
        self.i += 1
        return c == "1"


def arc_segment_error(r, sweep):
    """Max radial error of the standard one-cubic approximation of a circular arc."""
    q = sweep / 4.0
    return r * 2 * math.sin(q) ** 6 / (27 * math.cos(q) ** 2)


def arc_to_cubics(x1, y1, rx, ry, phi_deg, large, sweep, x2, y2, tol=DEFAULT_ARC_TOL):
    """SVG endpoint arc -> list of ('C', c1, c2, p) (SVG 1.1 F.6.5 / F.6.6), error <= tol."""
    if abs(x1 - x2) < 1e-12 and abs(y1 - y2) < 1e-12:
        return []
    rx, ry = abs(rx), abs(ry)
    if rx < 1e-12 or ry < 1e-12:
        return [("L", (x2, y2))]
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
    coef = math.sqrt(max(0.0, num / den)) if den > 0 else 0.0
    if large == sweep:
        coef = -coef
    cxp = coef * rx * y1p / ry
    cyp = -coef * ry * x1p / rx
    cx = cp * cxp - sp * cyp + (x1 + x2) / 2.0
    cy = sp * cxp + cp * cyp + (y1 + y2) / 2.0

    def ang(ux, uy, vx, vy):
        return math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)

    t1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dt = ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dt > 0:
        dt -= 2 * math.pi
    elif sweep and dt < 0:
        dt += 2 * math.pi
    r = max(rx, ry)
    n = max(1, int(math.ceil(abs(dt) / (math.pi / 2) - 1e-9)))
    while n < 256 and arc_segment_error(r, abs(dt) / n) > tol:
        n += 1

    def mp(u, v):
        return (cx + rx * u * cp - ry * v * sp, cy + rx * u * sp + ry * v * cp)

    out = []
    for k in range(n):
        a1 = t1 + dt * k / n
        a2 = t1 + dt * (k + 1) / n
        al = 4.0 / 3.0 * math.tan((a2 - a1) / 4.0)
        c1 = mp(math.cos(a1) - al * math.sin(a1), math.sin(a1) + al * math.cos(a1))
        c2 = mp(math.cos(a2) + al * math.sin(a2), math.sin(a2) - al * math.cos(a2))
        p = (x2, y2) if k == n - 1 else mp(math.cos(a2), math.sin(a2))
        out.append(("C", c1, c2, p))
    return out


def parse_path(d, arc_tol=DEFAULT_ARC_TOL):
    """SVG path data -> [Subpath] in absolute coordinates with only L and C segments."""
    r = _PathReader(d or "")
    subs = []
    cur = None
    x = y = sx = sy = 0.0
    cmd = None
    prev = None
    last_c2 = last_q = None
    while not r.eof():
        c = r.cmd()
        if c is None:
            if cmd is None or cmd in "Zz":
                raise ValueError(f"path data: command expected at {r.i}: {r.s[r.i:r.i + 24]!r}")
            c = cmd
        cmd = c
        up = c.upper()
        rel = c != up
        if up == "Z":
            if cur is not None:
                cur.closed = True
                cur = None
            x, y = sx, sy
            prev = "Z"
            last_c2 = last_q = None
            continue
        if up == "M":
            nx, ny = r.num(), r.num()
            if rel:
                nx += x
                ny += y
            cur = Subpath((nx, ny))
            subs.append(cur)
            x, y = sx, sy = nx, ny
            cmd = "l" if rel else "L"
            prev = "M"
            last_c2 = last_q = None
            continue
        if cur is None:
            if not subs:
                raise ValueError("path data must start with M")
            cur = Subpath((sx, sy))
            subs.append(cur)
        ox, oy = (x, y) if rel else (0.0, 0.0)
        if up == "L":
            nx, ny = r.num() + ox, r.num() + oy
            cur.segs.append(("L", (nx, ny)))
            last_c2 = last_q = None
        elif up == "H":
            nx, ny = r.num() + ox, y
            cur.segs.append(("L", (nx, ny)))
            last_c2 = last_q = None
        elif up == "V":
            nx, ny = x, r.num() + oy
            cur.segs.append(("L", (nx, ny)))
            last_c2 = last_q = None
        elif up == "C":
            c1 = (r.num() + ox, r.num() + oy)
            c2 = (r.num() + ox, r.num() + oy)
            nx, ny = r.num() + ox, r.num() + oy
            cur.segs.append(("C", c1, c2, (nx, ny)))
            last_c2, last_q = c2, None
        elif up == "S":
            c1 = (2 * x - last_c2[0], 2 * y - last_c2[1]) if prev in ("C", "S") and last_c2 else (x, y)
            c2 = (r.num() + ox, r.num() + oy)
            nx, ny = r.num() + ox, r.num() + oy
            cur.segs.append(("C", c1, c2, (nx, ny)))
            last_c2, last_q = c2, None
        elif up in ("Q", "T"):
            if up == "Q":
                q = (r.num() + ox, r.num() + oy)
            else:
                q = (2 * x - last_q[0], 2 * y - last_q[1]) if prev in ("Q", "T") and last_q else (x, y)
            nx, ny = r.num() + ox, r.num() + oy
            c1 = (x + 2.0 / 3.0 * (q[0] - x), y + 2.0 / 3.0 * (q[1] - y))
            c2 = (nx + 2.0 / 3.0 * (q[0] - nx), ny + 2.0 / 3.0 * (q[1] - ny))
            cur.segs.append(("C", c1, c2, (nx, ny)))
            last_c2, last_q = None, q
        elif up == "A":
            rx, ry, phi = r.num(), r.num(), r.num()
            fa, fs = r.flag(), r.flag()
            nx, ny = r.num() + ox, r.num() + oy
            cur.segs.extend(arc_to_cubics(x, y, rx, ry, phi, fa, fs, nx, ny, arc_tol))
            last_c2 = last_q = None
        x, y = nx, ny
        prev = up
    return subs


def transform_subpaths(subs, m):
    out = []
    for sp in subs:
        n = Subpath(mat_apply(m, *sp.start), closed=sp.closed)
        for seg in sp.segs:
            n.segs.append((seg[0],) + tuple(mat_apply(m, *p) for p in seg[1:]))
        out.append(n)
    return out


def subpaths_to_d(subs, nd=4):
    out = []
    for sp in subs:
        out.append(f"M{fmt(sp.start[0], nd)} {fmt(sp.start[1], nd)}")
        for seg in sp.segs:
            if seg[0] == "L":
                out.append(f"L{fmt(seg[1][0], nd)} {fmt(seg[1][1], nd)}")
            else:
                out.append("C" + " ".join(f"{fmt(p[0], nd)} {fmt(p[1], nd)}" for p in seg[1:]))
        if sp.closed:
            out.append("Z")
    return " ".join(out)


def svg_path_to_mlcz(d, transform=None, arc_tol=DEFAULT_ARC_TOL, nd=4):
    """Any SVG path data -> absolute M/L/C/Z path data (arcs -> cubics within arc_tol)."""
    m = transform if transform is not None else IDENT
    if isinstance(m, str):
        m = parse_transform(m)
    tol = arc_tol / max(mat_max_sv(m), 1e-12)
    return subpaths_to_d(transform_subpaths(parse_path(d, tol), m), nd)


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
            sq = math.sqrt(disc)
            ts += [(-b + sq) / (2 * a), (-b - sq) / (2 * a)]
    out = []
    for t in ts:
        if 0 < t < 1:
            mt = 1 - t
            out.append(mt ** 3 * p0 + 3 * mt * mt * t * p1 + 3 * mt * t * t * p2 + t ** 3 * p3)
    return out


def subpaths_bbox(subs):
    xs, ys = [], []
    for sp in subs:
        x, y = sp.start
        xs.append(x)
        ys.append(y)
        for seg in sp.segs:
            if seg[0] == "C":
                (c1x, c1y), (c2x, c2y), (px, py) = seg[1], seg[2], seg[3]
                xs += _cubic_extrema(x, c1x, c2x, px)
                ys += _cubic_extrema(y, c1y, c2y, py)
            px, py = seg[-1]
            xs.append(px)
            ys.append(py)
            x, y = px, py
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _near(p, q, eps=1e-9):
    return abs(p[0] - q[0]) <= eps and abs(p[1] - q[1]) <= eps


def subpath_points(sp, smooth_deg=0.1):
    """Subpath -> Illustrator-style point list [[ax, ay, lx, ly, rx, ry, smooth], ...].
    left = incoming handle, right = outgoing handle. A closed subpath whose last segment ends on
    its start point is folded into the first point, so there are no duplicate anchors."""
    a = sp.start
    pts = [[a[0], a[1], a[0], a[1], a[0], a[1]]]
    for seg in sp.segs:
        last = pts[-1]
        if seg[0] == "L":
            p = seg[1]
            if _near(p, last):
                continue
            pts.append([p[0], p[1], p[0], p[1], p[0], p[1]])
        else:
            c1, c2, p = seg[1], seg[2], seg[3]
            if _near(c1, last) and _near(c2, last) and _near(p, last):
                continue
            last[4], last[5] = c1
            pts.append([p[0], p[1], c2[0], c2[1], p[0], p[1]])
    if sp.closed and len(pts) > 1 and _near(pts[-1], pts[0], 1e-7):
        last = pts.pop()
        pts[0][2], pts[0][3] = last[2], last[3]
    cos_tol = math.cos(math.radians(smooth_deg))
    for q in pts:
        v1 = (q[0] - q[2], q[1] - q[3])
        v2 = (q[4] - q[0], q[5] - q[1])
        l1, l2 = math.hypot(*v1), math.hypot(*v2)
        smooth = 0
        if l1 > 1e-9 and l2 > 1e-9 and (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2) >= cos_tol:
            smooth = 1
        q.append(smooth)
    return pts


def points_to_subpath(pts, closed):
    """Inverse of subpath_points (round-trip helper)."""
    sp = Subpath((pts[0][0], pts[0][1]), closed=closed)
    n = len(pts)
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        q, r = pts[i], pts[(i + 1) % n]
        if _near((q[4], q[5]), q) and _near((r[2], r[3]), r):
            sp.segs.append(("L", (r[0], r[1])))
        else:
            sp.segs.append(("C", (q[4], q[5]), (r[2], r[3]), (r[0], r[1])))
    return sp


def reverse_subpath(sp):
    pts = subpath_points(sp)
    rev = [[q[0], q[1], q[4], q[5], q[2], q[3], q[6]] for q in reversed(pts)]
    return points_to_subpath(rev, sp.closed)


def flatten_subpath(sp, steps=16):
    pts = [sp.start]
    x, y = sp.start
    for seg in sp.segs:
        if seg[0] == "L":
            pts.append(seg[1])
        else:
            (c1x, c1y), (c2x, c2y), (px, py) = seg[1], seg[2], seg[3]
            for k in range(1, steps + 1):
                t = k / steps
                mt = 1 - t
                pts.append((mt ** 3 * x + 3 * mt * mt * t * c1x + 3 * mt * t * t * c2x + t ** 3 * px,
                            mt ** 3 * y + 3 * mt * mt * t * c1y + 3 * mt * t * t * c2y + t ** 3 * py))
        x, y = seg[-1]
    return pts


def signed_area(sp, steps=16):
    """Shoelace area of the flattened subpath (positive = clockwise on screen, y down)."""
    p = flatten_subpath(sp, steps)
    return 0.5 * sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1]
                     for i in range(len(p)))


def ellipse_d(cx, cy, rx, ry):
    return (f"M{fmt(cx + rx)} {fmt(cy)} A{fmt(rx)} {fmt(ry)} 0 0 1 {fmt(cx)} {fmt(cy + ry)} "
            f"A{fmt(rx)} {fmt(ry)} 0 0 1 {fmt(cx - rx)} {fmt(cy)} A{fmt(rx)} {fmt(ry)} 0 0 1 {fmt(cx)} {fmt(cy - ry)} "
            f"A{fmt(rx)} {fmt(ry)} 0 0 1 {fmt(cx + rx)} {fmt(cy)} Z")


def rect_d(x, y, w, h, rx=0.0, ry=0.0):
    if rx <= 0 or ry <= 0:
        return f"M{fmt(x)} {fmt(y)} H{fmt(x + w)} V{fmt(y + h)} H{fmt(x)} Z"
    a = f"A{fmt(rx)} {fmt(ry)} 0 0 1"
    return (f"M{fmt(x + rx)} {fmt(y)} H{fmt(x + w - rx)} {a} {fmt(x + w)} {fmt(y + ry)} "
            f"V{fmt(y + h - ry)} {a} {fmt(x + w - rx)} {fmt(y + h)} H{fmt(x + rx)} {a} {fmt(x)} {fmt(y + h - ry)} "
            f"V{fmt(y + ry)} {a} {fmt(x + rx)} {fmt(y)} Z")


def squircle_d(x, y, w, h, n=5.0):
    """Superellipse |u|^n + |v|^n = 1 in the box, one cubic per quadrant (handle length fitted)."""
    def err(k):
        worst = 0.0
        for i in range(1, 64):
            t = i / 64
            mt = 1 - t
            u = mt ** 3 + 3 * mt * mt * t + 3 * mt * t * t * k
            v = 3 * mt * mt * t * k + 3 * mt * t * t + t ** 3
            worst = max(worst, abs((abs(u) ** n + abs(v) ** n) ** (1 / n) - 1))
        return worst

    lo, hi = 0.3, 1.0
    for _ in range(60):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if err(m1) < err(m2):
            hi = m2
        else:
            lo = m1
    k = (lo + hi) / 2
    cx, cy, a, b = x + w / 2, y + h / 2, w / 2, h / 2

    def P(u, v):
        return f"{fmt(cx + a * u)} {fmt(cy + b * v)}"

    return (f"M{P(1, 0)} C{P(1, k)} {P(k, 1)} {P(0, 1)} C{P(-k, 1)} {P(-1, k)} {P(-1, 0)} "
            f"C{P(-1, -k)} {P(-k, -1)} {P(0, -1)} C{P(k, -1)} {P(1, -k)} {P(1, 0)} Z")


# ============================================================================ layered SVG reader
def _local(tag):
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


_SHAPES = ("path", "polygon", "polyline", "rect", "circle", "ellipse", "line")
_CONTAINERS = ("g", "a", "switch")
_SKIP = ("defs", "title", "desc", "metadata", "clipPath", "mask", "filter", "linearGradient",
         "radialGradient", "symbol", "style", "script", "pattern", "marker", "namedview")
_INHERITED = ("fill", "fill-opacity", "fill-rule", "clip-rule", "color", "visibility", "stroke")
_DEFAULT_INH = {"fill": "black", "fill-opacity": "1", "fill-rule": "nonzero", "clip-rule": "nonzero",
                "color": "black", "visibility": "visible", "stroke": "none"}


class _Reader:
    def __init__(self, svg_text, arc_tol, warnings):
        data = svg_text.encode("utf-8") if isinstance(svg_text, str) else svg_text
        self.root = ET.fromstring(data)
        self.w = warnings
        self.arc_tol = arc_tol
        self.ids = {}
        self._styles = {}
        for el in self.root.iter():
            i = el.get("id")
            if i and i not in self.ids:
                self.ids[i] = el
            if _local(el.tag) == "style":
                warnings.add("CSS <style> sheets are not supported (use attributes or style=\"\")")
        vb = self.root.get("viewBox")
        if vb:
            vx, vy, vw, vh = [float(t) for t in _NUM.findall(vb)][:4]
        else:
            vx = vy = 0.0
            vw = parse_length(self.root.get("width"), default=1024.0)
            vh = parse_length(self.root.get("height"), default=vw)
        self.size = (vw, vh)
        self.base = (1.0, 0.0, 0.0, 1.0, -vx, -vy)

    # ---- attributes
    def style(self, el):
        k = id(el)
        if k not in self._styles:
            d = {}
            for part in (el.get("style") or "").split(";"):
                if ":" in part:
                    a, b = part.split(":", 1)
                    d[a.strip()] = b.strip()
            self._styles[k] = d
        return self._styles[k]

    def prop(self, el, name):
        v = self.style(el).get(name)
        return v if v is not None else el.get(name)

    @staticmethod
    def name_of(el, default=None):
        for key in ("{%s}label" % INKSCAPE_NS, "data-name", "id"):
            v = el.get(key)
            if v and v.strip():
                return v.strip()
        return default

    def inherit(self, el, inh):
        d = dict(inh)
        for k in _INHERITED:
            v = self.prop(el, k)
            if v is not None and v.strip() not in ("inherit", ""):
                d[k] = v.strip()
        return d

    # ---- tree
    def walk(self, parent, ctm, inh):
        out = []
        for el in parent:
            if not isinstance(el.tag, str):
                continue
            r = self.node(el, ctm, inh)
            if r is None:
                continue
            out.extend(r if isinstance(r, list) else [r])
        return out

    def node(self, el, ctm, inh):
        tag = _local(el.tag)
        if tag in _SKIP:
            return None
        if (self.prop(el, "display") or "").strip() == "none":
            return None
        m = mat_mul(ctm, parse_transform(el.get("transform")))
        inh2 = self.inherit(el, inh)
        if tag in _CONTAINERS or tag == "svg":
            if tag == "svg":
                m = mat_mul(m, (1, 0, 0, 1, parse_length(el.get("x")), parse_length(el.get("y"))))
                if el.get("viewBox"):
                    self.w.add("nested <svg> viewBox ignored (treated as a group)")
            node = {"name": self.name_of(el, "Group"), "children": self.walk(el, m, inh2), "_m": m}
        elif tag == "use":
            href = el.get("{%s}href" % XLINK_NS) or el.get("href") or ""
            target = self.ids.get(href[1:]) if href.startswith("#") else None
            if target is None:
                self.w.add(f"<use> target {href!r} not found")
                return None
            m = mat_mul(m, (1, 0, 0, 1, parse_length(el.get("x")), parse_length(el.get("y"))))
            if _local(target.tag) == "symbol":
                kids = self.walk(target, m, self.inherit(target, inh2))
            else:
                k = self.node(target, m, inh2)
                kids = [] if k is None else (k if isinstance(k, list) else [k])
            node = {"name": self.name_of(el) or self.name_of(target, "Use"), "children": kids, "_m": m}
        elif tag in _SHAPES:
            node = self.leaf(el, tag, m, inh2)
        else:
            self.w.add(f"<{tag}> is not supported and was skipped")
            return None
        if node is None:
            return None
        return self.decorate(el, node, m)

    def shape_d(self, el, tag):
        W, H = self.size
        diag = math.sqrt((W * W + H * H) / 2)
        if tag == "path":
            return el.get("d")
        if tag in ("polygon", "polyline"):
            v = [float(t) for t in _NUM.findall(el.get("points") or "")]
            if len(v) < 4:
                return None
            pts = " L".join(f"{fmt(v[i], 6)} {fmt(v[i + 1], 6)}" for i in range(0, len(v) - 1, 2))
            return "M" + pts + (" Z" if tag == "polygon" else "")
        if tag == "rect":
            x, y = parse_length(el.get("x"), W), parse_length(el.get("y"), H)
            w, h = parse_length(el.get("width"), W), parse_length(el.get("height"), H)
            if w <= 0 or h <= 0:
                return None
            rx, ry = el.get("rx"), el.get("ry")
            rx = parse_length(rx, W) if rx not in (None, "auto") else None
            ry = parse_length(ry, H) if ry not in (None, "auto") else None
            rx = ry if rx is None else rx
            ry = rx if ry is None else ry
            rx, ry = min(rx or 0.0, w / 2), min(ry or 0.0, h / 2)
            return rect_d(x, y, w, h, rx, ry)
        if tag in ("circle", "ellipse"):
            cx, cy = parse_length(el.get("cx"), W), parse_length(el.get("cy"), H)
            if tag == "circle":
                rx = ry = parse_length(el.get("r"), diag)
            else:
                rx, ry = parse_length(el.get("rx"), W), parse_length(el.get("ry"), H)
            if rx <= 0 or ry <= 0:
                return None
            return ellipse_d(cx, cy, rx, ry)
        if tag == "line":
            self.w.add("<line> has no fill area and was skipped")
        return None

    def native_shape(self, el, tag, m):
        """Live-shape parameters in artwork coordinates when the transform keeps the shape
        axis-aligned (used for backgrounds)."""
        if abs(m[1]) > 1e-12 or abs(m[2]) > 1e-12 or m[0] <= 0 or m[3] <= 0:
            return None
        W, H = self.size
        if tag == "circle" or tag == "ellipse":
            cx, cy = mat_apply(m, parse_length(el.get("cx"), W), parse_length(el.get("cy"), H))
            if tag == "circle":
                r = parse_length(el.get("r"), math.sqrt((W * W + H * H) / 2))
                rx, ry = r * m[0], r * m[3]
            else:
                rx, ry = parse_length(el.get("rx"), W) * m[0], parse_length(el.get("ry"), H) * m[3]
            if abs(rx - ry) < 1e-9:
                return {"type": "circle", "cx": cx, "cy": cy, "r": rx}
            return {"type": "ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry}
        if tag == "rect":
            x, y = mat_apply(m, parse_length(el.get("x"), W), parse_length(el.get("y"), H))
            w, h = parse_length(el.get("width"), W) * m[0], parse_length(el.get("height"), H) * m[3]
            rx, ry = el.get("rx"), el.get("ry")
            rx = parse_length(rx, W) if rx not in (None, "auto") else None
            ry = parse_length(ry, H) if ry not in (None, "auto") else None
            rx = ry if rx is None else rx
            ry = rx if ry is None else ry
            rx = min((rx or 0.0) * m[0], w / 2)
            ry = min((ry or 0.0) * m[3], h / 2)
            return {"type": "rect", "x": x, "y": y, "w": w, "h": h, "rx": rx, "ry": ry}
        return None

    def leaf(self, el, tag, m, inh):
        if inh.get("visibility") in ("hidden", "collapse"):
            return None
        d_local = self.shape_d(el, tag)
        if not d_local:
            return None
        tol = self.arc_tol / max(mat_max_sv(m), 1e-12)
        name = self.name_of(el, tag)
        try:
            subs_local = parse_path(d_local, tol)
        except ValueError as e:
            self.w.add(f"{name}: {e}; element skipped")
            return None
        if not subs_local:
            return None
        bbox_local = subpaths_bbox(subs_local)
        stroke = inh.get("stroke")
        if stroke and stroke != "none":
            self.w.add(f"{name}: strokes are not supported (outline them first); stroke ignored")
        paint = self.paint(inh.get("fill", "black"), inh, bbox_local, m, name)
        if paint is None:
            return None
        node = {"name": name, "d": subpaths_to_d(transform_subpaths(subs_local, m)), "_m": m,
                "_bbox_local": bbox_local}
        fo = parse_opacity(inh.get("fill-opacity"))
        if paint[0] == "solid":
            node["fill"] = paint[1]
            fo *= paint[2]
        else:
            node["fill"] = paint[1]
        if fo < 1:
            node["fill_opacity"] = fo
        if inh.get("fill-rule") == "evenodd":
            node["fill_rule"] = "evenodd"
        shp = self.native_shape(el, tag, m)
        if shp:
            node["_shape"] = shp
        return node

    def paint(self, value, inh, bbox_local, m, name):
        v = (value or "black").strip()
        if v == "none":
            return None
        if v.startswith("url("):
            ref, fallback = _url(v)
            g = self.ids.get(ref)
            if g is not None and _local(g.tag) in ("linearGradient", "radialGradient"):
                return self.gradient(g, bbox_local, m, name)
            self.w.add(f"{name}: paint server #{ref} is not supported; using fallback")
            v = fallback or "none"
            if v == "none":
                return None
        if v == "currentColor":
            v = inh.get("color", "black")
        try:
            c = parse_color(v)
        except ValueError as e:
            self.w.add(f"{name}: {e}; painted black")
            c = ("#000000", 1.0)
        return None if c is None else ("solid", c[0], c[1])

    # ---- gradients
    def _chain(self, g):
        chain, seen = [g], {id(g)}
        while True:
            href = g.get("{%s}href" % XLINK_NS) or g.get("href")
            if not href or not href.startswith("#"):
                break
            t = self.ids.get(href[1:])
            if t is None or id(t) in seen:
                break
            chain.append(t)
            seen.add(id(t))
            g = t
        return chain

    @staticmethod
    def _gattr(chain, name, default=None):
        for g in chain:
            v = g.get(name)
            if v is not None:
                return v
        return default

    def _stops(self, chain):
        st = []
        for g in chain:
            st = [c for c in g if _local(c.tag) == "stop"]
            if st:
                break
        out, prev = [], 0.0
        for s in st:
            off = min(1.0, max(_fraction(self.prop(s, "offset"), 0.0), prev))
            prev = off
            try:
                c = parse_color(self.prop(s, "stop-color") or "black")
            except ValueError:
                c = ("#000000", 1.0)
            col, ca = c if c else ("#000000", 0.0)
            out.append((off, col, ca * parse_opacity(self.prop(s, "stop-opacity"))))
        return out

    def gradient(self, g, bbox_local, m, name):
        chain = self._chain(g)
        stops = self._stops(chain)
        if not stops:
            return None
        if len(stops) == 1:
            return ("solid", stops[0][1], stops[0][2])
        spread = self._gattr(chain, "spreadMethod", "pad")
        if spread != "pad":
            self.w.add(f"{name}: spreadMethod={spread} is drawn as pad")
        obb = self._gattr(chain, "gradientUnits", "objectBoundingBox") == "objectBoundingBox"
        W, H = self.size
        if obb:
            if bbox_local is None or bbox_local[2] - bbox_local[0] <= 0 or bbox_local[3] - bbox_local[1] <= 0:
                return None
            x0, y0, x1, y1 = bbox_local
            M = mat_mul(m, (x1 - x0, 0, 0, y1 - y0, x0, y0))
        else:
            M = m
        M = mat_mul(M, parse_transform(self._gattr(chain, "gradientTransform")))
        diag = math.sqrt((W * W + H * H) / 2)

        def co(attr, default, ref):
            v = self._gattr(chain, attr, default)
            return _fraction(v, 0.0) if obb else parse_length(v, ref)

        gid = chain[0].get("id")
        a, b, c, d = M[:4]
        det = a * d - b * c
        if abs(det) < 1e-12:
            return ("solid", stops[-1][1], stops[-1][2])
        if _local(g.tag) == "linearGradient":
            x1, y1 = co("x1", "0%", W), co("y1", "0%", H)
            x2, y2 = co("x2", "100%", W), co("y2", "0%", H)
            vx, vy = x2 - x1, y2 - y1
            L2 = vx * vx + vy * vy
            if L2 <= 1e-18:
                return ("solid", stops[-1][1], stops[-1][2])
            gx, gy = vx / L2, vy / L2                       # grad(t) in gradient space
            ux, uy = (d * gx - b * gy) / det, (-c * gx + a * gy) / det   # A^-T . grad
            n2 = ux * ux + uy * uy
            P1 = mat_apply(M, x1, y1)
            return ("grad", {"type": "linear", "x1": P1[0], "y1": P1[1],
                             "x2": P1[0] + ux / n2, "y2": P1[1] + uy / n2, "stops": stops, "id": gid})
        cx, cy = co("cx", "50%", W), co("cy", "50%", H)
        r = co("r", "50%", diag)
        fx = co("fx", self._gattr(chain, "cx", "50%"), W)
        fy = co("fy", self._gattr(chain, "cy", "50%"), H)
        if self._gattr(chain, "fr") not in (None, "0", "0%"):
            self.w.add(f"{name}: radialGradient fr is ignored")
        if r <= 0:
            return ("solid", stops[-1][1], stops[-1][2])
        C = mat_apply(M, cx, cy)
        F = mat_apply(M, fx, fy)
        r1, r2, ang = _ellipse_axes(a * r, b * r, c * r, d * r)
        return ("grad", {"type": "radial", "cx": C[0], "cy": C[1], "r": r1, "r2": r2, "angle": ang,
                         "fx": F[0], "fy": F[1], "stops": stops, "id": gid})

    # ---- clip / mask / filter
    def clip(self, ref, m, bbox_local, name):
        el = self.ids.get(ref)
        if el is None or _local(el.tag) != "clipPath":
            self.w.add(f"{name}: clip-path #{ref} not found; ignored")
            return None
        mm = mat_mul(m, parse_transform(el.get("transform")))
        if el.get("clipPathUnits") == "objectBoundingBox":
            if bbox_local is None:
                return {"empty": True}
            x0, y0, x1, y1 = bbox_local
            mm = mat_mul(mm, (x1 - x0, 0, 0, y1 - y0, x0, y0))
        if self.prop(el, "clip-path") not in (None, "none"):
            self.w.add(f"clip-path set on <clipPath id={ref}> is ignored")
        rule0 = self.prop(el, "clip-rule") or "nonzero"
        parts = []
        for ch in el:
            tag = _local(ch.tag)
            cm = mat_mul(mm, parse_transform(ch.get("transform")))
            if tag == "use":
                href = ch.get("{%s}href" % XLINK_NS) or ch.get("href") or ""
                tgt = self.ids.get(href[1:]) if href.startswith("#") else None
                if tgt is None:
                    continue
                cm = mat_mul(mat_mul(cm, (1, 0, 0, 1, parse_length(ch.get("x")), parse_length(ch.get("y")))),
                             parse_transform(tgt.get("transform")))
                ch, tag = tgt, _local(tgt.tag)
            if tag not in _SHAPES:
                continue
            if (self.prop(ch, "display") or "") == "none" or (self.prop(ch, "visibility") or "") == "hidden":
                continue
            d = self.shape_d(ch, tag)
            if not d:
                continue
            subs = transform_subpaths(parse_path(d, self.arc_tol / max(mat_max_sv(cm), 1e-12)), cm)
            parts.append((subs, self.prop(ch, "clip-rule") or rule0))
        return _combine_clip(parts, self.w, name)

    def mask(self, ref, m, bbox_local, name):
        el = self.ids.get(ref)
        if el is None or _local(el.tag) != "mask":
            self.w.add(f"{name}: mask #{ref} not found; ignored")
            return None
        mm = m
        if el.get("maskContentUnits") == "objectBoundingBox":
            if bbox_local is None:
                return {"clip": {"empty": True}}
            x0, y0, x1, y1 = bbox_local
            mm = mat_mul(m, (x1 - x0, 0, 0, y1 - y0, x0, y0))
        kind = (self.prop(el, "mask-type") or "luminance").strip()
        nodes = self.walk(el, mm, dict(_DEFAULT_INH))
        parts, ok = [], [True]

        def visit(n, op):
            o = op * n.get("opacity", 1.0)
            if n.get("clip") or n.get("mask") or n.get("effects"):
                ok[0] = False
                return
            if "children" in n:
                for c in n["children"]:
                    visit(c, o)
                return
            f = n.get("fill")
            a = o * n.get("fill_opacity", 1.0)
            if not isinstance(f, str) or a < 1 - 1e-9 or (kind == "luminance" and f.upper() != "#FFFFFF"):
                ok[0] = False
                return
            parts.append((parse_path(n["d"]), n.get("fill_rule", "nonzero")))

        for n in nodes:
            visit(n, 1.0)
        _strip_private(nodes)
        if ok[0] and parts:
            return {"clip": _combine_clip(parts, self.w, name)}
        if not nodes:
            return {"clip": {"empty": True}}
        return {"soft": nodes, "type": kind}

    def filter(self, ref, m, bbox_local, name):
        el = self.ids.get(ref)
        if el is None or _local(el.tag) != "filter":
            self.w.add(f"{name}: filter #{ref} not found; ignored")
            return []
        obb = el.get("primitiveUnits") == "objectBoundingBox"
        bw = bh = 1.0
        if obb and bbox_local:
            bw, bh = bbox_local[2] - bbox_local[0], bbox_local[3] - bbox_local[1]
        results = {}
        last = None
        src, alpha = ("src",), ("alpha",)

        def inp(nm):
            if not nm:
                return last if last is not None else src
            if nm == "SourceGraphic":
                return src
            if nm == "SourceAlpha":
                return alpha
            return results.get(nm, ("unknown", nm))

        def flood(p):
            try:
                c = parse_color(self.prop(p, "flood-color") or "black") or ("#000000", 0.0)
            except ValueError:
                c = ("#000000", 1.0)
            return c[0], c[1] * parse_opacity(self.prop(p, "flood-opacity"))

        for p in el:
            t = _local(p.tag)
            if not t.startswith("fe"):
                continue
            i1 = inp(p.get("in"))
            if t == "feGaussianBlur":
                sd = [float(v) for v in _NUM.findall(p.get("stdDeviation") or "0")] or [0.0]
                sx, sy = sd[0] * bw, (sd[1] if len(sd) > 1 else sd[0]) * bh
                if abs(sx - sy) > 1e-9:
                    self.w.add(f"{name}: anisotropic blur approximated by its mean")
                val = ("blur", i1, (sx + sy) / 2)
            elif t == "feOffset":
                val = ("offset", i1, float(p.get("dx") or 0) * bw, float(p.get("dy") or 0) * bh)
            elif t == "feFlood":
                val = ("flood",) + flood(p)
            elif t == "feComposite":
                op = p.get("operator") or "over"
                i2 = inp(p.get("in2"))
                if op == "in" and i1[0] == "flood":
                    val = ("colored", i2, i1[1], i1[2])
                elif op == "over":
                    val = ("merge", [i2, i1])
                else:
                    val = ("unknown", "feComposite " + op)
            elif t == "feMerge":
                val = ("merge", [inp(n.get("in")) for n in p if _local(n.tag) == "feMergeNode"])
            elif t == "feBlend":
                mode = p.get("mode") or "normal"
                val = ("merge", [inp(p.get("in2")), i1]) if mode == "normal" else ("unknown", "feBlend " + mode)
            elif t == "feDropShadow":
                sd = [float(v) for v in _NUM.findall(p.get("stdDeviation") or "2")] or [2.0]
                col, op = flood(p)
                sh = ("colored", ("blur", ("offset", alpha, float(p.get("dx") or 2) * bw,
                                           float(p.get("dy") or 2) * bh), sd[0] * bw), col, op)
                val = ("merge", [sh, i1])
            elif t == "feColorMatrix":
                val = _color_matrix(p, i1)
            else:
                val = ("unknown", t)
            r = p.get("result")
            if r:
                results[r] = val
            last = val
        effects = _classify_filter(last)
        lin = math.sqrt(abs(m[0] * m[3] - m[1] * m[2]))
        out = []
        for e in effects:
            if e["type"] == "unsupported":
                self.w.add(f"{name}: filter #{ref} ({e['desc']}) has no Adobe equivalent here; skipped")
                continue
            e["blur"] = e.get("blur", 0.0) * lin
            if "dx" in e:
                e["dx"], e["dy"] = m[0] * e["dx"] + m[2] * e["dy"], m[1] * e["dx"] + m[3] * e["dy"]
            out.append(e)
        return out

    def decorate(self, el, node, m):
        name = node["name"]
        op = parse_opacity(self.prop(el, "opacity"))
        is_group = "children" in node
        if is_group:
            try:
                bbox_local = _nodes_bbox(node["children"], mat_inv(m))
            except ValueError:
                bbox_local = None
        else:
            bbox_local = node.get("_bbox_local")
        clip_ref = _url(self.prop(el, "clip-path"))[0]
        mask_ref = _url(self.prop(el, "mask"))[0]
        filt_ref = _url(self.prop(el, "filter"))[0]
        decorated = False
        if op < 1:
            node["opacity"] = op
            decorated = True
        if clip_ref:
            c = self.clip(clip_ref, m, bbox_local, name)
            if c and c.get("empty"):
                return None
            if c:
                node["clip"] = c
                decorated = True
        if mask_ref:
            mk = self.mask(mask_ref, m, bbox_local, name)
            if mk and mk.get("clip", {}).get("empty"):
                return None
            if mk:
                node["mask"] = mk
                decorated = True
        if filt_ref:
            fx = self.filter(filt_ref, m, bbox_local, name)
            for e in fx:
                if e.get("color") is None:
                    e["color"] = _average_fill(node)
            if fx:
                node["effects"] = fx
                decorated = True
        if is_group:
            if not node["children"]:
                return None
            explicit = self.name_of(el) is not None
            if not explicit and not decorated:
                return node["children"]
        return node


def _ellipse_axes(a, b, c, d):
    """Image of the unit circle under [[a, c], [b, d]] -> (major, minor, major-axis angle deg)."""
    p, q, r = a * a + c * c, a * b + c * d, b * b + d * d
    tr, det = p + r, p * r - q * q
    disc = math.sqrt(max(0.0, tr * tr / 4 - det))
    l1, l2 = tr / 2 + disc, max(0.0, tr / 2 - disc)
    ang = 0.5 * math.degrees(math.atan2(2 * q, p - r))
    return math.sqrt(l1), math.sqrt(l2), ang


def _color_matrix(p, i1):
    typ = p.get("type") or "matrix"
    v = [float(t) for t in _NUM.findall(p.get("values") or "")]
    if typ != "matrix" or len(v) != 20:
        return ("unknown", "feColorMatrix " + typ)
    rgb_const = all(abs(v[k]) < 1e-9 for k in (0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13))
    alpha_only = all(abs(v[k]) < 1e-9 for k in (15, 16, 17, 19))
    if rgb_const and alpha_only:
        if v[18] >= 50 and all(abs(v[k]) < 1e-9 for k in (4, 9, 14)):
            return i1 if i1 == ("alpha",) else ("alpha",)     # Figma "hardAlpha"
        col = "#%02X%02X%02X" % tuple(max(0, min(255, int(round(v[k] * 255)))) for k in (4, 9, 14))
        return ("colored", i1, col, max(0.0, min(1.0, v[18])))
    return ("unknown", "feColorMatrix values")


def _shadow_of(expr):
    sig2 = dx = dy = 0.0
    color, op = None, 1.0
    while True:
        k = expr[0]
        if k == "colored":
            if color is None:
                color, op = expr[2], expr[3]
            else:
                op *= expr[3]
            expr = expr[1]
        elif k == "blur":
            sig2 += expr[2] ** 2
            expr = expr[1]
        elif k == "offset":
            dx += expr[2]
            dy += expr[3]
            expr = expr[1]
        elif k in ("alpha", "src"):
            if k == "alpha" and color is None:
                color = "#000000"
            break
        else:
            return None
    return {"blur": math.sqrt(sig2), "dx": dx, "dy": dy, "color": color, "opacity": op}


def _classify_filter(final):
    if final is None or final == ("src",):
        return []
    if final[0] == "merge" and final[1] and final[1][-1] == ("src",):
        out = []
        for u in final[1][:-1]:
            sh = _shadow_of(u)
            if sh is None:
                return [{"type": "unsupported", "desc": repr(u)[:80]}]
            sh["type"] = "drop_shadow" if (abs(sh["dx"]) > 1e-9 or abs(sh["dy"]) > 1e-9) else "outer_glow"
            out.append(sh)
        return out
    if final[0] == "blur" and final[1] == ("src",):
        return [{"type": "blur", "blur": final[2]}]
    return [{"type": "unsupported", "desc": repr(final)[:80]}]


def _average_fill(node):
    acc = [0.0, 0.0, 0.0, 0.0]

    def visit(n):
        if "children" in n:
            for c in n["children"]:
                visit(c)
            return
        f = n.get("fill")
        if isinstance(f, str):
            bb = subpaths_bbox(parse_path(n["d"]))
            a = (bb[2] - bb[0]) * (bb[3] - bb[1]) if bb else 0.0
            rgb = hex_rgb(f)
            for i in range(3):
                acc[i] += rgb[i] * a
            acc[3] += a

    visit(node)
    if acc[3] <= 0:
        return "#000000"
    return "#%02X%02X%02X" % tuple(int(round(acc[i] / acc[3])) for i in range(3))


def _nodes_bbox(nodes, minv):
    bb = None
    for n in nodes:
        if "children" in n:
            bb = _bbox_union(bb, _nodes_bbox(n["children"], minv))
        elif n.get("d"):
            bb = _bbox_union(bb, subpaths_bbox(transform_subpaths(parse_path(n["d"]), minv)))
    return bb


def _combine_clip(parts, warnings, name):
    """[(subpaths, rule)] -> {d, rule}: the union of several clip children (nonzero after
    orientation normalisation), or {'empty': True}."""
    parts = [(s, r) for s, r in parts if s]
    if not parts:
        return {"empty": True}
    if len(parts) == 1:
        return {"d": subpaths_to_d(parts[0][0]), "rule": parts[0][1]}
    out = []
    for subs, rule in parts:
        if rule == "evenodd" and len(subs) > 1:
            warnings.add(f"{name}: multi-child clip with an evenodd child is approximated (nonzero union)")
        big = max(subs, key=lambda s: abs(signed_area(s)))
        if signed_area(big) < 0:
            subs = [reverse_subpath(s) for s in subs]
        out += subs
    return {"d": subpaths_to_d(out), "rule": "nonzero"}


def _strip_private(nodes):
    for n in nodes:
        for k in [k for k in n if k.startswith("_")]:
            del n[k]
        if "children" in n:
            _strip_private(n["children"])
        mk = n.get("mask")
        if mk and mk.get("soft"):
            _strip_private(mk["soft"])


def svg_to_variant(svg_text, name=None, background="auto", arc_tol=DEFAULT_ARC_TOL, warnings=None):
    """Layered SVG -> variant dict (see module docstring). warnings: optional list to extend."""
    w = Warnings()
    r = _Reader(svg_text, arc_tol, w)
    nodes = r.walk(r.root, r.base, dict(_DEFAULT_INH))
    if name is None:
        t = next((el for el in r.root if _local(el.tag) == "title"), None)
        name = (t.text or "").strip() if t is not None and t.text else "Artboard"
    names = ()
    if background == "auto":
        names = BACKGROUND_NAMES
    elif isinstance(background, str):
        names = (background.lower(),)
    elif isinstance(background, (tuple, list)):
        names = tuple(x.lower() for x in background)
    bg = None
    if nodes and names:
        n0 = nodes[0]
        if ("children" not in n0 and n0["name"].lower() in names and isinstance(n0.get("fill"), str)
                and n0.get("fill_opacity", 1) >= 1 and n0.get("opacity", 1) >= 1
                and not (n0.get("clip") or n0.get("mask") or n0.get("effects"))):
            shp = n0.get("_shape") or {}
            t = shp.get("type")
            bg = {"type": "circle" if t == "circle" else ("rect" if t == "rect" else "path"),
                  "fill": n0["fill"], "d": n0["d"]}
            if t in ("circle", "rect"):
                bg.update({k: v for k, v in shp.items() if k != "type"})
            if n0.get("fill_rule"):
                bg["fill_rule"] = n0["fill_rule"]
            nodes = nodes[1:]
    _strip_private(nodes)
    if warnings is not None:
        warnings.extend(x for x in w if x not in warnings)
    return {"name": name, "artboard_size": [r.size[0], r.size[1]], "background": bg,
            "facets": nodes, "warnings": list(w)}


# ============================================================================ normalisation
def swatch_name(key):
    k = str(key).strip()
    if k.startswith("K "):
        return k
    words = re.split(r"[\s_\-]+", k)
    return "K " + " ".join(wd[:1].upper() + wd[1:] for wd in words if wd)


class _Model:
    """App-neutral normalisation shared by both generators."""

    def __init__(self, variants, palette, arc_tol=DEFAULT_ARC_TOL, warnings=None):
        self.w = warnings if warnings is not None else Warnings()
        self.arc_tol = arc_tol
        self.swatches = []
        self.hex2sw = {}
        for key, val in (palette or {}).items():
            hx = parse_color(val if isinstance(val, str) else val["hex"])[0]
            self._add_swatch(swatch_name(key), hx, key)
        self.gradients = []
        self._grad_keys = {}
        if not variants:
            raise ValueError("no variants")
        self.variants = [self.variant(v) for v in variants]

    def _add_swatch(self, name, hx, key=None):
        names = {s["name"] for s in self.swatches}
        base, i = name, 2
        while name in names:
            name = f"{base} {i}"
            i += 1
        self.swatches.append({"name": name, "hex": hx, "rgb": hex_rgb(hx), "key": key})
        self.hex2sw.setdefault(hx, len(self.swatches) - 1)
        return len(self.swatches) - 1

    def swatch_for(self, hx, where):
        if hx not in self.hex2sw:
            idx = self._add_swatch("K " + hx[1:], hx)
            self.w.add(f"{where}: colour {hx} is not in the palette; added global swatch "
                       f"\"{self.swatches[idx]['name']}\"")
        return self.hex2sw[hx]

    def fill(self, f, where):
        if f is None or (isinstance(f, str) and f.strip().lower() == "none"):
            return None
        if isinstance(f, str):
            c = parse_color(f)
            if c is None:
                return None
            hx = c[0]
            return {"sw": self.swatch_for(hx, where), "hex": hx, "alpha": c[1]}
        typ = f.get("type")
        stops = []
        prev = 0.0
        for s in f.get("stops") or []:
            if isinstance(s, dict):
                o, col, a = s.get("offset", 0), s.get("color"), s.get("opacity", 1.0)
            else:
                o, col, a = (list(s) + [1.0])[:3]
            o = min(1.0, max(float(o), prev))
            prev = o
            c = parse_color(col) or ("#000000", 0.0)
            stops.append((o, c[0], float(a) * c[1]))
        if len(stops) < 2:
            if not stops:
                return None
            return {"sw": self.swatch_for(stops[0][1], where), "hex": stops[0][1], "alpha": stops[0][2]}
        key = (typ, tuple((round(o, 6), c, round(a, 6)) for o, c, a in stops))
        if key not in self._grad_keys:
            nm = f.get("name") or f.get("id") or f"Gradient {len(self.gradients) + 1}"
            nm = swatch_name(nm)
            taken = {g["name"] for g in self.gradients} | {s["name"] for s in self.swatches}
            base, i = nm, 2
            while nm in taken:
                nm = f"{base} {i}"
                i += 1
            self._grad_keys[key] = len(self.gradients)
            self.gradients.append({"name": nm, "type": typ, "stops": stops})
        gi = self._grad_keys[key]
        if typ == "linear":
            geom = {k: float(f[k]) for k in ("x1", "y1", "x2", "y2")}
        elif typ == "radial":
            r = float(f["r"])
            geom = {"cx": float(f["cx"]), "cy": float(f["cy"]), "r": r, "r2": float(f.get("r2", r)),
                    "angle": float(f.get("angle", 0.0)), "fx": float(f.get("fx", f["cx"])),
                    "fy": float(f.get("fy", f["cy"]))}
        else:
            raise ValueError(f"{where}: unknown gradient type {typ!r}")
        return {"g": gi, "geom": geom}

    def subs(self, d, where):
        sps = parse_path(d, self.arc_tol)
        out = []
        for sp in sps:
            pts = subpath_points(sp)
            if len(pts) < 2 and not sp.segs:
                continue
            out.append({"c": bool(sp.closed), "pts": pts, "sp": sp})
        if not out:
            self.w.add(f"{where}: empty path skipped")
        return out

    def clip(self, c, where):
        if not c:
            return None
        if isinstance(c, str):
            c = {"d": c}
        if c.get("empty"):
            return {"empty": True}
        s = self.subs(c["d"], where + " (clip)")
        return {"subs": s, "eo": c.get("rule") == "evenodd"} if s else {"empty": True}

    def effects(self, fx, where):
        out = []
        for e in fx or []:
            t = e.get("type")
            if t in ("drop_shadow", "outer_glow", "blur"):
                e = dict(e)
                e.setdefault("blur", 0.0)
                e.setdefault("dx", 0.0)
                e.setdefault("dy", 0.0)
                e.setdefault("opacity", 1.0)
                e["color"] = parse_color(e.get("color") or "#000000")[0]
                out.append(e)
            else:
                self.w.add(f"{where}: effect {t!r} not supported; skipped")
        return out

    def item(self, it):
        name = str(it.get("name") or "Item")
        op = float(it.get("opacity", 1.0))
        fx = self.effects(it.get("effects"), name)
        clips = []
        soft = None
        c = self.clip(it.get("clip"), name)
        if c:
            if c.get("empty"):
                return None
            clips.append(c)
        mk = it.get("mask")
        if mk:
            if mk.get("clip"):
                c2 = self.clip(mk["clip"], name)
                if c2 and c2.get("empty"):
                    return None
                if c2:
                    clips.append(c2)
            elif mk.get("soft"):
                kids = [n for n in (self.item(x) for x in mk["soft"]) if n]
                if not kids:
                    return None
                soft = {"ch": kids, "type": mk.get("type", "luminance")}
        if "children" in it:
            ch = [n for n in (self.item(x) for x in it["children"]) if n]
            if not ch:
                return None
            core = {"k": "g", "n": name, "ch": ch, "op": 1.0, "fx": fx}
        else:
            subs = self.subs(it.get("d") or "", name)
            fill = self.fill(it.get("fill", "#000000"), name)
            if not subs or fill is None:
                return None
            fop = float(it.get("fill_opacity", 1.0)) * fill.get("alpha", 1.0)
            core = {"k": "p", "n": name, "subs": subs, "eo": it.get("fill_rule") == "evenodd",
                    "fill": fill, "op": 1.0, "fop": fop, "fx": fx}
        node = core
        if soft:
            node = {"k": "g", "n": name, "ch": [node], "op": 1.0, "fx": [], "mask": soft}
            if core["k"] == "g":
                core["n"] = name + " (content)"
        for cl in clips:
            if node["k"] == "g" and not node.get("clip") and not node.get("mask") and not node["fx"] \
                    and node is not core:
                node["clip"] = cl
                continue
            if node["k"] == "g" and node is core and not node["fx"] and not node.get("clip"):
                node["clip"] = cl
                continue
            if node is core and core["k"] == "g":
                core["n"] = name + " (content)"
            node = {"k": "g", "n": name, "ch": [node], "op": 1.0, "fx": [], "clip": cl}
        node["op"] = op
        return node

    def background(self, bg, W, H):
        t = (bg.get("type") or "rect").lower()
        fill = self.fill(bg.get("fill"), "Background")
        if fill is None or "g" in fill:
            self.w.add("background must have a solid fill; skipped")
            return None
        node = {"n": "Background", "fill": fill, "op": 1.0, "fop": fill.get("alpha", 1.0), "fx": [],
                "eo": bg.get("fill_rule") == "evenodd"}
        has_d = bool(bg.get("d"))
        if t == "circle" and ("r" in bg or not has_d):
            r = float(bg.get("r", min(W, H) / 2))
            cx, cy = float(bg.get("cx", W / 2)), float(bg.get("cy", H / 2))
            node.update(k="e", x=cx - r, y=cy - r, w=2 * r, h=2 * r)
            node["subs"] = self.subs(ellipse_d(cx, cy, r, r), "Background")
        elif t == "rect" and ("w" in bg or not has_d):
            x, y = float(bg.get("x", 0)), float(bg.get("y", 0))
            w, h = float(bg.get("w", W)), float(bg.get("h", H))
            rx = float(bg.get("rx", 0) or 0)
            ry = float(bg.get("ry", rx) or 0)
            node.update(k="r", x=x, y=y, w=w, h=h, rx=rx, ry=ry)
            node["subs"] = self.subs(rect_d(x, y, w, h, rx, ry), "Background")
        else:
            d = bg.get("d")
            if not d:
                if t != "squircle":
                    self.w.add(f"background type {t!r} without 'd'; using a squircle")
                d = squircle_d(0, 0, W, H)
            node.update(k="p", subs=self.subs(d, "Background"))
        return node if node["subs"] else None

    def variant(self, v):
        size = v.get("artboard_size", 1024)
        W, H = (float(size), float(size)) if isinstance(size, (int, float)) else (float(size[0]), float(size[1]))
        items = []
        if v.get("background"):
            b = self.background(v["background"], W, H)
            if b:
                items.append(b)
        for it in v.get("facets") or []:
            n = self.item(it)
            if n:
                items.append(n)
        if not items:
            self.w.add(f"variant {v.get('name')!r} has nothing to draw")
        return {"name": str(v.get("name") or "Artboard"), "W": W, "H": H, "items": items}


def _walk_nodes(nodes):
    for n in nodes:
        yield n
        if n["k"] == "g":
            yield from _walk_nodes(n["ch"])
            if n.get("mask"):
                yield from _walk_nodes(n["mask"]["ch"])


# ============================================================================ JS serialisation
def _js(v):
    """ES3-safe literal (quoted keys, no trailing commas, ASCII-only strings)."""
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError("non-finite number in script data")
        r = _rnd(v, 4)
        return str(r)
    if isinstance(v, str):
        out = ['"']
        for ch in v:
            o = ord(ch)
            if ch == '"':
                out.append('\\"')
            elif ch == "\\":
                out.append("\\\\")
            elif o < 0x20 or o > 0x7E:
                out.append("\\u%04x" % o if o <= 0xFFFF else "\\u%04x\\u%04x" % (
                    0xD800 + ((o - 0x10000) >> 10), 0xDC00 + ((o - 0x10000) & 0x3FF)))
            else:
                out.append(ch)
        out.append('"')
        return "".join(out)
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_js(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(_js(str(k)) + ":" + _js(x) for k, x in v.items()) + "}"
    raise TypeError(f"cannot serialise {type(v)}")


def _pts_js(pts):
    return [[_rnd(q[0]), _rnd(q[1]), _rnd(q[2]), _rnd(q[3]), _rnd(q[4]), _rnd(q[5]), q[6]] for q in pts]


def _slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-") or "variant"


def _grid(sizes, gap, columns):
    n = len(sizes)
    cols = columns or int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    cw = max(s[0] for s in sizes)
    ch = max(s[1] for s in sizes)
    gw = cols * cw + (cols - 1) * gap
    gh = rows * ch + (rows - 1) * gap
    cells = []
    for i, (w, h) in enumerate(sizes):
        r, c = divmod(i, cols)
        cells.append([c * (cw + gap) + (cw - w) / 2, r * (ch + gap) + (ch - h) / 2, w, h])
    return cells, (gw, gh)


# ============================================================================ Illustrator
def _ai_effect(e, s):
    """SVG effect -> (Illustrator LiveEffect XML, description)."""
    rgb = hex_rgb(e["color"])
    blur = 2 * e["blur"] * s          # Illustrator blur radius ~ 2 sigma (same as CSS / Figma)
    if e["type"] == "blur":
        xml = f'<LiveEffect name="Adobe PSL Gaussian Blur"><Dict data="R blur {fmt(e["blur"] * s, 3)} "/></LiveEffect>'
        return xml, f"Gaussian blur radius {fmt(e['blur'] * s, 2)} pt"
    black = rgb == [0, 0, 0]
    if black:
        colour = "I csrc 1 R dark 100 "
        entry = ""
    else:
        colour = "I csrc 0 R dark 0 "
        entry = (f'<Entry name="sclr" valueType="F"><Fill color="2 {fmt(rgb[0] / 255, 4)} '
                 f'{fmt(rgb[1] / 255, 4)} {fmt(rgb[2] / 255, 4)}"/></Entry>')
    if e["type"] == "outer_glow":
        xml = (f'<LiveEffect name="Adobe Outer Glow"><Dict data="R opac {fmt(e["opacity"], 4)} I mode 0 '
               f'R blur {fmt(blur, 3)} {colour}">{entry}</Dict></LiveEffect>')
        desc = f"outer glow {e['color']} {round(e['opacity'] * 100)}% blur {fmt(blur, 2)} pt"
    else:
        xml = (f'<LiveEffect name="Adobe Drop Shadow"><Dict data="R horz {fmt(e["dx"] * s, 3)} '
               f'R vert {fmt(e["dy"] * s, 3)} R blur {fmt(blur, 3)} R opac {fmt(e["opacity"], 4)} I mode 0 '
               f'B usePSLBlur 1 {colour}">{entry}</Dict></LiveEffect>')
        desc = (f"drop shadow {e['color']} {round(e['opacity'] * 100)}% x {fmt(e['dx'] * s, 2)} "
                f"y {fmt(e['dy'] * s, 2)} blur {fmt(blur, 2)} pt")
    return xml, desc


def _ai_subs(subs):
    return [{"c": 1 if s["c"] else 0, "p": _pts_js(s["pts"])} for s in subs]


def _ai_fill(f, model):
    if f is None:
        return None
    if "sw" in f:
        return {"sw": f["sw"]}
    g = f["geom"]
    if model.gradients[f["g"]]["type"] == "linear":
        v = {"t": "l", "x1": g["x1"], "y1": g["y1"], "x2": g["x2"], "y2": g["y2"]}
    else:
        v = {"t": "r", "cx": g["cx"], "cy": g["cy"], "r": g["r"], "r2": g["r2"], "a": g["angle"]}
        dx, dy = g["fx"] - g["cx"], g["fy"] - g["cy"]
        if math.hypot(dx, dy) > 1e-6 * max(g["r"], 1e-9):
            # focal point -> highlight (angle in Illustrator's y-up frame, length as a fraction of r)
            v["ha"] = math.degrees(math.atan2(-dy, dx)) + g["angle"]
            v["hl"] = math.hypot(dx, dy) / g["r"]
    return {"g": f["g"], "v": v}


def _ai_node(n, model, s, counts):
    out = {"k": n["k"], "n": n["n"]}
    o = n.get("op", 1.0) * (n.get("fop", 1.0) if n["k"] != "g" else 1.0)
    if o < 1 - 1e-9:
        out["o"] = round(o, 6)
    if n["k"] == "g":
        counts["groups"] += 1
        out["ch"] = [_ai_node(c, model, s, counts) for c in n["ch"]]
        if n.get("clip"):
            out["clip"] = {"subs": _ai_subs(n["clip"]["subs"]), "eo": 1 if n["clip"]["eo"] else 0}
            counts["paths"] += len(n["clip"]["subs"])
            counts["points"] += sum(len(x["pts"]) for x in n["clip"]["subs"])
            counts["clips"] += 1
        if n.get("mask"):
            out["m"] = {"ch": [_ai_node(c, model, s, counts) for c in n["mask"]["ch"]], "t": n["mask"]["type"]}
            counts["masks"] += 1
            if n["mask"]["type"] == "alpha":
                model.w.add(f"{n['n']}: alpha mask built as an Illustrator (luminosity) opacity mask")
    else:
        out["f"] = _ai_fill(n["fill"], model)
        if n["k"] == "e":
            out.update({k: n[k] for k in ("x", "y", "w", "h")})
            counts["paths"] += 1
            counts["points"] += 4
            counts["live"] += 1
        elif n["k"] == "r":
            out.update({k: n[k] for k in ("x", "y", "w", "h", "rx", "ry")})
            counts["paths"] += 1
            counts["points"] += 8 if (n["rx"] > 0 and n["ry"] > 0) else 4
            counts["live"] += 1
        else:
            out["subs"] = _ai_subs(n["subs"])
            out["eo"] = 1 if n["eo"] else 0
            counts["paths"] += len(n["subs"])
            counts["points"] += sum(len(x["pts"]) for x in n["subs"])
            counts["leaves"] += 1
    if n.get("fx"):
        out["fx"] = []
        for e in n["fx"]:
            xml, desc = _ai_effect(e, s)
            out["fx"].append({"xml": xml, "d": desc})
            counts["effects"] += 1
    return out


def illustrator_script(variants, palette, out_path, *, base=None, title=None, scale=1.0, gap=None,
                       columns=None, export_svg=True, export_pdf=True, arc_tol=DEFAULT_ARC_TOL,
                       swatch_group=SWATCH_GROUP):
    """Write an Illustrator ExtendScript that rebuilds `variants` natively. Returns a report."""
    w = Warnings()
    model = _Model(variants, palette, arc_tol, w)
    s = float(scale)
    sizes = [(v["W"] * s, v["H"] * s) for v in model.variants]
    gap = float(gap) * s if gap is not None else round(max(max(x) for x in sizes) / 8.0)
    cells, (gw, gh) = _grid(sizes, gap, columns)
    if max(gw, gh) > 16000:
        w.add(f"artboard grid is {gw:.0f} x {gh:.0f} pt; Illustrator's classic canvas is 16383 pt")
    counts = {k: 0 for k in ("paths", "points", "groups", "clips", "masks", "live", "leaves", "effects")}
    base = base or os.path.splitext(os.path.basename(out_path))[0]
    data = {
        "title": title or f"{base} - native Illustrator rebuild",
        "base": base, "scale": s, "grid": [gw, gh], "swatchGroup": swatch_group,
        "exportSvg": bool(export_svg), "exportPdf": bool(export_pdf),
        "swatches": [{"n": x["name"], "rgb": x["rgb"], "hex": x["hex"]} for x in model.swatches],
        "gradients": [],
        "variants": [],
    }
    for g in model.gradients:
        stops = []
        for o, hx, a in g["stops"]:
            stops.append({"o": round(o * 100, 4), "sw": model.hex2sw.get(hx, -1), "rgb": hex_rgb(hx),
                          "a": round(a * 100, 4)})
        data["gradients"].append({"n": g["name"], "t": g["type"], "stops": stops})
    sublayers = 0
    for v, cell in zip(model.variants, cells):
        items = [_ai_node(n, model, s, counts) for n in v["items"]]
        sublayers += len(items)
        data["variants"].append({"n": v["name"], "ab": cell, "items": items})
    data["expect"] = {"paths": counts["paths"], "points": counts["points"]}
    js = _AI_JS.replace("__DATA__", _js(data)).replace("__TITLE__", _comment_safe(data["title"]))
    _write(out_path, js)
    problems = es3_lint(js)
    if problems:
        raise RuntimeError("generated Illustrator script is not ES3-clean: " + "; ".join(problems[:5]))
    return {
        "path": out_path, "app": "illustrator", "base": base, "bytes": len(js.encode()),
        "artboards": [{"name": v["name"], "cell": c} for v, c in zip(model.variants, cells)],
        "grid": [gw, gh], "layers": len(model.variants), "sublayers": sublayers,
        "swatches": [x["name"] for x in model.swatches],
        "gradients": [g["name"] for g in model.gradients],
        "counts": counts, "warnings": list(w), "model": model,
    }


# ============================================================================ Photoshop
def _ps_rings(subs, steps=24):
    from shapely.geometry import Polygon
    out = []
    for s in subs:
        pts = flatten_subpath(s["sp"] if "sp" in s else points_to_subpath(s["pts"], s["c"]), steps)
        if len(pts) < 3:
            out.append(None)
            continue
        g = Polygon(pts)
        if not g.is_valid:
            g = g.buffer(0)
        out.append(g)
    return out


def ps_ops(subs, evenodd, warnings=None, where=""):
    """Photoshop path-component operations reproducing an SVG fill rule.
    Returns (order, ops): component indices in build order and 'a'/'s'/'x' per component."""
    n = len(subs)
    if evenodd or n == 1:
        return list(range(n)), ["x" if evenodd else "a"] * n
    rings = _ps_rings(subs)
    sign = []
    for s in subs:
        sp = s["sp"] if "sp" in s else points_to_subpath(s["pts"], s["c"])
        sign.append(1 if signed_area(sp) >= 0 else -1)
    depth, wind = [], []
    for i in range(n):
        if rings[i] is None or rings[i].is_empty:
            depth.append(0)
            wind.append(sign[i])
            continue
        cont = [j for j in range(n) if j != i and rings[j] is not None and not rings[j].is_empty
                and rings[j].buffer(1e-6).contains(rings[i])]
        for j in range(n):
            if (j != i and j not in cont and rings[j] is not None and not rings[j].is_empty
                    and rings[i].intersects(rings[j]) and not rings[i].buffer(1e-6).contains(rings[j])
                    and rings[i].intersection(rings[j]).area > 1e-6 and warnings is not None):
                warnings.add(f"{where}: overlapping subpaths with the nonzero rule are approximated in Photoshop")
        depth.append(len(cont))
        wind.append(sign[i] + sum(sign[j] for j in cont))
    order = sorted(range(n), key=lambda i: depth[i])
    ops = ["a" if wind[i] != 0 else "s" for i in range(n)]
    return order, ops


def ps_linear_params(P1, P2, W, H):
    """Assumed geometry of a Photoshop gradient fill layer with 'Align with layer' OFF: the ramp is
    centred on the canvas centre + offset (percent of W, H), at `angle` degrees (counter-clockwise,
    y up), and at 100 % scale spans W|cos a| + H|sin a| (the canvas projected on the ramp)."""
    dx, dy = P2[0] - P1[0], P2[1] - P1[1]
    ang = math.degrees(math.atan2(-dy, dx))
    a = math.radians(ang)
    L0 = W * abs(math.cos(a)) + H * abs(math.sin(a))
    mid = ((P1[0] + P2[0]) / 2, (P1[1] + P2[1]) / 2)
    return {"angle": ang, "scale": math.hypot(dx, dy) / L0 * 100,
            "ox": (mid[0] - W / 2) / W * 100, "oy": (mid[1] - H / 2) / H * 100}


def ps_linear_points(angle, scale, ox, oy, W, H):
    """Inverse of ps_linear_params -> (P1, P2) in pixels (y down)."""
    a = math.radians(angle)
    L = (W * abs(math.cos(a)) + H * abs(math.sin(a))) * scale / 100
    mx, my = W / 2 + ox / 100 * W, H / 2 + oy / 100 * H
    ux, uy = math.cos(a), -math.sin(a)
    return (mx - ux * L / 2, my - uy * L / 2), (mx + ux * L / 2, my + uy * L / 2)


def _ps_grad(model, f, s, W, H, where):
    g = model.gradients[f["g"]]
    geo = f["geom"]
    stops = [{"l": int(round(o * 4096)), "rgb": hex_rgb(hx), "a": round(a * 100, 3)} for o, hx, a in g["stops"]]
    if g["type"] == "linear":
        p = ps_linear_params((geo["x1"] * s, geo["y1"] * s), (geo["x2"] * s, geo["y2"] * s), W, H)
        t = "linear"
    else:
        r = geo["r"] * s
        if abs(geo["r2"] - geo["r"]) > 1e-6 * geo["r"]:
            model.w.add(f"{where}: elliptical radial gradient drawn as a circle in Photoshop")
            r = math.sqrt(geo["r"] * geo["r2"]) * s
        if math.hypot(geo["fx"] - geo["cx"], geo["fy"] - geo["cy"]) > 1e-6 * geo["r"]:
            model.w.add(f"{where}: radial gradient focal point ignored in Photoshop")
        c = (geo["cx"] * s, geo["cy"] * s)
        p = ps_linear_params(c, (c[0] + 2 * r, c[1]), W, H)     # radius at 100 % = W / 2 (assumed)
        p["ox"], p["oy"] = (c[0] - W / 2) / W * 100, (c[1] - H / 2) / H * 100
        t = "radial"
    if not 10 <= p["scale"] <= 150:
        model.w.add(f"{where}: gradient scale {p['scale']:.0f}% is outside 10-150% (older Photoshop clamps it)")
    out = {"t": t, "n": g["name"], "stops": stops}
    out.update({k: round(v, 4) for k, v in p.items()})
    return out


def _ps_effect(e, s):
    if e["type"] == "blur":
        return {"t": "blur", "r": round(e["blur"] * s, 3), "d": f"Gaussian blur {e['blur'] * s:.1f} px"}
    dx, dy = e["dx"] * s, e["dy"] * s
    out = {"t": "glow" if e["type"] == "outer_glow" else "shadow", "rgb": hex_rgb(e["color"]),
           "op": round(e["opacity"] * 100, 3), "size": round(2 * e["blur"] * s, 3),
           "dist": round(math.hypot(dx, dy), 3), "angle": round(math.degrees(math.atan2(dy, -dx)), 3)}
    out["d"] = f"{out['t']} {e['color']} {out['op']:.0f}% size {out['size']:.1f} px"
    return out


def _ps_subs(subs, eo, where, w):
    order, ops = ps_ops(subs, eo, w, where)
    return [{"c": 1 if subs[i]["c"] else 0, "o": ops[i], "p": _pts_js(subs[i]["pts"])} for i in order]


def _ps_node(n, model, s, W, H, counts):
    out = {"k": n["k"], "n": n["n"]}
    if n.get("op", 1.0) < 1 - 1e-9:
        out["o"] = round(n["op"], 6)
    if n["k"] == "g":
        counts["groups"] += 1
        out["ch"] = [_ps_node(c, model, s, W, H, counts) for c in n["ch"]]
        if n.get("clip"):
            out["clip"] = {"subs": _ps_subs(n["clip"]["subs"], n["clip"]["eo"], n["n"], model.w)}
            counts["clips"] += 1
            counts["points"] += sum(len(x["pts"]) for x in n["clip"]["subs"])
        if n.get("mask"):
            # Photoshop has no vector luminance mask: the mask art's silhouette becomes a vector mask
            subs = [x for c in _walk_nodes(n["mask"]["ch"]) if c["k"] != "g" for x in c["subs"]]
            model.w.add(f"{n['n']}: soft mask approximated by its silhouette (vector mask) in Photoshop")
            if subs:
                cl = {"subs": _ps_subs(subs, False, n["n"], model.w)}
                if "clip" in out:
                    out = {"k": "g", "n": n["n"], "ch": [out], "clip": cl}
                    counts["groups"] += 1
                else:
                    out["clip"] = cl
                counts["clips"] += 1
                counts["points"] += sum(len(x["pts"]) for x in subs)
    else:
        f = n["fill"]
        out["f"] = {"g": _ps_grad(model, f, s, W, H, n["n"])} if "g" in f else {"rgb": hex_rgb(f["hex"]), "hex": f["hex"]}
        if n.get("fop", 1.0) < 1 - 1e-9:
            out["fo"] = round(n["fop"], 6)
        out["subs"] = _ps_subs(n["subs"], n["eo"], n["n"], model.w)
        counts["points"] += sum(len(x["pts"]) for x in n["subs"])
        counts["layers"] += 1
        if "g" in f:
            counts["gradients"] += 1
        else:
            counts["solid"] += 1
        if n["k"] in ("e", "r"):
            out.update({k: n[k] for k in ("x", "y", "w", "h")})
            if n["k"] == "r":
                out["rx"], out["ry"] = n["rx"], n["ry"]
            out["live"] = 1 if (n["k"] == "e" or abs(n["rx"] - n["ry"]) < 1e-9) else 0
            counts["live"] += out["live"]
    if n.get("fx"):
        out["fx"] = [_ps_effect(e, s) for e in n["fx"]]
        counts["effects"] += len(out["fx"])
    return out


def _ps_samples(vm, s, W, H, model, margin_px=8.0, max_per_leaf=3):
    """Colour-check points: inside a facet, >= margin_px from every edge, where nothing painted
    later (including its effect spread) can change the colour. Expected = the facet's colour."""
    from shapely.affinity import scale as sh_scale
    from shapely.geometry import Point
    from shapely.ops import polylabel, unary_union

    def region(subs, eo):
        rings = _ps_rings(subs)
        order, ops = ps_ops(subs, eo)
        g = None
        for i in order:
            r = rings[i]
            if r is None or r.is_empty:
                continue
            if g is None:
                g = r
            elif ops[i] == "a":
                g = g.union(r)
            elif ops[i] == "s":
                g = g.difference(r)
            else:
                g = g.symmetric_difference(r)
        return g

    entries = []

    def spread(fx):
        out = 0.0
        for e in fx or []:
            out = max(out, 3 * e.get("blur", 0.0) + math.hypot(e.get("dx", 0.0), e.get("dy", 0.0)))
        return out

    def walk(nodes, clip, ok, extra):
        for n in nodes:
            ext = extra + spread(n.get("fx"))
            if n["k"] == "g":
                cl = clip
                if n.get("clip"):
                    cg = region(n["clip"]["subs"], n["clip"]["eo"])
                    cl = cg if cl is None else cl.intersection(cg)
                good = ok and n.get("op", 1.0) >= 1 and not n.get("mask") and not n.get("fx")
                if n.get("mask"):
                    mk = unary_union([region(c["subs"], c["eo"]) for c in _walk_nodes(n["mask"]["ch"]) if c["k"] != "g"])
                    cl = mk if cl is None else cl.intersection(mk)
                walk(n["ch"], cl, good, ext)
            else:
                g = region(n["subs"], n["eo"])
                if g is None:
                    continue
                if clip is not None:
                    g = g.intersection(clip)
                good = ok and n.get("op", 1.0) >= 1 and n.get("fop", 1.0) >= 1 and not n.get("fx")
                entries.append((n, g, good, ext))

    walk(vm["items"], None, True, 0.0)
    samples = []
    m_units = margin_px / s
    for i, (n, g, good, _) in enumerate(entries):
        if not good or g.is_empty:
            continue
        later = [e[1].buffer(e[3]) if e[3] > 0 else e[1] for e in entries[i + 1:] if not e[1].is_empty]
        free = g.difference(unary_union(later)) if later else g
        core = free.buffer(-m_units)
        if core.is_empty:
            continue
        f = n["fill"]
        if "sw" in f:
            if f.get("alpha", 1.0) < 1:
                continue
            geoms = sorted(getattr(core, "geoms", [core]), key=lambda x: -x.area)
            p = polylabel(geoms[0], tolerance=max(0.5, m_units / 4))
            if not core.contains(p):
                p = geoms[0].representative_point()
            samples.append({"x": round(p.x * s, 2), "y": round(p.y * s, 2), "rgb": hex_rgb(f["hex"]),
                            "w": n["n"]})
            continue
        grad = model.gradients[f["g"]]
        geo = f["geom"]
        if grad["type"] != "linear":
            continue
        k = 0
        for o, hx, a in grad["stops"]:
            if a < 1 or k >= max_per_leaf:
                continue
            p = Point(geo["x1"] + o * (geo["x2"] - geo["x1"]), geo["y1"] + o * (geo["y2"] - geo["y1"]))
            if core.contains(p):
                samples.append({"x": round(p.x * s, 2), "y": round(p.y * s, 2), "rgb": hex_rgb(hx),
                                "w": f"{n['n']} @ stop {o:g}"})
                k += 1
    return samples


def photoshop_script(variants, palette, out_path, px=4096, *, base=None, title=None, ppi=300,
                     sample_tol=2, arc_tol=DEFAULT_ARC_TOL, profile="sRGB IEC61966-2.1"):
    """Write a Photoshop ExtendScript that rebuilds each variant as a px x px layered document."""
    w = Warnings()
    model = _Model(variants, palette, arc_tol, w)
    base = base or os.path.splitext(os.path.basename(out_path))[0]
    counts = {k: 0 for k in ("layers", "solid", "gradients", "groups", "clips", "points", "live", "effects", "samples")}
    data = {"title": title or f"{base} - native Photoshop rebuild", "base": base, "ppi": ppi,
            "profile": profile, "sampleTol": sample_tol,
            "swatches": [{"n": x["name"], "rgb": x["rgb"]} for x in model.swatches], "variants": []}
    for v in model.variants:
        s = float(px) / max(v["W"], v["H"])
        Wp, Hp = int(round(v["W"] * s)), int(round(v["H"] * s))
        items = [_ps_node(n, model, s, Wp, Hp, counts) for n in v["items"]]
        samples = _ps_samples(v, s, Wp, Hp, model)
        counts["samples"] += len(samples)
        data["variants"].append({"n": v["name"], "doc": f"{base} {v['name']}",
                                 "file": f"{base}_{_slug(v['name'])}.psd", "w": Wp, "h": Hp,
                                 "s": round(s, 10), "items": items, "samples": samples})
    js = _PS_JS.replace("__DATA__", _js(data)).replace("__TITLE__", _comment_safe(data["title"]))
    _write(out_path, js)
    problems = es3_lint(js)
    if problems:
        raise RuntimeError("generated Photoshop script is not ES3-clean: " + "; ".join(problems[:5]))
    return {"path": out_path, "app": "photoshop", "base": base, "bytes": len(js.encode()), "px": px,
            "documents": [{"name": d["doc"], "file": d["file"], "w": d["w"], "h": d["h"], "scale": d["s"],
                           "samples": len(d["samples"])} for d in data["variants"]],
            "swatches": [x["name"] for x in model.swatches], "counts": counts, "warnings": list(w),
            "model": model}


def _comment_safe(s):
    return re.sub(r"[^\x20-\x7E]", "?", s).replace("*/", "* /")


def _write(path, text):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write(text)


# ============================================================================ ES3 lint
_ES3_RESERVED = set("""break case catch continue default delete do else finally for function if in
instanceof new return switch this throw try typeof var void while with abstract boolean byte char
class const debugger double enum export extends final float goto implements import int interface
long native package private protected public short static super synchronized throws transient
volatile""".split())
_BANNED_IDENT = {"let": "let declaration", "const": "const declaration", "class": "class syntax",
                 "async": "async function", "await": "await", "yield": "generator", "of": None,
                 "import": "module syntax", "export": "module syntax", "extends": "class syntax",
                 "super": "class syntax"}
_BANNED_MEMBER = {"forEach", "map", "filter", "reduce", "reduceRight", "some", "every", "indexOf",
                  "lastIndexOf", "trim", "trimStart", "trimEnd", "bind", "includes", "startsWith",
                  "endsWith", "find", "findIndex", "fill", "padStart", "padEnd", "repeat",
                  "toISOString", "keys", "values", "entries", "assign", "isArray", "now", "create",
                  "defineProperty", "defineProperties", "getPrototypeOf", "freeze", "seal",
                  "getOwnPropertyNames", "hypot", "sign", "trunc", "log10", "log2", "cbrt", "isFinite",
                  "isInteger", "isNaN", "EPSILON", "from", "of"}
_BANNED_GLOBAL = {"JSON", "Promise", "Symbol", "Map", "Set", "WeakMap", "WeakSet", "Proxy", "Reflect",
                  "globalThis"}


def _js_tokens(src):
    """Tiny JS tokenizer: yields (kind, text, line). Comments are dropped; strings, regex
    literals and template literals are single tokens."""
    i, n, line = 0, len(src), 1
    prev = None
    punct3 = ("===", "!==", "...", "**=", ">>>", "<<=", ">>=", "&&=", "||=", "??=")
    punct2 = ("=>", "==", "!=", "<=", ">=", "&&", "||", "++", "--", "+=", "-=", "*=", "/=", "%=",
              "&=", "|=", "^=", "<<", ">>", "**", "?.", "??")
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if c in " \t\r\f\v":
            i += 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            line += src.count("\n", i, j)
            i = j
            continue
        if c in "'\"`":
            j = i + 1
            while j < n and src[j] != c:
                if src[j] == "\\":
                    j += 1
                elif src[j] == "\n" and c != "`":
                    break
                j += 1
            tok = ("template" if c == "`" else "string", src[i:j + 1], line)
            line += src.count("\n", i, j + 1)
            i = j + 1
            prev = tok
            yield tok
            continue
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            m = re.match(r"0[xX][0-9a-fA-F]+|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", src[i:])
            tok = ("num", m.group(), line)
            i += m.end()
            prev = tok
            yield tok
            continue
        if c.isalpha() or c in "_$":
            m = re.match(r"[A-Za-z_$][A-Za-z0-9_$]*", src[i:])
            tok = ("ident", m.group(), line)
            i += m.end()
            prev = tok
            yield tok
            continue
        if c == "/":
            regex_ok = prev is None or (prev[0] == "punct" and prev[1] not in (")", "]", "}")) or \
                (prev[0] == "ident" and prev[1] in ("return", "typeof", "in", "of", "case", "throw", "new", "delete", "void"))
            if regex_ok:
                j, cls = i + 1, False
                while j < n and (src[j] != "/" or cls):
                    if src[j] == "\\":
                        j += 1
                    elif src[j] == "[":
                        cls = True
                    elif src[j] == "]":
                        cls = False
                    elif src[j] == "\n":
                        break
                    j += 1
                m = re.match(r"[a-z]*", src[j + 1:])
                tok = ("regex", src[i:j + 1 + m.end()], line)
                i = j + 1 + m.end()
                prev = tok
                yield tok
                continue
        for p in punct3 + punct2:
            if src.startswith(p, i):
                tok = ("punct", p, line)
                i += len(p)
                break
        else:
            tok = ("punct", c, line)
            i += 1
        prev = tok
        yield tok


def es3_lint(js):
    """Return a list of ES3 / ExtendScript conformance problems (empty = clean)."""
    toks = list(_js_tokens(js))
    probs = []

    def add(line, msg):
        probs.append(f"line {line}: {msg}")

    for k, (kind, text, line) in enumerate(toks):
        prv = toks[k - 1] if k else None
        nxt = toks[k + 1] if k + 1 < len(toks) else None
        if kind == "template":
            add(line, "template literal (backtick)")
        elif kind == "punct":
            if text in ("=>", "...", "**", "**=", "?.", "??", "&&=", "||=", "??="):
                add(line, f"ES2015+ operator {text}")
            if text == "," and nxt and nxt[1] in ("]", "}"):
                add(line, "trailing comma")
        elif kind == "ident":
            after_dot = prv is not None and prv[1] == "."
            if after_dot:
                if text in _BANNED_MEMBER:
                    add(line, f".{text} is not ES3 / ExtendScript")
                if text in _ES3_RESERVED:
                    add(line, f"reserved word .{text} as a property name")
                continue
            if text in _BANNED_IDENT and _BANNED_IDENT[text]:
                add(line, _BANNED_IDENT[text])
            if text == "of" and prv is not None and prv[0] == "ident" and nxt and nxt[1] != ":":
                j = k - 1
                while j >= 0 and toks[j][1] != "(":
                    j -= 1
                if j > 0 and toks[j - 1][1] == "for":
                    add(line, "for...of loop")
            if text in _BANNED_GLOBAL and not (nxt and nxt[1] == ":"):
                add(line, f"{text} is not available in ExtendScript (ES3)")
            if text in ("get", "set") and nxt and nxt[0] == "ident" and k + 2 < len(toks) and toks[k + 2][1] == "(" \
                    and prv is not None and prv[1] in ("{", ","):
                add(line, "getter/setter in object literal")
            if text in _ES3_RESERVED and nxt and nxt[1] == ":" and prv is not None and prv[1] in ("{", ",") \
                    and text not in ("default", "case"):
                add(line, f"reserved word {text} as an unquoted object key")
            if text == "function" and nxt and nxt[1] == "*":
                add(line, "generator function")
            if text == "function":
                j = k + 1
                if j < len(toks) and toks[j][0] == "ident":
                    j += 1
                if j < len(toks) and toks[j][1] == "(":
                    depth, j = 0, j
                    while j < len(toks):
                        if toks[j][1] == "(":
                            depth += 1
                        elif toks[j][1] == ")":
                            depth -= 1
                            if depth == 0:
                                break
                        elif toks[j][1] == "=" and depth == 1:
                            add(toks[j][2], "default parameter value")
                        elif toks[j][1] in ("{", "[") and depth == 1:
                            add(toks[j][2], "destructuring parameter")
                        j += 1
        elif kind == "regex" and re.search(r"/[a-z]*[uys][a-z]*$", text):
            add(line, f"ES2015+ regex flag in {text}")
    return probs


# ============================================================================ the trace
def trace_params():
    sys.path.insert(0, HERE)
    from kmark import KParams
    with open(os.path.join(HERE, "trace_params.json")) as fh:
        return KParams.from_dict(json.load(fh))


def trace_variants(k=None):
    """The logo trace as two variants: "K Mark" (transparent) and "K Badge" (on the disc)."""
    sys.path.insert(0, HERE)
    from kmark import FACETS, facets, poly_d
    k = k or trace_params()
    pal = dict(k.palette)
    f = facets(k)
    items = [{"name": n, "d": poly_d(f[n]), "fill": pal[n]} for n in FACETS]
    variants = [
        {"name": "K Mark", "artboard_size": k.size, "background": None, "facets": items},
        {"name": "K Badge", "artboard_size": k.size,
         "background": {"type": "circle", "fill": pal["disc"]}, "facets": [dict(i) for i in items]},
    ]
    palette = {n: pal[n] for n in FACETS}
    palette["disc"] = pal["disc"]
    return variants, palette


# ============================================================================ JSX runtimes
_AI_JS = r"""//@target illustrator
// __TITLE__
// Rebuilds the K brand artwork natively in Adobe Illustrator: one artboard and one layer per
// variant, a sub-layer per facet, GLOBAL process swatches (swatch group "K Brand") and exact
// Bezier geometry. Generated by brand/tools/illustrator_jsx.py - regenerate, do not hand-edit.
// Run: File > Scripts > Other Script... The .ai / .pdf / .svg files and a build log are
// written next to this script.
(function () {
var DATA = __DATA__;

var LOG = [];
var CHECK = [];
var SPOTS = [];
var GRADS = [];
var STATS = { layers: 0, sublayers: 0, groups: 0, paths: 0, points: 0, live: 0, clips: 0,
              maxDev: -1, checkFail: 0, grads: 0, gradDev: 0, gradFixed: 0, gradStopDev: 0,
              fx: 0, fxFail: 0, masks: 0, maskFail: 0, saved: [] };

function note(s) { LOG.push(s); }
function bigger(v, m) { v = Math.abs(v); return (v > m) ? v : m; }
function rgbColor(c) { var o = new RGBColor(); o.red = c[0]; o.green = c[1]; o.blue = c[2]; return o; }
function spotColor(i) { var o = new SpotColor(); o.spot = SPOTS[i]; o.tint = 100; return o; }

// Artwork (SVG, y down) -> document coordinates (y up), anchored on the artboard's top-left.
function Frame(doc, i) {
  var r = doc.artboards[i].artboardRect;
  this.i = i; this.L = r[0]; this.T = r[1]; this.s = DATA.scale;
}
Frame.prototype.X = function (x) { return this.L + x * this.s; };
Frame.prototype.Y = function (y) { return this.T - y * this.s; };
Frame.prototype.P = function (x, y) { return [this.L + x * this.s, this.T - y * this.s]; };

// ---------------------------------------------------------------- swatches and gradients
function makeSwatches(doc) {
  var grp = null, i, s, spot;
  try { grp = doc.swatchGroups.add(); grp.name = DATA.swatchGroup; }
  catch (e0) { grp = null; note('Swatch group not created: ' + e0); }
  for (i = 0; i < DATA.swatches.length; i++) {
    s = DATA.swatches[i];
    spot = doc.spots.add();
    spot.name = s.n;
    spot.colorType = ColorModel.PROCESS;      // a "spot" of process type = global process swatch
    spot.color = rgbColor(s.rgb);
    SPOTS.push(spot);
    if (grp) {
      try { grp.addSpot(spot); }
      catch (e1) {
        try { grp.addSwatch(doc.swatches.getByName(s.n)); }
        catch (e2) { note('Swatch ' + s.n + ' not added to ' + DATA.swatchGroup + ': ' + e2); }
      }
    }
  }
  for (i = 0; i < DATA.gradients.length; i++) GRADS.push(makeGradient(doc, DATA.gradients[i], grp));
}

function setStop(st, d) {
  st.rampPoint = d.o;
  st.midPoint = 50;
  st.color = (d.sw >= 0) ? spotColor(d.sw) : rgbColor(d.rgb);
  try { st.opacity = d.a; } catch (e) { if (d.a < 100) note('Gradient stop opacity not supported: ' + e); }
}

function makeGradient(doc, g, grp) {
  var gr = doc.gradients.add(), n = g.stops.length, first, last, i, got, dev = 0;
  gr.name = g.n;
  gr.type = (g.t === 'radial') ? GradientType.RADIAL : GradientType.LINEAR;
  try { while (gr.gradientStops.length > 2) gr.gradientStops[1].remove(); } catch (e0) {}
  // keep object references: Illustrator re-sorts the ramp whenever a rampPoint changes
  first = gr.gradientStops[0];
  last = gr.gradientStops[gr.gradientStops.length - 1];
  setStop(first, g.stops[0]);
  setStop(last, g.stops[n - 1]);
  for (i = 1; i < n - 1; i++) setStop(gr.gradientStops.add(), g.stops[i]);
  got = [];
  for (i = 0; i < gr.gradientStops.length; i++) got.push(gr.gradientStops[i].rampPoint);
  got.sort(function (a, b) { return a - b; });
  if (got.length !== n) { note('Gradient ' + g.n + ' has ' + got.length + ' stops, expected ' + n); dev = 100; }
  else { for (i = 0; i < n; i++) dev = bigger(got[i] - g.stops[i].o, dev); }
  STATS.gradStopDev = bigger(dev, STATS.gradStopDev);
  if (grp) { try { grp.addSwatch(doc.swatches.getByName(g.n)); } catch (e1) {} }
  return gr;
}

// ---------------------------------------------------------------- paths
function setPath(p, sub, F) {
  var pts = sub.p, n = pts.length, anchors = [], i, q, pp;
  try { p.pixelAligned = false; } catch (e) {}
  for (i = 0; i < n; i++) anchors.push(F.P(pts[i][0], pts[i][1]));
  p.setEntirePath(anchors);
  for (i = 0; i < n; i++) {
    q = pts[i];
    pp = p.pathPoints[i];
    pp.pointType = q[6] ? PointType.SMOOTH : PointType.CORNER;
    pp.anchor = anchors[i];
    pp.leftDirection = F.P(q[2], q[3]);
    pp.rightDirection = F.P(q[4], q[5]);
  }
  p.closed = sub.c ? true : false;
  STATS.paths++;
  STATS.points += n;
  CHECK.push({ item: p, pts: pts, ab: F.i });
  return p;
}

function makeShape(container, subs, F, eo) {
  var i, p, cp;
  if (subs.length === 1) {
    p = setPath(container.pathItems.add(), subs[0], F);
    p.evenodd = eo ? true : false;
    return p;
  }
  cp = container.compoundPathItems.add();
  for (i = 0; i < subs.length; i++) {
    p = setPath(cp.pathItems.add(), subs[i], F);
    p.evenodd = eo ? true : false;
  }
  return cp;
}

function members(item) {
  var out = [], i;
  if (item.typename === 'CompoundPathItem') { for (i = 0; i < item.pathItems.length; i++) out.push(item.pathItems[i]); }
  else out.push(item);
  return out;
}

function liveEllipse(container, n, F) {
  var p = container.pathItems.ellipse(F.Y(n.y), F.X(n.x), n.w * F.s, n.h * F.s, false, true);
  try { p.pixelAligned = false; } catch (e) {}
  STATS.paths++; STATS.live++; STATS.points += p.pathPoints.length;
  CHECK.push({ item: p, box: [n.x, n.y, n.w, n.h], ab: F.i });
  return p;
}

function liveRect(container, n, F) {
  var p;
  if (n.rx > 0 && n.ry > 0) p = container.pathItems.roundedRectangle(F.Y(n.y), F.X(n.x), n.w * F.s, n.h * F.s, n.rx * F.s, n.ry * F.s, false);
  else p = container.pathItems.rectangle(F.Y(n.y), F.X(n.x), n.w * F.s, n.h * F.s, false);
  try { p.pixelAligned = false; } catch (e) {}
  STATS.paths++; STATS.live++; STATS.points += p.pathPoints.length;
  CHECK.push({ item: p, box: [n.x, n.y, n.w, n.h], ab: F.i });
  return p;
}

// ---------------------------------------------------------------- paint
function gradWant(v, F) {
  var x1, y1, x2, y2;
  if (v.t === 'l') {
    x1 = F.X(v.x1); y1 = F.Y(v.y1); x2 = F.X(v.x2); y2 = F.Y(v.y2);
    return [x1, y1, Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI,
            Math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))];
  }
  return [F.X(v.cx), F.Y(v.cy), -v.a, v.r * F.s];
}
function readGrad(p) { var c = p.fillColor; return [c.origin[0], c.origin[1], c.angle, c.length]; }
function angDiff(a, b) { var d = (a - b) % 360; if (d > 180) d -= 360; if (d < -180) d += 360; return d; }
function gradErr(g, w) {
  var e = bigger(g[0] - w[0], 0);
  e = bigger(g[1] - w[1], e);
  e = bigger(g[3] - w[3], e);
  return bigger(angDiff(g[2], w[2]) * Math.PI / 180 * w[3], e);
}

function gradientFill(item, ps, f, F) {
  var v = f.v, want = gradWant(v, F), gc = new GradientColor(), i, got, d, k;
  gc.gradient = GRADS[f.g];
  try { gc.origin = [want[0], want[1]]; gc.angle = want[2]; gc.length = want[3]; } catch (e0) {}
  if (v.t === 'r' && v.hl) { try { gc.hiliteAngle = v.ha; gc.hiliteLength = v.hl; } catch (e1) {} }
  for (i = 0; i < ps.length; i++) ps[i].fillColor = gc;
  STATS.grads++;
  try {
    if (v.t === 'r' && Math.abs(v.r2 - v.r) > 1e-6 * v.r) {
      // elliptical radial: level the gradient, squash it (gradient only, not the art) - the
      // correction below then turns it onto the major axis and moves it back to the centre
      got = readGrad(ps[0]);
      item.rotate(angDiff(0, got[2]), false, false, true, false, Transformation.CENTER);
      item.resize(100, 100 * v.r2 / v.r, false, false, true, false, 100, Transformation.CENTER);
    }
    got = readGrad(ps[0]);
    if (gradErr(got, want) > 1e-3) {
      STATS.gradFixed++;
      d = angDiff(want[2], got[2]);
      if (Math.abs(d) > 1e-6) { item.rotate(d, false, false, true, false, Transformation.CENTER); got = readGrad(ps[0]); }
      if (got[3] > 0 && Math.abs(got[3] - want[3]) > 1e-6) {
        k = 100 * want[3] / got[3];
        item.resize(k, k, false, false, true, false, 100, Transformation.CENTER);
        got = readGrad(ps[0]);
      }
      if (Math.abs(got[0] - want[0]) > 1e-6 || Math.abs(got[1] - want[1]) > 1e-6) {
        item.translate(want[0] - got[0], want[1] - got[1], false, false, true, false);
        got = readGrad(ps[0]);
      }
    }
    STATS.gradDev = bigger(gradErr(got, want) / F.s, STATS.gradDev);
  } catch (e2) { note('Gradient on "' + item.name + '" not verified: ' + e2); }
}

function paint(item, f, F) {
  var ps = members(item), i, p;
  for (i = 0; i < ps.length; i++) {
    p = ps[i];
    p.stroked = false;
    if (!f) { p.filled = false; continue; }
    p.filled = true;
    if (f.sw !== undefined) p.fillColor = spotColor(f.sw);
  }
  if (f && f.g !== undefined) gradientFill(item, ps, f, F);
}

function applyFx(item, list) {
  var i, e;
  for (i = 0; i < list.length; i++) {
    e = list[i];
    try { item.applyEffect(e.xml); STATS.fx++; }
    catch (err) { STATS.fxFail++; note('Effect "' + e.d + '" on "' + item.name + '" not applied (' + err + ') - add it from the Effect menu'); }
    try { item.note = (item.note ? item.note + ' | ' : '') + 'SVG filter: ' + e.d; } catch (e2) {}
  }
}

function softMask(doc, holder, content, m, F) {
  var art = holder.groupItems.add(), i;
  art.name = holder.name + ' (mask)';
  for (i = 0; i < m.ch.length; i++) build(art, m.ch[i], F);
  try {
    doc.selection = null;
    content.selected = true;
    art.selected = true;
    app.executeMenuCommand('makeMask');
    doc.selection = null;
    STATS.masks++;
  } catch (e) {
    STATS.maskFail++;
    try { art.hidden = true; } catch (e2) {}
    note('Opacity mask on "' + holder.name + '" not made (' + e + '): select "' + content.name + '" and "' + art.name + '", then Transparency > Make Mask');
  }
}

// ---------------------------------------------------------------- tree
function build(container, n, F) {
  var item, holder, i, cl, ps;
  if (n.k === 'g') {
    item = container.groupItems.add();
    item.name = n.n;
    STATS.groups++;
    holder = item;
    if (n.m) { holder = item.groupItems.add(); holder.name = n.n + ' (content)'; }
    for (i = 0; i < n.ch.length; i++) build(holder, n.ch[i], F);
    if (n.clip) {
      cl = makeShape(item, n.clip.subs, F, n.clip.eo);       // added last = topmost = the clip
      cl.name = n.n + ' (clip)';
      ps = members(cl);
      for (i = 0; i < ps.length; i++) { ps[i].filled = false; ps[i].stroked = false; }
      try { if (cl.typename === 'CompoundPathItem') ps[0].clipping = true; else cl.clipping = true; } catch (e0) {}
      item.clipped = true;
      STATS.clips++;
    }
    if (n.m) softMask(app.activeDocument, item, holder, n.m, F);
  } else {
    if (n.k === 'e') item = liveEllipse(container, n, F);
    else if (n.k === 'r') item = liveRect(container, n, F);
    else item = makeShape(container, n.subs, F, n.eo);
    item.name = n.n;
    paint(item, n.f, F);
  }
  if (n.o !== undefined) item.opacity = n.o * 100;
  if (n.fx) applyFx(item, n.fx);
  return item;
}

// Re-reads every path against the FINAL artboard rectangles (frame independent).
function verify(doc) {
  var dev = 0, i, j, c, F, pp, q, a, gb;
  for (i = 0; i < CHECK.length; i++) {
    c = CHECK[i];
    try {
      F = new Frame(doc, c.ab);
      if (c.box) {
        gb = c.item.geometricBounds;
        dev = bigger(gb[0] - F.X(c.box[0]), dev);
        dev = bigger(gb[1] - F.Y(c.box[1]), dev);
        dev = bigger(gb[2] - F.X(c.box[0] + c.box[2]), dev);
        dev = bigger(gb[3] - F.Y(c.box[1] + c.box[3]), dev);
      } else {
        pp = c.item.pathPoints;
        if (pp.length !== c.pts.length) { STATS.checkFail++; note('Path "' + c.item.name + '" has ' + pp.length + ' points, expected ' + c.pts.length); }
        else {
          for (j = 0; j < pp.length; j++) {
            q = c.pts[j];
            a = pp[j].anchor; dev = bigger(a[0] - F.X(q[0]), dev); dev = bigger(a[1] - F.Y(q[1]), dev);
            a = pp[j].leftDirection; dev = bigger(a[0] - F.X(q[2]), dev); dev = bigger(a[1] - F.Y(q[3]), dev);
            a = pp[j].rightDirection; dev = bigger(a[0] - F.X(q[4]), dev); dev = bigger(a[1] - F.Y(q[5]), dev);
          }
        }
      }
    } catch (e) { STATS.checkFail++; }
  }
  return dev / DATA.scale;
}

// ---------------------------------------------------------------- output
function outFolder() {
  var f = null;
  try { f = File($.fileName).parent; } catch (e) { f = null; }
  if (!f || !f.exists) f = Folder.desktop;
  return f;
}

function save(doc, folder) {
  var base = folder.fsName + '/' + DATA.base, o;
  if (DATA.exportPdf) {
    try {
      o = new PDFSaveOptions();
      o.compatibility = PDFCompatibility.ACROBAT7;
      o.preserveEditability = true;
      o.generateThumbnails = true;
      o.viewAfterSaving = false;
      try { o.acrobatLayers = true; } catch (e0) {}
      doc.saveAs(new File(base + '.pdf'), o);
      STATS.saved.push(DATA.base + '.pdf');
    } catch (e1) { note('PDF not saved: ' + e1); }
  }
  if (DATA.exportSvg) {
    try {
      o = new ExportOptionsSVG();
      o.embedRasterImages = true;
      o.cssProperties = SVGCSSPropertyLocation.PRESENTATIONATTRIBUTES;
      o.coordinatePrecision = 3;
      o.documentEncoding = SVGDocumentEncoding.UTF8;
      o.saveMultipleArtboards = true;
      o.artboardRange = '';
      doc.exportFile(new File(base + '.svg'), ExportType.SVG, o);
      STATS.saved.push(DATA.base + '_<artboard>.svg');
    } catch (e2) { note('SVG not exported: ' + e2); }
  }
  try {
    o = new IllustratorSaveOptions();
    o.pdfCompatible = true;
    o.embedICCProfile = true;
    o.compressed = true;
    doc.saveAs(new File(base + '.ai'), o);       // last, so the open document is the .ai
    STATS.saved.push(DATA.base + '.ai');
  } catch (e3) { note('AI not saved: ' + e3); }
}

function summary(doc) {
  var L = [], i, names = [], sw = [];
  for (i = 0; i < DATA.variants.length; i++) names.push(DATA.variants[i].n);
  for (i = 0; i < DATA.swatches.length; i++) sw.push(DATA.swatches[i].n);
  L.push(DATA.title);
  L.push('Artboards: ' + doc.artboards.length + ' (' + names.join(', ') + ')');
  L.push('Layers: ' + STATS.layers + ' variant layers, ' + STATS.sublayers + ' sub-layers, ' + STATS.groups + ' groups');
  L.push('Paths: ' + STATS.paths + ' (expected ' + DATA.expect.paths + '), anchor points: ' + STATS.points +
         ' (expected ' + DATA.expect.points + '), live shapes: ' + STATS.live + ', clipping paths: ' + STATS.clips);
  L.push('Global process swatches in "' + DATA.swatchGroup + '": ' + sw.join(', '));
  L.push('Geometry check: max anchor/handle deviation ' + STATS.maxDev.toFixed(6) + ' artwork units' +
         (STATS.checkFail ? ' (' + STATS.checkFail + ' paths could not be checked)' : ''));
  if (STATS.grads) L.push('Gradients: ' + STATS.grads + ' fills, vector deviation ' + STATS.gradDev.toFixed(6) +
         ' units, stop deviation ' + STATS.gradStopDev.toFixed(4) + '%' + (STATS.gradFixed ? ', ' + STATS.gradFixed + ' re-aimed' : ''));
  if (STATS.fx || STATS.fxFail) L.push('Effects: ' + STATS.fx + ' applied, ' + STATS.fxFail + ' failed');
  if (STATS.masks || STATS.maskFail) L.push('Opacity masks: ' + STATS.masks + ' made, ' + STATS.maskFail + ' failed');
  L.push('Saved: ' + (STATS.saved.length ? STATS.saved.join(', ') : 'nothing'));
  if (LOG.length) { L.push('Notes:'); for (i = 0; i < LOG.length; i++) L.push(' - ' + LOG[i]); }
  return L.join('\n');
}

function writeLog(folder, text) {
  var f;
  try {
    f = new File(folder.fsName + '/' + DATA.base + '_illustrator-build.txt');
    f.encoding = 'UTF-8';
    f.lineFeed = 'Unix';
    if (f.open('w')) { f.write(text + '\n'); f.close(); }
  } catch (e) {}
}

function main() {
  var prevCS = app.coordinateSystem, prevUI = app.userInteractionLevel, doc, dflt, r0, cx, cy, gl, gt,
      i, j, a, rect, ab, v, F, lay, sub, folder = null, text;
  doc = app.documents.add(DocumentColorSpace.RGB, DATA.variants[0].ab[2], DATA.variants[0].ab[3]);
  try {
    app.coordinateSystem = CoordinateSystem.DOCUMENTCOORDINATESYSTEM;
    try { doc.rulerOrigin = [0, 0]; } catch (e0) { note('rulerOrigin not set: ' + e0); }
    dflt = doc.layers[0];
    // artboards in a grid centred on the default artboard (stays inside the canvas)
    r0 = doc.artboards[0].artboardRect;
    cx = (r0[0] + r0[2]) / 2;
    cy = (r0[1] + r0[3]) / 2;
    gl = cx - DATA.grid[0] / 2;
    gt = cy + DATA.grid[1] / 2;
    for (i = 0; i < DATA.variants.length; i++) {
      a = DATA.variants[i].ab;
      rect = [gl + a[0], gt - a[1], gl + a[0] + a[2], gt - a[1] - a[3]];
      if (i === 0) { ab = doc.artboards[0]; ab.artboardRect = rect; }
      else ab = doc.artboards.add(rect);
      ab.name = DATA.variants[i].n;
    }
    makeSwatches(doc);
    // last variant first, so the Layers panel lists the variants in artboard order
    for (i = DATA.variants.length - 1; i >= 0; i--) {
      v = DATA.variants[i];
      F = new Frame(doc, i);
      lay = doc.layers.add();
      lay.name = v.n;
      STATS.layers++;
      for (j = 0; j < v.items.length; j++) {
        sub = lay.layers.add();
        sub.name = v.items[j].n;
        STATS.sublayers++;
        build(sub, v.items[j], F);
      }
    }
    if (dflt.pageItems.length === 0 && dflt.layers.length === 0) dflt.remove();
    STATS.maxDev = verify(doc);
    doc.selection = null;
    try { doc.artboards.setActiveArtboardIndex(0); app.executeMenuCommand('fitall'); } catch (e1) {}
    folder = outFolder();
    app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;
    save(doc, folder);
  } finally {
    app.userInteractionLevel = prevUI;
    app.coordinateSystem = prevCS;
  }
  text = summary(doc);
  writeLog(folder, text);
  alert(text);
}

try { main(); }
catch (err) { alert('K brand Illustrator build failed: ' + err + (err.line ? ' (script line ' + err.line + ')' : '')); }
})();
"""

_PS_JS = r"""//@target photoshop
// __TITLE__
// Rebuilds the K brand artwork natively in Adobe Photoshop: one document per variant with a
// named layer group, every facet a solid-colour SHAPE LAYER made from an exact Bezier path.
// Generated by brand/tools/illustrator_jsx.py - regenerate, do not hand-edit.
// Run: File > Scripts > Browse... The .psd files and a build log are written next to this script.
(function () {
var DATA = __DATA__;

var LOG = [];
var OPS = null;
var PATHNO = 0;
var STATS = { docs: 0, layers: 0, groups: 0, points: 0, maxDev: 0, checkFail: 0, live: 0, clips: 0,
              fx: 0, fxFail: 0, swatches: 0, samples: 0, sampleFail: 0, sampleDev: 0, saved: [] };

function note(s) { LOG.push(s); }
function bigger(v, m) { v = Math.abs(v); return (v > m) ? v : m; }
function cid(s) { return charIDToTypeID(s); }
function sid(s) { return stringIDToTypeID(s); }
function rgbDesc(c) {
  var d = new ActionDescriptor();
  d.putDouble(cid('Rd  '), c[0]);
  d.putDouble(cid('Grn '), c[1]);
  d.putDouble(cid('Bl  '), c[2]);
  return d;
}

// ---------------------------------------------------------------- paths (pixels; doc at 72 ppi)
function buildPath(doc, subs, s) {
  var spis = [], i, j, sub, pts, list, q, ppi, spi, path;
  for (i = 0; i < subs.length; i++) {
    sub = subs[i];
    pts = sub.p;
    list = [];
    for (j = 0; j < pts.length; j++) {
      q = pts[j];
      ppi = new PathPointInfo();
      ppi.kind = q[6] ? PointKind.SMOOTHPOINT : PointKind.CORNERPOINT;
      ppi.anchor = [q[0] * s, q[1] * s];
      ppi.leftDirection = [q[2] * s, q[3] * s];
      ppi.rightDirection = [q[4] * s, q[5] * s];
      list.push(ppi);
    }
    spi = new SubPathInfo();
    spi.closed = sub.c ? true : false;
    spi.operation = OPS[sub.o];
    spi.entireSubPath = list;
    spis.push(spi);
    STATS.points += pts.length;
  }
  PATHNO++;
  path = doc.pathItems.add('K build path ' + PATHNO, spis);
  checkPath(path, subs, s);
  return path;
}

function checkPath(path, subs, s) {
  var i, j, sp, q, pp, a;
  try {
    if (path.subPathItems.length !== subs.length) { STATS.checkFail++; note('Path has ' + path.subPathItems.length + ' components, expected ' + subs.length); return; }
    for (i = 0; i < subs.length; i++) {
      sp = path.subPathItems[i];
      if (sp.pathPoints.length !== subs[i].p.length) { STATS.checkFail++; continue; }
      for (j = 0; j < subs[i].p.length; j++) {
        q = subs[i].p[j];
        pp = sp.pathPoints[j];
        a = pp.anchor; STATS.maxDev = bigger(a[0] / s - q[0], STATS.maxDev); STATS.maxDev = bigger(a[1] / s - q[1], STATS.maxDev);
        a = pp.leftDirection; STATS.maxDev = bigger(a[0] / s - q[2], STATS.maxDev); STATS.maxDev = bigger(a[1] / s - q[3], STATS.maxDev);
        a = pp.rightDirection; STATS.maxDev = bigger(a[0] / s - q[4], STATS.maxDev); STATS.maxDev = bigger(a[1] / s - q[5], STATS.maxDev);
      }
    }
  } catch (e) { STATS.checkFail++; note('Path check failed: ' + e); }
}

// ---------------------------------------------------------------- fill layers (Action Manager)
// "Make a solid colour fill layer from the current path", as ScriptingListener records it.
function makeFill(typeKey, fillDesc, shape) {
  var desc = new ActionDescriptor(), ref = new ActionReference(), using = new ActionDescriptor();
  ref.putClass(sid('contentLayer'));
  desc.putReference(cid('null'), ref);
  using.putObject(cid('Type'), sid(typeKey), fillDesc);
  if (shape) using.putObject(cid('Shp '), cid(shape.cls), shape.desc);
  desc.putObject(cid('Usng'), sid('contentLayer'), using);
  executeAction(cid('Mk  '), desc, DialogModes.NO);
  return app.activeDocument.activeLayer;
}

function solidFill(rgb) {
  var d = new ActionDescriptor();
  d.putObject(cid('Clr '), cid('RGBC'), rgbDesc(rgb));
  return d;
}

function gradientFill(g) {
  var d = new ActionDescriptor(), grad = new ActionDescriptor(), clrs = new ActionList(),
      trns = new ActionList(), off = new ActionDescriptor(), i, st, cs, ts;
  d.putEnumerated(cid('Type'), cid('GrdT'), cid(g.t === 'radial' ? 'Rdl ' : 'Lnr '));
  d.putUnitDouble(cid('Angl'), cid('#Ang'), g.angle);
  d.putBoolean(cid('Algn'), false);
  d.putUnitDouble(cid('Scl '), cid('#Prc'), g.scale);
  off.putUnitDouble(cid('Hrzn'), cid('#Prc'), g.ox);
  off.putUnitDouble(cid('Vrtc'), cid('#Prc'), g.oy);
  d.putObject(cid('Ofst'), cid('Pnt '), off);
  grad.putString(cid('Nm  '), g.n);
  grad.putEnumerated(cid('GrdF'), cid('GrdF'), cid('CstS'));
  grad.putDouble(cid('Intr'), 0);
  for (i = 0; i < g.stops.length; i++) {
    st = g.stops[i];
    cs = new ActionDescriptor();
    cs.putObject(cid('Clr '), cid('RGBC'), rgbDesc(st.rgb));
    cs.putEnumerated(cid('Type'), cid('Clry'), cid('UsrS'));
    cs.putInteger(cid('Lctn'), st.l);
    cs.putInteger(cid('Mdpn'), 50);
    clrs.putObject(cid('Clrt'), cs);
    ts = new ActionDescriptor();
    ts.putUnitDouble(cid('Opct'), cid('#Prc'), st.a);
    ts.putInteger(cid('Lctn'), st.l);
    ts.putInteger(cid('Mdpn'), 50);
    trns.putObject(cid('TrnS'), ts);
  }
  grad.putList(cid('Clrs'), clrs);
  grad.putList(cid('Trns'), trns);
  d.putObject(cid('Grad'), cid('Grdn'), grad);
  return d;
}

function shapeDesc(n, s) {
  var d = new ActionDescriptor(), r;
  d.putInteger(sid('unitValueQuadVersion'), 1);
  d.putUnitDouble(cid('Top '), cid('#Pxl'), n.y * s);
  d.putUnitDouble(cid('Left'), cid('#Pxl'), n.x * s);
  d.putUnitDouble(cid('Btom'), cid('#Pxl'), (n.y + n.h) * s);
  d.putUnitDouble(cid('Rght'), cid('#Pxl'), (n.x + n.w) * s);
  if (n.k === 'r' && n.rx > 0) {
    r = n.rx * s;
    d.putUnitDouble(sid('topRight'), cid('#Pxl'), r);
    d.putUnitDouble(sid('topLeft'), cid('#Pxl'), r);
    d.putUnitDouble(sid('bottomLeft'), cid('#Pxl'), r);
    d.putUnitDouble(sid('bottomRight'), cid('#Pxl'), r);
  }
  return { cls: (n.k === 'e') ? 'Elps' : 'Rctn', desc: d };
}

// Layer > Vector Mask > Current Path (on the active layer or group)
function vectorMask(doc, subs, s) {
  var path = buildPath(doc, subs, s), desc = new ActionDescriptor(), r1 = new ActionReference(),
      r2 = new ActionReference(), r3 = new ActionReference();
  path.select();
  r1.putClass(cid('Path'));
  desc.putReference(cid('null'), r1);
  r2.putEnumerated(cid('Path'), cid('Path'), sid('vectorMask'));
  desc.putReference(cid('At  '), r2);
  r3.putEnumerated(cid('Path'), cid('Ordn'), cid('Trgt'));
  desc.putReference(cid('Usng'), r3);
  executeAction(cid('Mk  '), desc, DialogModes.NO);
  path.remove();
}

// ---------------------------------------------------------------- effects
function addStyle(fx, e) {
  var d = new ActionDescriptor(), key = (e.t === 'glow') ? 'OrGl' : 'DrSh';
  d.putBoolean(cid('enab'), true);
  d.putEnumerated(cid('Md  '), cid('BlnM'), cid('Nrml'));
  d.putObject(cid('Clr '), cid('RGBC'), rgbDesc(e.rgb));
  d.putUnitDouble(cid('Opct'), cid('#Prc'), e.op);
  if (key === 'DrSh') {
    d.putBoolean(cid('uglg'), false);
    d.putUnitDouble(cid('lagl'), cid('#Ang'), e.angle);
    d.putUnitDouble(cid('Dstn'), cid('#Pxl'), e.dist);
  } else {
    d.putEnumerated(cid('GlwT'), cid('BETE'), cid('SfBL'));
  }
  d.putUnitDouble(cid('Ckmt'), cid('#Pxl'), 0);
  d.putUnitDouble(cid('blur'), cid('#Pxl'), e.size);
  d.putUnitDouble(cid('Nose'), cid('#Prc'), 0);
  d.putBoolean(cid('AntA'), false);
  fx.putObject(cid(key), cid(key), d);
}

function setStyles(fx) {
  var desc = new ActionDescriptor(), ref = new ActionReference();
  fx.putUnitDouble(cid('Scl '), cid('#Prc'), 100);
  ref.putProperty(cid('Prpr'), cid('Lefx'));
  ref.putEnumerated(cid('Lyr '), cid('Ordn'), cid('Trgt'));
  desc.putReference(cid('null'), ref);
  desc.putObject(cid('T   '), cid('Lefx'), fx);
  executeAction(cid('setd'), desc, DialogModes.NO);
}

function smartBlur(r) {
  var d = new ActionDescriptor();
  executeAction(sid('newPlacedLayer'), undefined, DialogModes.NO);
  d.putUnitDouble(cid('Rds '), cid('#Pxl'), r);
  executeAction(cid('GsnB'), d, DialogModes.NO);
}

function applyFx(doc, lyr, list) {
  var i, e, fx = null, n = 0, descs = [];
  doc.activeLayer = lyr;
  for (i = 0; i < list.length; i++) {
    e = list[i];
    if (e.t === 'blur') {
      try { smartBlur(e.r); STATS.fx++; lyr = doc.activeLayer; }
      catch (err) { STATS.fxFail++; note('Blur on "' + lyr.name + '" not applied: ' + err); }
    } else {
      if (!fx) fx = new ActionDescriptor();
      addStyle(fx, e);
      descs.push(e.d);
      n++;
    }
  }
  if (fx) {
    try { setStyles(fx); STATS.fx += n; }
    catch (err2) { STATS.fxFail += n; note('Layer style (' + descs.join(', ') + ') on "' + lyr.name + '" not applied: ' + err2); }
  }
  return lyr;
}

// ---------------------------------------------------------------- tree
// Children are built top-down and each is moved to the BOTTOM of its group, so the final
// stacking order equals the SVG paint order wherever Photoshop first created the layer.
function place(lyr, parent) { lyr.move(parent, ElementPlacement.PLACEATEND); }

function build(doc, parent, n, s) {
  var lyr = null, path, i, set, isGrad;
  if (n.k === 'g') {
    set = parent.layerSets.add();
    set.name = n.n;
    place(set, parent);
    STATS.groups++;
    for (i = n.ch.length - 1; i >= 0; i--) build(doc, set, n.ch[i], s);
    if (n.clip) { doc.activeLayer = set; vectorMask(doc, n.clip.subs, s); STATS.clips++; }
    lyr = set;
  } else {
    isGrad = (n.f.g !== undefined);
    if (n.live && !isGrad) {
      try { lyr = makeFill('solidColorLayer', solidFill(n.f.rgb), shapeDesc(n, s)); STATS.live++; }
      catch (e0) { lyr = null; note('Live shape "' + n.n + '" built from its path instead: ' + e0); }
    }
    if (!lyr) {
      path = buildPath(doc, n.subs, s);
      path.select();
      lyr = isGrad ? makeFill('gradientLayer', gradientFill(n.f.g), null) : makeFill('solidColorLayer', solidFill(n.f.rgb), null);
      path.remove();
    }
    lyr.name = n.n;
    place(lyr, parent);
    STATS.layers++;
    if (n.fo !== undefined) lyr.fillOpacity = n.fo * 100;
  }
  if (n.o !== undefined) lyr.opacity = n.o * 100;
  if (n.fx) lyr = applyFx(doc, lyr, n.fx);
  return lyr;
}

function addSwatches() {
  var i, s, desc, ref, sw;
  for (i = 0; i < DATA.swatches.length; i++) {
    s = DATA.swatches[i];
    try {
      desc = new ActionDescriptor();
      ref = new ActionReference();
      ref.putClass(cid('Clrs'));
      desc.putReference(cid('null'), ref);
      sw = new ActionDescriptor();
      sw.putString(cid('Nm  '), s.n);
      sw.putObject(cid('Clr '), cid('RGBC'), rgbDesc(s.rgb));
      desc.putObject(cid('Usng'), cid('Clrs'), sw);
      executeAction(cid('Mk  '), desc, DialogModes.NO);
      STATS.swatches++;
    } catch (e) { note('Swatch ' + s.n + ' not added: ' + e); }
  }
}

// Colour samplers at precomputed facet interiors: proves the shapes landed where they belong.
function sampleCheck(doc, v) {
  var i, smp, cs, c, d;
  try { doc.colorSamplers.removeAll(); } catch (e0) {}
  for (i = 0; i < v.samples.length; i++) {
    smp = v.samples[i];
    try {
      cs = doc.colorSamplers.add([smp.x, smp.y]);
      c = cs.color.rgb;
      d = bigger(c.red - smp.rgb[0], 0);
      d = bigger(c.green - smp.rgb[1], d);
      d = bigger(c.blue - smp.rgb[2], d);
      cs.remove();
      STATS.samples++;
      STATS.sampleDev = bigger(d, STATS.sampleDev);
      if (d > DATA.sampleTol) {
        STATS.sampleFail++;
        note('Colour check "' + smp.w + '" in ' + v.n + ' at ' + smp.x + ', ' + smp.y + ' px: got ' +
             Math.round(c.red) + '/' + Math.round(c.green) + '/' + Math.round(c.blue) + ', expected ' + smp.rgb.join('/'));
      }
    } catch (e) { note('Colour sampler failed: ' + e); }
  }
}

function outFolder() {
  var f = null;
  try { f = File($.fileName).parent; } catch (e) { f = null; }
  if (!f || !f.exists) f = Folder.desktop;
  return f;
}

function savePsd(doc, folder, v) {
  var o;
  try {
    o = new PhotoshopSaveOptions();
    o.layers = true;
    o.embedColorProfile = true;
    o.alphaChannels = true;
    o.annotations = false;
    o.spotColors = false;
    doc.saveAs(new File(folder.fsName + '/' + v.file), o, false, Extension.LOWERCASE);
    STATS.saved.push(v.file);
  } catch (e) { note('PSD not saved (' + v.file + '): ' + e); }
}

function summary() {
  var L = [], i, names = [];
  for (i = 0; i < DATA.variants.length; i++) names.push(DATA.variants[i].n);
  L.push(DATA.title);
  L.push('Documents: ' + STATS.docs + ' (' + names.join(', ') + ')');
  L.push('Layers: ' + STATS.layers + ' shape/fill layers (' + STATS.live + ' live shapes), ' + STATS.groups + ' groups, ' + STATS.clips + ' vector masks');
  L.push('Path check: ' + STATS.points + ' points, max deviation ' + STATS.maxDev.toFixed(6) + ' artwork units' +
         (STATS.checkFail ? ' (' + STATS.checkFail + ' not checked)' : ''));
  L.push('Colour check: ' + STATS.samples + ' samples, max difference ' + STATS.sampleDev + ' levels, ' + STATS.sampleFail + ' failed');
  if (STATS.fx || STATS.fxFail) L.push('Effects: ' + STATS.fx + ' applied, ' + STATS.fxFail + ' failed');
  L.push('Swatches added: ' + STATS.swatches);
  L.push('Saved: ' + (STATS.saved.length ? STATS.saved.join(', ') : 'nothing'));
  if (LOG.length) { L.push('Notes:'); for (i = 0; i < LOG.length; i++) L.push(' - ' + LOG[i]); }
  return L.join('\n');
}

function writeLog(folder, text) {
  var f;
  try {
    f = new File(folder.fsName + '/' + DATA.base + '_photoshop-build.txt');
    f.encoding = 'UTF-8';
    f.lineFeed = 'Unix';
    if (f.open('w')) { f.write(text + '\n'); f.close(); }
  } catch (e) {}
}

function main() {
  var prevUnits = app.preferences.rulerUnits, prevDialogs = app.displayDialogs, prevMax = null,
      folder = outFolder(), i, j, v, doc, root, first, text;
  OPS = { a: ShapeOperation.SHAPEADD, s: ShapeOperation.SHAPESUBTRACT, x: ShapeOperation.SHAPEXOR, i: ShapeOperation.SHAPEINTERSECT };
  app.preferences.rulerUnits = Units.PIXELS;
  app.displayDialogs = DialogModes.NO;
  try { prevMax = app.preferences.maximizeCompatibility; app.preferences.maximizeCompatibility = QueryStateType.ALWAYS; } catch (e0) { prevMax = null; }
  try {
    for (i = 0; i < DATA.variants.length; i++) {
      v = DATA.variants[i];
      doc = app.documents.add(v.w, v.h, 72, v.doc, NewDocumentMode.RGB, DocumentFill.TRANSPARENT, 1,
                              BitsPerChannelType.EIGHT, DATA.profile);
      app.activeDocument = doc;
      STATS.docs++;
      if (i === 0) addSwatches();
      first = doc.layers[0];
      root = doc.layerSets.add();
      root.name = v.n;
      for (j = v.items.length - 1; j >= 0; j--) build(doc, root, v.items[j], v.s);
      try { first.remove(); } catch (e1) {}
      sampleCheck(doc, v);
      if (DATA.ppi !== 72) {
        try { doc.resizeImage(undefined, undefined, DATA.ppi, ResampleMethod.NONE); }
        catch (e2) { note('Resolution not set: ' + e2); }
      }
      doc.activeLayer = root;
      savePsd(doc, folder, v);
    }
  } finally {
    app.preferences.rulerUnits = prevUnits;
    app.displayDialogs = prevDialogs;
    if (prevMax !== null) { try { app.preferences.maximizeCompatibility = prevMax; } catch (e3) {} }
  }
  text = summary();
  writeLog(folder, text);
  alert(text);
}

try { main(); }
catch (err) { alert('K brand Photoshop build failed: ' + err + (err.line ? ' (script line ' + err.line + ')' : '')); }
})();
"""


# ============================================================================ CLI
def _cli(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trace", help="scripts for the logo trace (K Mark + K Badge)")
    t.add_argument("out_dir")
    t.add_argument("--base", default="K-Logo")
    t.add_argument("--px", type=int, default=4096)
    s = sub.add_parser("svg", help="scripts for layered SVG files (one variant per file)")
    s.add_argument("out_dir")
    s.add_argument("svgs", nargs="+", help="file.svg or file.svg=Variant Name")
    s.add_argument("--palette", help="JSON file {key: hex}")
    s.add_argument("--base", default="K-Logo")
    s.add_argument("--px", type=int, default=4096)
    a = ap.parse_args(argv)
    if a.cmd == "trace":
        variants, palette = trace_variants()
    else:
        variants = []
        for spec in a.svgs:
            path, _, nm = spec.partition("=")
            with open(path, encoding="utf-8") as fh:
                variants.append(svg_to_variant(fh.read(), nm or os.path.splitext(os.path.basename(path))[0]))
        if a.palette:
            with open(a.palette) as fh:
                palette = json.load(fh)
        else:
            palette = dict(trace_params().palette)
    os.makedirs(a.out_dir, exist_ok=True)
    ai = illustrator_script(variants, palette, os.path.join(a.out_dir, f"{a.base}_Illustrator.jsx"), base=a.base)
    ps = photoshop_script(variants, palette, os.path.join(a.out_dir, f"{a.base}_Photoshop.jsx"), px=a.px, base=a.base)
    for r in (ai, ps):
        print(f"{r['path']}  ({r['bytes']} bytes)")
        print("   counts:", {k: v for k, v in r["counts"].items() if v})
        for x in r["warnings"]:
            print("   warning:", x)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
