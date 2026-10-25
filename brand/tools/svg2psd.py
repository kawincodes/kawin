#!/usr/bin/env python3
"""svg2psd — convert a *layered SVG* into a layered Adobe Photoshop document (.psd).

    convert(svg_text, out_path, px=4096, dpi=300, title=None, check=False) -> dict  (report)
    python3 svg2psd.py in.svg out.psd --px 4096 [--dpi 300] [--title T] [--report r.json] [--check]

Output: PSD v1, 8-bit RGB, px x px, sRGB ICC profile embedded (image resource 1039), resolution
(1005), version info (1057), XMP with dc:title (1060) and a JPEG thumbnail (1036).  Merged
image = the full rsvg-convert render (the reference renderer, see preview.py), RLE compressed
and, when it has transparency, stored matted on white with 'Mtrn' + a negative layer count,
as Photoshop does.  Layer channels are RLE (PackBits) compressed too.  The bytes are produced
by psd-tools' own low-level structures (psd_tools.psd.*) and its writer.

Rendering: every leaf is rendered by rsvg on its own (see isolated_svg); unstroked shape layers
are rendered only inside their Bezier hull (rsvg page offset = a pixel-identical window of the
full render).  check=True re-composites the written layers with psd-tools and compares them with
the merged image (report['consistency']).  Known reference-renderer bug: librsvg 2.58 mis-renders
an objectBoundingBox clip-path inside an opacity/filter/mask group at any render scale != 1;
convert() warns when the artwork contains that pattern (the layers are right, the merged image
is not; use clipPathUnits=userSpaceOnUse).

How the SVG maps to Photoshop
-----------------------------
  <g> named by data-name | inkscape:label | id (first wins)
        -> layer group (folder) with that name; SVG paint order is kept (PSD stores layers
           bottom-first, so Photoshop lists the top-most SVG element first).  Folders at depth
           0-1 are written open, deeper ones closed (override: data-psd-open="1|0").
  unnamed <g>
        -> transparent: its children are hoisted into the parent, its transform/clip still apply.
           If it carries opacity, mix-blend-mode, filter, mask or display:none it becomes an
           anonymous folder "Group n" instead, because those act on the group as a whole.
  path / polygon / polyline / rect (rx, ry) / circle / ellipse / line with a SOLID fill,
  no filter, no mask
        -> SHAPE layer ('SoCo' solid fill + 'vmsk' vector mask + 'vstk' stroke style), flags
           bit 3 + bit 4 ("pixel data irrelevant"), plus the anti-aliased pixels cropped to the
           layer bounds for readers that ignore vectors.  Lines, arcs and quadratic curves
           become exact cubic Bezier knots.  fill-rule is honoured with path operations
           (exclude for even-odd holes, combine for overlapping same-direction rings).
           A solid stroke (no dash) is written into 'vstk' (enabled, centred, width/cap/join).
  clip-path on the shape or on any ancestor group
        -> appended to the shape's vector mask as "intersect" components (exact Beziers).
           Pixel layers simply have the clip baked into their pixels.
  gradient / pattern fill, filter, mask, <text>, <image>, <use>, nested <svg>, dashed or
  gradient stroke, a stroke combined with a clip-path / fill-opacity / anisotropic transform
        -> PIXEL layer: a copy of the SVG with only that element visible (defs kept, every
           ancestor transform and clip kept) rendered by rsvg and cropped to its bounds.
  <g> with filter or mask
        -> folder (same name) whose visible content is ONE pixel layer "<name> (rendered …)"
           holding the rendered group; its editable children follow inside, hidden.
  opacity        -> layer / folder opacity.  A folder is 'pass through' like a plain SVG <g>,
                    unless the SVG group is an isolated group (opacity < 1, mask, filter,
                    mix-blend-mode, isolation:isolate - exactly the cases rsvg isolates): then
                    it is 'norm' (isolated) or its blend mode.
                    A solid-fill shape with opacity < 1 therefore stays a SHAPE layer with that
                    layer opacity (identical result, more editable than rasterising it); pixel
                    layers keep opacity / blend as layer properties too, never baked in.
  fill-opacity   -> shape fill opacity ('iOpa'); baked into pixels for pixel layers.
  mix-blend-mode -> PSD blend mode.     display:none / visibility:hidden -> hidden layer.
  CSS            -> presentation attributes < <style> rules (simple tag/.class/#id selectors,
                    by specificity) < style="" ; fill/stroke/... inherit as in SVG.

Rare geometry that cannot be expressed with sequential path operations on the original
Bezier subpaths (self-intersecting non-zero paths, multi-ring clip of a multi-ring shape) is
written as an exactly computed polygon outline (flattened at 0.25 px, simplified within
0.05 px); the report marks such layers with geometry='flattened'.

Spec references in comments are to "Adobe Photoshop File Formats Specification" (2016+):
  §FH  File Header Section            §IR  Image Resources Section / Image resource IDs
  §LM  Layer and Mask Information      §LR  Layer records    §CI  Channel image data
  §AL  Additional Layer Information (tagged blocks)          §PR  Path resource format
  §DS  Descriptor structure            §ID  Image Data Section
"""
from __future__ import annotations

import argparse
import copy
import datetime
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from xml.sax.saxutils import escape as _xml_escape

import numpy as np
from PIL import Image, ImageCms, ImageColor

from psd_tools.compression import compress
from psd_tools.constants import (BlendMode, ChannelID, Clipping, ColorMode, Compression, Resource,
                                 SectionDivider, Tag)
from psd_tools.psd import PSD
from psd_tools.psd.color_mode_data import ColorModeData
from psd_tools.psd.descriptor import (Bool, Descriptor, DescriptorBlock, Double, Enumerated, Integer,
                                      UnitFloat)
from psd_tools.psd.descriptor import List as DescList
from psd_tools.psd.header import FileHeader
from psd_tools.psd.image_data import ImageData
from psd_tools.psd.image_resources import (ImageResource, ImageResources, ResoulutionInfo,
                                           ThumbnailResource, VersionInfo)
from psd_tools.psd.layer_and_mask import (ChannelData, ChannelDataList, ChannelImageData, ChannelInfo,
                                          GlobalLayerMaskInfo, LayerAndMaskInformation,
                                          LayerBlendingRanges, LayerFlags, LayerInfo, LayerRecord,
                                          LayerRecords)
from psd_tools.psd.tagged_blocks import SectionDividerSetting, TaggedBlock, TaggedBlocks
from psd_tools.psd.vector import (ClosedKnotLinked, ClosedKnotUnlinked, ClosedPath, InitialFillRule,
                                  OpenKnotLinked, OpenKnotUnlinked, OpenPath, PathFillRule,
                                  VectorMaskSetting)
from psd_tools.psd.vector import Path as VPath
from psd_tools.terminology import Unit

try:  # the shared reference renderer (rsvg-convert); identical flags are used as a fallback
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preview import svg_to_png as _svg_to_png  # type: ignore
except Exception:  # pragma: no cover
    def _svg_to_png(svg: str, size: int) -> Image.Image:
        out = subprocess.run(["rsvg-convert", "-w", str(size), "-h", str(size), "-f", "png"],
                             input=svg.encode(), capture_output=True, check=True).stdout
        return Image.open(io.BytesIO(out)).convert("RGBA")

__all__ = ["convert", "render_svg", "isolated_svg", "parse_path", "SVG_NS"]

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
UID = "data-svg2psd-uid"
for _p, _u in (("", SVG_NS), ("xlink", XLINK_NS), ("inkscape", INKSCAPE_NS),
               ("sodipodi", "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd")):
    ET.register_namespace(_p, _u)

NON_RENDERING = {"defs", "clipPath", "mask", "linearGradient", "radialGradient", "pattern", "filter",
                 "marker", "symbol", "style", "title", "desc", "metadata", "script", "font",
                 "font-face", "cursor", "view", "animate", "set", "animateTransform",
                 "animateMotion", "animateColor", "solidColor", "hatch", "meshgradient"}
CONTAINERS = {"g", "a"}
SHAPES = {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon"}

OP_XOR, OP_UNION, OP_SUB, OP_ISECT = 0, 1, 2, 3   # §PR subpath "operation" (undocumented, CS6+)
OP_NAMES = {OP_XOR: "exclude", OP_UNION: "combine", OP_SUB: "subtract", OP_ISECT: "intersect"}
FLAT_STEP = 0.25        # px: flattening step for analysis / polygon fallback (document pixels)
UNSUPPORTED = object()  # sentinel: a clip-path we cannot express as vectors

BLEND_MAP = {  # CSS mix-blend-mode -> PSD blend mode key (§LR "Blend mode key")
    "normal": BlendMode.NORMAL, "multiply": BlendMode.MULTIPLY, "screen": BlendMode.SCREEN,
    "overlay": BlendMode.OVERLAY, "darken": BlendMode.DARKEN, "lighten": BlendMode.LIGHTEN,
    "color-dodge": BlendMode.COLOR_DODGE, "color-burn": BlendMode.COLOR_BURN,
    "hard-light": BlendMode.HARD_LIGHT, "soft-light": BlendMode.SOFT_LIGHT,
    "difference": BlendMode.DIFFERENCE, "exclusion": BlendMode.EXCLUSION, "hue": BlendMode.HUE,
    "saturation": BlendMode.SATURATION, "color": BlendMode.COLOR, "luminosity": BlendMode.LUMINOSITY,
}

ICC_CANDIDATES = ("/usr/share/color/icc/ghostscript/srgb.icc",
                  "/usr/share/ghostscript/iccprofiles/srgb.icc",
                  "/usr/share/color/icc/sRGB.icc")


# =========================================================================== small helpers
def lname(el) -> str:
    t = el.tag
    return t.rsplit("}", 1)[-1] if isinstance(t, str) else ""


def _in_svg_ns(el, root_ns) -> bool:
    t = el.tag
    if not isinstance(t, str):
        return False
    ns = t[1:].split("}", 1)[0] if t.startswith("{") else ""
    return ns == root_ns


def _href(el):
    return el.get("{%s}href" % XLINK_NS) or el.get("href")


_LEN_UNITS = {"": 1.0, "px": 1.0, "pt": 96 / 72, "pc": 16.0, "in": 96.0, "cm": 96 / 2.54,
              "mm": 96 / 25.4, "q": 96 / 101.6, "em": 16.0, "rem": 16.0, "ex": 8.0}


def parse_len(v, ref=None, default=0.0):
    """SVG <length> to user units (96 dpi CSS units; % relative to ``ref``)."""
    if v is None:
        return default
    m = re.fullmatch(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([a-zA-Z%]*)\s*", str(v))
    if not m:
        return default
    num, unit = float(m.group(1)), m.group(2).lower()
    if unit == "%":
        return num / 100.0 * (ref if ref is not None else 0.0)
    return num * _LEN_UNITS.get(unit, 1.0)


def _num_list(s):
    return [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", s or "")]


# --------------------------------------------------------------------------- affine matrices
IDENT = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)   # (a, b, c, d, e, f):  x' = a x + c y + e ; y' = b x + d y + f


def mul(m, n):
    """m · n  (n is applied first)."""
    a, b, c, d, e, f = m
    A, B, C, D, E, F = n
    return (a * A + c * B, b * A + d * B, a * C + c * D, b * C + d * D,
            a * E + c * F + e, b * E + d * F + f)


def apply(m, p):
    return (m[0] * p[0] + m[2] * p[1] + m[4], m[1] * p[0] + m[3] * p[1] + m[5])


def parse_transform(s):
    m = IDENT
    if not s:
        return m
    for name, args in re.findall(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)", s):
        v = _num_list(args)
        if name == "matrix" and len(v) == 6:
            t = tuple(v)
        elif name == "translate":
            t = (1, 0, 0, 1, v[0] if v else 0, v[1] if len(v) > 1 else 0)
        elif name == "scale":
            sx = v[0] if v else 1
            t = (sx, 0, 0, v[1] if len(v) > 1 else sx, 0, 0)
        elif name == "rotate":
            a = math.radians(v[0] if v else 0)
            r = (math.cos(a), math.sin(a), -math.sin(a), math.cos(a), 0, 0)
            if len(v) >= 3:
                t = mul(mul((1, 0, 0, 1, v[1], v[2]), r), (1, 0, 0, 1, -v[1], -v[2]))
            else:
                t = r
        elif name == "skewX":
            t = (1, 0, math.tan(math.radians(v[0] if v else 0)), 1, 0, 0)
        elif name == "skewY":
            t = (1, math.tan(math.radians(v[0] if v else 0)), 0, 1, 0, 0)
        else:
            continue
        m = mul(m, tuple(float(x) for x in t))
    return m


# --------------------------------------------------------------------------- colours / paint
def parse_color(v):
    """CSS colour -> ((r, g, b) 0-255 ints, alpha 0-1) or None."""
    v = (v or "").strip()
    if not v:
        return None
    if v.lower() == "transparent":
        return (0, 0, 0), 0.0
    m = re.fullmatch(r"rgba?\(\s*([^)]*)\)", v, flags=re.I)
    if m:
        parts = [p for p in re.split(r"[\s,/]+", m.group(1).strip()) if p]
        if len(parts) < 3:
            return None
        rgb = []
        for p in parts[:3]:
            rgb.append(round(float(p[:-1]) * 2.55) if p.endswith("%") else round(float(p)))
        a = 1.0
        if len(parts) > 3:
            a = float(parts[3][:-1]) / 100 if parts[3].endswith("%") else float(parts[3])
        return tuple(max(0, min(255, c)) for c in rgb), max(0.0, min(1.0, a))
    try:
        c = ImageColor.getrgb(v)
    except ValueError:
        return None
    return tuple(c[:3]), (c[3] / 255.0 if len(c) > 3 else 1.0)


def parse_paint(v, current_color="black"):
    """-> None | ('color', rgb, alpha) | ('url', id, fallback_str)"""
    v = (v or "").strip()
    if not v or v == "none":
        return None
    m = re.match(r"url\(\s*['\"]?#([^'\")]+)['\"]?\s*\)\s*(.*)$", v)
    if m:
        return ("url", m.group(1), m.group(2).strip() or None)
    if v.lower() == "currentcolor":
        v = current_color
    c = parse_color(v)
    if c is None:
        return None
    return ("color", c[0], c[1])


def _hex(rgb):
    return "#%02X%02X%02X" % tuple(rgb)


# =========================================================================== CSS / style cascade
STYLE_DEFAULTS = {
    "fill": "black", "fill-opacity": "1", "fill-rule": "nonzero", "stroke": "none",
    "stroke-width": "1", "stroke-opacity": "1", "stroke-linecap": "butt",
    "stroke-linejoin": "miter", "stroke-miterlimit": "4", "stroke-dasharray": "none",
    "stroke-dashoffset": "0", "clip-rule": "nonzero", "visibility": "visible", "color": "black",
    "paint-order": "normal", "vector-effect": "none",
    # not inherited:
    "opacity": "1", "clip-path": "none", "mask": "none", "filter": "none", "display": "inline",
    "mix-blend-mode": "normal", "isolation": "auto",
}
NOT_INHERITED = {"opacity", "clip-path", "mask", "filter", "display", "mix-blend-mode",
                 "vector-effect", "isolation"}


def _parse_decls(body):
    out = {}
    for decl in (body or "").split(";"):
        if ":" in decl:
            k, v = decl.split(":", 1)
            v = re.sub(r"!\s*important\s*$", "", v.strip()).strip()
            out[k.strip().lower()] = v
    return out


class Styler:
    """Presentation attributes < <style> rules (simple selectors, by specificity) < style attr."""

    def __init__(self, root, warn):
        self.rules = []   # (specificity, order, tag, classes, ids, decls)
        order = 0
        for st in root.iter():
            if lname(st) != "style":
                continue
            text = re.sub(r"/\*.*?\*/", "", "".join(st.itertext()), flags=re.S)
            for m in re.finditer(r"([^{}@;]+)\{([^{}]*)\}", text):
                decls = _parse_decls(m.group(2))
                for sel in m.group(1).split(","):
                    sel = sel.strip()
                    mm = re.fullmatch(r"(\*|[A-Za-z][\w-]*)?((?:[.#][\w-]+)*)", sel)
                    if not sel or not mm:
                        warn(f"CSS selector not supported (ignored): {sel!r}")
                        continue
                    tag = mm.group(1) if mm.group(1) not in (None, "*") else None
                    classes = re.findall(r"\.([\w-]+)", mm.group(2))
                    ids = re.findall(r"#([\w-]+)", mm.group(2))
                    spec = (len(ids), len(classes), 1 if tag else 0)
                    self.rules.append((spec, order, tag, set(classes), set(ids), decls))
                    order += 1
        self.rules.sort(key=lambda r: (r[0], r[1]))

    def specified(self, el):
        d = {}
        for k in STYLE_DEFAULTS:
            if el.get(k) is not None:
                d[k] = el.get(k).strip()
        if self.rules:
            cls = set((el.get("class") or "").split())
            eid = el.get("id")
            t = lname(el)
            for _, _, tag, classes, ids, decls in self.rules:
                if (tag is None or tag == t) and classes <= cls and (not ids or ids == {eid}):
                    d.update({k: v for k, v in decls.items() if k in STYLE_DEFAULTS})
        if el.get("style"):
            d.update({k: v for k, v in _parse_decls(el.get("style")).items() if k in STYLE_DEFAULTS})
        return d


# =========================================================================== geometry
class Sub:
    """One subpath in absolute coordinates: start point + cubic segments (c1, c2, p)."""
    __slots__ = ("p0", "segs", "closed", "flat")

    def __init__(self, p0, segs=None, closed=False, flat=False):
        self.p0 = (float(p0[0]), float(p0[1]))
        self.segs = segs if segs is not None else []
        self.closed = closed
        self.flat = flat   # True when this outline is a polygon computed by the fallback

    def end(self):
        return self.segs[-1][2] if self.segs else self.p0

    def transformed(self, m):
        f = lambda p: apply(m, p)  # noqa: E731
        return Sub(f(self.p0), [(f(a), f(b), f(c)) for a, b, c in self.segs], self.closed, self.flat)

    def closed_copy(self):
        return Sub(self.p0, list(self.segs), True, self.flat)

    @classmethod
    def polygon(cls, pts, flat=False):
        pts = [tuple(map(float, p)) for p in pts]
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
        s = cls(pts[0], [], True, flat)
        for p in pts[1:]:
            s.segs.append((s.end(), p, p))
        return s

    def control_points(self):
        pts = [self.p0]
        for a, b, c in self.segs:
            pts += [a, b, c]
        return pts

    def flatten(self, step=FLAT_STEP):
        """Polyline (N, 2); for closed subpaths the closing point is not repeated."""
        pts = [self.p0]
        cur = self.p0
        for c1, c2, p in self.segs:
            if c1 == cur and c2 == p:
                pts.append(p)
            else:
                L = math.dist(cur, c1) + math.dist(c1, c2) + math.dist(c2, p)
                n = int(min(2048, max(8, math.ceil(L / step))))
                t = np.linspace(0.0, 1.0, n + 1)[1:, None]
                P = [np.array(q, dtype=float) for q in (cur, c1, c2, p)]
                mt = 1 - t
                b = mt ** 3 * P[0] + 3 * mt * mt * t * P[1] + 3 * mt * t * t * P[2] + t ** 3 * P[3]
                pts.extend(map(tuple, b))
            cur = p
        arr = np.array(pts, dtype=float)
        if self.closed and len(arr) > 1 and np.allclose(arr[-1], arr[0], atol=1e-9):
            arr = arr[:-1]
        return arr

    def knots(self):
        """[(preceding, anchor, leaving)] in this subpath's coordinates (§PR Bezier knots)."""
        segs = [s for s in self.segs]
        if not segs:
            return []
        if self.closed:
            e = segs[-1][2]
            if math.dist(e, self.p0) > 1e-9:
                segs.append((e, self.p0, self.p0))              # implicit closing line
            else:
                c1, c2, _ = segs[-1]
                segs[-1] = (c1, c2, self.p0)
            if len(segs) == 1:                                 # one curve: split, PS needs 2 knots
                segs = list(_split_cubic(self.p0, *segs[0]))
            anchors = [self.p0] + [s[2] for s in segs[:-1]]
            leave = [s[0] for s in segs]
            pre = [segs[-1][1]] + [s[1] for s in segs[:-1]]
        else:
            anchors = [self.p0] + [s[2] for s in segs]
            leave = [s[0] for s in segs] + [anchors[-1]]
            pre = [anchors[0]] + [s[1] for s in segs]
        return list(zip(pre, anchors, leave))


def _split_cubic(p0, c1, c2, p3, t=0.5):
    lerp = lambda a, b: (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)  # noqa: E731
    a, b, c = lerp(p0, c1), lerp(c1, c2), lerp(c2, p3)
    d, e = lerp(a, b), lerp(b, c)
    f = lerp(d, e)
    return (a, d, f), (e, c, p3)


def _clean(sub):
    """Drop zero-length segments (duplicate knots)."""
    out = Sub(sub.p0, [], sub.closed, sub.flat)
    cur = sub.p0
    for c1, c2, p in sub.segs:
        if math.dist(cur, p) < 1e-9 and math.dist(cur, c1) < 1e-9 and math.dist(cur, c2) < 1e-9:
            continue
        out.segs.append((c1, c2, p))
        cur = p
    return out


def arc_to_cubics(p0, rx, ry, phi_deg, large, sweep, p1):
    """SVG elliptical arc (endpoint parameterisation, SVG 1.1 F.6.5/F.6.6) -> cubic segments."""
    x0, y0 = p0
    x1, y1 = p1
    if math.dist(p0, p1) < 1e-12:
        return []
    rx, ry = abs(rx), abs(ry)
    if rx < 1e-12 or ry < 1e-12:
        return [(p0, p1, p1)]
    phi = math.radians(phi_deg % 360)
    cph, sph = math.cos(phi), math.sin(phi)
    dx2, dy2 = (x0 - x1) / 2, (y0 - y1) / 2
    x1p = cph * dx2 + sph * dy2
    y1p = -sph * dx2 + cph * dy2
    lam = (x1p / rx) ** 2 + (y1p / ry) ** 2
    if lam > 1:
        s = math.sqrt(lam)
        rx, ry = rx * s, ry * s
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = math.sqrt(max(0.0, num / den)) if den else 0.0
    if bool(large) == bool(sweep):
        coef = -coef
    cxp, cyp = coef * rx * y1p / ry, -coef * ry * x1p / rx
    cx = cph * cxp - sph * cyp + (x0 + x1) / 2
    cy = sph * cxp + cph * cyp + (y0 + y1) / 2

    def ang(u, v):
        return math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])

    u = ((x1p - cxp) / rx, (y1p - cyp) / ry)
    v = ((-x1p - cxp) / rx, (-y1p - cyp) / ry)
    th1 = ang((1, 0), u)
    dth = ang(u, v)
    if not sweep and dth > 0:
        dth -= 2 * math.pi
    elif sweep and dth < 0:
        dth += 2 * math.pi
    n = max(1, math.ceil(abs(dth) / (math.pi / 2) - 1e-9))
    d = dth / n
    k = 4 / 3 * math.tan(d / 4)

    def pt(ux, uy):
        return (cx + rx * cph * ux - ry * sph * uy, cy + rx * sph * ux + ry * cph * uy)

    out = []
    for i in range(n):
        a1, a2 = th1 + i * d, th1 + (i + 1) * d
        c1 = pt(math.cos(a1) - k * math.sin(a1), math.sin(a1) + k * math.cos(a1))
        c2 = pt(math.cos(a2) + k * math.sin(a2), math.sin(a2) - k * math.cos(a2))
        end = p1 if i == n - 1 else pt(math.cos(a2), math.sin(a2))
        out.append((c1, c2, end))
    return out


_PATH_ARGS = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}


def parse_path(d):
    """SVG path data -> [Sub] (absolute coordinates, every segment a cubic)."""
    s = d or ""
    i, n = 0, len(s)
    num_re = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

    def skip():
        nonlocal i
        while i < n and (s[i].isspace() or s[i] == ","):
            i += 1

    def number():
        nonlocal i
        skip()
        m = num_re.match(s, i)
        if not m:
            raise ValueError(f"bad number at {i} in path data")
        i = m.end()
        return float(m.group(0))

    def flag():
        nonlocal i
        skip()
        if i < n and s[i] in "01":
            i += 1
            return int(s[i - 1])
        raise ValueError(f"bad arc flag at {i}")

    subs = []
    sub = None
    cur = (0.0, 0.0)
    start = (0.0, 0.0)
    cmd = None
    prev = None
    last_c2 = None      # for S/s
    last_q = None       # for T/t
    while True:
        skip()
        if i >= n:
            break
        ch = s[i]
        if ch.isalpha():
            if ch.upper() not in _PATH_ARGS:
                raise ValueError(f"unknown path command {ch!r}")
            cmd = ch
            i += 1
            if cmd in "Zz":
                if sub is not None:
                    sub.closed = True
                    cur = start
                    sub = None
                prev = cmd
                continue
        elif cmd is None or cmd in "Zz":
            raise ValueError("path data must start with a command")
        rel = cmd.islower()
        C = cmd.upper()
        if C == "M":
            x, y = number(), number()
            if rel:
                x, y = cur[0] + x, cur[1] + y
            cur = start = (x, y)
            sub = Sub(cur)
            subs.append(sub)
            cmd = "l" if rel else "L"     # implicit lineto after the first pair
            prev = "M"
            continue
        if sub is None:                     # drawing after Z (or before M): new subpath at start
            sub = Sub(cur)
            subs.append(sub)
            start = cur
        if C == "L":
            x, y = number(), number()
            p = (cur[0] + x, cur[1] + y) if rel else (x, y)
            sub.segs.append((cur, p, p))
        elif C == "H":
            x = number()
            p = (cur[0] + x if rel else x, cur[1])
            sub.segs.append((cur, p, p))
        elif C == "V":
            y = number()
            p = (cur[0], cur[1] + y if rel else y)
            sub.segs.append((cur, p, p))
        elif C == "C":
            v = [number() for _ in range(6)]
            if rel:
                v = [v[k] + cur[k % 2] for k in range(6)]
            c1, c2, p = (v[0], v[1]), (v[2], v[3]), (v[4], v[5])
            sub.segs.append((c1, c2, p))
            last_c2 = c2
        elif C == "S":
            v = [number() for _ in range(4)]
            if rel:
                v = [v[k] + cur[k % 2] for k in range(4)]
            c1 = (2 * cur[0] - last_c2[0], 2 * cur[1] - last_c2[1]) \
                if prev and prev.upper() in "CS" and last_c2 else cur
            c2, p = (v[0], v[1]), (v[2], v[3])
            sub.segs.append((c1, c2, p))
            last_c2 = c2
        elif C == "Q":
            v = [number() for _ in range(4)]
            if rel:
                v = [v[k] + cur[k % 2] for k in range(4)]
            q, p = (v[0], v[1]), (v[2], v[3])
            sub.segs.append(_quad(cur, q, p))
            last_q = q
        elif C == "T":
            v = [number(), number()]
            p = (cur[0] + v[0], cur[1] + v[1]) if rel else (v[0], v[1])
            q = (2 * cur[0] - last_q[0], 2 * cur[1] - last_q[1]) \
                if prev and prev.upper() in "QT" and last_q else cur
            sub.segs.append(_quad(cur, q, p))
            last_q = q
        elif C == "A":
            rx, ry, rot = number(), number(), number()
            large, sweep = flag(), flag()
            x, y = number(), number()
            p = (cur[0] + x, cur[1] + y) if rel else (x, y)
            sub.segs.extend(arc_to_cubics(cur, rx, ry, rot, large, sweep, p))
        cur = sub.segs[-1][2] if sub.segs else cur
        if C == "A" and not sub.segs:
            cur = p
        prev = cmd
    return [_clean(x) for x in subs if x.segs]


def _quad(p0, q, p):
    """Quadratic -> cubic (exact degree elevation)."""
    return ((p0[0] + 2 / 3 * (q[0] - p0[0]), p0[1] + 2 / 3 * (q[1] - p0[1])),
            (p[0] + 2 / 3 * (q[0] - p[0]), p[1] + 2 / 3 * (q[1] - p[1])), p)


def _points_attr(s):
    v = _num_list(s)
    return [(v[k], v[k + 1]) for k in range(0, len(v) - 1, 2)]


def element_geometry(el, vw=1024.0, vh=1024.0):
    """Basic shape / path -> [Sub] in the element's user space (before its own transform)."""
    t = lname(el)
    diag = math.hypot(vw, vh) / math.sqrt(2)
    if t == "path":
        return parse_path(el.get("d"))
    if t in ("polygon", "polyline"):
        pts = _points_attr(el.get("points"))
        if len(pts) < 2:
            return []
        s = Sub(pts[0], [], t == "polygon")
        for p in pts[1:]:
            s.segs.append((s.end(), p, p))
        return [_clean(s)]
    if t == "line":
        a = (parse_len(el.get("x1"), vw), parse_len(el.get("y1"), vh))
        b = (parse_len(el.get("x2"), vw), parse_len(el.get("y2"), vh))
        return [Sub(a, [(a, b, b)], False)]
    if t == "rect":
        x, y = parse_len(el.get("x"), vw), parse_len(el.get("y"), vh)
        w, h = parse_len(el.get("width"), vw), parse_len(el.get("height"), vh)
        if w <= 0 or h <= 0:
            return []
        rx_a, ry_a = el.get("rx"), el.get("ry")
        rx = parse_len(rx_a, vw) if rx_a not in (None, "auto") else None
        ry = parse_len(ry_a, vh) if ry_a not in (None, "auto") else None
        if rx is None and ry is None:
            rx = ry = 0.0
        elif rx is None:
            rx = ry
        elif ry is None:
            ry = rx
        rx, ry = min(max(rx, 0.0), w / 2), min(max(ry, 0.0), h / 2)
        if rx <= 0 or ry <= 0:
            return [Sub.polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])]
        s = Sub((x + rx, y), [], True)

        def line_to(p):
            s.segs.append((s.end(), p, p))

        def arc_to(p):
            s.segs.extend(arc_to_cubics(s.end(), rx, ry, 0, 0, 1, p))

        line_to((x + w - rx, y)); arc_to((x + w, y + ry))
        line_to((x + w, y + h - ry)); arc_to((x + w - rx, y + h))
        line_to((x + rx, y + h)); arc_to((x, y + h - ry))
        line_to((x, y + ry)); arc_to((x + rx, y))
        return [_clean(s)]
    if t in ("circle", "ellipse"):
        cx, cy = parse_len(el.get("cx"), vw), parse_len(el.get("cy"), vh)
        if t == "circle":
            rx = ry = parse_len(el.get("r"), diag)
        else:
            rx, ry = parse_len(el.get("rx"), vw), parse_len(el.get("ry"), vh)
            if el.get("rx") in (None, "auto"):
                rx = ry
            if el.get("ry") in (None, "auto"):
                ry = rx
        if rx <= 0 or ry <= 0:
            return []
        k = 4 / 3 * math.tan(math.pi / 8)   # 0.5522847498: quarter-ellipse cubic
        P = [(cx + rx, cy), (cx, cy + ry), (cx - rx, cy), (cx, cy - ry)]
        H = [((cx + rx, cy + k * ry), (cx + k * rx, cy + ry)),
             ((cx - k * rx, cy + ry), (cx - rx, cy + k * ry)),
             ((cx - rx, cy - k * ry), (cx - k * rx, cy - ry)),
             ((cx + k * rx, cy - ry), (cx + rx, cy - k * ry))]
        s = Sub(P[0], [], True)
        for q in range(4):
            s.segs.append((H[q][0], H[q][1], P[(q + 1) % 4]))
        return [s]
    raise ValueError(f"<{t}> has no vector geometry")


# --------------------------------------------------------------------------- regions
# A *region* is a list of (op, Sub) evaluated in order exactly like Photoshop path
# components: r = 0; for op, s: r = op(r, fill(s)).   First op is always "combine".

def _winding(ring, pts):
    """Winding number of every point in ``pts`` (M, 2) w.r.t. closed polyline ``ring``."""
    a = ring
    b = np.roll(ring, -1, axis=0)
    px, py = pts[:, 0][None, :], pts[:, 1][None, :]
    ax, ay, bx, by = a[:, 0][:, None], a[:, 1][:, None], b[:, 0][:, None], b[:, 1][:, None]
    is_left = (bx - ax) * (py - ay) - (px - ax) * (by - ay)
    up = (ay <= py) & (by > py) & (is_left > 0)
    down = (ay > py) & (by <= py) & (is_left < 0)
    return up.sum(0) - down.sum(0)


def region_geom(region, step=FLAT_STEP):
    """Shapely geometry of a region (flattened)."""
    from shapely.geometry import Polygon
    g = None
    for op, s in region:
        r = s.flatten(step) if s.closed else s.closed_copy().flatten(step)
        p = Polygon(r).buffer(0) if len(r) >= 3 else Polygon()
        if g is None:
            g = p
        elif op == OP_UNION:
            g = g.union(p)
        elif op == OP_XOR:
            g = g.symmetric_difference(p)
        elif op == OP_SUB:
            g = g.difference(p)
        elif op == OP_ISECT:
            g = g.intersection(p)
    return g if g is not None else Polygon()


def region_from_geom(g):
    """Exact polygon outline (polygons with holes) -> region; islands always after their hole.
    Outlines are simplified within 0.05 px (document pixels) to keep the knot count sane."""
    from shapely.geometry import Polygon
    if not g.is_empty:
        g = g.simplify(0.05, preserve_topology=True)
    polys = [g] if g.geom_type == "Polygon" else [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon"]
    polys = [p for p in polys if p.area > 1e-6]
    polys.sort(key=lambda p: -Polygon(p.exterior).area)
    region = []
    for p in polys:
        region.append((OP_UNION, Sub.polygon(list(p.exterior.coords), flat=True)))
        for hole in p.interiors:
            region.append((OP_SUB, Sub.polygon(list(hole.coords), flat=True)))
    return region


def fill_region(subs, rule="nonzero", step=FLAT_STEP, keep_open=False):
    """Fill area of a path under ``rule`` as a region, keeping the original Beziers whenever the
    rule can be expressed with sequential path operations.  Open subpaths are filled as if
    closed (SVG semantics); with ``keep_open`` they stay open in the region (stroked shapes)."""
    from shapely.geometry import LinearRing, LineString
    from shapely.ops import polygonize, unary_union
    rings, keep = [], []
    for s in subs:
        if not s.segs:
            continue
        r = s.closed_copy().flatten(step)
        if len(r) >= 3 and abs(_shoelace(r)) > 1e-9:
            rings.append(r)
            keep.append(s if keep_open else s.closed_copy())
    if not keep:
        return []
    simple = all(LinearRing(r).is_simple for r in rings)
    if len(keep) == 1 and simple:
        return [(OP_UNION, keep[0])]
    lines = unary_union([LineString(np.vstack([r, r[:1]])) for r in rings])
    faces = [f for f in polygonize(lines) if f.area > 1e-9]
    if not faces:
        return []
    pts = np.array([f.representative_point().coords[0] for f in faces])
    areas = np.array([f.area for f in faces])
    W = np.stack([_winding(r, pts) for r in rings])
    total = W.sum(0)
    target = (total != 0) if rule != "evenodd" else (total % 2 != 0)
    tol = 1e-6 * max(areas.sum(), 1.0)
    if simple:
        # prefer "combine" (disjoint or same-direction rings), then "exclude" (holes)
        uni = (W != 0).any(0)
        if areas[target != uni].sum() <= tol:
            return [(OP_UNION, s) for s in keep]
        xor = (np.abs(W) % 2).sum(0) % 2 == 1
        if areas[target != xor].sum() <= tol:
            return [(OP_UNION if k == 0 else OP_XOR, s) for k, s in enumerate(keep)]
    return region_from_geom(unary_union([f for f, t in zip(faces, target) if t]))


def _shoelace(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def region_union(regions):
    regions = [r for r in regions if r]
    if not regions:
        return []
    if len(regions) == 1:
        return regions[0]
    if all(len(r) == 1 for r in regions):
        return [(OP_UNION, r[0][1]) for r in regions]
    from shapely.ops import unary_union
    return region_from_geom(unary_union([region_geom(r) for r in regions]))


def region_intersect(a, b):
    if not a or not b:
        return []
    if len(b) == 1:
        return a + [(OP_ISECT, b[0][1])]
    if len(a) == 1:
        return b + [(OP_ISECT, a[0][1])]
    return region_from_geom(region_geom(a).intersection(region_geom(b)))


# =========================================================================== rendering
def render_svg(svg: str, px: int) -> np.ndarray:
    """rsvg-convert render, (px, px, 4) uint8 straight (non-premultiplied) RGBA."""
    return np.asarray(_svg_to_png(svg, px).convert("RGBA")).copy()


def render_svg_box(svg: str, px: int, box) -> np.ndarray:
    """The (x0, y0, x1, y1) window of the px x px rsvg render, rendered directly: same scaling
    flags as the full render, the page offset by the window origin (pixel-identical crop)."""
    x0, y0, x1, y1 = box
    out = subprocess.run(["rsvg-convert", "-w", str(px), "-h", str(px), f"--page-width={x1 - x0}",
                          f"--page-height={y1 - y0}", f"--left={-x0}", f"--top={-y0}", "-f", "png"],
                         input=svg.encode(), capture_output=True, check=True).stdout
    a = np.asarray(Image.open(io.BytesIO(out)).convert("RGBA"))
    if a.shape[:2] != (y1 - y0, x1 - x0):
        raise RuntimeError(f"rsvg-convert returned {a.shape[:2]} for window {box}")
    return a


def _crop(rgba):
    a = rgba[..., 3]
    rows = np.flatnonzero(a.any(axis=1))
    if rows.size == 0:
        return None, None
    cols = np.flatnonzero(a.any(axis=0))
    t, b, l, r = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    return (l, t, r, b), np.ascontiguousarray(rgba[t:b, l:r])


def _append_style(el, decls):
    st = el.get("style", "").strip()
    if st and not st.endswith(";"):
        st += ";"
    el.set("style", st + ";".join(f"{k}:{v}" for k, v in decls.items()))


def isolated_svg(root, uid, target_style=None, root_ns=SVG_NS):
    """Serialise a copy of ``root`` in which only element ``uid`` (and its subtree) paints.

    Every other drawable stays in the document with ``opacity:0`` instead of being deleted:
    it then paints nothing but still counts in its ancestors' bounding boxes, so clip-paths /
    masks in objectBoundingBox units on the target's ancestors resolve exactly as in the full
    render.  Every ancestor keeps its transform and clip-path; ancestors' opacity / blend /
    filter / mask are neutralised (they become PSD folder properties, or their own pixel layer),
    and so are the target's opacity and blend mode (PSD layer properties); ``target_style``
    adds further overrides.  <defs> and other non-rendering elements are untouched."""
    root = copy.deepcopy(root)
    parent = {c: p for p in root.iter() for c in p}
    target = next(e for e in root.iter() if e.get(UID) == str(uid))
    chain = []
    p = parent.get(target)
    while p is not None:
        chain.append(p)
        p = parent.get(p)
    keep = set(chain) | {target}
    sub_of_target = set(target.iter())

    def drawable(e):
        return isinstance(e.tag, str) and _in_svg_ns(e, root_ns) and lname(e) not in NON_RENDERING

    # content that <use> elements inside the target reference: give the reference a visible copy
    # in a fresh <defs>; the hidden original stays where it is (bounding boxes unchanged).
    by_id = {e.get("id"): e for e in root.iter() if e.get("id")}
    referenced, todo = [], [target]
    while todo:
        for d in todo.pop().iter():
            h = _href(d)
            if lname(d) == "use" and h and h.startswith("#") and h[1:] in by_id:
                r = by_id[h[1:]]
                if r not in referenced and r not in keep and r not in sub_of_target:
                    referenced.append(r)
                    todo.append(r)
    copies = [copy.deepcopy(r) for r in referenced]
    for r in referenced:
        r.attrib.pop("id", None)

    def hide(el):
        for ch in el:
            if ch is target or not drawable(ch):
                continue
            if ch in keep:
                hide(ch)
            else:
                _append_style(ch, {"opacity": "0"})
    if target is not root:
        hide(root)
    if copies:
        defs = ET.SubElement(root, f"{{{root_ns}}}defs" if root_ns else "defs")
        defs.extend(copies)
    neutral = {"opacity": "1", "mix-blend-mode": "normal", "filter": "none", "mask": "none",
               "display": "inline"}
    for a in chain:
        _append_style(a, neutral)
    ts = {"opacity": "1", "mix-blend-mode": "normal", "display": "inline", "visibility": "visible"}
    ts.update(target_style or {})
    _append_style(target, ts)
    return ET.tostring(root, encoding="unicode")


# =========================================================================== layer tree model
class LNode:
    def __init__(self, kind, name, uid=None):
        self.kind = kind            # 'group' | 'shape' | 'pixel'
        self.name = name
        self.uid = uid
        self.svg_id = None
        self.children = []
        self.opacity = 1.0
        self.blend = "normal"
        self.visible = True
        self.open = True
        self.info = {}
        self.target_style = None    # overrides for the isolated render
        self.bbox = (0, 0, 0, 0)
        self.pixels = None


class _Builder:
    def __init__(self, svg_text, px, warn):
        self.warn = warn
        self.root = ET.fromstring(svg_text)
        rt = self.root.tag
        self.root_ns = rt[1:].split("}", 1)[0] if rt.startswith("{") else ""
        if lname(self.root) != "svg":
            raise ValueError("root element is not <svg>")
        if not self.root_ns:
            self.root.set("xmlns", SVG_NS)
        for k, e in enumerate(self.root.iter()):           # uid = document-order index
            if isinstance(e.tag, str):
                e.set(UID, str(k))
        self.parent = {c: p for p in self.root.iter() for c in p}
        self.ids = {}
        for e in self.root.iter():
            if e.get("id") and e.get("id") not in self.ids:
                self.ids[e.get("id")] = e
        self.styler = Styler(self.root, warn)
        self._style_cache = {}
        self._clip_cache = {}
        self._obb_warned = set()
        self.px = px
        self.M0, self.vw, self.vh = self._root_matrix(px)
        self.anon = 0
        self.unnamed = {}

    # ------------------------------------------------------------------ root viewport
    def _root_matrix(self, px):
        r = self.root
        vb = _num_list(r.get("viewBox"))
        wa, ha = r.get("width"), r.get("height")
        wi = parse_len(wa) if wa and not wa.strip().endswith("%") else None
        hi = parse_len(ha) if ha and not ha.strip().endswith("%") else None
        if len(vb) == 4 and vb[2] > 0 and vb[3] > 0:
            minx, miny, vbw, vbh = vb
        else:
            minx, miny, vbw, vbh = 0.0, 0.0, (wi or px), (hi or px)
        wi = wi or vbw
        hi = hi or vbh
        # rsvg-convert -w px -h px scales the intrinsic size (wi x hi) to px x px ...
        S = (px / wi, 0.0, 0.0, px / hi, 0.0, 0.0)
        # ... and inside it the viewBox is fitted per preserveAspectRatio
        par = (r.get("preserveAspectRatio") or "xMidYMid meet").split()
        if par and par[0] == "none":
            V = (wi / vbw, 0.0, 0.0, hi / vbh, -minx * wi / vbw, -miny * hi / vbh)
        else:
            align = par[0] if par else "xMidYMid"
            slice_ = len(par) > 1 and par[1] == "slice"
            s = (max if slice_ else min)(wi / vbw, hi / vbh)
            fx = {"xMin": 0.0, "xMid": 0.5, "xMax": 1.0}.get(align[:4], 0.5)
            fy = {"YMin": 0.0, "YMid": 0.5, "YMax": 1.0}.get(align[4:], 0.5)
            V = (s, 0.0, 0.0, s, -minx * s + fx * (wi - vbw * s), -miny * s + fy * (hi - vbh * s))
        return mul(S, V), vbw, vbh

    # ------------------------------------------------------------------ style
    def style(self, el):
        k = id(el)
        if k in self._style_cache:
            return self._style_cache[k]
        par = self.parent.get(el)
        pst = self.style(par) if par is not None else STYLE_DEFAULTS
        spec = self.styler.specified(el)
        out = {}
        for prop, dflt in STYLE_DEFAULTS.items():
            v = spec.get(prop)
            if v is None or v == "":
                v = dflt if prop in NOT_INHERITED else pst[prop]
            elif v == "inherit":
                v = pst[prop]
            out[prop] = v
        if out["color"].lower() == "currentcolor":
            out["color"] = pst["color"]
        self._style_cache[k] = out
        return out

    def name_of(self, el):
        for key in ("data-name", "{%s}label" % INKSCAPE_NS, "id"):
            v = el.get(key)
            if v and v.strip():
                return v.strip()
        return None

    # ------------------------------------------------------------------ bounding boxes (user space)
    def bbox_user(self, el, m=IDENT):
        t = lname(el)
        if t in SHAPES:
            try:
                pts = [apply(m, p) for s in element_geometry(el, self.vw, self.vh)
                       for p in s.flatten(0.5 / max(1e-9, math.sqrt(abs(m[0] * m[3] - m[1] * m[2]))))]
            except ValueError:
                return None
            if not pts:
                return "empty"
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            return (min(xs), min(ys), max(xs), max(ys))
        if t in CONTAINERS or el is self.root:
            boxes = []
            for ch in el:
                ct = lname(ch)
                if not _in_svg_ns(ch, self.root_ns) or ct in NON_RENDERING:
                    continue
                b = self.bbox_user(ch, mul(m, parse_transform(ch.get("transform"))))
                if b is None:
                    return None
                if b != "empty":
                    boxes.append(b)
            if not boxes:
                return "empty"
            return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                    max(b[2] for b in boxes), max(b[3] for b in boxes))
        return None   # text, image, use ... : not computed

    # ------------------------------------------------------------------ clip-paths
    def clip_for(self, el, ctm):
        """Clip region (document px) of ``el``'s clip-path, None (no clip) or UNSUPPORTED."""
        v = self.style(el)["clip-path"]
        if not v or v == "none":
            return None
        m = re.match(r"url\(\s*['\"]?#([^'\")]+)['\"]?\s*\)", v)
        cp = self.ids.get(m.group(1)) if m else None
        if cp is None or lname(cp) != "clipPath":
            self.warn(f"clip-path {v!r} on {self.name_of(el) or lname(el)} is not a <clipPath>; ignored")
            return None
        if (cp.get("clipPathUnits") or "") == "objectBoundingBox":
            self._check_rsvg_obb_bug(el)
        key = (id(cp), tuple(round(x, 9) for x in ctm), id(el))
        if key in self._clip_cache:
            return self._clip_cache[key]
        res = self._clip_region(cp, el, ctm)
        self._clip_cache[key] = res
        return res

    def _check_rsvg_obb_bug(self, el):
        """librsvg 2.58 mis-places an objectBoundingBox clip-path on an element inside a group that
        needs an offscreen layer (opacity < 1, filter, mask) whenever the render scale is not 1:
        the element is dropped or clipped wrongly in the *full* render (= the merged image, and a
        rendered filter/mask group).  Its own layer is unaffected (isolated renders neutralise
        those ancestors).  Warn so the artwork can use userSpaceOnUse clips instead."""
        scale = math.sqrt(abs(self.M0[0] * self.M0[3] - self.M0[1] * self.M0[2]))
        if abs(scale - 1.0) < 1e-6 or id(el) in self._obb_warned:
            return
        p = self.parent.get(el)
        while p is not None:
            st = self.style(p)
            if _clamp01(st["opacity"]) < 1 or st["filter"] not in ("none", "") or st["mask"] not in ("none", ""):
                self._obb_warned.add(id(el))
                self.warn(f"{self.name_of(el) or lname(el)}: objectBoundingBox clip-path inside an opacity/filter/"
                          f"mask group at {scale:.3g}x render scale - librsvg mis-renders this in the full render, "
                          f"so the merged image (and any rendered filter/mask group) shows it wrongly; the layer "
                          f"itself is correct. Use clipPathUnits=userSpaceOnUse to avoid it.")
                return
            p = self.parent.get(p)

    def _clip_region(self, cp, el, ctm):
        if self.style(cp)["clip-path"] not in ("none", ""):
            return UNSUPPORTED                       # nested clip on the clipPath itself
        M = mul(ctm, parse_transform(cp.get("transform")))
        if (cp.get("clipPathUnits") or "userSpaceOnUse") == "objectBoundingBox":
            bb = self.bbox_user(el)
            if bb is None:
                return UNSUPPORTED
            if bb == "empty":
                return []
            M = mul(M, (bb[2] - bb[0], 0.0, 0.0, bb[3] - bb[1], bb[0], bb[1]))
        regions = []
        for ch in cp:
            t = lname(ch)
            if not _in_svg_ns(ch, self.root_ns) or t in NON_RENDERING:
                continue
            if t not in SHAPES:
                return UNSUPPORTED                   # <use>/<text> clip children
            cst = self.style(ch)
            if cst["display"] == "none" or cst["visibility"] in ("hidden", "collapse"):
                continue
            if cst["clip-path"] not in ("none", ""):
                return UNSUPPORTED
            try:
                subs = element_geometry(ch, self.vw, self.vh)
            except ValueError:
                return UNSUPPORTED
            cm = mul(M, parse_transform(ch.get("transform")))
            regions.append(fill_region([s.transformed(cm) for s in subs], cst["clip-rule"]))
        return region_union(regions)

    # ------------------------------------------------------------------ tree walk
    def build(self):
        return self.visit(self.root, self.M0, [], 0)

    def visit(self, el, ctm, clips, depth):
        if not isinstance(el.tag, str) or not _in_svg_ns(el, self.root_ns):
            return []
        t = lname(el)
        if t in NON_RENDERING:
            return []
        st = self.style(el)
        uid = int(el.get(UID))
        if t in CONTAINERS or el is self.root:
            return self._visit_group(el, t, st, uid, ctm, clips, depth)
        return self._visit_leaf(el, t, st, uid, ctm, clips)

    def _visit_group(self, el, t, st, uid, ctm, clips, depth):
        m = ctm if el is self.root else mul(ctm, parse_transform(el.get("transform")))
        name = None if el is self.root else self.name_of(el)
        opacity = _clamp01(st["opacity"])
        blend = st["mix-blend-mode"].strip().lower()
        if blend not in BLEND_MAP:
            self.warn(f"mix-blend-mode {blend!r} not supported; using normal")
            blend = "normal"
        filt = st["filter"] not in ("none", "")
        mask = st["mask"] not in ("none", "")
        hidden = st["display"] == "none"
        c = self.clip_for(el, m)
        clips2 = clips + [c] if c is not None else clips
        # Isolated group (PSD folder 'norm' / its blend mode) exactly when the reference renderer
        # isolates it: opacity < 1, mask, filter, a group blend mode or isolation:isolate.
        # clip-path does not isolate in rsvg, so a clipped group stays pass-through.
        isolated = opacity < 1 or blend != "normal" or filt or mask or st["isolation"] == "isolate"
        need_folder = name is not None or opacity < 1 or blend != "normal" or filt or mask or hidden
        if not need_folder and isolated and self._has_blend_below(el):
            need_folder = True                    # hoisting would leak the blend past the group
        children = []
        for ch in el:
            children += self.visit(ch, m, clips2, depth + (1 if need_folder else 0))
        if filt or mask:
            for n in children:
                n.visible = False                 # editable source, hidden
            why = "+".join(w for w, f in (("filter", filt), ("mask", mask)) if f)
            r = LNode("pixel", f"{name or 'Group'} (rendered {why})", uid)
            r.svg_id = el.get("id")
            r.info = {"reason": f"group {why}", "render": "group"}
            children.append(r)
        if not need_folder:
            return children
        if name is None:
            self.anon += 1
            name = f"Group {self.anon}"
        g = LNode("group", name, uid)
        g.svg_id = el.get("id")
        g.children = children
        g.opacity, g.blend, g.visible = opacity, blend, not hidden
        g.info = {"isolated": bool(isolated)}
        o = el.get("data-psd-open")
        g.open = (o.strip().lower() in ("1", "true", "yes", "open")) if o else depth < 2
        return [g]

    def _has_blend_below(self, el):
        for d in el.iter():
            if d is not el and isinstance(d.tag, str) and \
                    self.style(d)["mix-blend-mode"].strip().lower() not in ("normal", ""):
                return True
        return False

    def _visit_leaf(self, el, t, st, uid, ctm, clips):
        name = self.name_of(el)
        if name is None:
            self.unnamed[t] = self.unnamed.get(t, 0) + 1
            name = f"{t} {self.unnamed[t]}"
        m = mul(ctm, parse_transform(el.get("transform")))
        c = self.clip_for(el, m)
        clips = clips + [c] if c is not None else clips
        opacity = _clamp01(st["opacity"])
        blend = st["mix-blend-mode"].strip().lower()
        if blend not in BLEND_MAP:
            blend = "normal"
        visible = st["display"] != "none" and st["visibility"] not in ("hidden", "collapse")
        node = self._classify(el, t, st, m, clips, name, uid)
        if node is None:
            return []
        node.svg_id = el.get("id")
        node.opacity, node.blend, node.visible = opacity, blend, visible
        return [node]

    def _classify(self, el, t, st, m, clips, name, uid):
        def pixel(reason):
            n = LNode("pixel", name, uid)
            n.info = {"reason": reason, "render": "element"}
            return n

        if t not in SHAPES:
            return pixel(f"<{t}> element")
        if st["filter"] not in ("none", ""):
            return pixel("filter")
        if st["mask"] not in ("none", ""):
            return pixel("mask")
        fill = parse_paint(st["fill"], st["color"])
        stroke = parse_paint(st["stroke"], st["color"])
        if fill and fill[0] == "url":
            ref = self.ids.get(fill[1])
            if ref is None:
                fill = parse_paint(fill[2], st["color"]) if fill[2] else None
            else:
                return pixel(f"{lname(ref)} fill")
        if stroke and stroke[0] == "url":
            ref = self.ids.get(stroke[1])
            if ref is None:
                stroke = parse_paint(stroke[2], st["color"]) if stroke[2] else None
            else:
                return pixel(f"{lname(ref)} stroke")
        sw = parse_len(st["stroke-width"], math.hypot(self.vw, self.vh) / math.sqrt(2), 1.0)
        if stroke and sw <= 0:
            stroke = None
        if stroke and stroke[2] <= 0:
            stroke = None
        if t == "line":
            fill = None                           # a line has no interior to fill
        if fill is None and stroke is None:
            self.warn(f"{name}: no fill and no stroke - nothing painted, skipped")
            return None
        fill_op = (fill[2] if fill else 1.0) * _clamp01(st["fill-opacity"])
        if stroke:
            if st["stroke-dasharray"] not in ("none", "") and any(v > 0 for v in _num_list(st["stroke-dasharray"])):
                return pixel("dashed stroke")
            if any(c is not None for c in clips):
                return pixel("stroke + clip-path")
            if fill and fill_op < 1:
                return pixel("stroke + fill-opacity")
            po = st["paint-order"].split()
            if po and po[0] == "stroke" and fill:
                return pixel("paint-order stroke first")
            a, b, c_, d = m[:4]
            s1 = math.hypot(a, b)
            s2 = math.hypot(c_, d)
            if abs(s1 - s2) > 1e-3 * max(s1, s2) or abs(a * c_ + b * d) > 1e-3 * s1 * s2:
                return pixel("stroke under anisotropic transform")
        if any(c is UNSUPPORTED for c in clips):
            return pixel("clip-path not expressible as vectors")
        try:
            subs = [s.transformed(m) for s in element_geometry(el, self.vw, self.vh)]
        except ValueError as e:
            self.warn(f"{name}: {e}; rasterised")
            return pixel("geometry not parsable")
        if fill:
            region = fill_region(subs, st["fill-rule"], keep_open=bool(stroke))
            if not region and stroke:
                fill = None                       # zero-area outline: only the stroke paints
            elif stroke and (any(s.flat for _, s in region) or len(region) != len([s for s in subs if s.segs])):
                return pixel("stroke on a path needing polygon fallback")
        if not fill:
            region = [(OP_UNION, s) for s in subs if s.segs]
        for c in clips:
            if c is not None:
                region = region_intersect(region, c)
        if not region:
            self.warn(f"{name}: empty after fill-rule/clip - nothing painted, skipped")
            return None
        n = LNode("shape", name, uid)
        stroke_info = None
        if stroke:
            scale = math.sqrt(abs(m[0] * m[3] - m[1] * m[2]))
            if st["vector-effect"] == "non-scaling-stroke":
                scale = self.M0[0]
            stroke_info = {
                "rgb": stroke[1], "opacity": stroke[2] * _clamp01(st["stroke-opacity"]),
                "width": sw * scale, "cap": st["stroke-linecap"], "join": st["stroke-linejoin"],
                "miter": float(_num_list(st["stroke-miterlimit"])[0]) if _num_list(st["stroke-miterlimit"]) else 4.0,
            }
        n.info = {
            "region": region,
            "fill": fill[1] if fill else (stroke[1] if stroke else (0, 0, 0)),
            "fill_enabled": fill is not None,
            "fill_opacity": fill_op if fill else 1.0,
            "stroke": stroke_info,
            "fill_rule": st["fill-rule"],
        }
        ts = {}
        if fill and not stroke:
            ts = {"fill": _hex(fill[1]), "fill-opacity": "1"}
            # every painted pixel lies inside the control-point hull of the region (+AA margin)
            pts = np.array([p for _, sp in region for p in sp.control_points()])
            x0, y0 = np.floor(pts.min(axis=0)).astype(int) - 2
            x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 2
            n.info["render_box"] = (max(0, int(x0)), max(0, int(y0)), min(self.px, int(x1)), min(self.px, int(y1)))
        n.target_style = ts
        return n


def _clamp01(v):
    try:
        f = float(str(v).strip().rstrip("%")) / (100.0 if str(v).strip().endswith("%") else 1.0)
    except ValueError:
        return 1.0
    return max(0.0, min(1.0, f))


# =========================================================================== PSD structures
def _fixed(v):
    """Snap to the 8.24 fixed-point grid so psd-tools' int() encoding is exact (§PR knots)."""
    return round(v * 16777216.0) / 16777216.0


def _vector_mask(region, w, h):
    """'vmsk' Vector mask setting (§AL "Vector mask setting"): version 3, flags, path records.

    §PR path records (26 bytes each): fill-rule record (6), initial fill (8, 0 = start empty),
    then per subpath a length record (0 closed / 3 open: knot count + operation) followed by
    its Bezier knots (1/2 closed linked/unlinked, 4/5 open). Knot = preceding, anchor, leaving
    control points, each (vertical, horizontal) as signed 8.24 fixed fractions of the
    document height / width."""
    items = [PathFillRule(), InitialFillRule(value=0)]
    n_knots = 0
    for op, sub in region:
        ks = sub.knots()
        if len(ks) < 2 and sub.closed:
            continue
        knots = []
        for pre, anc, lea in ks:
            u = (pre[0] - anc[0], pre[1] - anc[1])
            v = (lea[0] - anc[0], lea[1] - anc[1])
            nu, nv = math.hypot(*u), math.hypot(*v)
            linked = (nu > 1e-6 and nv > 1e-6 and abs(u[0] * v[1] - u[1] * v[0]) <= 1e-4 * nu * nv
                      and (u[0] * v[0] + u[1] * v[1]) < 0)
            kls = (ClosedKnotLinked if linked else ClosedKnotUnlinked) if sub.closed else \
                  (OpenKnotLinked if linked else OpenKnotUnlinked)
            yx = lambda p: (_fixed(p[1] / h), _fixed(p[0] / w))  # noqa: E731
            knots.append(kls(preceding=yx(pre), anchor=yx(anc), leaving=yx(lea)))
        n_knots += len(knots)
        rec = (ClosedPath if sub.closed else OpenPath)(items=knots, operation=op, index=0)
        items.append(rec)
    return VectorMaskSetting(version=3, flags=0, path=VPath(items)), n_knots


def _rgbc(rgb):
    # §DS Descriptor 'RGBC' {Rd, Grn, Bl: doub 0..255}
    return Descriptor(classID=b"RGBC", items=[(b"Rd  ", Double(float(rgb[0]))),
                                              (b"Grn ", Double(float(rgb[1]))),
                                              (b"Bl  ", Double(float(rgb[2])))])


def _soco(rgb):
    # §AL 'SoCo' Solid color sheet setting: version 16 + descriptor {Clr : RGBC}
    return DescriptorBlock(version=16, classID=b"null", items=[(b"Clr ", _rgbc(rgb))])


def _vstk(stroke, fill_enabled, dpi):
    """'vstk' Vector stroke data: descriptor 'strokeStyle' with the keys Photoshop CS6+ writes."""
    s = stroke or {}
    cap = {"round": b"strokeStyleRoundCap", "square": b"strokeStyleSquareCap"}.get(
        s.get("cap"), b"strokeStyleButtCap")
    join = {"round": b"strokeStyleRoundJoin", "bevel": b"strokeStyleBevelJoin"}.get(
        s.get("join"), b"strokeStyleMiterJoin")
    content = Descriptor(classID=b"solidColorLayer", items=[(b"Clr ", _rgbc(s.get("rgb", (0, 0, 0))))])
    items = [
        (b"strokeStyleVersion", Integer(2)),
        (b"strokeEnabled", Bool(bool(stroke))),
        (b"fillEnabled", Bool(bool(fill_enabled))),
        (b"strokeStyleLineWidth", UnitFloat(value=float(s.get("width", 1.0)), unit=Unit.Pixels)),
        (b"strokeStyleLineDashOffset", UnitFloat(value=0.0, unit=Unit.Points)),
        (b"strokeStyleMiterLimit", Double(float(s.get("miter", 100.0)))),
        (b"strokeStyleLineCapType", Enumerated(typeID=b"strokeStyleLineCapType", enum=cap)),
        (b"strokeStyleLineJoinType", Enumerated(typeID=b"strokeStyleLineJoinType", enum=join)),
        (b"strokeStyleLineAlignment", Enumerated(typeID=b"strokeStyleLineAlignment",
                                                 enum=b"strokeStyleAlignCenter")),
        (b"strokeStyleScaleLock", Bool(False)),
        (b"strokeStyleStrokeAdjust", Bool(False)),
        (b"strokeStyleLineDashSet", DescList(items=[])),
        (b"strokeStyleBlendMode", Enumerated(typeID=b"BlnM", enum=b"Nrml")),
        (b"strokeStyleOpacity", UnitFloat(value=100.0 * float(s.get("opacity", 1.0)), unit=Unit.Percent)),
        (b"strokeStyleContent", content),
        (b"strokeStyleResolution", Double(float(dpi))),
    ]
    return DescriptorBlock(version=16, classID=b"strokeStyle", items=items)


def _pascal(name):
    """Legacy Pascal layer name (§LR: MacRoman, <= 255 bytes); the full name goes in 'luni'."""
    b = name.encode("mac_roman", "replace")[:255]
    return b.decode("mac_roman")


def _blocks(name, lid, extra=()):
    tb = TaggedBlocks()
    tb.set_data(Tag.UNICODE_LAYER_NAME, name)          # 'luni' §AL Unicode layer name
    tb.set_data(Tag.LAYER_ID, lid)                      # 'lyid' §AL Layer ID
    tb.set_data(Tag.BLEND_CLIPPING_ELEMENTS, 1)         # 'clbl' §AL Blend clipping elements
    tb.set_data(Tag.BLEND_INTERIOR_ELEMENTS, 0)         # 'infx' §AL Blend interior elements
    tb.set_data(Tag.KNOCKOUT_SETTING, 0)                # 'knko' §AL Knockout setting
    for key, data in extra:
        tb[key] = TaggedBlock(key=key, data=data)
    return tb


def _channels(rgba, rgb_const=None):
    """§CI channel image data: per channel 2-byte compression (1 = RLE/PackBits) + rows."""
    data = ChannelDataList()
    infos = []
    order = (ChannelID.TRANSPARENCY_MASK, ChannelID.CHANNEL_0, ChannelID.CHANNEL_1, ChannelID.CHANNEL_2)
    if rgba is None or rgba.size == 0:
        for cid in order:
            infos.append(ChannelInfo(id=cid, length=2))
            data.append(ChannelData(compression=Compression.RAW, data=b""))
        return infos, data
    h, w = rgba.shape[:2]
    planes = {ChannelID.TRANSPARENCY_MASK: rgba[..., 3]}
    for k, cid in enumerate(order[1:]):
        planes[cid] = (np.full((h, w), rgb_const[k], np.uint8) if rgb_const is not None
                       else rgba[..., k])
    for cid in order:
        cd = ChannelData(compression=Compression.RLE)
        cd.data = compress(np.ascontiguousarray(planes[cid]).tobytes(), Compression.RLE, w, h, 8, 1)
        infos.append(ChannelInfo(id=cid, length=0))       # length filled in at write time
        data.append(cd)
    return infos, data


def _record(name, bbox, infos, blend, opacity, visible, irrelevant, blocks):
    """§LR layer record: rect (top, left, bottom, right), channels, '8BIM' + blend key, opacity,
    clipping, flags (bit1 = hidden, bit3 = bit-4-is-valid, bit4 = pixel data irrelevant)."""
    l, t, r, b = bbox
    return LayerRecord(
        top=t, left=l, bottom=b, right=r, channel_info=infos, blend_mode=blend,
        opacity=int(round(255 * max(0.0, min(1.0, opacity)))), clipping=Clipping.BASE,
        flags=LayerFlags(visible=visible, photoshop_v5_later=True, pixel_data_irrelevant=irrelevant),
        mask_data=None, blending_ranges=LayerBlendingRanges(), name=_pascal(name), tagged_blocks=blocks)


def _load_icc(path=None):
    for p in ([path] if path else []) + list(ICC_CANDIDATES):
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                data = f.read()
            try:
                desc = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(data))).strip()
            except Exception:
                desc = "sRGB"
            return data, desc
    prof = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    return prof.tobytes(), "sRGB built-in (LittleCMS)"


def _xmp(title, dpi, icc_desc, now):
    t = _xml_escape(title or "")
    return ('<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/" '
            'xmlns:tiff="http://ns.adobe.com/tiff/1.0/">'
            f'<dc:format>application/vnd.adobe.photoshop</dc:format>'
            f'<dc:title><rdf:Alt><rdf:li xml:lang="x-default">{t}</rdf:li></rdf:Alt></dc:title>'
            f'<xmp:CreatorTool>svg2psd</xmp:CreatorTool><xmp:CreateDate>{now}</xmp:CreateDate>'
            f'<xmp:ModifyDate>{now}</xmp:ModifyDate><xmp:MetadataDate>{now}</xmp:MetadataDate>'
            f'<photoshop:ColorMode>3</photoshop:ColorMode>'
            f'<photoshop:ICCProfile>{_xml_escape(icc_desc)}</photoshop:ICCProfile>'
            f'<tiff:XResolution>{round(dpi * 1000)}/1000</tiff:XResolution>'
            f'<tiff:YResolution>{round(dpi * 1000)}/1000</tiff:YResolution>'
            f'<tiff:ResolutionUnit>2</tiff:ResolutionUnit>'
            '</rdf:Description></rdf:RDF></x:xmpmeta>\n<?xpacket end="w"?>').encode("utf-8")


def _thumbnail(comp_rgba):
    """§IR 1036 thumbnail: kJpegRGB (format 1), max 160 px, flattened on white."""
    img = Image.fromarray(comp_rgba, "RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, (0, 0), img)
    bg.thumbnail((160, 160), Image.LANCZOS)
    buf = io.BytesIO()
    bg.save(buf, "JPEG", quality=85)
    w, h = bg.size
    row = (w * 24 + 31) // 32 * 4
    return ThumbnailResource(fmt=1, width=w, height=h, row=row, total_size=row * h, bits=24,
                             planes=1, data=buf.getvalue())


# =========================================================================== convert
def convert(svg_text: str, out_path: str, px: int = 4096, dpi: int = 300, title: str = None,
            workers: int = None, icc_path: str = None, check: bool = False) -> dict:
    """Convert layered SVG text to a layered PSD at ``out_path``; returns a report dict.

    ``check=True`` re-opens the written file with psd-tools, composites the layer stack
    (stored pixels, folders, opacity, blend modes) and compares it with the merged image;
    the numbers land in ``report['consistency']`` and a mismatch adds a warning."""
    t0 = time.time()
    px = int(px)
    dpi = float(dpi)
    if dpi <= 0 or dpi >= 32768:
        raise ValueError("dpi must be in (0, 32768)")
    if not 1 <= px <= 30000:
        raise ValueError("px must be 1..30000 (PSD limit; PSB not supported)")
    warnings = []
    warn = warnings.append
    b = _Builder(svg_text, px, warn)
    tree = b.build()
    if title is None:
        tnode = next((e for e in b.root if lname(e) == "title"), None)
        title = ("".join(tnode.itertext()).strip() if tnode is not None else "") or \
            os.path.splitext(os.path.basename(out_path))[0]

    # ---------------------------------------------------------------- render (parallel)
    leaves = []

    def collect(nodes):
        for n in nodes:
            if n.kind == "group":
                collect(n.children)
            else:
                leaves.append(n)
    collect(tree)

    if b.root_ns:
        ref_svg = svg_text
    else:                                    # rsvg needs the SVG namespace: serialise with xmlns
        r = copy.deepcopy(b.root)
        for e in r.iter():
            e.attrib.pop(UID, None)
        ref_svg = ET.tostring(r, encoding="unicode")

    def job(n):
        if n is None:
            return render_svg(ref_svg, px)
        svg = isolated_svg(b.root, n.uid, n.target_style, b.root_ns)
        box = n.info.get("render_box")
        if box is not None:
            if box[2] <= box[0] or box[3] <= box[1]:
                return None, None                       # entirely off the canvas
            try:
                win = render_svg_box(svg, px, box)
            except (RuntimeError, subprocess.CalledProcessError, OSError):
                win = None
            if win is not None:
                a = win[..., 3]
                cut = ((box[0] > 0 and a[:, 0].any()) or (box[1] > 0 and a[0].any()) or
                       (box[2] < px and a[:, -1].any()) or (box[3] < px and a[-1].any()))
                if not cut:                             # else: fall back to the full render
                    bb, crop = _crop(win)
                    if bb is None:
                        return None, None
                    return (bb[0] + box[0], bb[1] + box[1], bb[2] + box[0], bb[3] + box[1]), crop
        bbox, crop = _crop(render_svg(svg, px))
        return bbox, crop

    workers = workers or max(1, min(4, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_comp = ex.submit(job, None)
        futs = [(n, ex.submit(job, n)) for n in leaves]
        for n, f in futs:
            bbox, crop = f.result()
            n.bbox = bbox or (0, 0, 0, 0)
            n.pixels = crop
            if bbox is None:
                warn(f"{n.name}: renders no pixels at {px}px")
        comp = fut_comp.result()

    # ---------------------------------------------------------------- layer records (bottom first)
    records, channels = [], ChannelImageData()
    lid = [0]

    def next_id():
        lid[0] += 1
        return lid[0]

    def emit(nodes, path, depth):
        for n in nodes:
            full = f"{path}/{n.name}" if path else n.name
            entry = {"name": n.name, "kind": n.kind, "path": full, "depth": depth, "uid": n.uid,
                     "svg_id": n.svg_id, "visible": n.visible, "opacity": round(n.opacity, 4),
                     "blend": n.blend}
            if n.kind == "group":
                # §LM section divider: '</Layer group>' record (lsct 3) below the children,
                # the folder record (lsct 1 open / 2 closed) above them.
                infos, data = _channels(None)
                div = _record("</Layer group>", (0, 0, 0, 0), infos, BlendMode.NORMAL, 1.0, True, True,
                              _blocks("</Layer group>", next_id(), [(Tag.SECTION_DIVIDER_SETTING,
                                      SectionDividerSetting(kind=SectionDivider.BOUNDING_SECTION_DIVIDER))]))
                records.append(div)
                channels.append(data)
                emit(n.children, full, depth + 1)
                mode = BLEND_MAP[n.blend]
                if not n.info.get("isolated"):
                    lsct_mode = rec_mode = BlendMode.PASS_THROUGH      # §LR / §AL lsct: 'pass'
                else:
                    lsct_mode = rec_mode = mode
                kind = SectionDivider.OPEN_FOLDER if n.open else SectionDivider.CLOSED_FOLDER
                infos, data = _channels(None)
                rec = _record(n.name, (0, 0, 0, 0), infos, rec_mode, n.opacity, n.visible, True,
                              _blocks(n.name, next_id(), [(Tag.SECTION_DIVIDER_SETTING,
                                      SectionDividerSetting(kind=kind, signature=b"8BIM",
                                                            blend_mode=lsct_mode))]))
                records.append(rec)
                channels.append(data)
                entry.update({"open": n.open, "psd_blend": lsct_mode.name})
                n.entry = entry
                continue
            mode = BLEND_MAP[n.blend]
            if n.kind == "shape":
                i = n.info
                vmsk, nk = _vector_mask(i["region"], px, px)
                extra = []
                if i["fill_opacity"] < 1:
                    extra.append((Tag.BLEND_FILL_OPACITY, _byte(int(round(255 * i["fill_opacity"])))))
                extra += [(Tag.SOLID_COLOR_SHEET_SETTING, _soco(i["fill"])),
                          (Tag.VECTOR_MASK_SETTING1, vmsk),
                          (Tag.VECTOR_STROKE_DATA, _vstk(i["stroke"], i["fill_enabled"], dpi))]
                const = i["fill"] if (i["stroke"] is None and i["fill_enabled"]) else None
                infos, data = _channels(n.pixels, const)
                rec = _record(n.name, n.bbox, infos, mode, n.opacity, n.visible, True,
                              _blocks(n.name, next_id(), extra))
                ops = [OP_NAMES[op] for op, _ in i["region"]]
                entry.update({"bbox": list(n.bbox), "fill": _hex(i["fill"]) if i["fill_enabled"] else None,
                              "fill_opacity": round(i["fill_opacity"], 4), "subpaths": len(i["region"]),
                              "knots": nk, "ops": ops,
                              "geometry": "flattened" if any(s.flat for _, s in i["region"]) else "bezier",
                              "stroke": (dict(i["stroke"], rgb=_hex(i["stroke"]["rgb"])) if i["stroke"] else None)})
            else:
                infos, data = _channels(n.pixels)
                rec = _record(n.name, n.bbox, infos, mode, n.opacity, n.visible, False,
                              _blocks(n.name, next_id()))
                entry.update({"bbox": list(n.bbox), "reason": n.info.get("reason"),
                              "render": n.info.get("render")})
            records.append(rec)
            channels.append(data)
            n.entry = entry

    emit(tree, "", 0)
    report_layers = _panel_order(tree)      # Photoshop panel order: top-most first

    # ---------------------------------------------------------------- merged image (§ID)
    alpha = comp[..., 3]
    has_alpha = bool((alpha < 255).any())
    nch = 4 if has_alpha else 3
    header = FileHeader(version=1, channels=nch, height=px, width=px, depth=8, color_mode=ColorMode.RGB)
    if has_alpha:
        # Photoshop stores a transparent merged image matted on white; readers un-blend it.
        a = alpha.astype(np.uint32)
        planes = [((comp[..., k].astype(np.uint32) * a + 255 * (255 - a) + 127) // 255).astype(np.uint8)
                  for k in range(3)] + [alpha]
    else:
        planes = [comp[..., k] for k in range(3)]
    image_data = ImageData(compression=Compression.RLE)
    image_data.set_data([np.ascontiguousarray(p).tobytes() for p in planes], header)

    # ---------------------------------------------------------------- image resources (§IR)
    icc, icc_desc = _load_icc(icc_path)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    res = [
        # 1005 ResolutionInfo: hRes/vRes Fixed 16.16, unit 1 = pixels per inch, display unit 1 = in
        (Resource.RESOLUTION_INFO, ResoulutionInfo(horizontal=int(round(dpi * 65536)), horizontal_unit=1, width_unit=1,
                                                   vertical=int(round(dpi * 65536)), vertical_unit=1, height_unit=1)),
        (Resource.THUMBNAIL_RESOURCE, _thumbnail(comp)),                   # 1036 JPEG thumbnail
        (Resource.ICC_PROFILE, icc),                                       # 1039 ICC profile (raw)
        (Resource.VERSION_INFO, VersionInfo(version=1, has_composite=True, writer="svg2psd",
                                            reader="svg2psd", file_version=1)),  # 1057
        (Resource.XMP_METADATA, _xmp(title, dpi, icc_desc, now)),          # 1060 XMP
    ]
    resources = ImageResources([(k, ImageResource(key=k, data=d)) for k, d in res])

    # ---------------------------------------------------------------- layer & mask info (§LM)
    gblocks = TaggedBlocks()
    if has_alpha:
        gblocks.set_data(Tag.SAVING_MERGED_TRANSPARENCY)   # 'Mtrn': merged image has transparency
    n_layers = len(records)
    layer_info = LayerInfo(layer_count=(-n_layers if has_alpha else n_layers),   # <0: 1st alpha = merged transparency
                           layer_records=LayerRecords(records), channel_image_data=channels)
    lmi = LayerAndMaskInformation(layer_info=layer_info,
                                  global_layer_mask_info=GlobalLayerMaskInfo(overlay_color=[0, 65535, 0, 0, 0],
                                                                             opacity=50, kind=128),
                                  tagged_blocks=gblocks)
    psd = PSD(header=header, color_mode_data=ColorModeData(), image_resources=resources,
              layer_and_mask_information=lmi, image_data=image_data)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".psd.part", dir=out_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            psd.write(f)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)          # mkstemp creates 0600; use the normal file mode
        os.replace(tmp, out_path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

    counts = {"group": 0, "shape": 0, "pixel": 0}
    for e in report_layers:
        counts[e["kind"]] += 1
    consistency = None
    if check:
        from psd_tools import PSDImage
        lay = np.asarray(PSDImage.open(out_path).composite(ignore_preview=True).convert("RGBA"))
        consistency = _diff_stats(lay, comp)
        if not consistency["ok"]:
            warn(f"layer composite differs from the merged image: {consistency}")
    return {
        "out": os.path.abspath(out_path), "px": px, "dpi": dpi, "title": title,
        "bytes": os.path.getsize(out_path), "merged_channels": nch, "merged_alpha": has_alpha,
        "icc": icc_desc, "records": n_layers, "counts": counts, "layers": report_layers,
        "warnings": warnings, "consistency": consistency, "renderer": _rsvg_version(),
        "seconds": round(time.time() - t0, 2),
    }


def _rsvg_version():
    try:
        return subprocess.run(["rsvg-convert", "--version"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def _diff_stats(a, b, tol=10 / 255):
    """Premultiplied RGBA difference; pixels next to a colour/alpha edge of ``b`` are AA edges."""
    def pm(x):
        x = x.astype(np.float32) / 255.0
        x[..., :3] *= x[..., 3:4]
        return x
    A, B = pm(a), pm(b)
    d = np.abs(A - B).max(axis=2)
    hi = B.copy()
    lo = B.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sh = np.roll(np.roll(B, dy, 0), dx, 1)
            np.maximum(hi, sh, out=hi)
            np.minimum(lo, sh, out=lo)
    edge = (hi - lo).max(axis=2) > 8 / 255
    edge2 = edge.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            edge2 |= np.roll(np.roll(edge, dy, 0), dx, 1)
    bad = int(((d > tol) & ~edge2).sum())
    mean = float(np.abs(A - B).mean())
    return {"mean_abs_x255": round(mean * 255, 4), "max_abs_x255": round(float(d.max()) * 255, 2),
            "px_gt10_off_edge": bad, "ok": bool(mean < 2 / 255 and bad == 0)}


def _byte(v):
    from psd_tools.psd.base import ByteElement
    return ByteElement(v)   # 'iOpa' §AL Fill opacity: 1 byte + 3 padding


def _panel_order(tree):
    """Report entries in Photoshop's panel order (top-most first, depth-first)."""
    out = []

    def walk(nodes):
        for n in reversed(nodes):
            out.append(n.entry)
            if n.kind == "group":
                walk(n.children)
    walk(tree)
    return out


def format_tree(report) -> str:
    lines = []
    for e in report["layers"]:
        pad = "  " * e["depth"]
        flags = ("" if e["visible"] else " hidden") + (f" op={e['opacity']}" if e["opacity"] < 1 else "") + \
                (f" blend={e['blend']}" if e["blend"] != "normal" else "")
        if e["kind"] == "group":
            lines.append(f"{pad}[group] {e['name']} ({'open' if e['open'] else 'closed'}{flags})")
        elif e["kind"] == "shape":
            lines.append(f"{pad}[shape] {e['name']} fill={e['fill']} subpaths={e['subpaths']} knots={e['knots']}"
                         f" ops={','.join(e['ops'])} {e['geometry']} bbox={e['bbox']}{flags}")
        else:
            lines.append(f"{pad}[pixel] {e['name']} ({e['reason']}) bbox={e['bbox']}{flags}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Layered SVG -> layered PSD (shape layers + pixels).")
    ap.add_argument("svg")
    ap.add_argument("psd")
    ap.add_argument("--px", type=int, default=4096)
    ap.add_argument("--dpi", type=float, default=300)
    ap.add_argument("--title", default=None)
    ap.add_argument("--report", default=None, help="write the JSON report here")
    ap.add_argument("--check", action="store_true",
                    help="composite the written layers with psd-tools and compare with the merged image")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    with open(a.svg, encoding="utf-8") as f:
        rep = convert(f.read(), a.psd, px=a.px, dpi=a.dpi, title=a.title, check=a.check)
    if a.report:
        with open(a.report, "w") as f:
            json.dump(rep, f, indent=2)
    if not a.quiet:
        print(format_tree(rep))
        print(f"wrote {rep['out']}  {rep['bytes'] / 1e6:.2f} MB  {rep["px"]}px @ {rep["dpi"]:g}dpi  "
              f"{rep['counts']}  in {rep['seconds']}s")
        if rep["consistency"]:
            print("consistency (layer composite vs merged image):", rep["consistency"])
        for w in rep["warnings"]:
            print("warning:", w)
    return rep


if __name__ == "__main__":
    main()
