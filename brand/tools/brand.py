"""The final K mark: master geometry, palette and every variant as layered SVG.

Decisions (design panel of three directions + three judges, then the client):
  * Geometry  - the "precision" direction: arm and leg share one 4:3 slope (53.130 deg) and one
                width (96.5% of the stem, optical compensation for diagonals), their right
                extremes align, J and C mirror about P. Scaled back to the original footprint
                (cap height 58.7% of the disc) and optically centred (+14 x / -6 y).
  * Corners   - sharp everywhere (client request). Only containers (disc, app tile) are curved.
  * Palette   - the client's own colours, deepened and moved away from Google's hues
                (blue +4.5 deg, green -4.5 deg, yellow +4.5 deg in OKLCH).

Every variant is an SVG document whose named <g> elements (id + data-name) become layers in the
PSD / AI exports. Editable masters use the exact facet polygons; raster output uses trapped
facets and supersampling so anti-aliased shared edges never show a seam.
"""
from __future__ import annotations

import io
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from PIL import Image  # noqa: E402
from shapely.geometry import LineString, Point, Polygon  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

import kmark  # noqa: E402
from kmark import FACETS, PAINT_ORDER, KParams, facets, fmt, points, poly_d, silhouette, svg_doc  # noqa: E402
from okcolor import mix, oklch_to_hex, tone  # noqa: E402

NAME = "K Logo"
SIZE = 1024.0

# ------------------------------------------------------------------ palette
PALETTE_OKLCH = {
    "red": (0.615, 0.215, 31.0),
    "yellow": (0.835, 0.170, 88.5),
    "green": (0.625, 0.180, 144.0),
    "blue": (0.595, 0.200, 264.5),
}
PALETTE = {n: oklch_to_hex(*v) for n, v in PALETTE_OKLCH.items()}
NEUTRAL = {
    "white": "#FFFFFF",
    "ink": "#141A29",      # brand near-black for type and one-colour use on screen
    "night": "#070D1A",    # tech / dark-mode background
    "black": "#000000",
}
FACET_LABEL = {"red": "Red - top", "yellow": "Yellow - wedge", "blue": "Blue - arm", "green": "Green - stem and leg"}
# greyscale tints of ink, ordered like the colours' lightness so the fold survives in grey
GREY_TINT = {"yellow": 0.30, "red": 0.55, "blue": 0.78, "green": 1.0}


# ------------------------------------------------------------------ geometry
def _precision() -> KParams:
    slope = math.degrees(math.atan2(4, 3))                  # 53.1301 deg
    return KParams(size=SIZE, x0=289.2, w=137.3925, yT=210.0, yB=802.0, yJ=437.3038,
                   arm_angle=slope, arm_w=132.5838, yC=574.6962, leg_angle=slope, leg_w=132.5838)


ORIGINAL_CAP = 812.5888334211539 - 211.43467611778317     # trace cap height (1024 grid)


def transform(k: KParams, s: float, cx: float, cy: float, dx: float = 0.0, dy: float = 0.0) -> KParams:
    """Scale the construction by s about (cx, cy), then translate; geometry stays baked in."""
    d = dict(k.__dict__)
    for key in ("x0", "yT", "yB", "yJ", "yC"):
        c = cx if key == "x0" else cy
        d[key] = c + (d[key] - c) * s + (dx if key == "x0" else dy)
    for key in ("w", "arm_w", "leg_w"):
        d[key] *= s
    d["palette"] = dict(k.palette)
    return KParams(**d)


def master() -> KParams:
    k = _precision()
    x0, y0, x1, y1 = kmark.bbox(k)
    k = transform(k, ORIGINAL_CAP / (y1 - y0), (x0 + x1) / 2, (y0 + y1) / 2)
    k.r_convex = k.r_concave = 0.0                           # sharp corners, always
    k.palette = dict(PALETTE, disc=NEUTRAL["white"], ink=NEUTRAL["ink"], dark_bg=NEUTRAL["night"])
    return k


def scaled_about_centre(k: KParams, s: float) -> KParams:
    return transform(k, s, k.size / 2, k.size / 2)


# ------------------------------------------------------------------ shape helpers
def _polys(g):
    if g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    return [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon"]


def shape_d(g, min_area=0.0):
    out = []
    for p in _polys(g):
        if p.area <= min_area:
            continue
        out.append(poly_d([tuple(c) for c in p.exterior.coords][:-1]))
        for hole in p.interiors:
            out.append(poly_d([tuple(c) for c in hole.coords][:-1]))
    return " ".join(out)


def trapped(k: KParams, pieces=None, t=None):
    """[(name, d)] in paint order; each piece grows by t only under pieces painted later.
    Detached slivers that buffering can leave at J are dropped."""
    if pieces is None:
        pieces = {n: Polygon(p) for n, p in facets(k).items()}
    t = k.size / 512.0 if t is None else t
    order = [n for n in PAINT_ORDER if n in pieces]
    min_area = 25.0 * (k.size / 1024.0) ** 2
    out = []
    for i, n in enumerate(order):
        allowed = unary_union([pieces[m] for m in order[i:]])
        g = pieces[n].buffer(t, join_style=2, mitre_limit=4.0).intersection(allowed)
        out.append((n, shape_d(g, min_area)))
    return out


def gapped(k: KParams, gap=None):
    """Facet pieces separated by gaps cut along the fold line and the arm's inner edge, so a
    one-colour mark still shows the construction (~0.8% of cap height)."""
    p = points(k)
    gap = 0.008 * (k.yB - k.yT) if gap is None else gap
    cuts = unary_union([LineString([p["foldL"], p["P"]]), LineString([p["innerL"], p["J"]])])
    band = cuts.buffer(gap / 2, cap_style=3, join_style=2)
    return {n: Polygon(f).difference(band) for n, f in facets(k).items()}


def _attrs(gid, name):
    return f'id="{gid}" data-name="{name}"'


def _lin(gid, p1, p2, stops):
    s = "".join(f'<stop offset="{fmt(o)}" stop-color="{c}"' + ("" if a is None or a >= 1 else f' stop-opacity="{fmt(a)}"') + "/>"
                for o, c, a in stops)
    return (f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{fmt(p1[0])}" y1="{fmt(p1[1])}" '
            f'x2="{fmt(p2[0])}" y2="{fmt(p2[1])}">{s}</linearGradient>')


def _rad(gid, c, r, stops):
    s = "".join(f'<stop offset="{fmt(o)}" stop-color="{col}"' + ("" if a is None or a >= 1 else f' stop-opacity="{fmt(a)}"') + "/>"
                for o, col, a in stops)
    return (f'<radialGradient id="{gid}" gradientUnits="userSpaceOnUse" cx="{fmt(c[0])}" cy="{fmt(c[1])}" '
            f'r="{fmt(r)}">{s}</radialGradient>')


def _circle_d(c, r):
    x, y = c
    return (f"M{fmt(x - r)} {fmt(y)} A{fmt(r)} {fmt(r)} 0 1 0 {fmt(x + r)} {fmt(y)} "
            f"A{fmt(r)} {fmt(r)} 0 1 0 {fmt(x - r)} {fmt(y)} Z")


def _rrect_d(x, y, w, h, r):
    return (f"M{fmt(x + r)} {fmt(y)} H{fmt(x + w - r)} A{fmt(r)} {fmt(r)} 0 0 1 {fmt(x + w)} {fmt(y + r)} "
            f"V{fmt(y + h - r)} A{fmt(r)} {fmt(r)} 0 0 1 {fmt(x + w - r)} {fmt(y + h)} H{fmt(x + r)} "
            f"A{fmt(r)} {fmt(r)} 0 0 1 {fmt(x)} {fmt(y + h - r)} V{fmt(y + r)} A{fmt(r)} {fmt(r)} 0 0 1 {fmt(x + r)} {fmt(y)} Z")


# ------------------------------------------------------------------ mark groups
def mark_group(k: KParams, colours=None, raster=False, gid="K", gaps=False):
    """Group of the four facets. colours: {facet: hex}; raster=True -> trapped facets."""
    colours = colours or {n: k.palette[n] for n in FACETS}
    if gaps:
        pieces = gapped(k)
        paths = [(n, shape_d(pieces[n])) for n in PAINT_ORDER]
    elif raster:
        paths = trapped(k)
    else:
        f = facets(k)
        paths = [(n, poly_d(f[n])) for n in PAINT_ORDER]
    body = "".join(f'<path {_attrs(f"{gid}-{n}", FACET_LABEL[n])} fill="{colours[n]}" d="{d}"/>' for n, d in paths)
    return f'<g {_attrs(gid, "K mark")}>{body}</g>'


def mono_colours(hexc):
    return {n: hexc for n in FACETS}


def grey_colours(ink=NEUTRAL["black"]):
    return {n: mix("#FFFFFF", ink, GREY_TINT[n]) for n in FACETS}


# light model for the dimensional mark: each facet tilts away from the light toward its lower
# crease, so every crease is a shaded edge meeting a lit one. Steps are small (<= 0.04 OKLCH L)
# and hue drift is held to 3 deg so each facet still reads as one brand colour.
SHADE = {           # (highlight dL, dC, dh), (shadow dL, dC, dh)
    "red": ((0.022, -0.004, 1), (-0.028, 0.000, -1)),
    "yellow": ((0.028, -0.008, 2), (-0.030, 0.005, -3)),
    "blue": ((0.035, -0.008, -2), (-0.040, 0.004, 1)),
    "green": ((0.035, 0.000, -2), (-0.040, 0.000, 2)),
}


def _foot(q, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    s = ((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / (dx * dx + dy * dy)
    return (a[0] + s * dx, a[1] + s * dy)


def dimensional_group(k: KParams, uid="dim"):
    """Folded-ribbon finish for light backgrounds: one gradient per facet, darkest at its lower
    crease, plus a fine highlight on the lit side of the fold that peaks at J. No masks, no
    filters, no drop shadow - converts 1:1 to PDF / AI / EPS / PSD."""
    pal = k.palette
    p = points(k)
    J, P = p["J"], p["P"]
    f = facets(k)
    defs, body = [], []
    fills = {}
    crease = {"red": (p["foldL"], J), "yellow": (J, p["innerL"]), "blue": (J, P)}
    for c, (a, b) in crease.items():
        far = max(f[c], key=lambda q: math.dist(q, _foot(q, a, b)))
        gid = f"{uid}-{c}-fill"
        defs.append(_lin(gid, far, _foot(far, a, b), [(0, tone(pal[c], *SHADE[c][0]), None),
                                                       (1, tone(pal[c], *SHADE[c][1]), None)]))
        fills[c] = f"url(#{gid})"
    gid = f"{uid}-green-fill"
    defs.append(_lin(gid, J, (J[0] + 0.25 * (k.yB - J[1]), k.yB),
                     [(0, tone(pal["green"], *SHADE["green"][0]), None), (1, tone(pal["green"], *SHADE["green"][1]), None)]))
    fills["green"] = f"url(#{gid})"
    for n, d in trapped(k):
        body.append(f'<path {_attrs(f"{uid}-{n}", FACET_LABEL[n])} fill="{fills[n]}" d="{d}"/>')
    # fold highlight: light catching the crease, fading away from it into yellow/green
    w = 0.018 * (k.yB - k.yT)
    lower = unary_union([Polygon(f["yellow"]), Polygon(f["green"])])
    a, b = p["foldL"], P
    dx, dy = b[0] - a[0], b[1] - a[1]
    n_ = math.hypot(dx, dy)
    nx, ny = -dy / n_, dx / n_
    if ny < 0:
        nx, ny = -nx, -ny                                     # point down, into yellow/green
    band = Polygon([a, b, (b[0] + nx * w, b[1] + ny * w), (a[0] + nx * w, a[1] + ny * w)]).intersection(lower)
    defs.append(_lin(f"{uid}-fold-light", J, (J[0] + nx * w, J[1] + ny * w),
                     [(0, "#FFFFFF", 0.34), (0.35, "#FFFFFF", 0.12), (1, "#FFFFFF", 0.0)]))
    body.append(f'<path {_attrs(f"{uid}-fold-light", "Fold highlight")} fill="url(#{uid}-fold-light)" d="{shape_d(band)}"/>')
    return "".join(defs), f'<g {_attrs(uid, "K mark - dimensional")}>' + "".join(body) + "</g>"


# ------------------------------------------------------------------ containers
def disc(k: KParams, fill=None, gid="disc"):
    c = k.size / 2
    return f'<circle {_attrs(gid, "Disc")} cx="{fmt(c)}" cy="{fmt(c)}" r="{fmt(c)}" fill="{fill or k.palette["disc"]}"/>'


# ------------------------------------------------------------------ documents
def doc_mark(k=None, style="color", raster=False, artboard="square", title=None):
    """style: color | dimensional | black | white | ink | grey | black-gapped | white-gapped."""
    k = k or master()
    defs = ""
    if style == "dimensional":
        defs, body = dimensional_group(k)
    elif style == "color":
        body = mark_group(k, raster=raster)
    elif style == "grey":
        body = mark_group(k, colours=grey_colours(), raster=raster)
    else:
        base = style.replace("-gapped", "")
        colour = {"black": NEUTRAL["black"], "white": NEUTRAL["white"], "ink": NEUTRAL["ink"]}[base]
        body = mark_group(k, colours=mono_colours(colour), raster=raster, gaps=style.endswith("-gapped"))
    title = title or f"{NAME} - mark ({style})"
    if artboard == "square":
        return svg_doc(body, defs, size=k.size, title=title)
    # tight or clear-space artboard around the mark
    x0, y0, x1, y1 = kmark.bbox(k)
    m = k.w if artboard == "clearspace" else 0.0
    vb = (x0 - m, y0 - m, x1 - x0 + 2 * m, y1 - y0 + 2 * m)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'viewBox="{fmt(vb[0])} {fmt(vb[1])} {fmt(vb[2])} {fmt(vb[3])}" width="{fmt(vb[2])}" height="{fmt(vb[3])}">'
            f'<title>{title}</title>' + (f"<defs>{defs}</defs>" if defs else "") + body + "</svg>")


def doc_badge(k=None, style="color", raster=False, disc_fill=None):
    k = k or master()
    if style == "dimensional":
        defs, body = dimensional_group(k)
    else:
        defs, body = "", mark_group(k, raster=raster)
    inner = f'<g {_attrs("badge", "Badge")}>{disc(k, disc_fill)}{body}</g>'
    return svg_doc(inner, defs, size=k.size, title=f"{NAME} - avatar badge ({style})")


# ------------------------------------------------------------------ tech icon
TILE_R = 0.2237             # app-tile corner radius as a fraction of the tile


def _tech_layers(k: KParams, detail="full", shape="tile"):
    """Returns (defs, [group strings]) for the tech icon drawn on a k.size square."""
    s = k.size
    e = s / 512.0                                            # opaque edge width
    pal = k.palette
    night = NEUTRAL["night"]
    p = points(k)
    J, P = p["J"], p["P"]
    f = facets(k)
    H = k.yB - k.yT
    u = H / 20.0                                             # icon unit: 1/20 of the cap height
    defs, groups = [], []

    # 1 - tile: an opaque pre-blended edge (so the plate never vanishes on dark UIs) + plate
    edge = mix(night, "#FFFFFF", 0.085)
    top = tone(night, dL=0.075, dC=0.012)
    defs.append(_lin("tech-plate", (0, 0), (0, s), [(0, top, None), (0.55, tone(night, dL=0.02), None), (1, night, None)]))
    if shape == "tile":
        outer, inner = _rrect_d(0, 0, s, s, TILE_R * s), _rrect_d(e, e, s - 2 * e, s - 2 * e, TILE_R * s - e)
        groups.append(f'<g {_attrs("tile", "Tile")}><path {_attrs("tile-edge", "Tile edge")} fill="{edge}" d="{outer}"/>'
                      f'<path {_attrs("tile-plate", "Plate")} fill="url(#tech-plate)" d="{inner}"/></g>')
    else:                                                    # full-bleed square for app stores
        groups.append(f'<g {_attrs("tile", "Tile")}><rect {_attrs("tile-plate", "Plate")} width="{fmt(s)}" '
                      f'height="{fmt(s)}" fill="url(#tech-plate)"/></g>')

    if detail == "full":
        # 2 - construction grid anchored on J, fading out from the centre (filled hairlines)
        g = k.w / 2
        R = 0.47 * s
        cxy = (s / 2, s / 2)
        circle = Point(cxy).buffer(R, resolution=64)
        hw = 0.6 * s / 1024
        strips = []
        for i in range(-12, 13):
            x = J[0] + i * g
            y = J[1] + i * g
            strips.append(LineString([(x, 0), (x, s)]).buffer(hw, cap_style=2))
            strips.append(LineString([(0, y), (s, y)]).buffer(hw, cap_style=2))
        grid = unary_union(strips).intersection(circle)
        defs.append(_rad("tech-grid", cxy, R, [(0, "#FFFFFF", 0.085), (0.6, "#FFFFFF", 0.04), (1, "#FFFFFF", 0.0)]))
        groups.append(f'<g {_attrs("grid", "Construction grid")}><path fill="url(#tech-grid)" d="{shape_d(grid)}"/></g>')

        # 3 - ambient light pooled behind the mark
        ctr = ((k.x0 + p["armTR"][0]) / 2, (k.yT + k.yB) / 2)
        glow = tone(pal["blue"], dL=0.04, dC=-0.03)
        defs.append(_rad("tech-ambient", ctr, 0.40 * s, [(0, glow, 0.24), (0.5, glow, 0.08), (1, glow, 0.0)]))
        groups.append(f'<g {_attrs("ambient", "Ambient light")}><path fill="url(#tech-ambient)" d="{_circle_d(ctr, 0.40 * s)}"/></g>')

        # 4 - traces: the fold line and the arm's inner edge run on past the letter to pads
        sig = tone(pal["blue"], dL=0.26, dC=-0.11)
        tw = 0.11 * u
        runs = [(p["foldL"], (p["foldL"][0] - P[0], p["foldL"][1] - P[1]), 2.4 * u),
                (P, (P[0] - J[0], P[1] - J[1]), 2.2 * u),
                (p["innerL"], (p["innerL"][0] - J[0], p["innerL"][1] - J[1]), 2.4 * u)]
        tr = []
        for i, (a, d, L) in enumerate(runs):
            n_ = math.hypot(*d)
            d = (d[0] / n_, d[1] / n_)
            b = (a[0] + d[0] * L, a[1] + d[1] * L)
            pad_r = 0.32 * u
            end = (b[0] - d[0] * pad_r, b[1] - d[1] * pad_r)
            line = LineString([a, end]).buffer(tw / 2, cap_style=2)
            defs.append(_lin(f"tech-trace{i}", a, b, [(0, sig, 0.75), (1, sig, 0.35)]))
            tr.append(f'<path {_attrs(f"trace{i}", f"Trace {i + 1}")} fill="url(#tech-trace{i})" d="{shape_d(line)}"/>')
            ring = Point(b).buffer(pad_r, resolution=48).difference(Point(b).buffer(pad_r - tw, resolution=48))
            tr.append(f'<path {_attrs(f"pad{i}", f"Pad {i + 1}")} fill="{sig}" fill-opacity="0.5" d="{shape_d(ring)}"/>')
        groups.append(f'<g {_attrs("traces", "Traces")}>' + "".join(tr) + "</g>")

    # 5 - the K, lit from the node at J (hot stop capped at +0.05 L so colours never wash out)
    reach = max(math.dist(J, q) for q in silhouette(k))
    mk = []
    for n, d in trapped(k):
        if detail == "full":
            defs.append(f'<radialGradient id="tech-{n}" gradientUnits="userSpaceOnUse" cx="{fmt(J[0])}" cy="{fmt(J[1])}" '
                        f'r="{fmt(reach)}"><stop offset="0" stop-color="{tone(pal[n], dL=min(0.05, 0.2 * (1 - PALETTE_OKLCH[n][0])))}"/>'
                        f'<stop offset="0.4" stop-color="{tone(pal[n], dL=0.015)}"/>'
                        f'<stop offset="1" stop-color="{tone(pal[n], dL=-0.03, dC=0.005)}"/></radialGradient>')
            fill = f"url(#tech-{n})"
        else:
            fill = pal[n]
        mk.append(f'<path {_attrs(f"tech-{n}", FACET_LABEL[n])} fill="{fill}" d="{d}"/>')
    if detail == "full":
        sil = Polygon(silhouette(k))
        # rim light on the edges that face the light: stem's left edge and cap, the arm's inner edge and cap
        lit = unary_union([LineString([p["BL"], p["TL"], p["TR"]]), LineString([J, p["armTL"], p["armTR"]])])
        rim = lit.buffer(0.07 * u, cap_style=2, join_style=2).intersection(sil)
        y0, y1 = k.yT, k.yB
        defs.append(_lin("tech-rim", (0, y0), (0, y1), [(0, "#FFFFFF", 0.5), (0.12, "#FFFFFF", 0.2), (0.5, "#FFFFFF", 0.04), (1, "#FFFFFF", 0.0)]))
        mk.append(f'<path {_attrs("tech-rim", "Rim light")} fill="url(#tech-rim)" d="{shape_d(rim)}"/>')
        # the fold as a line of light on the flap side, brightest at J
        a, b = p["foldL"], P
        dx, dy = b[0] - a[0], b[1] - a[1]
        n_ = math.hypot(dx, dy)
        nx, ny = dy / n_, -dx / n_
        if ny > 0:
            nx, ny = -nx, -ny
        w = 0.06 * u
        band = Polygon([a, b, (b[0] + nx * w, b[1] + ny * w), (a[0] + nx * w, a[1] + ny * w)]).intersection(sil)
        tj = math.dist(a, J) / n_
        defs.append(_lin("tech-crease", a, b, [(0, "#FFFFFF", 0.08), (tj, "#FFFFFF", 0.5), (1, "#FFFFFF", 0.12)]))
        mk.append(f'<path {_attrs("tech-crease", "Fold light")} fill="url(#tech-crease)" d="{shape_d(band)}"/>')
    groups.append(f'<g {_attrs("K", "K mark")}>' + "".join(mk) + "</g>")

    if detail == "full":
        # 6 - the node at J: a soft bloom, a dark ring cut into the letter and a bright core
        defs.append(_rad("tech-bloom", J, 1.4 * u, [(0, "#FFFFFF", 0.22), (0.3, "#FFFFFF", 0.08), (1, "#FFFFFF", 0.0)]))
        groups.append(f'<g {_attrs("node", "Node")}>'
                      f'<path {_attrs("node-bloom", "Bloom")} fill="url(#tech-bloom)" d="{_circle_d(J, 1.4 * u)}"/>'
                      f'<path {_attrs("node-ring", "Ring")} fill="{night}" d="{_circle_d(J, 0.34 * u)}"/>'
                      f'<path {_attrs("node-core", "Core")} fill="#FFFFFF" d="{_circle_d(J, 0.2 * u)}"/></g>')
    return defs, groups


def doc_tech(k=None, detail="full", shape="tile"):
    """shape: tile (rounded, transparent corners) | square (full-bleed, opaque: App Store / Play)
    | macos (824 px tile on a 1024 canvas, Apple's macOS icon grid)."""
    k = k or master()
    if shape == "macos":
        defs, groups = _tech_layers(k, detail, "tile")
        sc = 824.0 / 1024.0
        off = (k.size - k.size * sc) / 2
        body = f'<g {_attrs("macos", "macOS icon")} transform="translate({fmt(off)} {fmt(off)}) scale({fmt(sc)})">' + "".join(groups) + "</g>"
    else:
        defs, groups = _tech_layers(k, detail, shape)
        body = "".join(groups)
    return svg_doc(f'<g {_attrs("tech-icon", "Tech icon")}>{body}</g>', "".join(defs), size=k.size,
                   title=f"{NAME} - tech icon ({detail}, {shape})")


def doc_android_foreground(k=None):
    """Adaptive-icon foreground (108 dp canvas): the K scaled to sit inside the 66 dp safe circle."""
    k = scaled_about_centre(k or master(), 0.78)
    return svg_doc(mark_group(k), size=k.size, title=f"{NAME} - Android adaptive foreground")


def doc_android_background(k=None):
    k = k or master()
    night = NEUTRAL["night"]
    s = k.size
    defs = _lin("bg", (0, 0), (0, s), [(0, tone(night, dL=0.075, dC=0.012), None), (1, night, None)])
    return svg_doc(f'<rect {_attrs("background", "Background")} width="{fmt(s)}" height="{fmt(s)}" fill="url(#bg)"/>',
                   defs, size=s, title=f"{NAME} - Android adaptive background")


# ------------------------------------------------------------------ raster
def render(svg: str, size: int, height: int = None) -> Image.Image:
    """Render with librsvg, supersampled for small sizes and box-filtered in premultiplied alpha
    (no dark fringes, no conflation seams)."""
    height = height or size
    ss = 8 if max(size, height) <= 128 else 4 if max(size, height) <= 512 else 2 if max(size, height) <= 1024 else 1
    out = subprocess.run(["rsvg-convert", "-w", str(size * ss), "-h", str(height * ss), "-f", "png"],
                         input=svg.encode(), capture_output=True, check=True).stdout
    im = Image.open(io.BytesIO(out)).convert("RGBA")
    if ss > 1:
        im = im.convert("RGBa").reduce(ss).convert("RGBA")
    return im


def params_json(k=None) -> str:
    k = k or master()
    p = points(k)
    out = {"params": json.loads(k.to_json()), "points": {n: [round(x, 4), round(y, 4)] for n, (x, y) in p.items()},
           "palette_oklch": PALETTE_OKLCH, "palette": PALETTE, "neutrals": NEUTRAL}
    return json.dumps(out, indent=2)


if __name__ == "__main__":
    k = master()
    print(params_json(k))
