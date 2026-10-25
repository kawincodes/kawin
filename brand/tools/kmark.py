"""Parametric construction of the K mark.

Every variant of the logo is generated from one small set of parameters, so the
trace, the redesign and every exported format share exactly the same geometry.

Construction (y grows downward, artboard is ``size`` x ``size``):
  * Stem    – rectangle x0..x0+w, yT..yB.
  * Arm     – band rising to the upper right at ``arm_angle``°; its inner (lower-left)
              edge passes through the junction J=(x1, yJ) on the stem's right edge.
  * Leg     – band falling to the lower right at ``leg_angle``°; its inner edge
              starts at the crotch C=(x1, yC).
  * P       – where the arm's outer edge meets the leg's outer edge.
  * Facets  – the line J→P (the "fold") and the arm's inner edge, both extended
              across the stem, split the letter into red / yellow / blue / green.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

FACETS = ("red", "yellow", "green", "blue")


@dataclass
class KParams:
    size: float = 1024.0
    x0: float = 0.0
    w: float = 0.0
    yT: float = 0.0
    yB: float = 0.0
    yJ: float = 0.0
    arm_angle: float = 0.0
    arm_w: float = 0.0
    yC: float = 0.0
    leg_angle: float = 0.0
    leg_w: float = 0.0
    # optional corner rounding of the outer silhouette (design units)
    r_convex: float = 0.0
    r_concave: float = 0.0
    palette: dict = field(default_factory=dict)

    @property
    def x1(self):
        return self.x0 + self.w

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def scaled(self, s, size=None):
        d = asdict(self)
        for k in ("x0", "w", "yT", "yB", "yJ", "arm_w", "yC", "leg_w", "r_convex", "r_concave"):
            d[k] *= s
        d["size"] = size if size is not None else self.size * s
        return KParams(**d)


# --------------------------------------------------------------------------- geometry
def _isect(p, d, q, e):
    """Intersection of lines p + t*d and q + u*e."""
    den = d[0] * e[1] - d[1] * e[0]
    t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return (p[0] + t * d[0], p[1] + t * d[1])


def _y_at_x(p, d, x):
    t = (x - p[0]) / d[0]
    return p[1] + t * d[1]


def _x_at_y(p, d, y):
    t = (y - p[1]) / d[1]
    return p[0] + t * d[0]


def points(k: KParams) -> dict:
    """Named construction points."""
    a = math.radians(k.arm_angle)
    l = math.radians(k.leg_angle)
    da = (math.cos(a), -math.sin(a))          # arm direction (up-right)
    na = (math.sin(a), math.cos(a))           # arm's outer edge lies down-right of its inner edge
    dl = (math.cos(l), math.sin(l))           # leg direction (down-right)
    nl = (math.sin(l), -math.cos(l))          # leg's outer edge lies up-right of its inner edge
    J = (k.x1, k.yJ)
    C = (k.x1, k.yC)
    armO = (J[0] + na[0] * k.arm_w, J[1] + na[1] * k.arm_w)   # point on arm outer edge
    legO = (C[0] + nl[0] * k.leg_w, C[1] + nl[1] * k.leg_w)   # point on leg outer edge
    P = _isect(armO, da, legO, dl)
    pts = {
        "TL": (k.x0, k.yT), "TR": (k.x1, k.yT), "J": J,
        "armTL": (_x_at_y(J, da, k.yT), k.yT), "armTR": (_x_at_y(armO, da, k.yT), k.yT),
        "P": P, "legBR": (_x_at_y(legO, dl, k.yB), k.yB), "legBL": (_x_at_y(C, dl, k.yB), k.yB),
        "C": C, "BR": (k.x1, k.yB), "BL": (k.x0, k.yB),
    }
    fold = (P[0] - J[0], P[1] - J[1])
    pts["foldL"] = (k.x0, _y_at_x(J, fold, k.x0))      # red/yellow split on stem's left edge
    pts["innerL"] = (k.x0, _y_at_x(J, da, k.x0))       # yellow/green split on stem's left edge
    return pts


def facets(k: KParams) -> dict:
    p = points(k)
    return {
        "red": [p["TL"], p["TR"], p["J"], p["foldL"]],
        "yellow": [p["foldL"], p["J"], p["innerL"]],
        "blue": [p["armTL"], p["armTR"], p["P"], p["J"]],
        "green": [p["innerL"], p["J"], p["P"], p["legBR"], p["legBL"], p["C"], p["BR"], p["BL"]],
    }


def silhouette(k: KParams) -> list:
    p = points(k)
    return [p[n] for n in ("TL", "TR", "J", "armTL", "armTR", "P", "legBR", "legBL", "C", "BR", "BL")]


def bbox(k: KParams):
    s = silhouette(k)
    xs = [q[0] for q in s]
    ys = [q[1] for q in s]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_area_centroid(poly):
    A = cx = cy = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        c = x0 * y1 - x1 * y0
        A += c
        cx += (x0 + x1) * c
        cy += (y0 + y1) * c
    A *= 0.5
    return A, (cx / (6 * A), cy / (6 * A))


# --------------------------------------------------------------------------- svg paths
def fmt(v):
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def poly_d(poly):
    return "M" + " L".join(f"{fmt(x)} {fmt(y)}" for x, y in poly) + " Z"


def _is_convex(prev, cur, nxt, orient):
    cr = (cur[0] - prev[0]) * (nxt[1] - cur[1]) - (cur[1] - prev[1]) * (nxt[0] - cur[0])
    return (cr * orient) > 0


def rounded_d(poly, r_convex=0.0, r_concave=0.0):
    """Polygon path with circular fillets. Radii are clamped so fillets never overlap."""
    if r_convex <= 0 and r_concave <= 0:
        return poly_d(poly)
    n = len(poly)
    area, _ = polygon_area_centroid(poly)
    orient = 1 if area > 0 else -1
    segs = []
    for i in range(n):
        prev, cur, nxt = poly[i - 1], poly[i], poly[(i + 1) % n]
        convex = _is_convex(prev, cur, nxt, orient)
        r = r_convex if convex else r_concave
        v1 = (prev[0] - cur[0], prev[1] - cur[1])
        v2 = (nxt[0] - cur[0], nxt[1] - cur[1])
        l1 = math.hypot(*v1)
        l2 = math.hypot(*v2)
        u1 = (v1[0] / l1, v1[1] / l1)
        u2 = (v2[0] / l2, v2[1] / l2)
        cosang = max(-1.0, min(1.0, u1[0] * u2[0] + u1[1] * u2[1]))
        ang = math.acos(cosang)
        if r <= 0 or ang < 1e-6 or abs(math.pi - ang) < 1e-6:
            segs.append((cur, cur, 0, 0))
            continue
        t = r / math.tan(ang / 2)
        tmax = 0.45 * min(l1, l2)
        if t > tmax:
            t = tmax
            r = t * math.tan(ang / 2)
        a = (cur[0] + u1[0] * t, cur[1] + u1[1] * t)
        b = (cur[0] + u2[0] * t, cur[1] + u2[1] * t)
        cr = u1[0] * u2[1] - u1[1] * u2[0]
        sweep = 1 if cr < 0 else 0
        segs.append((a, b, r, sweep))
    out = []
    for i, (a, b, r, sweep) in enumerate(segs):
        out.append(("M" if i == 0 else "L") + f"{fmt(a[0])} {fmt(a[1])}")
        if r > 0:
            out.append(f"A{fmt(r)} {fmt(r)} 0 0 {sweep} {fmt(b[0])} {fmt(b[1])}")
    return " ".join(out) + " Z"


def silhouette_d(k: KParams):
    return rounded_d(silhouette(k), k.r_convex, k.r_concave)


# --------------------------------------------------------------------------- raster-safe facets
PAINT_ORDER = ("green", "yellow", "blue", "red")


def trapped_facets(k: KParams, t: float = None):
    """Facets for raster output, in paint order. Each facet is grown by ``t`` only *under*
    the facets painted after it (classic trapping), so anti-aliased shared edges never
    show a background-coloured seam while the outer silhouette stays exact."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    if t is None:
        t = k.size / 512.0
    f = {n: Polygon(p) for n, p in facets(k).items()}
    out = []
    for i, n in enumerate(PAINT_ORDER):
        allowed = unary_union([f[m] for m in PAINT_ORDER[i:]])
        g = f[n].buffer(t, join_style=2, mitre_limit=4.0).intersection(allowed)
        polys = [g] if g.geom_type == "Polygon" else [p for p in g.geoms if p.geom_type == "Polygon"]
        d = " ".join(poly_d(list(p.exterior.coords)[:-1]) for p in polys if p.area > 1e-6)
        out.append((n, d))
    return out


def flat_group(k: KParams, palette=None, trap=True, gid="K"):
    """SVG <g> of the flat mark. ``trap=False`` emits the exact, editable facet polygons."""
    pal = palette or k.palette
    if trap:
        body = "".join(f'<path id="{gid}-{n}" fill="{pal[n]}" d="{d}"/>' for n, d in trapped_facets(k))
    else:
        body = "".join(f'<path id="{gid}-{n}" fill="{pal[n]}" d="{poly_d(facets(k)[n])}"/>' for n in FACETS)
    if k.r_convex > 0 or k.r_concave > 0:
        # rounded corners: facets are clipped by the filleted silhouette
        return (f'<g id="{gid}"><clipPath id="{gid}-clip"><path d="{silhouette_d(k)}"/></clipPath>'
                f'<g clip-path="url(#{gid}-clip)">{body}</g></g>')
    return f'<g id="{gid}">{body}</g>'


def svg_doc(body: str, defs: str = "", size: float = 1024, title: str = None) -> str:
    t = f"<title>{title}</title>" if title else ""
    d = f"<defs>{defs}</defs>" if defs else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'viewBox="0 0 {fmt(size)} {fmt(size)}" width="{fmt(size)}" height="{fmt(size)}">{t}{d}{body}</svg>')
