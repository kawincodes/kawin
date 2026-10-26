"""Brand guidelines: an 11-slide HTML document and its vector PDF, regenerated from brand.py.

    python3 brand/tools/build_guidelines.py              # build, then verify and print the numbers
    python3 brand/tools/build_guidelines.py --no-verify  # build only

Outputs (under brand/guidelines/):
  K-Logo_Brand_Guidelines.html   self-contained: inline SVG artwork, CSS and the fonts (subset,
                                 base64) - it renders offline, on machines without Inter too
  K-Logo_Brand_Guidelines.pdf    the same HTML printed by Playwright Chromium (page.pdf), one
                                 1920 x 1080 px slide per page (1440 x 810 pt, exactly 16:9)

Production decisions
  * Artwork     - every logo on every page is an inline SVG document from brand.py (doc_mark,
                  doc_badge, doc_tech, doc_android_*, mark_group) or a drawing built from
                  kmark.points(brand.master()). Nothing is a screenshot, so the PDF stays vector:
                  the verification counts 0 image XObjects. Flat marks use raster=True (trapped
                  facets): a PDF is rasterised by its viewer, and trapped facets show no seams.
  * Ids         - several brand documents share one HTML page and url(#id) resolves page-wide,
                  so inline() prefixes every id of every document (tech-plate, dim-red-fill ...).
  * Effects     - Chromium rasterises CSS / SVG filters in PDF output, so the one shadow in the
                  document (the misuse example) is drawn as stacked translucent vector outlines.
  * Favicons    - the zoomed and 1x favicons are build_raster's grid-fitted pixels (the files the
                  kit ships), drawn as vector squares: one path per colour, so no seams.
  * Fonts       - Inter 400 / 500 / 600 and JetBrains Mono 400, subset to the characters the
                  document uses. Inter ships as CFF (.otf), which Chromium's PDF writer embeds as
                  Type 3; the subsets are converted to TrueType outlines (fontTools cu2qu) so the
                  PDF embeds real TrueType (CID) fonts. Labels in drawings clear the guides behind
                  them with measured backing rects, not stroked halos: Chromium embeds stroked
                  text as Type 3 too.
  * Colour      - CMYK via vector_export.srgb_to_cmyk (an ICC transform, LittleCMS sRGB ->
                  Ghostscript's default_cmyk.icc, relative colorimetric + black-point
                  compensation); if that module cannot be imported the same transform is built
                  here with Pillow ImageCms. The profile name printed on page 7 is read from it.
  * Numbers     - every number on pages 3, 4 and 7 is computed here from brand.master(),
                  trace_params.json and the colour maths; verify() recomputes them on its own and
                  finds each one in the text of the right PDF page.
"""
from __future__ import annotations

import base64
import glob
import html
import io
import itertools
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from shapely.geometry import Polygon  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

import brand  # noqa: E402
import kmark  # noqa: E402
from brand import NEUTRAL, PALETTE, PALETTE_OKLCH  # noqa: E402
from kmark import FACETS, KParams, fmt, points, poly_d, silhouette  # noqa: E402
from okcolor import contrast, hex_to_oklch, hex_to_rgb, luminance, oklch_to_hex, tone  # noqa: E402

OUT_DIR = "guidelines"
BASENAME = "K-Logo_Brand_Guidelines"
TITLE = "K — Brand Mark Guidelines"
VERSION = "1.0"
DATE = "2026-10-25"
TRACE_SVG = os.path.join(ROOT, "source", "K-Logo_trace.svg")
TRACE_PARAMS = os.path.join(HERE, "trace_params.json")

W, H = 1920, 1080               # slide in CSS px; 1440 x 810 pt in the PDF (Chromium rounds page
MX = 120                        # sizes to 1/300 in, and 1080 px is a whole number of those)
PAGES = 11

# Google's brand palette: only for the misuse page and the "distance from Google" numbers
GOOGLE = {"red": "#EA4335", "yellow": "#FBBC05", "green": "#34A853", "blue": "#4285F4"}

# ------------------------------------------------------------------ document colours
# Everything but the brand colours is a tone of ink, moved in OKLCH so it keeps ink's cool hue.
INK, NIGHT, WHITE = NEUTRAL["ink"], NEUTRAL["night"], NEUTRAL["white"]
_INK_HUE = hex_to_oklch(INK)[2]


def _grey(L: float, C: float = 0.006) -> str:
    return oklch_to_hex(L, C, _INK_HUE)


TEXT2 = tone(INK, dL=0.25)       # secondary text, 6.9:1 on white
TEXT3 = tone(INK, dL=0.30)       # captions and footers, 5.5:1 on white
RULE = _grey(0.905)              # hairlines
RULE2 = _grey(0.80)              # construction guides
SURFACE = _grey(0.972, 0.004)    # artwork tiles on white slides
SURFACE2 = _grey(0.940, 0.006)   # a second, deeper light neutral
CHECK_RED, CHECK_GREEN = PALETTE["red"], PALETTE["green"]

FONTS = [  # (family, weight, file)
    ("Inter", 400, "/usr/share/fonts/opentype/inter/Inter-Regular.otf"),
    ("Inter", 500, "/usr/share/fonts/opentype/inter/Inter-Medium.otf"),
    ("Inter", 600, "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"),
    ("JetBrains Mono", 400, "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf"),
]


# ------------------------------------------------------------------ number formatting
def f2(v: float, nd: int = 2) -> str:
    """Fixed decimals with a true minus sign."""
    return f"{v:.{nd}f}".replace("-", "−")


def sg(v: float, nd: int = 2) -> str:
    """Signed fixed decimals: +14.00 / −6.00. A value that rounds to zero prints +0.00, never
    the negative zero (−0.000) that a tiny negative difference would otherwise show."""
    v = round(v, nd) + 0.0                                  # + 0.0 turns -0.0 into 0.0
    return f"{v:+.{nd}f}".replace("-", "−")


def esc(s: str) -> str:
    return html.escape(s, quote=True)


# ------------------------------------------------------------------ inline SVG
_uid = itertools.count(1)


def inline(svg: str, w: float = None, h: float = None, cls: str = None, style: str = None) -> str:
    """A complete SVG document as an inline <svg> of w x h CSS px (one given: aspect kept).
    Every id gets a per-document prefix, so gradients / clips of different documents never
    resolve to each other's definitions."""
    pre = f"s{next(_uid)}-"
    svg = re.sub(r"<title>.*?</title>", "", svg, flags=re.S)
    svg = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{pre}{m.group(1)}"', svg)
    svg = re.sub(r"url\(#([^)]+)\)", lambda m: f"url(#{pre}{m.group(1)})", svg)
    svg = re.sub(r'href="#([^"]+)"', lambda m: f'href="#{pre}{m.group(1)}"', svg)
    head = re.match(r"<svg\b[^>]*>", svg).group(0)
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', head).group(1).replace(",", " ").split()]
    if w is None and h is None:
        w, h = vb[2], vb[3]
    elif w is None:
        w = h * vb[2] / vb[3]
    elif h is None:
        h = w * vb[3] / vb[2]
    new = re.sub(r'\s(?:width|height|xmlns:xlink)="[^"]*"', "", head)
    extra = f' width="{fmt(w)}" height="{fmt(h)}" aria-hidden="true"'
    extra += f' class="{cls}"' if cls else ""
    extra += f' style="{style}"' if style else ""
    return new[:4] + extra + new[4:] + svg[len(head):]


def doc(body: str, vb, defs: str = "") -> str:
    """A bare SVG document around body, viewBox vb = (x, y, w, h)."""
    d = f"<defs>{defs}</defs>" if defs else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{" ".join(fmt(v) for v in vb)}" '
            f'width="{fmt(vb[2])}" height="{fmt(vb[3])}">{d}{body}</svg>')


def tight_vb(k: KParams, margin: float = 0.0):
    x0, y0, x1, y1 = kmark.bbox(k)
    return (x0 - margin, y0 - margin, x1 - x0 + 2 * margin, y1 - y0 + 2 * margin)


def mark(k: KParams = None, colours=None, gaps=False) -> str:
    """The flat mark as a <g>, trapped (seam-free in any PDF viewer) unless gaps are asked for."""
    k = k or brand.master()
    return brand.mark_group(k, colours=colours, raster=not gaps, gaps=gaps)


_METRICS = {}


def text_width(s: str, size: float, weight: int = 400) -> float:
    """Advance width of s in Inter at size px (no kerning: callers pad)."""
    from fontTools.ttLib import TTFont
    path = next(p for fam, wt, p in FONTS if fam == "Inter" and wt == weight)
    if path not in _METRICS:
        f = TTFont(path)
        _METRICS[path] = (f.getBestCmap(), f["hmtx"].metrics, f["head"].unitsPerEm)
    cmap, hm, upm = _METRICS[path]
    return sum(hm[cmap.get(ord(c), ".notdef")][0] for c in s) * size / upm


def verdict(ok: bool, size: int = 30) -> str:
    """Round check / cross badge: green check for correct use, red cross for misuse."""
    c, s = size / 2, size
    if ok:
        glyph = (f'<path d="M{fmt(s * .29)} {fmt(s * .52)} L{fmt(s * .44)} {fmt(s * .66)} L{fmt(s * .72)} {fmt(s * .36)}" '
                 f'fill="none" stroke="#FFFFFF" stroke-width="{fmt(s * .095)}" stroke-linecap="round" stroke-linejoin="round"/>')
    else:
        a, b = s * .34, s * .66
        glyph = (f'<path d="M{fmt(a)} {fmt(a)} L{fmt(b)} {fmt(b)} M{fmt(b)} {fmt(a)} L{fmt(a)} {fmt(b)}" fill="none" '
                 f'stroke="#FFFFFF" stroke-width="{fmt(s * .095)}" stroke-linecap="round"/>')
    label = "correct" if ok else "incorrect"
    return (f'<svg class="verdict" width="{s}" height="{s}" viewBox="0 0 {s} {s}" role="img" aria-label="{label}">'
            f'<circle cx="{fmt(c)}" cy="{fmt(c)}" r="{fmt(c)}" fill="{CHECK_GREEN if ok else CHECK_RED}"/>{glyph}</svg>')


class Draw:
    """An annotated drawing in design units shown at a known screen scale: px(n) converts screen
    px to units, so hairlines and labels keep their screen size whatever the drawing's scale."""

    def __init__(self, vb, width: float = None, height: float = None):
        self.vb = vb
        self.s = width / vb[2] if width else height / vb[3]
        self.items, self.defs = [], []

    def px(self, n: float) -> float:
        return n / self.s

    def add(self, *items):
        self.items.extend(items)
        return self

    def line(self, a, b, col=INK, w=1.0, dash=None, op=None):
        extra = f' stroke-dasharray="{fmt(self.px(dash[0]))} {fmt(self.px(dash[1]))}"' if dash else ""
        extra += f' stroke-opacity="{fmt(op)}"' if op is not None else ""
        return self.add(f'<path d="M{fmt(a[0])} {fmt(a[1])} L{fmt(b[0])} {fmt(b[1])}" fill="none" stroke="{col}" '
                        f'stroke-width="{fmt(self.px(w))}"{extra}/>')

    def text(self, p, s, size=15, weight=500, col=INK, anchor="start", rotate=None, halo="#FFFFFF"):
        """Label at baseline point p. halo: a backing rect (measured from the font's advance
        widths) that clears guides behind the label - a stroked 'paint-order' halo would make
        Chromium embed the font as Type 3."""
        fs = self.px(size)
        back = ""
        if halo:
            w = self.px(text_width(s, size, weight))
            x = p[0] - {"start": 0, "middle": w / 2, "end": w}[anchor]
            pad = self.px(3)
            back = (f'<rect x="{fmt(x - pad)}" y="{fmt(p[1] - 0.76 * fs - pad)}" width="{fmt(w + 2 * pad)}" '
                    f'height="{fmt(0.98 * fs + 2 * pad)}" fill="{halo}"/>')
        t = (f'<text x="{fmt(p[0])}" y="{fmt(p[1])}" font-family="Inter" font-size="{fmt(fs)}" font-weight="{weight}" '
             f'fill="{col}" text-anchor="{anchor}" style="font-variant-numeric:tabular-nums">{esc(s)}</text>')
        if rotate:
            return self.add(f'<g transform="rotate({fmt(rotate)} {fmt(p[0])} {fmt(p[1])})">{back}{t}</g>')
        return self.add(back + t)

    def arrows(self, a, b, col=INK, w=1.1, size=6, run=16):
        """A dimension too short for ticks: arrowheads outside a and b pointing in, with tails."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy)
        ux, uy = dx / n, dy / n
        s, r = self.px(size), self.px(run)
        for tip, sgn_ in ((a, -1), (b, 1)):
            ox, oy = ux * sgn_, uy * sgn_                       # pointing away from the gap
            self.line(tip, (tip[0] + ox * r, tip[1] + oy * r), col, w)
            base = (tip[0] + ox * s, tip[1] + oy * s)
            self.add(f'<path d="M{fmt(tip[0])} {fmt(tip[1])} L{fmt(base[0] - oy * s * .45)} {fmt(base[1] + ox * s * .45)} '
                     f'L{fmt(base[0] + oy * s * .45)} {fmt(base[1] - ox * s * .45)} Z" fill="{col}"/>')
        return self

    def dot(self, p, r=4.5, fill="#FFFFFF", col=INK, w=1.6):
        return self.add(f'<circle cx="{fmt(p[0])}" cy="{fmt(p[1])}" r="{fmt(self.px(r))}" fill="{fill}" '
                        f'stroke="{col}" stroke-width="{fmt(self.px(w))}"/>')

    def dim(self, a, b, col=INK, w=1.1, tick=7):
        """Dimension line a-b with short perpendicular end ticks."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy)
        nx, ny = -dy / n * self.px(tick), dx / n * self.px(tick)
        self.line(a, b, col, w)
        for p in (a, b):
            self.line((p[0] - nx, p[1] - ny), (p[0] + nx, p[1] + ny), col, w)
        return self

    def arc(self, c, r, a0, a1, col=INK, w=1.1):
        """Arc around c from angle a0 to a1 (degrees, SVG orientation: y down, clockwise +)."""
        p0 = (c[0] + r * math.cos(math.radians(a0)), c[1] + r * math.sin(math.radians(a0)))
        p1 = (c[0] + r * math.cos(math.radians(a1)), c[1] + r * math.sin(math.radians(a1)))
        return self.add(f'<path d="M{fmt(p0[0])} {fmt(p0[1])} A{fmt(r)} {fmt(r)} 0 0 {1 if a1 > a0 else 0} '
                        f'{fmt(p1[0])} {fmt(p1[1])}" fill="none" stroke="{col}" stroke-width="{fmt(self.px(w))}"/>')

    def svg(self) -> str:
        return doc("".join(self.items), self.vb, "".join(self.defs))

    def html(self, cls=None) -> str:
        return inline(self.svg(), self.vb[2] * self.s, self.vb[3] * self.s, cls=cls)


# ------------------------------------------------------------------ measurements
def _geometry(k: KParams) -> dict:
    """The construction numbers of one K (master or trace), in units of its 1024 grid."""
    p = points(k)
    x0, y0, x1, y1 = kmark.bbox(k)
    return {
        "w": k.w, "arm_w": k.arm_w, "leg_w": k.leg_w,
        "arm_pct": 100 * k.arm_w / k.w, "leg_pct": 100 * k.leg_w / k.w,
        "arm_angle": k.arm_angle, "leg_angle": k.leg_angle,
        "arm_x": p["armTR"][0], "leg_x": p["legBR"][0],
        "above": p["P"][1] - p["J"][1], "below": p["C"][1] - p["P"][1],
        "cap": y1 - y0, "cap_pct": 100 * (y1 - y0) / k.size, "width": x1 - x0,
        "dx": (x0 + x1) / 2 - k.size / 2, "dy": (y0 + y1) / 2 - k.size / 2,
        "fold": math.degrees(math.atan2(p["P"][1] - p["J"][1], p["P"][0] - p["J"][0])),
        "J": p["J"], "P": p["P"], "C": p["C"], "r": max(k.r_convex, k.r_concave),
    }


def trace_k() -> KParams:
    with open(TRACE_PARAMS) as f:
        return KParams.from_dict(json.load(f))


def measure() -> dict:
    k, t = brand.master(), trace_k()
    m, tr = _geometry(k), _geometry(t)
    colours = {}
    for n in FACETS:
        a, b, g = hex_to_oklch(t.palette[n]), PALETTE_OKLCH[n], hex_to_oklch(GOOGLE[n])
        colours[n] = {"trace": t.palette[n], "master": PALETTE[n], "dL": b[0] - a[0], "dC": b[1] - a[1],
                      "dh": b[2] - a[2], "g_trace": a[2] - g[2], "g_master": b[2] - g[2]}
    return {"k": k, "t": t, "m": m, "tr": tr, "colours": colours, "trace_disc": t.palette["disc"],
            "disc": k.palette["disc"]}


# ------------------------------------------------------------------ colour: CMYK + contrast
def _icc_fallback():
    """Pillow ImageCms: sRGB -> Ghostscript's default CMYK profile, relative colorimetric + BPC."""
    from PIL import Image, ImageCms
    cands = sorted(glob.glob("/usr/share/ghostscript/**/default_cmyk.icc", recursive=True)
                   + glob.glob("/usr/share/color/icc/ghostscript/default_cmyk.icc"))
    prof = ImageCms.getOpenProfile(cands[0])
    srgb, lab = ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB")
    kw = {"renderingIntent": ImageCms.Intent.RELATIVE_COLORIMETRIC, "flags": ImageCms.Flags.BLACKPOINTCOMPENSATION}
    fwd = ImageCms.buildTransform(srgb, prof, "RGB", "CMYK", **kw)
    rgb_lab = ImageCms.buildTransform(srgb, lab, "RGB", "LAB", renderingIntent=kw["renderingIntent"])
    cmyk_lab = ImageCms.buildTransform(prof, lab, "CMYK", "LAB", **kw)

    def labpx(p):
        return (p[0] * 100 / 255, p[1] - 128, p[2] - 128)

    def to_cmyk(h):
        px = ImageCms.applyTransform(Image.new("RGB", (1, 1), hex_to_rgb(h)), fwd).getpixel((0, 0))
        return tuple(round(v / 255 * 100) / 100 for v in px)

    def de(h, c):
        a = labpx(ImageCms.applyTransform(Image.new("RGB", (1, 1), hex_to_rgb(h)), rgb_lab).getpixel((0, 0)))
        b = labpx(ImageCms.applyTransform(Image.new("CMYK", (1, 1), tuple(round(v * 255) for v in c)), cmyk_lab).getpixel((0, 0)))
        return math.dist(a, b)

    return to_cmyk, de, ImageCms.getProfileDescription(prof).strip(), cands[0], "Pillow ImageCms (fallback)"


def cmyk_converter():
    """(to_cmyk, delta_e, profile name, profile path, source). Prefers vector_export's transform."""
    try:
        import vector_export as ve
        ve.srgb_to_cmyk("#808080")
        return ve.srgb_to_cmyk, ve.delta_e76, ve.icc_profile_name(), ve.CMYK_PROFILE, "vector_export.srgb_to_cmyk"
    except Exception as e:  # pragma: no cover - only when the other engineer's module is broken
        print(f"  vector_export unavailable ({type(e).__name__}: {e}); using the ImageCms fallback")
        return _icc_fallback()


def colour_specs() -> dict:
    to_cmyk, de, name, path, source = cmyk_converter()
    rows = []
    for n, h in list(PALETTE.items()) + [("ink", INK), ("night", NIGHT), ("white", WHITE)]:
        L, C, hh = PALETTE_OKLCH[n] if n in PALETTE_OKLCH else hex_to_oklch(h)
        c = to_cmyk(h)
        rows.append({"name": n, "hex": h, "rgb": hex_to_rgb(h), "oklch": (L, C, hh),
                     "cmyk": tuple(round(v * 100) for v in c), "de": de(h, c),
                     "on_white": contrast(h, WHITE), "on_night": contrast(h, NIGHT)})
    return {"rows": rows, "profile": name, "profile_file": os.path.basename(path), "source": source}


def oklch_str(L, C, h) -> str:
    """CSS oklch() order: lightness %, chroma, hue (none for a neutral without chroma)."""
    return f"{L * 100:.1f}% {C:.3f} " + (f"{h:.1f}" if C >= 0.0005 else "none")


def cmyk_str(c) -> str:
    return " ".join(f"{ch}{v}" for ch, v in zip("CMYK", c))


def usage_on_white(r: float) -> str:
    return "text on white" if r >= 4.5 else "large text, graphics" if r >= 3 else "inside the mark only"


def gamut_note(de: float) -> str:
    return "in gamut" if de < 3 else "just outside the gamut" if de < 15 else "outside the CMYK gamut"


# ------------------------------------------------------------------ favicon pixels
def favicon_pixels(n: int):
    """The kit's grid-fitted favicon as an n x n RGBA pixel array (build_raster.favicon); the same
    hinting is rebuilt here if that module cannot be imported."""
    try:
        import build_raster
        im = build_raster.favicon(n)
    except Exception as e:  # pragma: no cover
        print(f"  build_raster unavailable ({type(e).__name__}); grid-fitting the favicon here")
        k = brand.master()
        Hc = k.yB - k.yT
        x0, y0, x1, y1 = kmark.bbox(k)
        cap = round(n * 13 / 16)
        stem = round(cap * k.w / Hc)
        left = round((n - cap * (x1 - x0) / Hc) / 2 + ((x0 + x1) / 2 - k.size / 2) / k.size * n)
        top = round((n - cap) / 2 + ((y0 + y1) / 2 - k.size / 2) / k.size * n)
        u = k.size / n
        s = cap * u / Hc
        k = brand.transform(k, s, 0, 0, left * u - k.x0 * s, top * u - k.yT * s)
        f = stem * u / k.w
        k.w, k.arm_w, k.leg_w = k.w * f, k.arm_w * f, k.leg_w * f
        f = math.ceil(1024 / n)
        im = brand.render(brand.doc_mark(k, raster=True), n * f).convert("RGBa").reduce(f).convert("RGBA")
    return im


def pixel_art(im, scale: float, bg: str = None, grid: str = None) -> str:
    """Pixels as vector squares: one path per RGBA value (no seams between same-colour pixels),
    optional background and pixel grid. Returns an inline <svg> of n*scale px."""
    n = im.size[0]
    by = {}
    for y in range(n):
        for x in range(n):
            r, g, b, a = im.getpixel((x, y))
            if a:
                by.setdefault((r, g, b, a), []).append(f"M{x} {y}h1v1h-1z")
    body = [f'<rect width="{n}" height="{n}" fill="{bg}"/>'] if bg else []
    for (r, g, b, a), d in sorted(by.items(), key=lambda kv: kv[0][3]):
        op = "" if a == 255 else f' fill-opacity="{fmt(a / 255)}"'
        body.append(f'<path fill="#{r:02X}{g:02X}{b:02X}"{op} d="{"".join(d)}"/>')
    if grid:
        hw = 0.5 / scale
        lines = "".join(f"M{i} 0V{n}M0 {i}H{n}" for i in range(1, n))
        body.append(f'<path d="{lines}" stroke="{grid}" stroke-width="{fmt(2 * hw)}" fill="none"/>')
    return inline(doc("".join(body), (0, 0, n, n)), n * scale, n * scale)


# ------------------------------------------------------------------ page scaffolding
def slide(no: int, eyebrow: str, title: str, lead: str, body: str, top: int = 330, cls: str = "") -> str:
    lead_html = f'<p class="lead">{lead}</p>' if lead else ""
    return (f'<section class="slide {cls}" id="p{no}">'
            f'<header class="head"><div class="eyebrow"><span class="no">{no:02d}</span><span>{esc(eyebrow)}</span></div>'
            f'<h1>{title}</h1>{lead_html}</header>'
            f'<div class="body" style="top:{top}px">{body}</div>'
            f'<footer class="foot"><span>{esc(TITLE)}</span><span>Version {VERSION} · {DATE}</span>'
            f'<span class="pg">{no:02d} / {PAGES}</span></footer></section>')


def tile(art: str, bg: str = SURFACE, h: int = 226, border: bool = False, extra: str = "", style: str = "") -> str:
    b = f"border:1px solid {RULE};" if border else ""
    return f'<div class="tile" style="background:{bg};height:{h}px;{b}{style}">{art}{extra}</div>'


# ------------------------------------------------------------------ 1 cover
def page_cover() -> str:
    k = brand.master()
    art = inline(doc(mark(k), tight_vb(k)), h=560)
    return (f'<section class="slide cover" id="p1">'
            f'<div class="cover-panel">{art}</div>'
            f'<div class="cover-text"><div class="eyebrow"><span>Brand mark guidelines</span></div>'
            f'<h1 class="cover-title">K — Brand Mark<br>Guidelines</h1>'
            f'<dl class="cover-meta"><div><dt>Version</dt><dd>{VERSION}</dd></div>'
            f'<div><dt>Date</dt><dd>{DATE}</dd></div><div><dt>Contents</dt><dd>{PAGES} pages</dd></div></dl></div>'
            f'</section>')


# ------------------------------------------------------------------ 2 the idea
def page_idea(M) -> str:
    k = brand.master()
    p = points(k)
    J, P = p["J"], p["P"]
    vb = (120, 112, 720, 776)
    sil = poly_d(silhouette(k))

    def ext(a, b, t0, t1):
        """Points of the line a->b at parameters t0 and t1 (0 = a, 1 = b)."""
        return [(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t) for t in (t0, t1)]

    fold = ext(J, P, -1.42, 1.0)                      # P back through J and across the stem
    edge = ext(J, p["armTL"], -0.72, 1.32)            # the arm's inner edge, both ways
    panels = []
    # 1 - the letter
    d1 = Draw(vb, width=440)
    d1.add(f'<path d="{sil}" fill="{INK}"/>')
    # 2 - two lines through J
    d2 = Draw(vb, width=440)
    d2.add(f'<path d="{sil}" fill="{SURFACE2}"/>')
    d2.line(*fold, INK, 2.0).line(*edge, INK, 2.0)
    d2.dot(J, 6, fill=INK, col=INK).dot(P, 4.5)
    d2.text((J[0] - d2.px(22), J[1] + d2.px(6)), "J", 17, 600, anchor="end", halo=None)   # in the wedge, between the lines
    d2.text((P[0] + d2.px(12), P[1] + d2.px(20)), "P", 17, 600, halo=None)
    d2.text((fold[0][0] - d2.px(8), fold[0][1] - d2.px(12)), "fold line", 15, 500, TEXT2, halo=None)
    d2.text((edge[1][0] + d2.px(10), edge[1][1] + d2.px(5)), "arm edge", 15, 500, TEXT2, halo=None)
    # 3 - four colours meet at J; the label sits in the notch above J (between stem and arm)
    d3 = Draw(vb, width=440)
    d3.add(mark(k))
    lead_dir = (math.cos(math.radians(74)), -math.sin(math.radians(74)))
    tip = (J[0] + lead_dir[0] * d3.px(44), J[1] + lead_dir[1] * d3.px(44))
    d3.line((J[0] + lead_dir[0] * d3.px(7), J[1] + lead_dir[1] * d3.px(7)), tip, INK, 1.2)
    d3.dot(J, 6, fill="#FFFFFF", col=INK, w=2)
    d3.text((tip[0] - d3.px(4), tip[1] - d3.px(7)), "J", 17, 600, halo=None)
    caps = [("1", "The letter", "A stem, an arm and a leg. Straight edges, sharp corners."),
            ("2", "Two lines through J",
             f"The fold line J–P ({f2(M['m']['fold'])}° down) and the arm’s inner edge "
             f"({f2(M['m']['arm_angle'])}° up) run straight on across the stem."),
            ("3", "Four colours meet at J",
             "Red on top, yellow in the wedge, blue for the arm, green for the stem and leg.")]
    for d, (n, t, s) in zip((d1, d2, d3), caps):
        panels.append(f'<figure class="step">{tile(d.html(), SURFACE, 500)}'
                      f'<figcaption><span class="stepno">{n}</span><div><h3>{t}</h3><p>{s}</p></div></figcaption></figure>')
    arrow = ('<div class="arrow"><svg width="28" height="28" viewBox="0 0 28 28"><path d="M4 14H23M16 7l7 7-7 7" '
             f'fill="none" stroke="{TEXT3}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></div>')
    body = f'<div class="steps">{panels[0]}{arrow}{panels[1]}{arrow}{panels[2]}</div>'
    return slide(2, "The idea", "Two lines, one point, four colours",
                 "Two straight lines through one point — the fold line and the arm’s edge — cut the K into "
                 "four colours that meet at the junction J. Nothing is added to the letter; the colours are its own construction.",
                 body, top=318)


# ------------------------------------------------------------------ 3 construction
def page_construction(M) -> str:
    k = brand.master()
    p = points(k)
    m = M["m"]
    J, P, C = p["J"], p["P"], p["C"]
    x0, y0, x1, y1 = kmark.bbox(k)
    a = math.radians(k.arm_angle)
    da, na = (math.cos(a), -math.sin(a)), (math.sin(a), math.cos(a))       # arm direction / normal
    dl, nl = (math.cos(a), math.sin(a)), (math.sin(a), -math.cos(a))        # leg direction / normal
    d = Draw((-150, -36, 1300, 1092), height=900)
    G = RULE2
    # badge disc and canvas centre
    d.add(f'<circle cx="512" cy="512" r="512" fill="none" stroke="{G}" stroke-width="{fmt(d.px(1.1))}" '
          f'stroke-dasharray="{fmt(d.px(5))} {fmt(d.px(5))}"/>')
    d.line((512, 4), (512, 1020), G, 1, dash=(5, 5)).line((4, 512), (1020, 512), G, 1, dash=(5, 5))
    # the mark
    d.add(mark(k))
    # the two cutting lines, continued past the letter
    fl = (P[0] - J[0], P[1] - J[1])
    fold_end = (x0 - 70, p["foldL"][1] - 70 * fl[1] / fl[0])
    d.line(p["foldL"], fold_end, INK, 1.1, dash=(6, 4))
    edge_lo = (p["innerL"][0] - 52, p["innerL"][1] - 52 * da[1] / da[0])   # 52 units left of the stem
    d.line(p["innerL"], edge_lo, INK, 1.1, dash=(6, 4))
    d.text((fold_end[0] - d.px(8), fold_end[1] + d.px(5)), "fold line", 14, 500, TEXT2, anchor="end")
    d.text((edge_lo[0] - d.px(8), edge_lo[1] + d.px(5)), "arm edge", 14, 500, TEXT2, anchor="end")
    # stem width w (k.x0 .. k.x1 - not the bbox, whose right side is the arm's extreme)
    yd = y0 - 46
    d.line((k.x0, y0 - 6), (k.x0, yd - 14), INK, 0.8).line((k.x1, y0 - 6), (k.x1, yd - 14), INK, 0.8)
    d.dim((k.x0, yd), (k.x1, yd))
    d.text(((k.x0 + k.x1) / 2, yd - d.px(12)), f"w = {f2(m['w'])}", 15, 600, anchor="middle")
    # arm and leg width d (perpendicular), beyond the cap line and the baseline
    t = 128
    A = (p["armTL"][0] + da[0] * t, p["armTL"][1] + da[1] * t)
    B = (A[0] + na[0] * k.arm_w, A[1] + na[1] * k.arm_w)
    d.line(p["armTL"], (A[0] + da[0] * 14, A[1] + da[1] * 14), INK, 0.8, dash=(4, 3))
    tb = (B[0] - p["armTR"][0]) / da[0]
    d.line(p["armTR"], (p["armTR"][0] + da[0] * (tb + 20), p["armTR"][1] + da[1] * (tb + 20)), INK, 0.8, dash=(4, 3))
    d.dim(A, B)
    d.text((B[0] + d.px(24), B[1] - d.px(4)), f"d = {f2(m['arm_w'])}", 15, 600)
    d.text((B[0] + d.px(24), B[1] + d.px(15)), f"{f2(m['arm_pct'], 1)}% of w", 13, 500, TEXT2)
    A2 = (p["legBL"][0] + dl[0] * t, p["legBL"][1] + dl[1] * t)
    B2 = (A2[0] + nl[0] * k.leg_w, A2[1] + nl[1] * k.leg_w)
    d.line(p["legBL"], (A2[0] + dl[0] * 14, A2[1] + dl[1] * 14), INK, 0.8, dash=(4, 3))
    tb = (B2[0] - p["legBR"][0]) / dl[0]
    d.line(p["legBR"], (p["legBR"][0] + dl[0] * (tb + 14), p["legBR"][1] + dl[1] * (tb + 14)), INK, 0.8, dash=(4, 3))
    d.dim(A2, B2)
    d.text((B2[0] + d.px(14), B2[1] + d.px(6)), f"d = {f2(m['leg_w'])}", 15, 600)
    d.text((B2[0] + d.px(14), B2[1] + d.px(25)), f"{f2(m['leg_pct'], 1)}% of w", 13, 500, TEXT2)
    # the shared slope: arcs in the notch (arm) and the counter (leg)
    ang = k.arm_angle
    d.arc(p["armTL"], 64, 180 - ang, 180)
    d.text((p["armTL"][0] - 64 - d.px(8), p["armTL"][1] + d.px(36)), f"{f2(ang)}°", 14, 600, anchor="end")
    d.arc(p["legBL"], 64, 180, 180 + ang)
    d.text((p["legBL"][0] - 64 - d.px(8), p["legBL"][1] - d.px(24)), f"{f2(k.leg_angle)}°", 14, 600, anchor="end")
    # right extremes aligned
    xr = p["armTR"][0]
    d.line((xr, y0 - 2), (xr, y1 + 2), INK, 1.0, dash=(6, 4))
    d.line((xr, y0), (xr + 112, y0), INK, 0.8).line((xr, y1), (xr + 112, y1), INK, 0.8)
    d.text((xr + d.px(16), 650), f"arm and leg end at x {f2(m['arm_x'])}", 14, 500, TEXT2, anchor="middle", rotate=90)
    # J, P, C and the mirror about P
    xm = 664
    for q in (J, C):
        d.line(q, (xm + 12, q[1]), INK, 0.8, dash=(3, 3), op=0.75)
    d.line(P, (xm + 12, P[1]), INK, 0.8, dash=(3, 3), op=0.75)
    d.dim((xm, J[1]), (xm, P[1])).dim((xm, P[1]), (xm, C[1]))
    d.text((xm + d.px(12), (J[1] + P[1]) / 2 + d.px(5)), f2(m["above"]), 14, 600)
    d.text((xm + d.px(12), (P[1] + C[1]) / 2 + d.px(5)), f2(m["below"]), 14, 600)
    for q, lab, off in ((J, "J", (16, -26)), (P, "P", (12, -10)), (C, "C", (16, 44))):
        d.dot(q, 5)
        lp = (q[0] + d.px(off[0]), q[1] + d.px(off[1]))
        if lab != "P":
            d.line((q[0] + d.px(4), q[1] + (d.px(-4) if lab == "J" else d.px(4))),
                   (lp[0] - d.px(2), lp[1] + (d.px(4) if lab == "J" else d.px(-14))), INK, 0.9)
        d.text(lp, lab, 16, 600, halo=None)
    # cap height and badge diameter. The mark's horizontal centre line (optical centring, below)
    # crosses the cap-height label, so it is drawn first and the label's backing rect clears it.
    xc = xr + 96
    xb = xc + 60                                             # x of the vertical centring arrows
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    d.line((x1 + 8, cy), (xb + 12, cy), INK, 0.8, dash=(3, 3), op=0.75)
    d.dim((xc, y0), (xc, y1))
    d.text((xc + d.px(20), (y0 + y1) / 2), f"cap height {f2(m['cap'])} = {f2(m['cap_pct'], 1)}% of 1024", 15, 600,
           anchor="middle", rotate=90)
    xD = 1084
    d.line((512, 0), (xD + 14, 0), G, 1).line((512, 1024), (xD + 14, 1024), G, 1)
    d.dim((xD, 0), (xD, 1024))
    d.text((xD + d.px(20), 512), "badge diameter 1024", 15, 600, anchor="middle", rotate=90)
    # optical centring: the mark's centre against the canvas centre. The x label sits left of the
    # arrows, under the stem: on the right it would land on the leg's dashed width extension.
    yb = y1 + 62
    d.line((cx, y1 + 8), (cx, yb + 12), INK, 0.8)
    d.arrows((512, yb), (cx, yb))
    d.text((512 - d.px(26), yb + d.px(5)), f"mark centre {sg(m['dx'])} x", 14, 600, anchor="end")
    d.text((512 + d.px(8), 992), "canvas centre", 13, 500, TEXT3)
    d.arrows((xb, 512), (xb, cy))
    d.text((xb + d.px(12), cy - d.px(18)), f"{sg(m['dy'])} y", 14, 600)
    drawing = d.html()

    rows = [
        ("Stem width w", f"{f2(m['w'])}"),
        ("Arm and leg width d", f"{f2(m['arm_w'])} = {f2(m['arm_pct'], 1)}% of w"),
        ("Arm and leg slope", f"{f2(m['arm_angle'])}° = rise 4 : run 3"),
        ("Junction J", f"({f2(m['J'][0])}, {f2(m['J'][1])})"),
        ("Fold point P", f"({f2(m['P'][0])}, {f2(m['P'][1])})"),
        ("Crotch C", f"({f2(m['C'][0])}, {f2(m['C'][1])})"),
        ("J and C mirror about P", f"{f2(m['above'])} above, {f2(m['below'])} below"),
        ("Fold line J–P", f"{f2(m['fold'])}° below horizontal"),
        ("Right extremes", f"arm {f2(m['arm_x'])}, leg {f2(m['leg_x'])}"),
        ("Cap height", f"{f2(m['cap'])} = {f2(m['cap_pct'], 1)}% of the badge"),
        ("Optical centring", f"{sg(m['dx'])} x, {sg(m['dy'])} y from centre"),
        ("Corners", "all sharp, radius 0"),
    ]
    table = "".join(f'<tr><th>{esc(a)}</th><td>{esc(b)}</td></tr>' for a, b in rows)
    return (f'<section class="slide" id="p3">'
            f'<div class="cons-draw">{drawing}</div>'
            f'<div class="cons-side"><div class="eyebrow"><span class="no">03</span><span>Construction</span></div>'
            f'<h1>One stem, one slope,<br>one width</h1>'
            f'<p class="lead small">Measured on the 1024 grid of the badge (diameter 1024), in design units.</p>'
            f'<table class="spec">{table}</table></div>'
            f'<footer class="foot"><span>{esc(TITLE)}</span><span>Version {VERSION} · {DATE}</span>'
            f'<span class="pg">03 / {PAGES}</span></footer></section>')


# ------------------------------------------------------------------ 4 refinement
def trace_pieces():
    """The trace's facet polygons and fills, read from source/K-Logo_trace.svg."""
    with open(TRACE_SVG) as f:
        s = f.read()
    pieces, fills = {}, {}
    for pid, fill, d in re.findall(r'<path id="(\w+)" fill="(#[0-9A-Fa-f]{6})" d="([^"]+)"', s):
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", d)]
        pieces[pid] = Polygon(list(zip(nums[0::2], nums[1::2])))
        fills[pid] = fill
    disc = re.search(r'<circle id="disc"[^>]*fill="(#[0-9A-Fa-f]{6})"', s).group(1)
    return pieces, fills, disc


def page_refinement(M) -> str:
    k, m, tr = M["k"], M["m"], M["tr"]
    pieces, fills, disc = trace_pieces()
    before = "".join(f'<path fill="{fills[n]}" d="{d}"/>' for n, d in brand.trapped(k, pieces=pieces))
    before = inline(doc(f'<circle cx="512" cy="512" r="512" fill="{disc}"/>{before}', (0, 0, 1024, 1024)), 226, 226)
    after = inline(brand.doc_badge(k, raster=True), 226, 226)
    # overlay: master silhouette, trace outline on top
    ts = unary_union(list(pieces.values()))
    ov = Draw((0, 0, 1024, 1024), width=226)
    ov.add('<circle cx="512" cy="512" r="512" fill="#FFFFFF"/>')
    ov.add(f'<path d="{poly_d(silhouette(k))}" fill="{_grey(0.86)}"/>')
    ov.add(f'<path d="{brand.shape_d(ts)}" fill="none" stroke="{PALETTE["red"]}" stroke-width="{fmt(ov.px(1.4))}" '
           f'stroke-dasharray="{fmt(ov.px(4))} {fmt(ov.px(3))}"/>')
    ov.line((512, 470), (512, 554), INK, 0.8).line((470, 512), (554, 512), INK, 0.8)
    figs = [("Before", "Trace of the original", before), ("After", "The master", after),
            ("Overlay", "Trace (dashed) on the master", ov.html())]
    badges = "".join(f'<figure class="ba"><div class="ba-art">{a}</div><figcaption><b>{t}</b><span>{s}</span></figcaption></figure>'
                     for t, s, a in figs)

    def pct(v):
        return f"{f2(v, 1)}%"

    rows = [
        ("Arm angle", f"{f2(tr['arm_angle'])}°", f"{f2(m['arm_angle'])}°"),
        ("Leg angle", f"{f2(tr['leg_angle'])}°", f"{f2(m['leg_angle'])}°"),
        ("Stem width w", f2(tr["w"]), f2(m["w"])),
        ("Arm width", f"{f2(tr['arm_w'])} · {pct(tr['arm_pct'])} w", f"{f2(m['arm_w'])} · {pct(m['arm_pct'])} w"),
        ("Leg width", f"{f2(tr['leg_w'])} · {pct(tr['leg_pct'])} w", f"{f2(m['leg_w'])} · {pct(m['leg_pct'])} w"),
        ("Right extremes, arm / leg", f"{f2(tr['arm_x'])} / {f2(tr['leg_x'])}", f"{f2(m['arm_x'])} / {f2(m['leg_x'])}"),
        ("P to J / P to C", f"{f2(tr['above'])} / {f2(tr['below'])}", f"{f2(m['above'])} / {f2(m['below'])}"),
        ("Cap height", f"{f2(tr['cap'])} · {pct(tr['cap_pct'])}", f"{f2(m['cap'])} · {pct(m['cap_pct'])}"),
        ("Mark width", f2(tr["width"]), f2(m["width"])),
        ("Offset from centre, x / y", f"{sg(tr['dx'])} / {sg(tr['dy'])}", f"{sg(m['dx'])} / {sg(m['dy'])}"),
        ("Corners", "sharp" if tr["r"] == 0 else f2(tr["r"]), "sharp, kept" if m["r"] == 0 else f2(m["r"])),
        ("Disc", M["trace_disc"], M["disc"]),
    ]
    table = "".join(f"<tr><th>{esc(a)}</th><td>{esc(b)}</td><td>{esc(c)}</td></tr>" for a, b, c in rows)
    table = f'<table class="cmp"><thead><tr><th></th><th>Trace</th><th>Master</th></tr></thead><tbody>{table}</tbody></table>'
    cols = []
    for n in ("red", "yellow", "green", "blue"):
        c = M["colours"][n]
        cols.append(f'<div class="cchg"><div class="chips"><i style="background:{c["trace"]}"></i>'
                    f'<i style="background:{c["master"]}"></i></div>'
                    f'<div class="cname">{n.capitalize()}</div>'
                    f'<div class="mono">{c["trace"]} → {c["master"]}</div>'
                    f'<div class="cdelta">L {sg(c["dL"], 3)} · C {sg(c["dC"], 3)} · h {sg(c["dh"])}°</div>'
                    f'<div class="clabel">Hue from Google’s {n}</div>'
                    f'<div class="cdelta">{sg(c["g_trace"], 1)}° → {sg(c["g_master"], 1)}°</div></div>')
    body = (f'<div class="ref-left"><div class="ba-row">{badges}</div>'
            f'<h3 class="sub">Colour, trace → master (OKLCH)</h3><div class="cchg-row">{"".join(cols)}</div></div>'
            f'<div class="ref-right">{table}</div>')
    return slide(4, "Refinement", "From the original to the master",
                 "The same letter and the same footprint, made exact: one slope, one diagonal weight, aligned extremes, "
                 "a centred fold and colours of its own. Numbers in units of the 1024 grid.", body, top=318)


# ------------------------------------------------------------------ 5 clear space and minimum size
def page_clearspace(M) -> str:
    k = brand.master()
    x0, y0, x1, y1 = kmark.bbox(k)
    w = k.w
    vb = (x0 - w - 40, y0 - w - 40, x1 - x0 + 2 * w + 80, y1 - y0 + 2 * w + 80)
    d = Draw(vb, height=600)
    tint = _grey(0.955, 0.012)
    outer = (x0 - w, y0 - w, x1 - x0 + 2 * w, y1 - y0 + 2 * w)
    d.add(f'<path fill-rule="evenodd" fill="{tint}" d="M{fmt(outer[0])} {fmt(outer[1])}h{fmt(outer[2])}v{fmt(outer[3])}'
          f'h{fmt(-outer[2])}z M{fmt(x0)} {fmt(y0)}V{fmt(y1)}H{fmt(x1)}V{fmt(y0)}z"/>')
    d.add(f'<rect x="{fmt(outer[0])}" y="{fmt(outer[1])}" width="{fmt(outer[2])}" height="{fmt(outer[3])}" fill="none" '
          f'stroke="{INK}" stroke-width="{fmt(d.px(1.1))}" stroke-dasharray="{fmt(d.px(6))} {fmt(d.px(4))}"/>')
    d.add(f'<rect x="{fmt(x0)}" y="{fmt(y0)}" width="{fmt(x1 - x0)}" height="{fmt(y1 - y0)}" fill="none" '
          f'stroke="{RULE2}" stroke-width="{fmt(d.px(1))}"/>')
    d.add(mark(k))
    # one stem-width square on each side
    sq = [(x0, y0 - w), (x0 - w, y0), (x1 - w, y1), (x1, y1 - w)]
    for sx, sy in sq:
        d.add(f'<rect x="{fmt(sx)}" y="{fmt(sy)}" width="{fmt(w)}" height="{fmt(w)}" fill="none" stroke="{INK}" '
              f'stroke-width="{fmt(d.px(1.2))}"/>')
        d.text((sx + w / 2, sy + w / 2 + d.px(6)), "w", 17, 600, anchor="middle", halo=None)
    d.text((x0 + w + d.px(14), y0 - w / 2 + d.px(6)), f"w = stem width = {f2(w)}", 14, 500, TEXT2, halo=None)
    drawing = d.html()

    fav16 = favicon_pixels(16)
    fav_1x = pixel_art(fav16, 1)
    fav_4x = pixel_art(fav16, 4)
    badge24 = inline(brand.doc_badge(k, raster=True), 24, 24)
    mm_h = 8 / 25.4 * 96                                    # 8 mm in CSS px (96 px per inch)
    print_mark = inline(doc(mark(k), tight_vb(k)), h=mm_h)
    ink_g = {n: INK for n in FACETS}

    def one(gaps: bool, px_: int) -> str:
        """One-colour ink mark, px_ = mark height in CSS px."""
        return inline(doc(mark(k, ink_g, gaps=gaps), tight_vb(k)), h=px_)

    def samples(items):
        return "".join(f'<figure class="ms"><div class="ms-art">{a}</div><figcaption>{c}</figcaption></figure>'
                       for a, c in items)

    rows = [
        ("Screen", "Smallest digital use: the grid-fitted favicon at 16 px; the badge at 24 px.",
         samples([(fav_1x, "16 px favicon"), (fav_4x, "16 px, shown 4×"), (badge24, "24 px badge")])),
        ("Print", "Mark height at least 8 mm. Shown here at 8 mm for a print at 100%.",
         samples([(print_mark, "8 mm mark height")])),
        ("One colour", "Construction gaps from 32 px up. Below 32 px use the solid one-colour mark: the gaps would close.",
         samples([(one(True, 48), "48 px gapped"), (one(True, 32), "32 px gapped"),
                  (one(False, 24), "24 px solid"), (one(False, 16), "16 px solid")])),
    ]
    mins = "".join(f'<div class="minrow"><div class="mintext"><h3>{t}</h3><p>{s}</p></div>'
                   f'<div class="minsamples">{x}</div></div>' for t, s, x in rows)
    mins = f'<div class="mins">{mins}</div>'
    body = (f'<div class="cs-left">{tile(drawing, "#FFFFFF", 640, border=True)}</div>'
            f'<div class="cs-right"><h3 class="sub">Clear space</h3><p class="note">Keep one stem width, w, free on every '
            f'side of the mark’s bounds: no text, edges, images or other marks inside the tinted zone. On the badge, '
            f'the disc is the clear space.</p><h3 class="sub" style="margin-top:34px">Minimum sizes</h3>{mins}</div>')
    return slide(5, "Clear space and minimum size", "Room to breathe, sizes that hold",
                 "Clear space keeps the mark legible; minimum sizes keep its construction visible.", body, top=286)


# ------------------------------------------------------------------ 6 variants
def page_variants() -> str:
    k = brand.master()
    vb = tight_vb(k)
    hm = 168

    def m_(colours=None, gaps=False):
        return inline(doc(mark(k, colours, gaps), vb), h=hm)

    split = (f'<div class="split"><div style="background:#FFFFFF">{m_({n: NEUTRAL["black"] for n in FACETS})}</div>'
             f'<div style="background:{NIGHT}">{m_({n: WHITE for n in FACETS})}</div></div>')
    cards = [
        ("Primary, full colour", "The default, on white and light backgrounds.",
         tile(m_(), "#FFFFFF", border=True)),
        ("Badge", "Avatars, social profiles, photography and dark or busy backgrounds.",
         tile(inline(brand.doc_badge(k, raster=True), 200, 200), NIGHT)),
        ("Premium shaded", "Hero use on light backgrounds: covers, packaging, signage.",
         tile(inline(brand.doc_mark(k, style="dimensional", artboard="tight"), h=hm), "#FFFFFF", border=True)),
        ("Tech icon", "App and product icon. At 64 px and below use its small version.",
         tile(inline(brand.doc_tech(k), 200, 200), SURFACE)),
        ("One colour, black", "Single-ink print, engraving, forms. The gaps keep the construction.",
         tile(m_({n: NEUTRAL["black"] for n in FACETS}, gaps=True), "#FFFFFF", border=True)),
        ("One colour, white", "Reversed out of dark colour where full colour cannot print.",
         tile(m_({n: WHITE for n in FACETS}, gaps=True), NIGHT)),
        ("One colour, solid", "Stamps, embroidery, die-cuts, and any one-colour use below 32 px.",
         tile(split, "#FFFFFF", border=True, style="padding:0;overflow:hidden")),
        ("Greyscale", "Black-only print that can hold tints, e.g. newsprint.",
         tile(m_(brand.grey_colours()), "#FFFFFF", border=True)),
    ]
    body = '<div class="grid4">' + "".join(
        f'<figure class="card">{t}<figcaption><h3>{esc(a)}</h3><p>{esc(b)}</p></figcaption></figure>'
        for a, b, t in cards) + "</div>"
    count = {6: "six", 7: "seven", 8: "eight", 9: "nine"}[len(cards)]
    return slide(6, "Variants", f"One mark, {count} variants",
                 "Pick by medium and background. Each variant is the same geometry; none is redrawn.", body, top=286)


# ------------------------------------------------------------------ 7 colour
def page_colour(CS) -> str:
    rows = {r["name"]: r for r in CS["rows"]}

    def spec(r, brandc=True):
        L, C, h = r["oklch"]
        items = [("HEX", r["hex"]), ("RGB", ", ".join(str(v) for v in r["rgb"])),
                 ("OKLCH", oklch_str(L, C, h)), ("CMYK", cmyk_str(r["cmyk"])),
                 ("On white", f"{r['on_white']:.2f}:1"), ("On night", f"{r['on_night']:.2f}:1")]
        if brandc:
            items.append(("Print ΔE", f"{r['de']:.1f} · {gamut_note(r['de'])}"))
        return "".join(f"<tr><th>{a}</th><td>{esc(b)}</td></tr>" for a, b in items)

    names = {"red": "Red", "yellow": "Yellow", "green": "Green", "blue": "Blue"}
    roles = {"red": "the top", "yellow": "the wedge", "green": "stem and leg", "blue": "the arm"}
    cards = []
    for n in ("red", "yellow", "green", "blue"):
        r = rows[n]
        tag = usage_on_white(r["on_white"])
        cards.append(f'<div class="ccard"><div class="sw" style="background:{r["hex"]}"></div>'
                     f'<div class="cn"><h3>{names[n]}</h3><span>{roles[n]}</span></div>'
                     f'<table class="cspec">{spec(r)}</table><div class="use">On white: {esc(tag)}</div></div>')
    neut = []
    for n, lab, role in (("ink", "Ink", "type, one-colour mark on screen"), ("night", "Night", "dark backgrounds, app tile"),
                         ("white", "White", "backgrounds, the badge disc")):
        r = rows[n]
        bd = f"border:1px solid {RULE};" if n == "white" else ""
        neut.append(f'<div class="ncard"><div class="nsw" style="background:{r["hex"]};{bd}"></div>'
                    f'<div><div class="cn"><h3>{lab}</h3><span>{role}</span></div>'
                    f'<table class="cspec">{spec(r, False)}</table></div></div>')
    blue, green = rows["blue"], rows["green"]
    note = (f'<div class="pnote"><h3>Print</h3><p>CMYK from sRGB through <b>{esc(CS["profile"])}</b> '
            f'({esc(CS["profile_file"])}, Ghostscript), relative colorimetric with black-point compensation, '
            f'whole percent. ΔE is CIE76 between screen and print.</p>'
            f'<p><b>Blue is outside the CMYK gamut</b> (ΔE {blue["de"]:.1f}). For print, match it with a spot ink '
            f'chosen against a physical swatch book, and approve it on a proof on the real stock. '
            f'Green is {gamut_note(green["de"])} (ΔE {green["de"]:.1f}): check it on the proof too.</p></div>')
    body = (f'<div class="grid4 colours">{"".join(cards)}</div>'
            f'<div class="neutrals">{"".join(neut)}{note}</div>')
    return slide(7, "Colour", "Four colours, three neutrals",
                 "The client’s own colours, deepened and moved away from Google’s hues. Defined in OKLCH.",
                 body, top=286)


# ------------------------------------------------------------------ 8 backgrounds
def photo_svg(w: int, h: int, badge: bool) -> str:
    """An illustrative 'photograph' (vector: sky, sun, hills, water) with the badge or the bare mark."""
    k = brand.master()
    sky = brand._lin("sky", (0, 0), (0, h * .62), [(0, "#7F9CC4", None), (0.6, "#C9B9C9", None), (1, "#F2C7A2", None)])
    sun = brand._rad("sun", (w * .66, h * .5), h * .55, [(0, "#FFF1D6", 0.95), (0.25, "#FBD9AE", 0.55), (1, "#F2C7A2", 0)])
    water = brand._lin("water", (0, h * .7), (0, h), [(0, "#A8A6BC", None), (1, "#4B5873", None)])
    defs = sky + sun + water
    hills = []
    for col, base, amp, ph in (("#8C8FB0", .56, .09, .3), ("#5F6F8F", .63, .08, 1.7), ("#33425C", .70, .07, 3.1)):
        pts = [(x, h * (base - amp * (0.6 * math.sin(x / w * 7 + ph) + 0.4 * math.sin(x / w * 17 + 2 * ph))))
               for x in [w * j / 60 for j in range(61)]]
        d = f"M0 {fmt(h)} " + " ".join(f"L{fmt(x)} {fmt(y)}" for x, y in pts) + f" L{fmt(w)} {fmt(h)} Z"
        hills.append(f'<path fill="{col}" d="{d}"/>')
    body = (f'<rect width="{w}" height="{h}" fill="url(#sky)"/><rect width="{w}" height="{h}" fill="url(#sun)"/>'
            + "".join(hills) + f'<rect y="{fmt(h * .74)}" width="{w}" height="{fmt(h * .26)}" fill="url(#water)"/>')
    for i in range(7):                                     # ripples on the water
        y = h * (.78 + i * .03)
        body += (f'<rect x="{fmt(w * (.1 + .09 * (i % 3)))}" y="{fmt(y)}" width="{fmt(w * (.25 + .05 * (i % 2)))}" '
                 f'height="1.2" fill="#FFFFFF" fill-opacity="0.18"/>')
    if badge:                                              # in the calm sky, top left
        s = h * .34
        body += f'<g transform="translate({fmt(h * .09)} {fmt(h * .09)}) scale({fmt(s / 1024)})">{brand.disc(k)}{mark(k)}</g>'
    else:
        x0, y0, x1, y1 = kmark.bbox(k)
        s = h * .46 / (y1 - y0)
        tx, ty = w * .5 - (x0 + x1) / 2 * s, h * .52 - (y0 + y1) / 2 * s     # over the busy hills
        body += f'<g transform="translate({fmt(tx)} {fmt(ty)}) scale({fmt(s)})">{mark(k)}</g>'
    return inline(doc(body, (0, 0, w, h), defs), w, h)


def page_backgrounds() -> str:
    k = brand.master()
    vb = tight_vb(k)

    def mk(colours=None, gaps=False, h=120):
        return inline(doc(mark(k, colours, gaps), vb), h=h)

    TH = 250                                               # tile height

    def v(ok):
        return f'<div class="vd">{verdict(ok, 28)}</div>'

    yellow_bad = inline(doc(f'<rect x="0" y="0" width="300" height="200" fill="#FFFFFF"/>'
                            f'<rect x="40" y="52" width="150" height="22" rx="3" fill="{PALETTE["yellow"]}"/>'
                            f'<rect x="40" y="94" width="220" height="10" rx="2" fill="{PALETTE["yellow"]}"/>'
                            f'<rect x="40" y="116" width="190" height="10" rx="2" fill="{PALETTE["yellow"]}"/>'
                            f'<circle cx="250" cy="63" r="14" fill="{PALETTE["yellow"]}"/>', (0, 0, 300, 200)), 270, 180)
    cols = [
        ("Light", "Full colour on white and light neutrals. The premium shaded mark is made for light backgrounds too.",
         [tile(mk() + v(True), "#FFFFFF", TH, border=True), tile(mk() + v(True), SURFACE2, TH)]),
        ("Dark", "On night or other dark fields: the full-colour mark, or the white one-colour mark with gaps.",
         [tile(mk() + v(True), NIGHT, TH), tile(mk({n: WHITE for n in FACETS}, gaps=True) + v(True), NIGHT, TH)]),
        ("Photography", "Use the badge: its white disc keeps every colour true on any image. Never the bare mark on a photo.",
         [tile(photo_svg(390, TH, True) + v(True), "#000", TH, style="padding:0;overflow:hidden"),
          tile(photo_svg(390, TH, False) + v(False), "#000", TH, style="padding:0;overflow:hidden")]),
        ("Yellow", f"Brand yellow never stands alone on white ({contrast(PALETTE['yellow'], WHITE):.2f}:1): "
                   "it appears only inside the mark.",
         [tile(yellow_bad + v(False), "#FFFFFF", TH, border=True), tile(mk() + v(True), "#FFFFFF", TH, border=True)]),
    ]
    body = '<div class="grid4 bgs">' + "".join(
        f'<div class="bgcol"><h3>{t}</h3>{"".join(ts)}<p>{esc(s)}</p></div>' for t, s, ts in cols) + "</div>"
    return slide(8, "Backgrounds", "Where the mark sits",
                 "Contrast first: the four colours need a calm, light or dark field to read as one letter.", body, top=286)


# ------------------------------------------------------------------ 9 misuse
def low_contrast_grey(target: str) -> str:
    """The ink-hued grey of the same luminance as target (contrast 1.0:1): bisection on OKLCH L."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if luminance(_grey(mid, 0.01)) < luminance(target) else (lo, mid)
    return _grey(lo, 0.01)


def page_misuse() -> str:
    k = brand.master()
    x0, y0, x1, y1 = kmark.bbox(k)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    vb = (cx - 400, cy - 400, 800, 800)
    S = 212

    def cell(body, defs="", v=vb):
        return inline(doc(body, v, defs), S, S)

    pal = {n: PALETTE[n] for n in FACETS}
    rnd = KParams(**dict(k.__dict__, r_convex=44.0, r_concave=44.0, palette=dict(k.palette)))
    outline = "".join(f'<path fill="none" stroke="{PALETTE[n]}" stroke-width="7" stroke-linejoin="miter" d="{poly_d(f)}"/>'
                      for n, f in kmark.facets(k).items())
    sil = Polygon(silhouette(k))
    shadow = "".join(f'<path fill="{INK}" fill-opacity="0.045" d="{brand.shape_d(sil.buffer(r, join_style=1))}" '
                     f'transform="translate(0 26)"/>' for r in (2, 7, 13, 20, 28, 37, 47))
    grey = low_contrast_grey(PALETTE["green"])
    crop = brand.scaled_about_centre(k, 1.55)
    clip = '<clipPath id="disc-clip"><circle cx="512" cy="512" r="512"/></clipPath>'
    items = [
        (True, "Correct", "The master, as supplied", cell(mark(k))),
        (False, "Rounded corners", "Every corner is sharp", cell(kmark.flat_group(rnd))),
        (False, "Rearranged colours", "Colours stay on their facets",
         cell(mark(k, dict(pal, red=pal["blue"], blue=pal["red"], yellow=pal["green"], green=pal["yellow"])))),
        (False, "Google’s palette", "Only the brand colours", cell(mark(k, GOOGLE))),
        (False, "Stretched", "Scale in proportion only",
         cell(f'<g transform="translate({fmt(cx)} {fmt(cy)}) scale(1.32 0.92) translate({fmt(-cx)} {fmt(-cy)})">{mark(k)}</g>')),
        (False, "Rotated", "The mark stands upright", cell(f'<g transform="rotate(-14 {fmt(cx)} {fmt(cy)})">{mark(k)}</g>')),
        (False, "Outlined", "Facets are solid fills", cell(outline)),
        (False, "Drop shadow", "No shadow, glow or bevel", cell(shadow + mark(k))),
        (False, "Low contrast", f"Green on this grey: {contrast(PALETTE['green'], grey):.2f}:1", cell(mark(k)), grey),
        (False, "Cropped badge", "The K sits whole inside the disc",
         cell(f'{brand.disc(k)}<g clip-path="url(#disc-clip)">{mark(crop)}</g>', defs=clip, v=(0, 0, 1024, 1024)), SURFACE2),
    ]
    body = '<div class="grid5">' + "".join(
        f'<figure class="mis"><div class="mis-art{" ok" if it[0] else ""}" style="background:{it[4] if len(it) > 4 else "#FFFFFF"}">'
        f'{it[3]}<div class="vd">{verdict(it[0], 30)}</div></div>'
        f'<figcaption><h3>{esc(it[1])}</h3><p>{esc(it[2])}</p></figcaption></figure>' for it in items) + "</div>"
    return slide(9, "Misuse", "Never alter the mark",
                 "Each example is the real geometry, changed in one way — and each change breaks the mark.",
                 body, top=286)


# ------------------------------------------------------------------ 10 app icons and favicons
def page_icons() -> str:
    k = brand.master()
    tech_tile = inline(brand.doc_tech(k), 200, 200)
    tech_sq = inline(brand.doc_tech(k, shape="square"), 200, 200)
    small = "".join(f'<figure class="sz"><div class="sz-art" style="height:64px">{inline(brand.doc_tech(k, detail="small"), n, n)}</div>'
                    f'<figcaption>{n}</figcaption></figure>' for n in (64, 48, 32, 24))
    # Android adaptive: 108 dp canvas, 72 dp visible, 66 dp safe circle
    u = 1024 / 108
    bg = brand.doc_android_background(k)
    fg = brand.doc_android_foreground(k)
    inner = lambda s: re.sub(r"^<svg\b[^>]*>|</svg>$", "", re.sub(r"<title>.*?</title>", "", s))
    layers = inner(bg) + inner(fg)
    ad = Draw((-8, -8, 1040, 1040), width=236)
    ad.add(layers)
    ad.add(f'<rect x="{fmt(18 * u)}" y="{fmt(18 * u)}" width="{fmt(72 * u)}" height="{fmt(72 * u)}" fill="none" stroke="#FFFFFF" '
           f'stroke-opacity="0.75" stroke-width="{fmt(ad.px(1.2))}" stroke-dasharray="{fmt(ad.px(5))} {fmt(ad.px(4))}"/>')
    ad.add(f'<circle cx="512" cy="512" r="{fmt(33 * u)}" fill="none" stroke="#FFFFFF" stroke-width="{fmt(ad.px(1.4))}"/>')
    ad.add(f'<rect x="0" y="0" width="1024" height="1024" fill="none" stroke="{RULE2}" stroke-width="{fmt(ad.px(1))}"/>')
    lab = _grey(0.86)
    ad.text((ad.px(9), ad.px(17)), "108 dp", 11, 500, lab, halo=None)
    ad.text((18 * u + ad.px(6), 18 * u + ad.px(15)), "72 dp", 11, 500, lab, halo=None)
    ad.text((512, 512 + 33 * u - ad.px(7)), "66 dp", 11, 500, lab, anchor="middle", halo=None)
    masks = []
    c, r = 512, 36 * u
    sq = 2 * r
    n_ = 4.0
    squircle = " ".join(
        f"{'M' if i == 0 else 'L'}{fmt(c + r * math.copysign(abs(math.cos(t)) ** (2 / n_), math.cos(t)))} "
        f"{fmt(c + r * math.copysign(abs(math.sin(t)) ** (2 / n_), math.sin(t)))}"
        for i, t in enumerate([2 * math.pi * j / 128 for j in range(128)])) + " Z"
    shapes = [("Circle", f'<circle cx="512" cy="512" r="{fmt(r)}"/>'),
              ("Squircle", f'<path d="{squircle}"/>'),
              ("Rounded square", f'<rect x="{fmt(c - r)}" y="{fmt(c - r)}" width="{fmt(sq)}" height="{fmt(sq)}" rx="{fmt(0.18 * sq)}"/>')]
    for name, shp in shapes:
        dd = f'<clipPath id="m">{shp}</clipPath>'
        art = inline(doc(f'<g clip-path="url(#m)">{layers}</g>', (c - r, c - r, sq, sq), dd), 96, 96)
        masks.append(f'<figure class="sz"><div class="sz-art" style="height:96px">{art}</div><figcaption>{name}</figcaption></figure>')
    fav16, fav32 = favicon_pixels(16), favicon_pixels(32)
    fav = (f'<div class="favs"><div class="fav1x">'
           f'<div class="chip" style="background:#FFFFFF;border:1px solid {RULE}">{pixel_art(fav16, 1)}{pixel_art(fav32, 1)}</div>'
           f'<div class="chip" style="background:{NIGHT}">{pixel_art(fav16, 1)}{pixel_art(fav32, 1)}</div>'
           f'<p>1×: 16 and 32 px</p></div>'
           f'<figure class="zoom">{pixel_art(fav16, 11, "#FFFFFF", RULE)}<figcaption>16 px, 11×</figcaption></figure>'
           f'<figure class="zoom">{pixel_art(fav32, 5.5, "#FFFFFF", RULE)}<figcaption>32 px, 5.5×</figcaption></figure></div>')
    badge = brand.doc_badge(k, raster=True)

    def crops(cap_cls: str = "") -> str:
        """The badge at avatar sizes. Built per strip: inline() gives each copy its own ids (one
        string inserted twice would duplicate every id in the page)."""
        cc = f' class="{cap_cls}"' if cap_cls else ""
        return "".join(f'<figure class="sz"><div class="sz-art" style="height:96px">{inline(badge, n, n)}</div>'
                       f'<figcaption{cc}>{n}</figcaption></figure>' for n in (96, 64, 40, 24))
    up = Draw((-40, -40, 1104, 1104), width=150)
    up.add(f'<rect x="0" y="0" width="1024" height="1024" fill="{SURFACE2}"/>')
    up.add(re.sub(r"^<svg\b[^>]*>|</svg>$", "", re.sub(r"<title>.*?</title>", "", badge)))
    up.add(f'<rect x="0" y="0" width="1024" height="1024" fill="none" stroke="{INK}" stroke-width="{fmt(up.px(1.1))}" '
           f'stroke-dasharray="{fmt(up.px(5))} {fmt(up.px(4))}"/>')
    quads = [
        ("Tech icon", "Rounded tile for apps and docks; the full-bleed square for stores, which apply their own mask. "
                      "At 64 px and below, the small version: no grid, traces or node.",
         f'<div class="qrow">{tile(tech_tile, SURFACE, 228, style="width:228px")}{tile(tech_sq, SURFACE, 228, style="width:228px")}'
         f'<div class="smalls"><div class="strip">{small}</div><p>Small version, px</p></div></div>'),
        ("Android adaptive", "Foreground and background layers on the 108 dp canvas. The K stays inside the 66 dp "
                             "safe circle (solid), so every launcher mask of the 72 dp view (dashed) keeps it whole.",
         f'<div class="qrow"><div class="adap">{ad.html()}</div><div class="strip masks">{"".join(masks)}</div></div>'),
        ("Favicon", "Grid-fitted: cap and base on pixel rows, stem on whole pixels. Shown at 1× and enlarged, "
                    "pixel for pixel as shipped.", fav),
        ("Avatar badge", "Upload the square; the platform’s circle crop lands on the disc edge and the K stays whole.",
         f'<div class="qrow"><div class="upload">{up.html()}<p>Square upload, circle crop</p></div>'
         # light strip on a light neutral, not white: on white the disc and its crop edge vanish
         f'<div class="strip crops" style="background:{SURFACE2}">{crops()}</div>'
         f'<div class="strip crops" style="background:{NIGHT}">{crops("inv")}</div></div>'),
    ]
    body = '<div class="quads">' + "".join(f'<section class="quad"><h3>{t}</h3><p class="qnote">{esc(s)}</p>{x}</section>'
                                           for t, s, x in quads) + "</div>"
    return slide(10, "App icons and favicons", "Icons for every screen",
                 "The tech icon for apps, the badge for avatars and the bare, grid-fitted K for favicons.", body, top=282)


# ------------------------------------------------------------------ 11 file guide
FILES = [
    ("logo/svg", "SVG", "The editable masters: 17 variants with named groups. The source for web, code and every other format."),
    ("logo/ai", "AI (PDF-compatible)", "Illustrator. It opens as one layer holding named groups. To rebuild natively, with layers "
                                       "and global swatches, run the scripts in logo/scripts/*.jsx inside Illustrator."),
    ("logo/psd", "PSD", "Photoshop: editable vector shape layers, one per facet, grouped by variant."),
    ("logo/eps", "EPS", "Vendors that ask for EPS: sign makers, embroidery, promotional print."),
    ("logo/pdf", "PDF", "Vector print artwork, RGB and CMYK, for printers and documents."),
    ("logo/png", "PNG, 256–4096 px", "Transparent rasters for slides, office documents and quick use."),
    ("icons/", "PNG, ICO, ICNS, XML, SVG", "App and web icons: iOS, macOS, Android adaptive, favicons and touch icons."),
    ("social/", "PNG", "Avatars and banners sized for social platforms, plus the Open Graph image."),
    ("web/", "SVG, TSX, JS, CSS, JSON", "Developer kit: optimised SVGs, React and web components, design tokens."),
    ("tools/", "Python", "The reproducible build. Every file above is generated from brand.py."),
    ("guidelines/", "HTML, PDF", "This document."),
]


def page_files() -> str:
    rows = "".join(f'<tr><td class="mono">{esc(a)}</td><td>{esc(b)}</td><td>{esc(c)}</td></tr>' for a, b, c in FILES)
    body = (f'<table class="files"><thead><tr><th>Folder</th><th>Formats</th><th>Use it for</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>'
            f'<div class="cmd"><span>Rebuild the whole kit</span><code>python3 brand/tools/build.py all</code></div>')
    return slide(11, "File guide", "What’s in the kit",
                 "Pick the folder by the program you work in. Never redraw the artwork: change brand.py and rebuild.",
                 body, top=290)


# ------------------------------------------------------------------ fonts
def _cff_to_glyf(f):
    """Replace a font's CFF outlines by TrueType (quadratic) outlines, in place."""
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import newTable
    order = f.getGlyphOrder()
    gs = f.getGlyphSet()
    glyf = newTable("glyf")
    glyf.glyphOrder, glyf.glyphs = order, {}
    for g in order:
        pen = TTGlyphPen(gs)
        gs[g].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyf.glyphs[g] = pen.glyph()
    f["glyf"], f["loca"] = glyf, newTable("loca")
    for t in ("CFF ", "VORG"):
        if t in f:
            del f[t]
    f.sfntVersion = "\x00\x01\x00\x00"
    mp = f["maxp"]
    mp.tableVersion = 0x00010000
    for a in ("maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs", "maxStackElements",
              "maxSizeOfInstructions", "maxComponentElements"):
        setattr(mp, a, 0)
    mp.maxZones = 1
    f["head"].glyphDataFormat = 0
    post = f["post"]
    post.formatType, post.extraNames, post.mapping = 2.0, [], {}


def font_css(chars: set) -> str:
    from fontTools import subset
    from fontTools.ttLib import TTFont
    text = "".join(sorted(chars | {chr(c) for c in range(0x20, 0x7F)}))
    out = []
    for fam, wt, path in FONTS:
        f = TTFont(path, recalcTimestamp=False)             # keep head.modified: same bytes every build
        o = subset.Options()
        o.layout_features, o.name_IDs, o.notdef_outline, o.hinting = ["*"], ["*"], True, True
        s = subset.Subsetter(o)
        s.populate(text=text)
        s.subset(f)
        if "CFF " in f:
            _cff_to_glyf(f)
        f.flavor = "woff"
        b = io.BytesIO()
        f.save(b)
        out.append(f"@font-face{{font-family:'{fam}';font-weight:{wt};font-style:normal;font-display:block;"
                   f"src:url(data:font/woff;base64,{base64.b64encode(b.getvalue()).decode()}) format('woff')}}")
    return "\n".join(out)


# ------------------------------------------------------------------ CSS
def css() -> str:
    return f"""
:root{{--ink:{INK};--t2:{TEXT2};--t3:{TEXT3};--rule:{RULE};--rule2:{RULE2};--surface:{SURFACE};--surface2:{SURFACE2};--night:{NIGHT}}}
@page{{size:{W}px {H}px;margin:0}}
*{{box-sizing:border-box}}
html,body{{margin:0;padding:0}}
body{{font-family:Inter,sans-serif;color:var(--ink);background:#FFFFFF;-webkit-print-color-adjust:exact;print-color-adjust:exact;
  text-rendering:geometricPrecision}}
h1,h2,h3,p,figure,dl,dd{{margin:0}}
figure{{display:block}}
.slide{{position:relative;width:{W}px;height:{H}px;overflow:hidden;background:#FFFFFF;break-after:page;page-break-after:always}}
.slide:last-child{{break-after:auto;page-break-after:auto}}
.head{{position:absolute;left:{MX}px;top:92px;width:{W - 2 * MX}px}}
.eyebrow{{display:flex;gap:14px;align-items:baseline;font-size:14px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--t3)}}
.eyebrow .no{{color:var(--ink);font-variant-numeric:tabular-nums}}
h1{{margin-top:18px;font-size:54px;line-height:1.08;font-weight:600;letter-spacing:-.024em;text-wrap:balance}}
.lead{{margin-top:18px;font-size:21px;line-height:1.55;color:var(--t2);max-width:1100px;text-wrap:balance}}
.lead.small{{font-size:18px;max-width:none}}
.body{{position:absolute;left:{MX}px;width:{W - 2 * MX}px;bottom:118px}}
.foot{{position:absolute;left:{MX}px;right:{MX}px;bottom:46px;display:grid;grid-template-columns:1fr auto 1fr;font-size:13px;
  color:var(--t3);letter-spacing:.01em}}
.foot span:nth-child(2){{text-align:center}}
.foot .pg{{text-align:right;font-variant-numeric:tabular-nums;color:var(--ink);font-weight:500}}
.tile{{position:relative;display:flex;align-items:center;justify-content:center;border-radius:14px;padding:24px}}
.vd{{position:absolute;right:14px;top:14px;line-height:0}}
h3{{font-size:18px;line-height:1.3;font-weight:600;letter-spacing:-.005em}}
h3.sub{{font-size:15px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--t3)}}
p{{font-size:16px;line-height:1.55;color:var(--t2);text-wrap:pretty}}
.mono{{font-family:'JetBrains Mono',monospace;font-size:.92em;letter-spacing:-.01em}}
/* 1 cover */
.cover-panel{{position:absolute;left:960px;top:0;width:960px;height:{H}px;background:var(--surface);display:flex;align-items:center;justify-content:center}}
.cover-text{{position:absolute;left:{MX}px;bottom:120px;width:760px}}
.cover-title{{margin-top:26px;font-size:92px;line-height:1.02;letter-spacing:-.035em;font-weight:600}}
.cover-meta{{display:flex;gap:56px;margin-top:64px;padding-top:24px;border-top:1px solid var(--rule)}}
.cover-meta dt{{font-size:13px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--t3)}}
.cover-meta dd{{margin-top:8px;font-size:20px;font-weight:500;font-variant-numeric:tabular-nums}}
/* 2 idea */
.steps{{display:grid;grid-template-columns:1fr 56px 1fr 56px 1fr;align-items:start}}
.arrow{{height:500px;display:flex;align-items:center;justify-content:center}}
.step figcaption{{display:flex;gap:16px;margin-top:26px;padding-right:12px}}
.stepno{{flex:none;width:30px;height:30px;border-radius:50%;border:1.5px solid var(--ink);display:flex;align-items:center;justify-content:center;
  font-size:14px;font-weight:600}}
.step p{{margin-top:6px;font-size:16px}}
/* 3 construction */
.cons-draw{{position:absolute;left:64px;top:84px}}
.cons-side{{position:absolute;left:1172px;top:92px;width:628px}}
.cons-side .lead{{margin-top:16px}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums}}
.spec{{margin-top:32px;width:100%}}
.spec th,.spec td{{text-align:left;padding:11px 0;border-top:1px solid var(--rule);font-size:16px;line-height:1.35;vertical-align:baseline}}
.spec th{{font-weight:500;color:var(--t2);width:250px}}
.spec td{{font-weight:500}}
.spec tr:last-child th,.spec tr:last-child td{{border-bottom:1px solid var(--rule)}}
/* 4 refinement */
.ref-left{{position:absolute;left:0;top:0;width:870px}}
.ba-row{{display:grid;grid-template-columns:repeat(3,226px);gap:46px;padding:30px 34px 24px;background:var(--surface);border-radius:14px}}
.ba-art{{height:226px;line-height:0}}
.ba figcaption{{margin-top:14px;font-size:15px;color:var(--t2);text-align:center;white-space:nowrap}}
.ba figcaption b{{display:block;color:var(--ink);font-weight:600;font-size:16px}}
.ba figcaption span{{display:block;margin-top:2px}}
.ref-left h3.sub{{margin-top:40px}}
.cchg-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:24px;margin-top:18px}}
.chips{{display:flex;height:44px;border-radius:8px;overflow:hidden}}
.chips i{{flex:1}}
.cname{{margin-top:12px;font-size:16px;font-weight:600}}
.cchg .mono{{margin-top:4px;font-size:13.5px}}
.cdelta{{margin-top:4px;font-size:13.5px;color:var(--t2);font-variant-numeric:tabular-nums;line-height:1.4}}
.clabel{{margin-top:8px;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--t3);font-weight:500}}
.ref-right{{position:absolute;left:950px;top:0;width:730px}}
.cmp{{width:100%}}
.cmp th,.cmp td{{text-align:left;padding:10.5px 0;border-bottom:1px solid var(--rule);font-size:15.5px;line-height:1.3;font-weight:500}}
.cmp thead th{{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:var(--t3);font-weight:600;padding-top:0}}
.cmp tbody th{{color:var(--t2);width:250px}}
.cmp td:nth-child(2){{color:var(--t2);width:230px}}
/* 5 clear space */
.cs-left{{position:absolute;left:0;top:0;width:820px}}
.cs-left .tile{{padding:0}}
.cs-right{{position:absolute;left:900px;top:0;width:780px}}
.note{{margin-top:12px;font-size:17px;max-width:740px}}
.mins{{margin-top:14px}}
.minrow{{display:grid;grid-template-columns:250px 1fr;gap:32px;padding:14px 0;border-top:1px solid var(--rule);align-items:center}}
.minrow:last-child{{border-bottom:1px solid var(--rule)}}
.mintext h3{{font-size:17px}}
.mintext p{{margin-top:4px;font-size:14.5px;line-height:1.5}}
.minsamples{{display:flex;gap:36px;align-items:flex-end;padding:18px 24px 14px;background:var(--surface);border-radius:12px;min-height:122px}}
.ms{{display:flex;flex-direction:column;align-items:flex-start}}
.ms-art{{height:72px;display:flex;align-items:flex-end;line-height:0}}
.ms figcaption{{margin-top:10px;font-size:13px;color:var(--t3);white-space:nowrap}}
/* 6 variants */
.grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:40px}}
.card figcaption{{margin-top:16px}}
.card p{{margin-top:4px;font-size:15px;line-height:1.5}}
.split{{display:grid;grid-template-columns:1fr 1fr;width:100%;height:100%}}
.split>div{{display:flex;align-items:center;justify-content:center}}
.split svg{{height:120px;width:auto}}
/* 7 colour */
.colours{{gap:40px}}
.sw{{height:150px;border-radius:14px}}
.cn{{display:flex;align-items:baseline;gap:10px;margin-top:16px}}
.cn h3{{font-size:19px}}
.cn span{{font-size:14px;color:var(--t3)}}
.cspec{{margin-top:8px;width:100%}}
.cspec th,.cspec td{{text-align:left;padding:4px 0;font-size:14.5px;line-height:1.35;font-weight:500}}
.cspec th{{width:92px;color:var(--t3);font-weight:500}}
.use{{margin-top:8px;font-size:13.5px;color:var(--t2);padding-top:8px;border-top:1px solid var(--rule)}}
.neutrals{{display:grid;grid-template-columns:repeat(3,1fr) 1.62fr;gap:32px;margin-top:34px;padding-top:28px;border-top:1px solid var(--rule)}}
.ncard{{display:flex;gap:18px}}
.nsw{{flex:none;width:64px;height:64px;border-radius:12px;margin-top:16px}}
.ncard .cspec th{{width:78px}}
.ncard .cspec th,.ncard .cspec td{{font-size:13.5px;padding:2px 0}}
.pnote{{padding-left:28px;border-left:1px solid var(--rule)}}
.pnote h3{{font-size:17px;margin-top:12px}}
.pnote p{{margin-top:6px;font-size:14px;line-height:1.5}}
.pnote b{{color:var(--ink);font-weight:600}}
/* 8 backgrounds */
.bgcol h3{{margin-bottom:16px}}
.bgcol .tile+.tile{{margin-top:16px}}
.bgcol p{{margin-top:18px;font-size:15px}}
/* 9 misuse */
.grid5{{display:grid;grid-template-columns:repeat(5,1fr);gap:28px 32px}}
.mis-art{{position:relative;height:250px;border-radius:14px;border:1px solid var(--rule);display:flex;align-items:center;justify-content:center}}
.mis-art.ok{{border-color:{CHECK_GREEN}}}
.mis figcaption{{margin-top:12px}}
.mis h3{{font-size:16.5px}}
.mis p{{margin-top:2px;font-size:14px;line-height:1.45}}
/* 10 icons */
.quads{{display:grid;grid-template-columns:1fr 1fr;gap:34px 72px}}
.quad h3{{font-size:18px}}
.qnote{{margin-top:4px;font-size:14.5px;line-height:1.5;max-width:800px;min-height:44px}}
.qrow{{display:flex;gap:24px;align-items:center;margin-top:16px}}
.smalls p{{margin-top:8px;font-size:13px;color:var(--t3)}}
.strip{{display:flex;gap:18px;align-items:flex-end;padding:14px 18px;border-radius:12px;background:var(--surface)}}
.sz{{display:flex;flex-direction:column;align-items:center}}
.sz-art{{display:flex;align-items:flex-end;line-height:0}}
.sz figcaption{{margin-top:8px;font-size:12.5px;color:var(--t3);font-variant-numeric:tabular-nums}}
.sz figcaption.inv{{color:{_grey(0.78)}}}
.adap{{line-height:0}}
.masks{{background:var(--surface)}}
.favs{{display:flex;gap:30px;align-items:flex-start;margin-top:16px}}
.fav1x{{display:flex;flex-direction:column;gap:10px}}
.fav1x p{{font-size:13px;color:var(--t3)}}
.chip{{display:flex;gap:14px;align-items:center;padding:12px 16px;border-radius:10px;line-height:0}}
.zoom{{line-height:0}}
.zoom figcaption{{line-height:1.4;margin-top:8px;font-size:13px;color:var(--t3)}}
.upload{{line-height:0}}
.upload p{{line-height:1.4;margin-top:8px;font-size:13px;color:var(--t3);white-space:nowrap}}
.crops{{gap:16px;padding:14px 16px}}
/* 11 files */
.files{{width:100%}}
.files th,.files td{{text-align:left;padding:11px 24px 11px 0;border-bottom:1px solid var(--rule);font-size:16.5px;line-height:1.45;vertical-align:baseline}}
.files thead th{{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:var(--t3);font-weight:600;padding-top:0}}
.files td:first-child{{width:220px;font-weight:500}}
.files td:nth-child(2){{width:300px;color:var(--t2)}}
.cmd{{display:flex;gap:24px;align-items:baseline;margin-top:34px}}
.cmd span{{font-size:15px;color:var(--t3)}}
.cmd code{{font-family:'JetBrains Mono',monospace;font-size:16px;padding:10px 16px;border-radius:8px;background:var(--surface2)}}
/* screen: fit the deck to the window */
@media screen{{
  body{{background:{_grey(0.93)};padding:32px 0}}
  .deck{{zoom:var(--fit,1)}}
  .slide{{margin:0 auto 32px}}
}}
"""


FIT_JS = ("(function(){function f(){var s=Math.min(1,(window.innerWidth-48)/%d);"
          "document.documentElement.style.setProperty('--fit',s)}f();window.addEventListener('resize',f)})();" % W)


# ------------------------------------------------------------------ assembly
def build_html() -> str:
    global _uid
    _uid = itertools.count(1)                               # same ids on every build, in any process
    M = measure()
    CS = colour_specs()
    pages = [page_cover(), page_idea(M), page_construction(M), page_refinement(M), page_clearspace(M), page_variants(),
             page_colour(CS), page_backgrounds(), page_misuse(), page_icons(), page_files()]
    assert len(pages) == PAGES
    body = "\n".join(pages)
    chars = set(html.unescape(re.sub(r"<[^>]+>", " ", body)))
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(TITLE)}</title><meta name="generator" content="brand/tools/build_guidelines.py">'
            f'<style>{font_css(chars)}\n{css()}</style></head>'
            f'<body><main class="deck">{body}</main><script>{FIT_JS}</script></body></html>')


# Playwright: lay the page out for print, lint it (every box inside its slide, no text overflowing
# its box, no two text boxes overlapping), then print the PDF.
NODE_PDF = r"""
const { chromium } = require('playwright');
const [htmlPath, pdfPath] = process.argv.slice(1);
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: %(W)d, height: %(H)d } });
  await page.goto('file://' + htmlPath, { waitUntil: 'load' });
  await page.emulateMedia({ media: 'print' });
  await page.evaluate(() => document.fonts.ready);
  const report = await page.evaluate(() => {
    const out = { slides: 0, fonts: [], issues: [] };
    const FIXED = '.slide, .body, .tile, .mis-art, .sw, .chip, .strip, .ba-art, .minsamples';
    document.fonts.forEach(f => out.fonts.push(f.family + ' ' + f.weight + ' ' + f.status));
    const slides = [...document.querySelectorAll('.slide')];
    out.slides = slides.length;
    const textRects = el => {
      if (el.namespaceURI === 'http://www.w3.org/2000/svg') return [el.getBoundingClientRect()];
      const r = document.createRange(); r.selectNodeContents(el);
      return [...r.getClientRects()].filter(q => q.width > 0.5 && q.height > 0.5);
    };
    slides.forEach((s, i) => {
      const R = s.getBoundingClientRect();
      const all = [...s.querySelectorAll('*')];
      for (const e of all) {
        const inSvg = e.closest('svg') && e.tagName.toLowerCase() !== 'svg';
        if (inSvg && e.tagName.toLowerCase() !== 'text') continue;
        const r = e.getBoundingClientRect();
        if (r.width === 0 && r.height === 0) continue;
        if (r.left < R.left - 0.5 || r.top < R.top - 0.5 || r.right > R.right + 0.5 || r.bottom > R.bottom + 0.5)
          out.issues.push(`p${i + 1}: <${e.tagName.toLowerCase()} class="${e.getAttribute('class') || ''}"> leaves the slide`);
        if (inSvg || e.tagName.toLowerCase() === 'svg' || e.clientWidth === 0) continue;
        const tag = `p${i + 1}: <${e.tagName.toLowerCase()} class="${e.getAttribute('class') || ''}">`;
        if (e.scrollWidth > e.clientWidth + 1 && getComputedStyle(e).overflowX === 'visible')
          out.issues.push(`${tag} ${e.scrollWidth - e.clientWidth}px too wide (${e.textContent.trim().slice(0, 40)})`);
        if (e.matches(FIXED) && e.scrollHeight > e.clientHeight + 1)
          out.issues.push(`${tag} ${e.scrollHeight - e.clientHeight}px too tall (${e.textContent.trim().slice(0, 40)})`);
      }
      const texts = all.filter(e => [...e.childNodes].some(n => n.nodeType === 3 && n.textContent.trim()));
      const boxes = texts.map(e => ({ e, rs: textRects(e) }));
      for (let a = 0; a < boxes.length; a++) for (let b = a + 1; b < boxes.length; b++) {
        const A = boxes[a], B = boxes[b];
        if (A.e.contains(B.e) || B.e.contains(A.e)) continue;
        for (const p of A.rs) for (const q of B.rs) {
          const w = Math.min(p.right, q.right) - Math.max(p.left, q.left), h = Math.min(p.bottom, q.bottom) - Math.max(p.top, q.top);
          if (w > 1 && h > 1) out.issues.push(`p${i + 1}: text "${A.e.textContent.trim().slice(0, 30)}" overlaps "${B.e.textContent.trim().slice(0, 30)}"`);
        }
      }
    });
    return out;
  });
  // tagged: a structure tree (headings, reading order) for screen readers. No outline: Chromium
  // drops the space at every line wrap of a heading, so two-line titles become 'MarkGuidelines'.
  await page.pdf({ path: pdfPath, preferCSSPageSize: true, printBackground: true, tagged: true, outline: false });
  console.log(JSON.stringify(report));
  await browser.close();
})();
""" % {"W": W, "H": H}


def print_pdf(html_path: str, pdf_path: str) -> dict:
    res = subprocess.run(["node", "-e", NODE_PDF, html_path, pdf_path], capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        raise RuntimeError(f"Playwright failed:\n{res.stderr[-4000:]}")
    return json.loads(res.stdout.strip().splitlines()[-1])


LAYOUT = {}


def run(root: str = ROOT) -> list:
    out_dir = os.path.join(root, OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, f"{BASENAME}.html")
    pdf_path = os.path.join(out_dir, f"{BASENAME}.pdf")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html())
    LAYOUT.update(print_pdf(html_path, pdf_path))
    return [html_path, pdf_path]


# ------------------------------------------------------------------ verification
def verify(root: str = ROOT, png_dir: str = None) -> bool:
    """Opens the PDF, checks pages / size / fonts / images and the layout lint, renders every page
    to PNG (png_dir) for review, and recomputes the numbers of pages 3, 4 and 7 independently of
    the build code to find them in the text of their page."""
    import pypdf
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {msg}")

    html_path = os.path.join(root, OUT_DIR, f"{BASENAME}.html")
    pdf_path = os.path.join(root, OUT_DIR, f"{BASENAME}.pdf")
    print("\n== HTML")
    src = open(html_path, encoding="utf-8").read()
    check(not re.search(r"""(?:src|href)=["'](?!data:|#)""", src) and "url(http" not in src and "url(file" not in src,
          f"self-contained: no external src / href / url() ({len(src) / 1e6:.2f} MB)")
    check(src.count("<section class=\"slide") == PAGES, f"{PAGES} slides")
    print("\n== layout lint (Chromium, print media)")
    if LAYOUT:
        check(LAYOUT.get("slides") == PAGES, f"Chromium laid out {LAYOUT.get('slides')} slides")
        check(all(s.endswith("loaded") for s in LAYOUT.get("fonts", [])), "embedded fonts loaded: " + ", ".join(LAYOUT.get("fonts", [])))
        check(not LAYOUT.get("issues"), f"{len(LAYOUT.get('issues', []))} layout issues")
        for s in LAYOUT.get("issues", [])[:40]:
            print("       ", s)

    print("\n== PDF")
    r = pypdf.PdfReader(pdf_path)
    check(len(r.pages) == PAGES, f"{len(r.pages)} pages")
    sizes = {(float(p.mediabox.width), float(p.mediabox.height)) for p in r.pages}
    check(sizes == {(W * 0.75, H * 0.75)}, f"page size {sizes} pt = {W} x {H} px, 16:9")

    # every indirect object in the file, not only the pages' XObject resources: a rasterised
    # effect can hide in a pattern's or a soft mask's resources. Skia's form XObjects are the
    # transparency groups of gradients with alpha (soft masks), which are vector.
    kinds = {"/Image": 0, "/Form": 0}
    nums = {num for gen in r.xref.values() for num in gen} | set(getattr(r, "xref_objStm", {}))
    for num in sorted(nums):
        try:
            o = r.get_object(pypdf.generic.IndirectObject(num, 0, r))
        except Exception:                                    # pragma: no cover - free / broken entries
            continue
        st = o.get("/Subtype") if hasattr(o, "get") else None
        if st in kinds:
            kinds[st] += 1
    imgs, forms = kinds["/Image"], kinds["/Form"]
    check(imgs == 0, f"{imgs} image XObjects anywhere in the file ({forms} form XObjects: soft-mask groups): "
                     "all artwork is vector")
    fonts = subprocess.run(["pdffonts", pdf_path], capture_output=True, text=True).stdout.strip().splitlines()[2:]
    names = sorted({ln.split()[0].split("+")[-1] for ln in fonts})
    types = sorted({" ".join(ln.split()[1:-5]) for ln in fonts})
    emb = all(ln.split()[-5] == "yes" for ln in fonts)
    check(emb and fonts, f"{len(fonts)} font objects, all embedded: {', '.join(names)} ({'; '.join(types)})")
    check(all(n.startswith(("Inter", "JetBrainsMono")) for n in names), "only Inter and JetBrains Mono")
    check(all(ln.split()[1] != "Type" or ln.split()[2] != "3" for ln in fonts), "no Type 3 fonts (TrueType outlines)")
    check("/StructTreeRoot" in r.trailer["/Root"], "tagged PDF: structure tree for screen readers")
    meta_title = (r.metadata or {}).get("/Title")
    print(f"  info title: {meta_title}")

    texts = [re.sub(r"\s+", "", p.extract_text() or "").replace("−", "-") for p in r.pages]

    def find(page, tokens, what):
        t = texts[page - 1]
        miss = [x for x in tokens if x.replace("−", "-").replace(" ", "") not in t]
        check(not miss, f"p{page} {what}: {len(tokens) - len(miss)}/{len(tokens)} numbers found" + (f", missing {miss}" if miss else ""))

    # ---- page 3: recomputed from kmark.points(brand.master()) with plain arithmetic
    print("\n== numbers recomputed")
    k = brand.master()
    pt = kmark.points(k)
    xs = [q[0] for q in kmark.silhouette(k)]
    ys = [q[1] for q in kmark.silhouette(k)]
    J, P, C = pt["J"], pt["P"], pt["C"]
    slope = math.degrees(math.atan(4 / 3))
    check(abs(k.arm_angle - slope) < 1e-9 and abs(k.leg_angle - slope) < 1e-9, f"arm and leg slope = atan(4/3) = {slope:.4f} deg")
    check(abs(pt["armTR"][0] - pt["legBR"][0]) < 1e-9, "right extremes of arm and leg are equal")
    check(abs((P[1] - J[1]) - (C[1] - P[1])) < 1e-9, "J and C mirror about P")
    check(abs(round(100 * k.arm_w / k.w, 1) - 96.5) < 1e-9, "diagonal width = 96.5% of stem")
    p3 = [f"{k.w:.2f}", f"{k.arm_w:.2f}", f"{k.leg_w:.2f}", "96.5%", f"{slope:.2f}", f"{J[0]:.2f}", f"{J[1]:.2f}",
          f"{P[0]:.2f}", f"{P[1]:.2f}", f"{C[0]:.2f}", f"{C[1]:.2f}", f"{P[1] - J[1]:.2f}", f"{C[1] - P[1]:.2f}",
          f"{math.degrees(math.atan2(P[1] - J[1], P[0] - J[0])):.2f}", f"{pt['armTR'][0]:.2f}", f"{pt['legBR'][0]:.2f}",
          f"{max(ys) - min(ys):.2f}", f"{100 * (max(ys) - min(ys)) / 1024:.1f}%", "1024",
          f"{(min(xs) + max(xs)) / 2 - 512:+.2f}", f"{(min(ys) + max(ys)) / 2 - 512:+.2f}"]
    find(3, p3, "construction")
    # ---- page 4: trace numbers straight from trace_params.json
    with open(TRACE_PARAMS) as f:
        tj = json.load(f)
    t = KParams.from_dict(tj)
    tp = kmark.points(t)
    tx = [q[0] for q in kmark.silhouette(t)]
    ty = [q[1] for q in kmark.silhouette(t)]
    p4 = [f"{tj['arm_angle']:.2f}", f"{tj['leg_angle']:.2f}", f"{tj['w']:.2f}", f"{tj['arm_w']:.2f}", f"{tj['leg_w']:.2f}",
          f"{100 * tj['arm_w'] / tj['w']:.1f}%", f"{100 * tj['leg_w'] / tj['w']:.1f}%",
          f"{tp['armTR'][0]:.2f}", f"{tp['legBR'][0]:.2f}", f"{tp['P'][1] - tp['J'][1]:.2f}", f"{tp['C'][1] - tp['P'][1]:.2f}",
          f"{max(ty) - min(ty):.2f}", f"{max(tx) - min(tx):.2f}", f"{(min(tx) + max(tx)) / 2 - 512:+.2f}",
          f"{(min(ty) + max(ty)) / 2 - 512:+.2f}", tj["palette"]["disc"],
          f"{k.w:.2f}", f"{k.arm_w:.2f}", f"{pt['armTR'][0]:.2f}", f"{max(xs) - min(xs):.2f}"]
    for n in FACETS:
        a = hex_to_oklch(tj["palette"][n])
        b = brand.PALETTE_OKLCH[n]
        g = hex_to_oklch(GOOGLE[n])
        p4 += [tj["palette"][n], brand.PALETTE[n], f"{b[0] - a[0]:+.3f}", f"{b[2] - a[2]:+.2f}",
               f"{a[2] - g[2]:+.1f}", f"{b[2] - g[2]:+.1f}"]
    find(4, p4, "refinement")
    # ---- page 7: colour, with an ICC transform built here (not the converter the page used)
    from PIL import Image, ImageCms
    prof_path = (glob.glob("/usr/share/color/icc/ghostscript/default_cmyk.icc")
                 or glob.glob("/usr/share/ghostscript/**/default_cmyk.icc", recursive=True))[0]
    prof = ImageCms.getOpenProfile(prof_path)
    fwd = ImageCms.buildTransform(ImageCms.createProfile("sRGB"), prof, "RGB", "CMYK",
                                  renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                                  flags=ImageCms.Flags.BLACKPOINTCOMPENSATION)
    p7 = [ImageCms.getProfileDescription(prof).strip()]
    for n, h in list(brand.PALETTE.items()) + [("ink", INK), ("night", NIGHT), ("white", WHITE)]:
        rgb = tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
        p7 += [h, ", ".join(map(str, rgb)), f"{contrast(h, '#FFFFFF'):.2f}:1", f"{contrast(h, NIGHT):.2f}:1"]
        if n in brand.PALETTE:
            L, Cc, hh = brand.PALETTE_OKLCH[n]
            c = ImageCms.applyTransform(Image.new("RGB", (1, 1), rgb), fwd).getpixel((0, 0))
            p7 += [f"{L * 100:.1f}% {Cc:.3f} {hh:.1f}", "".join(f"{ch}{round(v / 255 * 100)}" for ch, v in zip("CMYK", c))]
    find(7, p7, "colour")
    try:
        import vector_export as ve
        de = ve.delta_e76(brand.PALETTE["blue"], ve.srgb_to_cmyk(brand.PALETTE["blue"]))
        find(7, [f"{de:.1f}"], "blue print delta E")
        check(de > 15, f"blue is outside the CMYK gamut (dE {de:.1f}) and the page says so")
        check("outsidetheCMYKgamut" in texts[6], "page 7 states the blue is outside the CMYK gamut")
    except ImportError:
        pass
    check("Pantone" not in "".join(texts), "no Pantone numbers claimed")

    if png_dir:
        os.makedirs(png_dir, exist_ok=True)
        subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", "-r72", "-dTextAlphaBits=4",
                        "-dGraphicsAlphaBits=4", f"-sOutputFile={png_dir}/page-%02d.png", pdf_path], check=True)
        print(f"\n  pages rendered by Ghostscript at 72 dpi -> {png_dir}/page-NN.png")
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    paths = run(ROOT)
    for p in paths:
        print(p, f"{os.path.getsize(p) / 1e6:.2f} MB")
    if "--no-verify" not in sys.argv:
        png = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--png=")), None)
        sys.exit(0 if verify(ROOT, png) else 1)
