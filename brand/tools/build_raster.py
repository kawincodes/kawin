"""Every raster deliverable of the brand kit, regenerated from brand.py.

    python3 brand/tools/build_raster.py              # build everything, then verify and print the numbers
    python3 brand/tools/build_raster.py --no-verify  # build only

Outputs (under brand/):
  logo/png/                       K-Logo_<Variant>_<px>.png, 256 ... 4096 px wide, transparent
  icons/web/                      favicons, touch / PWA icons, site.webmanifest, head-snippet.html
  icons/ios/AppIcon.appiconset/   single-size universal 1024 app icon (Xcode 14+) + Contents.json
  icons/macos/                    1024 master, AppIcon.iconset (16 ... 512@2x) and K-Logo.icns
  icons/android/                  adaptive icon (anydpi-v26 XML + mipmaps), legacy icons, Play Store icon
  social/                         avatars, Open Graph image, X / LinkedIn / YouTube banners
  icons/K-Logo_Icons_Overview.png every icon and social image in context

Production decisions
  * Rendering   - everything goes through brand.render (librsvg, premultiplied box filter). Below
                  1024 px we render through it at >= 1024 px and box-reduce with the same filter:
                  brand.render's own 8x supersampling at <= 128 px makes the 2-unit facet trap
                  thinner than one sample, which lets interior alpha dip to 252 at 16 px.
  * Favicons    - 16 / 32 / 48 px are the bare K, grid-fitted: brand.transform puts the cap and
                  base on pixel rows and the stem's left edge on a pixel column; the stem weight
                  is then snapped to whole pixels (<= 0.6 %, what TrueType hinting does) with arm
                  and leg kept at 96.5 % of it. Cap 13 / 26 / 39 px, stem 3 / 6 / 9 px.
  * favicon.svg - the 16 px grid fit on a 16-unit grid, so it is crisp at 16, 32, 48 and 64 px,
                  with a trap of one 16-grid pixel so browser anti-aliasing shows no facet seams.
                  No prefers-color-scheme rule: the K's large facets measure 3.3-4.2:1 on a white
                  tab and 2.9-4.9:1 on dark tab strips (#202124 ... #35363A) - it reads the same on
                  both, and the only possible dark rule would recolour or outline the facets.
  * Safe zones  - maskable: K within 0.35 of the side (safe circle 0.40); Android: K within the
                  66 dp circle of the 108 dp canvas; social: see SOCIAL below.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from PIL import Image, ImageChops, ImageDraw, ImageFont  # noqa: E402
from PIL.PngImagePlugin import PngInfo  # noqa: E402
from shapely.geometry import LineString, Polygon, box  # noqa: E402
from shapely.ops import polylabel  # noqa: E402

import brand  # noqa: E402
import kmark  # noqa: E402
from brand import FACET_LABEL, NEUTRAL, PALETTE, TILE_R  # noqa: E402
from kmark import FACETS, PAINT_ORDER, fmt, poly_d, svg_doc  # noqa: E402
from okcolor import contrast, mix, tone  # noqa: E402

LOGO_SIZES = (256, 512, 1024, 2048, 4096)
SMALL = 64                      # at or below this many px the tech icon uses detail="small"
FG_SCALE = 0.78                 # brand.doc_android_foreground's scale (K inside the 66 dp circle)
MASKABLE_SCALE = 0.90           # PWA maskable icon: K reaches 0.35 of the side, safe circle is 0.40
DENSITIES = {"mdpi": 1.0, "hdpi": 1.5, "xhdpi": 2.0, "xxhdpi": 3.0, "xxxhdpi": 4.0}

# raster variants of the editable masters (names follow build.VARIANTS); flat marks are trapped
LOGO_PNG = {
    "K-Logo_Mark_Color": lambda: brand.doc_mark(raster=True),
    "K-Logo_Mark_Color_Tight": lambda: brand.doc_mark(raster=True, artboard="tight"),
    "K-Logo_Badge_Color": lambda: brand.doc_badge(raster=True),
    "K-Logo_Mark_Dimensional": lambda: brand.doc_mark(style="dimensional"),
    "K-Logo_Badge_Dimensional": lambda: brand.doc_badge(style="dimensional"),
    "K-Logo_Mark_Black": lambda: brand.doc_mark(style="black-gapped", raster=True),
    "K-Logo_Mark_White": lambda: brand.doc_mark(style="white-gapped", raster=True),
    "K-Logo_Mark_Greyscale": lambda: brand.doc_mark(style="grey", raster=True),
    "K-Logo_Tech_Icon": lambda: brand.doc_tech(),
}

FONT_DIR = "/usr/share/fonts/opentype/inter"


# ------------------------------------------------------------------ rendering + writing
def px(svg: str, w: int, h: int = None) -> Image.Image:
    """brand.render at >= 1024 px, box-reduced in premultiplied alpha to w x h (see docstring)."""
    h = h or w
    f = max(1, math.ceil(1024 / max(w, h)))
    im = brand.render(svg, w * f, h * f)
    return im if f == 1 else im.convert("RGBa").reduce(f).convert("RGBA")


def on(im: Image.Image, colour) -> Image.Image:
    """Flatten onto an opaque colour (RGB, no alpha channel)."""
    bg = Image.new("RGBA", im.size, colour)
    bg.alpha_composite(im)
    return bg.convert("RGB")


def save_png(im: Image.Image, path: str, opaque: bool = False) -> str:
    """PNG, optimize=True, sRGB chunk (perceptual intent) and 72 dpi. opaque=True writes RGB and
    refuses an image that is not fully opaque."""
    if opaque and im.mode == "RGBA":
        assert im.getchannel("A").getextrema() == (255, 255), f"{path} is not opaque"
        im = im.convert("RGB")
    info = PngInfo()
    info.add(b"sRGB", b"\x00")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path, optimize=True, pnginfo=info, dpi=(72, 72))
    return path


def save_text(text: str, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path


def tech(n: int, shape="tile", k=None) -> str:
    """Tech icon document for an n px render: the simplified detail at <= 64 px."""
    return brand.doc_tech(k, detail="small" if n <= SMALL else "full", shape=shape)


def trapped_group(k, t: float, colours=None, gid="K") -> str:
    """brand.mark_group(raster=True) with an explicit trap width t (design units)."""
    colours = colours or {n: k.palette[n] for n in FACETS}
    body = "".join(f'<path id="{gid}-{n}" data-name="{FACET_LABEL[n]}" fill="{colours[n]}" d="{d}"/>'
                   for n, d in brand.trapped(k, t=t))
    return f'<g id="{gid}" data-name="K mark">{body}</g>'


def solid_d(k) -> str:
    """The four facets merged into one outline (one-colour solid mark, no internal edges)."""
    return poly_d(kmark.silhouette(k))


# ------------------------------------------------------------------ 1 logo PNGs
def stage_logo_png(root):
    out = []
    for name, fn in LOGO_PNG.items():
        svg = fn()
        for n in LOGO_SIZES:
            if name.endswith("_Tight"):
                x0, y0, x1, y1 = kmark.bbox(brand.master())
                im = px(svg, n, round(n * (y1 - y0) / (x1 - x0)))   # width = n, aspect kept (4:5)
            else:
                im = px(svg, n)
            out.append(save_png(im, os.path.join(root, "logo", "png", f"{name}_{n}.png")))
    return out


# ------------------------------------------------------------------ 2 web
def grid_fit(n: int):
    """The bare K grid-fitted to an n px square. Returns (KParams, layout in px)."""
    k = brand.master()
    H = k.yB - k.yT
    x0, y0, x1, y1 = kmark.bbox(k)
    cap = round(n * 13 / 16)                                  # 13 / 26 / 39 px
    stem = round(cap * k.w / H)                               # 3 / 6 / 9 px
    ox = ((x0 + x1) / 2 - k.size / 2) / k.size                # the master's optical offset (+14, -6)
    oy = ((y0 + y1) / 2 - k.size / 2) / k.size
    left = round((n - cap * (x1 - x0) / H) / 2 + ox * n)
    top = round((n - cap) / 2 + oy * n)
    u = k.size / n                                            # design units per pixel
    s = cap * u / H
    k = brand.transform(k, s, 0, 0, left * u - k.x0 * s, top * u - k.yT * s)
    f = stem * u / k.w                                        # hint: whole-pixel stem
    k.w *= f
    k.arm_w *= f
    k.leg_w *= f
    return k, {"cap": cap, "stem": stem, "left": left, "top": top}


def favicon_svg() -> str:
    k, _ = grid_fit(16)
    return svg_doc(trapped_group(k, t=k.size / 16), size=k.size, title=f"{brand.NAME} - favicon")


def favicon(n: int) -> Image.Image:
    k, _ = grid_fit(n)
    return px(brand.doc_mark(k, raster=True), n)


def safari_svg() -> str:
    k = brand.master()
    x0, y0, x1, y1 = kmark.bbox(k)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{fmt(x0)} {fmt(y0)} {fmt(x1 - x0)} {fmt(y1 - y0)}">'
            f'<title>{brand.NAME} - Safari pinned tab</title><path fill="#000000" d="{solid_d(k)}"/></svg>')


MASK_COLOUR = NEUTRAL["ink"]          # one-colour use on screen (brand rule)
THEME_COLOUR = NEUTRAL["night"]       # matches the app tile, so splash / title bar and icon agree


def manifest() -> str:
    icons = [{"src": "android-chrome-192x192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
             {"src": "android-chrome-512x512.png", "sizes": "512x512", "type": "image/png", "purpose": "any"},
             {"src": "icon-maskable-512x512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"}]
    return json.dumps({"name": "K", "short_name": "K", "icons": icons, "start_url": "/", "display": "standalone",
                       "theme_color": THEME_COLOUR, "background_color": THEME_COLOUR}, indent=2) + "\n"


def head_snippet() -> str:
    return "\n".join([
        "<!-- K logo: favicons and app icons. Serve the files of icons/web/ from the site root. -->",
        '<link rel="icon" href="/favicon.ico" sizes="16x16 32x32 48x48">',
        '<link rel="icon" href="/favicon.svg" type="image/svg+xml">',
        '<link rel="icon" href="/favicon-48x48.png" type="image/png" sizes="48x48">',
        '<link rel="icon" href="/favicon-32x32.png" type="image/png" sizes="32x32">',
        '<link rel="icon" href="/favicon-16x16.png" type="image/png" sizes="16x16">',
        '<link rel="apple-touch-icon" href="/apple-touch-icon.png" sizes="180x180">',
        f'<link rel="mask-icon" href="/safari-pinned-tab.svg" color="{MASK_COLOUR}">',
        '<link rel="manifest" href="/site.webmanifest">',
        f'<meta name="theme-color" content="{THEME_COLOUR}">', ""])


def stage_web(root):
    d = os.path.join(root, "icons", "web")
    out = []
    favs = {n: favicon(n) for n in (16, 32, 48)}
    for n, im in favs.items():
        out.append(save_png(im, os.path.join(d, f"favicon-{n}x{n}.png")))
    ico = os.path.join(d, "favicon.ico")
    favs[48].save(ico, format="ICO", sizes=[(16, 16), (32, 32), (48, 48)], append_images=[favs[16], favs[32]])
    out.append(ico)
    out.append(save_text(favicon_svg(), os.path.join(d, "favicon.svg")))
    square = brand.doc_tech(shape="square")
    out.append(save_png(px(square, 180), os.path.join(d, "apple-touch-icon.png"), opaque=True))
    for n in (192, 512):
        out.append(save_png(px(tech(n), n), os.path.join(d, f"android-chrome-{n}x{n}.png")))
    k = brand.scaled_about_centre(brand.master(), MASKABLE_SCALE)
    out.append(save_png(px(brand.doc_tech(k, shape="square"), 512), os.path.join(d, "icon-maskable-512x512.png"), opaque=True))
    out.append(save_text(safari_svg(), os.path.join(d, "safari-pinned-tab.svg")))
    out.append(save_text(manifest(), os.path.join(d, "site.webmanifest")))
    out.append(save_text(head_snippet(), os.path.join(d, "head-snippet.html")))
    return out


# ------------------------------------------------------------------ 3 iOS
def stage_ios(root):
    d = os.path.join(root, "icons", "ios", "AppIcon.appiconset")
    out = [save_png(px(brand.doc_tech(shape="square"), 1024), os.path.join(d, "AppIcon-1024.png"), opaque=True)]
    contents = {"images": [{"filename": "AppIcon-1024.png", "idiom": "universal", "platform": "ios", "size": "1024x1024"}],
                "info": {"author": "xcode", "version": 1}}
    out.append(save_text(json.dumps(contents, indent=2) + "\n", os.path.join(d, "Contents.json")))
    return out


# ------------------------------------------------------------------ 4 macOS
ICONSET = (  # (icns type, file, px); the order iconutil writes
    ("icp4", "icon_16x16.png", 16), ("ic11", "icon_16x16@2x.png", 32),
    ("icp5", "icon_32x32.png", 32), ("ic12", "icon_32x32@2x.png", 64),
    ("ic07", "icon_128x128.png", 128), ("ic13", "icon_128x128@2x.png", 256),
    ("ic08", "icon_256x256.png", 256), ("ic14", "icon_256x256@2x.png", 512),
    ("ic09", "icon_512x512.png", 512), ("ic10", "icon_512x512@2x.png", 1024),
)


def write_icns(path, entries):
    """ICNS container of PNG payloads [(type, png bytes)]. Pillow's own writer omits the 16 and 32
    px @1x slots (it would scale the 32 / 64 px art), so the container is written here."""
    blocks = [t.encode() + struct.pack(">I", 8 + len(b)) + b for t, b in entries]
    toc = b"TOC " + struct.pack(">I", 8 + 8 * len(entries)) + b"".join(blk[:8] for blk in blocks)
    data = toc + b"".join(blocks)
    with open(path, "wb") as f:
        f.write(b"icns" + struct.pack(">I", 8 + len(data)) + data)
    return path


def stage_macos(root):
    d = os.path.join(root, "icons", "macos")
    out = [save_png(px(brand.doc_tech(shape="macos"), 1024), os.path.join(d, "K-Logo_macOS_1024.png"))]
    entries = []
    for t, name, n in ICONSET:
        p = save_png(px(tech(n, "macos"), n), os.path.join(d, "AppIcon.iconset", name))
        out.append(p)
        with open(p, "rb") as f:
            entries.append((t, f.read()))
    out.append(write_icns(os.path.join(d, "K-Logo.icns"), entries))
    return out


# ------------------------------------------------------------------ 5 Android
def android_fg_k():
    return brand.scaled_about_centre(brand.master(), FG_SCALE)


def android_fg_svg() -> str:
    """brand.doc_android_foreground with trapped facets for raster output."""
    k = android_fg_k()
    return svg_doc(brand.mark_group(k, raster=True), size=k.size, title=f"{brand.NAME} - Android adaptive foreground")


def android_mono_svg() -> str:
    """Themed-icon layer (Android 13+): the solid one-colour K at the foreground's scale. Only the
    alpha is used by the system, which tints it."""
    k = android_fg_k()
    return svg_doc(f'<path id="K" data-name="K mark - solid" fill="{NEUTRAL["white"]}" d="{solid_d(k)}"/>',
                   size=k.size, title=f"{brand.NAME} - Android monochrome")


def _inner(svg: str) -> str:
    """Contents of an SVG document (drops the root element)."""
    return re.sub(r"^<svg[^>]*>|</svg>$", "", svg)


def mask_d(shape: str, x: float, y: float, s: float) -> str:
    """Launcher / icon masks on an s-square at (x, y): circle | squircle | rounded | tile."""
    if shape == "circle":
        return brand._circle_d((x + s / 2, y + s / 2), s / 2)
    if shape == "squircle":                                     # superellipse, n = 4
        pts = []
        for i in range(256):
            a = 2 * math.pi * i / 256
            c, sn = math.cos(a), math.sin(a)
            pts.append((x + s / 2 + s / 2 * math.copysign(abs(c) ** 0.5, c), y + s / 2 + s / 2 * math.copysign(abs(sn) ** 0.5, sn)))
        return poly_d(pts)
    r = {"rounded": 0.16, "tile": TILE_R}[shape]
    return brand._rrect_d(x, y, s, s, r * s)


def adaptive_svg(shape: str, inset: float = 0.0, mask_inset: float = None, layers=("bg", "fg"),
                 size: float = 1024.0, tint=None) -> str:
    """The adaptive icon as a launcher draws it: the 72 dp viewport of the 108 dp layers, masked.
    inset: margin around the viewport, mask_inset: margin around the mask (fractions of the
    canvas; legacy icons use 2 / 48 dp and a 40 dp tile). tint=(plate, ink) draws the themed
    icon: the monochrome layer's alpha in `ink` on a `plate`, as Android 13 does."""
    c = 1024.0
    vb = (c / 6, c / 6, c * 2 / 3, c * 2 / 3)                 # 72 dp of 108 dp
    m = inset * size
    mm = m if mask_inset is None else mask_inset * size
    if tint:
        plate, ink = tint
        layers_svg = (f'<rect width="{fmt(c)}" height="{fmt(c)}" fill="{plate}"/>'
                      + _inner(android_mono_svg()).replace(f'fill="{NEUTRAL["white"]}"', f'fill="{ink}"'))
    else:
        parts = {"bg": _inner(brand.doc_android_background()), "fg": _inner(android_fg_svg())}
        layers_svg = "".join(parts[n] for n in layers)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {fmt(size)} {fmt(size)}" width="{fmt(size)}" height="{fmt(size)}">'
            f'<defs><clipPath id="launcher-mask"><path d="{mask_d(shape, mm, mm, size - 2 * mm)}"/></clipPath></defs>'
            f'<g clip-path="url(#launcher-mask)"><svg x="{fmt(m)}" y="{fmt(m)}" width="{fmt(size - 2 * m)}" height="{fmt(size - 2 * m)}" '
            f'viewBox="{" ".join(fmt(v) for v in vb)}">{layers_svg}</svg></g></svg>')


ANDROID_XML = """<?xml version="1.0" encoding="utf-8"?>
<!-- K logo: adaptive launcher icon (API 26+). <monochrome> is the Android 13+ themed icon. -->
<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">
    <background android:drawable="@mipmap/ic_launcher_background" />
    <foreground android:drawable="@mipmap/ic_launcher_foreground" />
    <monochrome android:drawable="@mipmap/ic_launcher_monochrome" />
</adaptive-icon>
"""


def stage_android(root):
    d = os.path.join(root, "icons", "android")
    res = os.path.join(d, "res")
    out = [save_text(ANDROID_XML, os.path.join(res, "mipmap-anydpi-v26", f"{n}.xml")) for n in ("ic_launcher", "ic_launcher_round")]
    fg, mono, bg = android_fg_svg(), android_mono_svg(), brand.doc_android_background()
    # legacy (API < 26): the adaptive icon pre-masked on Material keylines - the 72 dp viewport
    # drawn at 44 dp of 48 dp, masked by a 44 dp circle or a 40 dp brand tile (equal visual area)
    legacy = {"ic_launcher": adaptive_svg("tile", inset=2 / 48, mask_inset=4 / 48),
              "ic_launcher_round": adaptive_svg("circle", inset=2 / 48)}
    for dens, sc in DENSITIES.items():
        m = os.path.join(res, f"mipmap-{dens}")
        n = round(108 * sc)
        out.append(save_png(px(fg, n), os.path.join(m, "ic_launcher_foreground.png")))
        out.append(save_png(px(bg, n), os.path.join(m, "ic_launcher_background.png"), opaque=True))
        out.append(save_png(px(mono, n), os.path.join(m, "ic_launcher_monochrome.png")))
        for name, svg in legacy.items():
            out.append(save_png(px(svg, round(48 * sc)), os.path.join(m, f"{name}.png")))
    out.append(save_png(px(brand.doc_tech(shape="square"), 512), os.path.join(d, "play-store-512.png"), opaque=True))
    return out


# ------------------------------------------------------------------ 6 social
# Canvas, where the K sits (centre, cap height in px) and the zones it must keep clear of or stay
# inside. Avatar zones are measured from each platform's desktop and mobile profile layouts
# (avatar circle + border, scaled to the upload size) and rounded up.
SOCIAL = {
    "og-image-1200x630": {"size": (1200, 630), "centre": (600, 315), "cap": 250,
                          "inside": [("centre square crop (chat apps)", (285, 0, 915, 630))], "avoid": []},
    "x-header-1500x500": {"size": (1500, 500), "centre": (1030, 250), "cap": 220,
                          "inside": [("mobile top / bottom crop", (0, 60, 1500, 440))],
                          "avoid": [("profile photo (bottom-left)", (0, 290, 440, 500)),
                                    ("mobile header buttons (top-right)", (1320, 0, 1500, 110))]},
    "linkedin-banner-1584x396": {"size": (1584, 396), "centre": (1150, 198), "cap": 180,
                                 "inside": [("mobile side crop", (100, 0, 1484, 396))],
                                 "avoid": [("profile photo (bottom-left)", (0, 150, 580, 396))]},
    "youtube-banner-2560x1440": {"size": (2560, 1440), "centre": (1280, 720), "cap": 260,
                                 "inside": [("safe area, all devices (1546 x 423)", (507, 508, 2053, 932))], "avoid": []},
}


def _tech_groups(k):
    """The tech icon's layers keyed by group id: tile, grid, ambient, traces, K, node."""
    defs, groups = brand._tech_layers(k, "full", "square")
    return "".join(defs), {re.match(r'<g id="([^"]+)"', g).group(1): g for g in groups}


def _to_edges(a, b, W, H):
    """The line through a and b, clipped to the canvas."""
    d = (b[0] - a[0], b[1] - a[1])
    n = math.hypot(*d)
    d = (d[0] / n * 4 * (W + H), d[1] / n * 4 * (W + H))
    g = LineString([(a[0] - d[0], a[1] - d[1]), (a[0] + d[0], a[1] + d[1])]).intersection(box(0, 0, W, H))
    return list(g.coords) if not g.is_empty else []


def scene_placement(spec):
    """(scale, dx, dy) mapping the 1024 design frame onto the canvas, plus the K's bounds there."""
    k = brand.master()
    f = spec["cap"] / (k.yB - k.yT)
    x0, y0, x1, y1 = kmark.bbox(k)
    dx = spec["centre"][0] - f * (x0 + x1) / 2
    dy = spec["centre"][1] - f * (y0 + y1) / 2
    return f, dx, dy


def scene_extent(spec):
    """Canvas bounds of everything the K brings along (letter, traces and pads, node bloom)."""
    k = brand.master()
    f, dx, dy = scene_placement(spec)
    p = kmark.points(k)
    u = (k.yB - k.yT) / 20.0
    pts = kmark.silhouette(k)
    for a, b, L in ((p["foldL"], p["P"], 2.4), (p["P"], p["J"], 2.2), (p["innerL"], p["J"], 2.4)):
        d = (a[0] - b[0], a[1] - b[1])
        n = math.hypot(*d)
        pts.append((a[0] + d[0] / n * (L + 0.1) * u, a[1] + d[1] / n * (L + 0.1) * u))
    xs = [dx + f * q[0] for q in pts]
    ys = [dy + f * q[1] for q in pts]
    return min(xs), min(ys), max(xs), max(ys)


def scene_svg(spec) -> str:
    """Night canvas, the construction grid and lines of the K running out into negative space,
    and the tech icon's lit K with its traces and node. No text."""
    W, H = spec["size"]
    k = brand.master()
    f, dx, dy = scene_placement(spec)
    m = lambda q: (dx + f * q[0], dy + f * q[1])                                  # noqa: E731
    p = {n: m(q) for n, q in kmark.points(k).items()}
    J = p["J"]
    cx, cy = spec["centre"]
    cap = spec["cap"]
    night = NEUTRAL["night"]
    lift = tone(night, dL=0.075, dC=0.012)
    glow = tone(PALETTE["blue"], dL=0.04, dC=-0.03)
    tdefs, layers = _tech_groups(k)
    reach = max(W, H) * 0.75
    defs = [
        # background: night, lifted around the mark and toward the top (the app tile's light)
        brand._lin("sc-bg", (0, 0), (0, H), [(0, tone(night, dL=0.03, dC=0.006), None), (1, night, None)]),
        f'<radialGradient id="sc-lift" gradientUnits="userSpaceOnUse" cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(reach)}" '
        f'gradientTransform="translate({fmt(cx)} {fmt(cy)}) scale(1 0.62) translate({fmt(-cx)} {fmt(-cy)})">'
        f'<stop offset="0" stop-color="{lift}" stop-opacity="0.9"/><stop offset="0.45" stop-color="{lift}" stop-opacity="0.25"/>'
        f'<stop offset="1" stop-color="{lift}" stop-opacity="0"/></radialGradient>',
        brand._rad("sc-glow", (cx, cy), 1.6 * cap, [(0, glow, 0.16), (0.5, glow, 0.05), (1, glow, 0.0)]),
        # grid and construction lines fade out from the mark (elliptical, wider than tall)
        f'<radialGradient id="sc-fade" gradientUnits="userSpaceOnUse" cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(2.6 * cap)}" '
        f'gradientTransform="translate({fmt(cx)} {fmt(cy)}) scale(1.5 1) translate({fmt(-cx)} {fmt(-cy)})">'
        f'<stop offset="0" stop-color="#FFFFFF" stop-opacity="0.075"/><stop offset="0.55" stop-color="#FFFFFF" stop-opacity="0.035"/>'
        f'<stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/></radialGradient>',
        f'<radialGradient id="sc-lines" gradientUnits="userSpaceOnUse" cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(4.2 * cap)}" '
        f'gradientTransform="translate({fmt(cx)} {fmt(cy)}) scale(1.6 1) translate({fmt(-cx)} {fmt(-cy)})">'
        f'<stop offset="0" stop-color="#FFFFFF" stop-opacity="0.16"/><stop offset="0.4" stop-color="#FFFFFF" stop-opacity="0.07"/>'
        f'<stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/></radialGradient>',
    ]
    hair = max(1.0, W / 1600)
    # construction grid: the tech icon's module (half a stem) anchored on J, across the canvas
    g = f * k.w / 2
    d = []
    i0, i1 = math.floor(-J[0] / g), math.ceil((W - J[0]) / g)
    d += [f"M{fmt(J[0] + i * g)} 0V{fmt(H)}" for i in range(i0, i1 + 1)]
    i0, i1 = math.floor(-J[1] / g), math.ceil((H - J[1]) / g)
    d += [f"M0 {fmt(J[1] + i * g)}H{fmt(W)}" for i in range(i0, i1 + 1)]
    grid = f'<path id="grid" data-name="Construction grid" fill="none" stroke="url(#sc-fade)" stroke-width="{fmt(hair)}" d="{"".join(d)}"/>'
    # construction lines: the fold, the arm's inner edge, the leg's inner edge, cap and base lines
    runs = [(p["foldL"], p["P"]), (p["innerL"], J), (p["C"], p["legBL"]), (p["TL"], p["armTR"]), (p["BL"], p["legBR"])]
    d = []
    for a, b in runs:
        c = _to_edges(a, b, W, H)
        if c:
            d.append(f"M{fmt(c[0][0])} {fmt(c[0][1])}L{fmt(c[-1][0])} {fmt(c[-1][1])}")
    lines = (f'<path id="construction" data-name="Construction lines" fill="none" stroke="url(#sc-lines)" '
             f'stroke-width="{fmt(hair)}" d="{"".join(d)}"/>')
    mark = (f'<g id="mark" data-name="Tech mark" transform="translate({fmt(dx)} {fmt(dy)}) scale({fmt(f)})">'
            + layers["ambient"] + layers["traces"] + layers["K"] + layers["node"] + "</g>")
    body = (f'<rect id="background" width="{W}" height="{H}" fill="url(#sc-bg)"/>'
            f'<rect id="lift" width="{W}" height="{H}" fill="url(#sc-lift)"/>'
            f'<rect id="glow" width="{W}" height="{H}" fill="url(#sc-glow)"/>' + grid + lines + mark)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" '
            f'width="{W}" height="{H}"><title>{brand.NAME} - social banner</title><defs>{"".join(defs)}{tdefs}</defs>{body}</svg>')


def avatar(n: int) -> Image.Image:
    """The badge, opaque: the disc's white runs to the corners, so any circular crop keeps it."""
    return on(px(brand.doc_badge(raster=True), n), NEUTRAL["white"])


def stage_social(root):
    d = os.path.join(root, "social")
    out = [save_png(avatar(n), os.path.join(d, f"avatar-{n}.png")) for n in (1080, 400)]
    for name, spec in SOCIAL.items():
        W, H = spec["size"]
        out.append(save_png(px(scene_svg(spec), W, H), os.path.join(d, f"{name}.png"), opaque=True))
    return out
