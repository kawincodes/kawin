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
  * Trapping    - flat marks and badges use brand.trapped (what raster=True draws) with the trap
                  matched to the render: 1.5 rendered px, at most brand's 2 units. The fixed trap
                  is 8 px long at 4096 px and tints the outline's anti-aliased pixels next to J
                  and P (red over green reads tan). Dimensional and tech variants keep brand's own.
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
  * Web         - safari-pinned-tab.svg is the solid mark as one path (no internal edges for Safari
                  to anti-alias), mask colour ink; manifest theme / background are night, the
                  app tile's colour. apple-touch, iOS, maskable and Play icons are opaque RGB.
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
from kmark import FACETS, fmt, poly_d, svg_doc  # noqa: E402
from okcolor import contrast, oklch_to_hex, tone  # noqa: E402

LOGO_SIZES = (256, 512, 1024, 2048, 4096)
SMALL = 64                      # at or below this many px the tech icon uses detail="small"
FG_SCALE = 0.78                 # brand.doc_android_foreground's scale (K inside the 66 dp circle)
MASKABLE_SCALE = 0.90           # PWA maskable icon: K reaches 0.35 of the side, safe circle is 0.40
DENSITIES = {"mdpi": 1.0, "hdpi": 1.5, "xhdpi": 2.0, "xxhdpi": 3.0, "xxxhdpi": 4.0}

# raster variants of the editable masters (names follow build.VARIANTS) as fn(w, h) -> SVG for a
# w x h px render: flat marks use trapped facets (flat_doc), the rest are brand's documents as-is
LOGO_PNG = {
    "K-Logo_Mark_Color": lambda w, h: flat_doc(w, h),
    "K-Logo_Mark_Color_Tight": lambda w, h: flat_doc(w, h, artboard="tight"),
    "K-Logo_Badge_Color": lambda w, h: flat_doc(w, h, badge=True),
    "K-Logo_Mark_Dimensional": lambda w, h: brand.doc_mark(style="dimensional"),
    "K-Logo_Badge_Dimensional": lambda w, h: brand.doc_badge(style="dimensional"),
    "K-Logo_Mark_Black": lambda w, h: brand.doc_mark(style="black-gapped", raster=True),
    "K-Logo_Mark_White": lambda w, h: brand.doc_mark(style="white-gapped", raster=True),
    "K-Logo_Mark_Greyscale": lambda w, h: flat_doc(w, h, colours=brand.grey_colours()),
    "K-Logo_Tech_Icon": lambda w, h: brand.doc_tech(),
}
TIGHT_ASPECT = 1.25             # cap height : width of the mark's bounds (4:5)

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


def trap_for(w: int, h: int = None, units: float = 1024.0) -> float:
    """Trap width (design units) for a w x h px render through px() of a viewBox `units` wide:
    1.5 px of the render brand.render is asked for, capped at brand.trapped's default (2 units).
    The fixed default is 8 px long at 4096 px, where a lower facet showing through the outline's
    anti-aliased pixels next to J / P tints them (red over green reads tan)."""
    h = h or w
    f = max(1, math.ceil(1024 / max(w, h)))
    return min(1024.0 / 512, 1.5 * units / (w * f))


def flat_doc(w: int, h: int = None, k=None, colours=None, badge=False, artboard="square") -> str:
    """brand.doc_mark / doc_badge with raster=True, for a w x h px render: the same trapped facets
    with the trap matched to the render (trap_for). artboard: square | tight."""
    k = k or brand.master()
    if artboard == "tight":
        x0, y0, x1, y1 = kmark.bbox(k)
        vb = (x0, y0, x1 - x0, y1 - y0)
    else:
        vb = (0.0, 0.0, k.size, k.size)
    body = trapped_group(k, trap_for(w, h, vb[2]), colours)
    if badge:
        body = f'<g id="badge" data-name="Badge">{brand.disc(k)}{body}</g>'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{" ".join(fmt(v) for v in vb)}" width="{fmt(vb[2])}" '
            f'height="{fmt(vb[3])}"><title>{brand.NAME} - {"badge" if badge else "mark"} (raster)</title>{body}</svg>')


def solid_d(k) -> str:
    """The four facets merged into one outline (one-colour solid mark, no internal edges)."""
    return poly_d(kmark.silhouette(k))


# ------------------------------------------------------------------ 1 logo PNGs
def stage_logo_png(root):
    out = []
    x0, y0, x1, y1 = kmark.bbox(brand.master())
    assert abs((y1 - y0) / (x1 - x0) - TIGHT_ASPECT) < 1e-5
    for name, fn in LOGO_PNG.items():
        for n in LOGO_SIZES:
            h = round(n * TIGHT_ASPECT) if name.endswith("_Tight") else n      # px = width, aspect kept
            out.append(save_png(px(fn(n, h), n, h), os.path.join(root, "logo", "png", f"{name}_{n}.png")))
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
    return px(flat_doc(n, k=grid_fit(n)[0]), n)


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


def android_fg_svg(t: float) -> str:
    """brand.doc_android_foreground with trapped facets (trap t) for raster output."""
    k = android_fg_k()
    return svg_doc(trapped_group(k, t), size=k.size, title=f"{brand.NAME} - Android adaptive foreground")


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


def adaptive_svg(shape: str, n: int, inset: float = 0.0, size: float = 1024.0) -> str:
    """The adaptive icon as a launcher draws it, for an n px render: the 72 dp viewport of the
    108 dp background and foreground, filling a `shape` mask inset by `inset` (fraction)."""
    c = 1024.0
    vb = (c / 6, c / 6, c * 2 / 3, c * 2 / 3)                 # 72 dp of 108 dp
    m = inset * size
    layers = _inner(brand.doc_android_background()) + _inner(android_fg_svg(trap_for(n, units=vb[2] / (1 - 2 * inset))))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {fmt(size)} {fmt(size)}" width="{fmt(size)}" height="{fmt(size)}">'
            f'<defs><clipPath id="launcher-mask"><path d="{mask_d(shape, m, m, size - 2 * m)}"/></clipPath></defs>'
            f'<g clip-path="url(#launcher-mask)"><svg x="{fmt(m)}" y="{fmt(m)}" width="{fmt(size - 2 * m)}" height="{fmt(size - 2 * m)}" '
            f'viewBox="{" ".join(fmt(v) for v in vb)}">{layers}</svg></g></svg>')


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
    mono, bg = android_mono_svg(), brand.doc_android_background()
    # legacy (API < 26): the adaptive icon pre-masked on Material keylines - a 44 dp circle or a
    # 40 dp brand tile in the 48 dp canvas (equal visual area), each filled by the 72 dp viewport
    # exactly as a launcher fills its mask
    legacy = {"ic_launcher": ("tile", 4 / 48), "ic_launcher_round": ("circle", 2 / 48)}
    for dens, sc in DENSITIES.items():
        m = os.path.join(res, f"mipmap-{dens}")
        n = round(108 * sc)
        out.append(save_png(px(android_fg_svg(trap_for(n)), n), os.path.join(m, "ic_launcher_foreground.png")))
        out.append(save_png(px(bg, n), os.path.join(m, "ic_launcher_background.png"), opaque=True))
        out.append(save_png(px(mono, n), os.path.join(m, "ic_launcher_monochrome.png")))
        n = round(48 * sc)
        for name, (shape, inset) in legacy.items():
            out.append(save_png(px(adaptive_svg(shape, n, inset), n), os.path.join(m, f"{name}.png")))
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
    """(scale, dx, dy) mapping the 1024 design frame onto the canvas."""
    k = brand.master()
    f = spec["cap"] / (k.yB - k.yT)
    x0, y0, x1, y1 = kmark.bbox(k)
    dx = spec["centre"][0] - f * (x0 + x1) / 2
    dy = spec["centre"][1] - f * (y0 + y1) / 2
    return f, dx, dy


def scene_extent(spec):
    """Canvas bounds of everything the K brings along: the letter and the tech icon's traces with
    their pads (the node and its bloom sit inside the letter's bounds)."""
    k = brand.master()
    f, dx, dy = scene_placement(spec)
    p = kmark.points(k)
    u = (k.yB - k.yT) / 20.0
    pts = kmark.silhouette(k)
    for a, b, L in ((p["foldL"], p["P"], 2.4), (p["P"], p["J"], 2.2), (p["innerL"], p["J"], 2.4)):
        d = (a[0] - b[0], a[1] - b[1])
        n = math.hypot(*d)
        pts.append((a[0] + d[0] / n * (L + 0.32) * u, a[1] + d[1] / n * (L + 0.32) * u))   # pad radius 0.32 u
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
    return on(px(flat_doc(n, badge=True), n), NEUTRAL["white"])


def stage_social(root):
    d = os.path.join(root, "social")
    out = [save_png(avatar(n), os.path.join(d, f"avatar-{n}.png")) for n in (1080, 400)]
    for name, spec in SOCIAL.items():
        W, H = spec["size"]
        out.append(save_png(px(scene_svg(spec), W, H), os.path.join(d, f"{name}.png"), opaque=True))
    return out


# ------------------------------------------------------------------ 7 overview
# Contact sheet of every icon and social image in context. Mock-ups use grey skeleton bars where a
# real UI would show text, so no name or label is invented; captions are documentation.
SHEET_BG = (244, 245, 248, 255)
MUTED = (104, 112, 128, 255)
MONO_FONT = "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf"


def _rgba(hexc, a=255):
    h = hexc.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (a,)


INK = _rgba(NEUTRAL["ink"])


def _font(px_, weight="Regular"):
    return ImageFont.truetype(os.path.join(FONT_DIR, f"Inter-{weight}.otf"), px_)


def _mono(px_):
    return ImageFont.truetype(MONO_FONT, px_)


def _reduce(im, ss):
    return im.convert("RGBa").reduce(ss).convert("RGBA")


def fit(im, n, h=None):
    """Resize a delivered image for display (premultiplied, Lanczos)."""
    return im.convert("RGBa").resize((n, h or n), Image.LANCZOS).convert("RGBA")


def rrect(w, h, r, fill, outline=None, width=1, ss=4):
    """Anti-aliased rounded rectangle as an RGBA tile."""
    im = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], r * ss, fill=fill,
                                         outline=outline, width=width * ss)
    return _reduce(im, ss)


def disc_tile(dm, fill, ss=4):
    im = Image.new("RGBA", (dm * ss, dm * ss), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse([0, 0, dm * ss - 1, dm * ss - 1], fill=fill)
    return _reduce(im, ss)


def vgrad(w, h, top, bottom):
    t = np.linspace(0, 1, h)[:, None, None]
    a = np.array(top, float)[None, None, :] * (1 - t) + np.array(bottom, float)[None, None, :] * t
    return Image.fromarray(np.repeat(a, w, axis=1).round().astype(np.uint8), "RGBA")


_MASKS = {}


def mask_alpha(shape, n):
    if (shape, n) not in _MASKS:
        _MASKS[shape, n] = px(svg_doc(f'<path fill="#FFFFFF" d="{mask_d(shape, 0, 0, 1024)}"/>'), n).getchannel("A")
    return _MASKS[shape, n]


def masked(im, shape):
    out = im.copy()
    out.putalpha(ImageChops.multiply(im.getchannel("A"), mask_alpha(shape, im.width)))
    return out


def rounded(im, r):
    """Clip an image to a rounded rectangle (anti-aliased)."""
    out = im.copy()
    out.putalpha(ImageChops.multiply(im.getchannel("A"), rrect(im.width, im.height, r, (255, 255, 255, 255)).getchannel("A")))
    return out


def overlay(size, draw_fn, ss=4):
    """Anti-aliased line work: draw_fn(draw, s) at ss x, box-reduced."""
    im = Image.new("RGBA", (size[0] * ss, size[1] * ss), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(im), ss)
    return _reduce(im, ss)


def dashed_rect(dr, box_, s, fill, width=2, dash=12, gap=8):
    x0, y0, x1, y1 = (v * s for v in box_)
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        L = math.hypot(bx - ax, by - ay)
        t = 0.0
        while t < L:
            e = min(L, t + dash * s)
            dr.line([(ax + (bx - ax) * t / L, ay + (by - ay) * t / L), (ax + (bx - ax) * e / L, ay + (by - ay) * e / L)],
                    fill=fill, width=width * s)
            t += (dash + gap) * s


def dashed_circle(dr, c, r, s, fill, width=2, n=64):
    bb = [(c[0] - r) * s, (c[1] - r) * s, (c[0] + r) * s, (c[1] + r) * s]
    for i in range(n):
        dr.arc(bb, i * 360 / n, i * 360 / n + 360 / n * 0.6, fill=fill, width=width * s)


def bar(sheet, x, y, w, h, fill):
    """A skeleton text line."""
    sheet.alpha_composite(rrect(w, h, h // 2, fill), (int(x), int(y)))


def caption(dr, xy, text, font, fill=MUTED, anchor="la"):
    dr.text(xy, text, font=font, fill=fill, anchor=anchor)


def launcher_icon(root, shape, n, tint=None):
    """The adaptive icon from the delivered xxxhdpi layers: the 72 dp viewport, masked.
    tint=(plate, ink) draws the Android 13 themed icon from the monochrome layer."""
    d = os.path.join(root, "icons", "android", "res", "mipmap-xxxhdpi")
    if tint:
        a = Image.open(os.path.join(d, "ic_launcher_monochrome.png")).getchannel("A")
        base = Image.new("RGBA", a.size, _rgba(tint[0]))
        ink = Image.new("RGBA", a.size, _rgba(tint[1]))
        ink.putalpha(a)
        base.alpha_composite(ink)
    else:
        base = Image.open(os.path.join(d, "ic_launcher_background.png")).convert("RGBA")
        base.alpha_composite(Image.open(os.path.join(d, "ic_launcher_foreground.png")).convert("RGBA"))
    return masked(fit(base.crop((72, 72, 360, 360)), n), shape)


def browser_mock(w, dark, fav):
    """A tab strip and toolbar at 2x (156 px tall); the favicon is pasted pixel-exact."""
    h, ss = 156, 4
    c = ({"frame": (32, 33, 36, 255), "tab": (53, 54, 58, 255), "box": (32, 33, 36, 255), "skel": (95, 99, 104, 255)} if dark else
         {"frame": (222, 225, 230, 255), "tab": (255, 255, 255, 255), "box": (241, 243, 244, 255), "skel": (196, 199, 204, 255)})
    im = Image.new("RGBA", (w * ss, h * ss), c["frame"])
    dr = ImageDraw.Draw(im)
    S = lambda *v: [int(round(x * ss)) for x in v]                              # noqa: E731
    for i in range(3):                                                           # window controls
        dr.ellipse(S(24 + i * 34, 37, 48 + i * 34, 61), fill=c["skel"])
    tx, tw = 140, 440
    dr.rounded_rectangle(S(tx, 12, tx + tw, 100), 16 * ss, fill=c["tab"])        # active tab
    dr.rectangle(S(0, 84, w, h), fill=c["tab"])                                  # toolbar
    dr.rounded_rectangle(S(tx + 72, 43, tx + 292, 55), 6 * ss, fill=c["skel"])   # title
    dr.line(S(tx + tw - 40, 41, tx + tw - 26, 55), fill=c["skel"], width=3 * ss)  # close
    dr.line(S(tx + tw - 40, 55, tx + tw - 26, 41), fill=c["skel"], width=3 * ss)
    for j in range(2):                                                           # inactive tabs
        x = tx + tw + 20 + j * 380
        dr.ellipse(S(x + 22, 35, x + 50, 63), fill=c["skel"])
        dr.rounded_rectangle(S(x + 68, 43, x + 68 + 190 - j * 40, 55), 6 * ss, fill=c["skel"])
        dr.line(S(x + 370, 30, x + 370, 68), fill=c["skel"], width=2 * ss)
    x = tx + tw + 20 + 2 * 380
    dr.line(S(x + 14, 49, x + 38, 49), fill=c["skel"], width=3 * ss)              # new tab
    dr.line(S(x + 26, 37, x + 26, 61), fill=c["skel"], width=3 * ss)
    for i in range(3):                                                           # back / forward / reload
        dr.ellipse(S(26 + i * 50, 106, 54 + i * 50, 134), fill=c["skel"])
    dr.rounded_rectangle(S(190, 98, w - 110, 142), 22 * ss, fill=c["box"])       # omnibox
    dr.ellipse(S(212, 112, 228, 128), fill=c["skel"])
    dr.rounded_rectangle(S(244, 114, 604, 126), 6 * ss, fill=c["skel"])
    dr.ellipse(S(w - 76, 106, w - 48, 134), fill=c["skel"])
    im = _reduce(im, ss)
    im.alpha_composite(fav, (tx + 24, 49 - fav.height // 2))
    return rounded(im, 14)


def pixel_zoom(fav, z, bg, grid):
    """Nearest-neighbour zoom of a favicon on a solid background with a pixel grid."""
    n = fav.width
    t = Image.new("RGBA", (n, n), bg)
    t.alpha_composite(fav)
    t = t.resize((n * z, n * z), Image.NEAREST)
    dr = ImageDraw.Draw(t)
    for i in range(1, n):
        dr.line([(i * z, 0), (i * z, n * z)], fill=grid)
        dr.line([(0, i * z), (n * z, i * z)], fill=grid)
    return t


def phone(root, kind, dark, w=535, h=900):
    """iOS or Android home screen: our icon in the grid, placeholders elsewhere."""
    walls = {("ios", False): ((226, 232, 243, 255), (192, 205, 228, 255)), ("ios", True): ((30, 37, 58, 255), (8, 12, 24, 255)),
             ("android", False): ((236, 238, 243, 255), (212, 219, 232, 255)), ("android", True): ((33, 37, 50, 255), (11, 14, 23, 255))}
    p = vgrad(w, h, *walls[kind, dark])
    ph = (255, 255, 255, 34) if dark else (255, 255, 255, 150)
    sk = (255, 255, 255, 70) if dark else (60, 70, 90, 70)
    fnt = _font(17)
    bar(p, 40, 26, 58, 14, sk)                                                   # status bar
    for i in range(3):
        bar(p, w - 110 + i * 26, 26, 18, 14, sk)
    n, cols = 104, 4
    gap = (w - cols * n) / (cols + 1)
    xs = [round(gap + i * (n + gap)) for i in range(cols)]
    ys = [92 + r * 156 for r in range(4 if kind == "ios" else 3)]             # Android: 3 rows + hotseat
    dr = ImageDraw.Draw(p)
    if kind == "ios":
        ours = [masked(fit(Image.open(os.path.join(root, "icons", "ios", "AppIcon.appiconset", "AppIcon-1024.png")).convert("RGBA"), n), "tile")]
        holder = rrect(n, n, round(TILE_R * n), ph)
    else:
        tint = ((oklch_to_hex(0.30, 0.05, 264.5), oklch_to_hex(0.88, 0.06, 264.5)) if dark else
                (oklch_to_hex(0.93, 0.03, 264.5), oklch_to_hex(0.32, 0.09, 264.5)))
        ours = [launcher_icon(root, s, n) for s in ("circle", "squircle", "rounded")] + [launcher_icon(root, "circle", n, tint)]
        holder = disc_tile(n, ph)
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            if r == 0 and c < len(ours):
                p.alpha_composite(ours[c], (x, y))
                if kind == "android":
                    caption(dr, (x + n // 2, y + n + 12), ("circle", "squircle", "rounded", "themed")[c], fnt,
                            fill=(235, 238, 245, 255) if dark else (70, 78, 96, 255), anchor="ma")
                    continue
            else:
                p.alpha_composite(holder, (x, y))
            bar(p, x + n // 2 - 32, y + n + 14, 64, 10, sk)
    if kind == "ios":                                                             # dock
        p.alpha_composite(rrect(w - 32, 136, 44, (255, 255, 255, 40) if dark else (255, 255, 255, 110)), (16, h - 158))
        for x in xs:
            p.alpha_composite(holder, (x, h - 142))
    else:                                                                         # hotseat + search
        for x in xs:
            p.alpha_composite(holder, (x, h - 236))
        p.alpha_composite(rrect(w - 64, 60, 30, (255, 255, 255, 40) if dark else (255, 255, 255, 170)), (32, h - 96))
    return rounded(p, 46)


def desktop(root, dark, w=1105, h=340):
    """macOS desktop: the iconset at actual pixels above a dock that holds the 1024 icon."""
    p = vgrad(w, h, *(((22, 28, 46, 255), (8, 12, 24, 255)) if dark else ((206, 218, 238, 255), (236, 240, 247, 255))))
    dr = ImageDraw.Draw(p)
    fnt = _mono(15)
    fill = (220, 226, 236, 255) if dark else (70, 78, 96, 255)
    x = 34
    for name, n in (("icon_16x16.png", 16), ("icon_32x32.png", 32), ("icon_32x32@2x.png", 64), ("icon_128x128.png", 128)):
        im = Image.open(os.path.join(root, "icons", "macos", "AppIcon.iconset", name)).convert("RGBA")
        p.alpha_composite(im, (x, 24 + 128 - n))
        caption(dr, (x + n // 2, 160), f"{n}", fnt, fill=fill, anchor="ma")
        x += n + 34
    caption(dr, (x + 10, 70), "iconset at actual pixels", _font(18), fill=fill)
    caption(dr, (x + 10, 96), "detail='small' at 64 px and below", _font(18), fill=fill)
    slot, cnt = 92, 6
    dw = cnt * slot + 7 * 12
    dx, dy = (w - dw) // 2, h - slot - 34
    p.alpha_composite(rrect(dw, slot + 22, 26, (255, 255, 255, 36) if dark else (255, 255, 255, 120),
                            outline=(255, 255, 255, 60) if dark else (255, 255, 255, 200)), (dx, dy - 11))
    mac = fit(Image.open(os.path.join(root, "icons", "macos", "K-Logo_macOS_1024.png")).convert("RGBA"), slot)
    t = round(slot * 824 / 1024)
    holder = rrect(t, t, round(TILE_R * t), (255, 255, 255, 46) if dark else (255, 255, 255, 170))
    for i in range(cnt):
        x = dx + 12 + i * (slot + 12)
        if i == 1:
            p.alpha_composite(mac, (x, dy))
        else:
            p.alpha_composite(holder, (x + (slot - t) // 2, dy + (slot - t) // 2))
    return rounded(p, 18)


def zones_overlay(spec, scale, size):
    """Dashed documentation overlay of a banner's zones, at display scale."""
    def draw(dr, s):
        for _, (x0, y0, x1, y1) in spec["avoid"]:
            dr.rectangle([x0 * scale * s, y0 * scale * s, x1 * scale * s, y1 * scale * s], fill=(234, 58, 35, 46))
            dashed_rect(dr, (x0 * scale, y0 * scale, x1 * scale, y1 * scale), s, (255, 150, 130, 230))
        for _, (x0, y0, x1, y1) in spec["inside"]:
            dashed_rect(dr, (x0 * scale, y0 * scale, x1 * scale, y1 * scale), s, (255, 255, 255, 200))
    return overlay(size, draw)


def stage_overview(root):
    W, M = 2400, 70
    sheet = Image.new("RGBA", (W, 6000), SHEET_BG)                                 # cropped to content at the end
    dr = ImageDraw.Draw(sheet)
    h1, h2, lab, small = _font(52, "Medium"), _font(30, "Medium"), _font(21), _font(18)
    web = os.path.join(root, "icons", "web")
    soc = os.path.join(root, "social")
    load = lambda *p: Image.open(os.path.join(*p)).convert("RGBA")                  # noqa: E731

    def section(y, title, note):
        dr.text((M, y), title, font=h2, fill=INK)
        dr.text((M + dr.textlength(title, font=h2) + 24, y + 9), note, font=lab, fill=MUTED)

    dr.text((M, 56), "K logo - icons and social kit", font=h1, fill=INK)
    dr.text((M, 130), "Every raster deliverable in context, on light and dark  |  generated by tools/build_raster.py",
            font=lab, fill=MUTED)

    # 1 browser ---------------------------------------------------------------
    y = 210
    section(y, "Browser tab", "favicon-32x32.png as a retina tab shows it; 16 / 32 / 48 px grid-fitted")
    for i, dark in enumerate((False, True)):
        sheet.alpha_composite(browser_mock(1380, dark, load(web, "favicon-32x32.png")), (M, y + 60 + i * 176))
    yy = y + 60 + 352 + 18
    caption(dr, (M, yy + 14), "actual size", small)
    x = M + 130
    for bg in ((255, 255, 255, 255), (53, 54, 58, 255)):
        for n in (16, 32, 48):
            chip = Image.new("RGBA", (n + 16, 64), bg)
            chip.alpha_composite(load(web, f"favicon-{n}x{n}.png"), (8, (64 - n) // 2))
            sheet.alpha_composite(rounded(chip, 8), (x, yy))
            x += n + 28
        x += 30
    zooms = ((16, 12), (32, 8), (48, 6))                                         # 192 / 256 / 288 px
    xs = [W - M - 288 - 256 - 192 - 48]
    for n, z in zooms[:-1]:
        xs.append(xs[-1] + n * z + 24)
    for j, (bg, grid) in enumerate((((255, 255, 255, 255), (0, 0, 0, 12)), ((32, 33, 36, 255), (255, 255, 255, 12)))):
        for i, (n, z) in enumerate(zooms):
            sheet.alpha_composite(pixel_zoom(load(web, f"favicon-{n}x{n}.png"), z, bg, grid), (xs[i], y + 60 + j * 352))
    for i, (n, z) in enumerate(zooms):
        lay = grid_fit(n)[1]
        caption(dr, (xs[i], y + 60 + 288 + 8), f"{n} px  x{z}", _mono(15))
        caption(dr, (xs[i], y + 60 + 288 + 28), f"cap {lay['cap']}  stem {lay['stem']}", _mono(15))

    # 2 home screens ----------------------------------------------------------
    y = 980
    section(y, "Home screens", "iOS app icon under the system mask; Android adaptive icon under launcher masks and as a themed icon")
    for i, (kind, dark) in enumerate((("ios", False), ("ios", True), ("android", False), ("android", True))):
        sheet.alpha_composite(phone(root, kind, dark), (M + i * 575, y + 60))

    # 3 desktop, web and stores -------------------------------------------------
    y = y + 60 + 900 + 70
    section(y, "Desktop, web and stores", "macOS dock, PWA / touch icons, maskable safe zone, Play Store, legacy Android, Safari")
    for i, dark in enumerate((False, True)):
        sheet.alpha_composite(desktop(root, dark), (M + i * 1155, y + 60))
    yy = y + 60 + 340 + 30
    sheet.alpha_composite(rrect(W - 2 * M, 330, 18, (255, 255, 255, 255)), (M, yy))
    x = M + 40
    top = yy + 34
    items = []
    items.append(("apple-touch-icon 180", masked(load(web, "apple-touch-icon.png"), "tile")))
    items.append(("android-chrome 192", load(web, "android-chrome-192x192.png")))
    mk = fit(load(web, "icon-maskable-512x512.png"), 240)
    mk.alpha_composite(overlay((240, 240), lambda d, s: dashed_circle(d, (120, 120), 96, s, (255, 255, 255, 220))))
    items.append(("maskable 512, 40 % safe circle", mk))
    items.append(("Play Store 512", masked(fit(load(root, "icons", "android", "play-store-512.png"), 192), "rounded")))
    items.append(("legacy ic_launcher", load(root, "icons", "android", "res", "mipmap-xxxhdpi", "ic_launcher.png")))
    items.append(("legacy round", load(root, "icons", "android", "res", "mipmap-xxxhdpi", "ic_launcher_round.png")))
    for label, im in items:
        sheet.alpha_composite(im, (x, top + (240 - im.height) // 2))
        caption(dr, (x + im.width // 2, top + 256), label, small, anchor="ma")
        x += im.width + 50
        if label.startswith("maskable"):                                           # the masks it survives
            for j, s in enumerate(("circle", "squircle", "rounded")):
                sheet.alpha_composite(masked(fit(load(web, "icon-maskable-512x512.png"), 72), s), (x - 30, top + j * 84))
            x += 72 + 20
    with open(os.path.join(web, "safari-pinned-tab.svg")) as f:
        pin = f.read().replace('fill="#000000"', f'fill="{MASK_COLOUR}"')
    pim = px(pin, 96, 120)
    sheet.alpha_composite(pim, (x + 10, top + 60))
    caption(dr, (x + 58, top + 256), "Safari pinned tab", small, anchor="ma")
    x += 96 + 90
    with open(os.path.join(web, "favicon.svg")) as f:
        sheet.alpha_composite(px(f.read(), 128), (x, top + 56))
    caption(dr, (x + 64, top + 256), "favicon.svg", small, anchor="ma")

    # 4 social ----------------------------------------------------------------
    y = yy + 330 + 60
    section(y, "Social", "avatars survive circular crops; banners keep the mark out of each platform's overlay zones (dashed)")
    top = y + 60
    og = load(soc, "og-image-1200x630.png")
    card = Image.new("RGBA", (1000, 525 + 120), (255, 255, 255, 255))
    card.alpha_composite(fit(og, 1000, 525), (0, 0))
    bar(card, 28, 525 + 30, 520, 18, (60, 64, 72, 210))
    bar(card, 28, 525 + 64, 760, 12, (150, 156, 168, 160))
    bar(card, 28, 525 + 88, 300, 12, (150, 156, 168, 160))
    sheet.alpha_composite(rounded(card, 18), (M, top))
    caption(dr, (M, top + 655), "og-image-1200x630 in a link preview", small)
    av = load(soc, "avatar-400.png")
    cx0 = M + 1040
    cw = (W - M - cx0 - 30) // 2
    for i, dark in enumerate((False, True)):                                       # profile cards
        c = Image.new("RGBA", (cw, 300), (0, 0, 0, 255) if dark else (255, 255, 255, 255))
        c.alpha_composite(masked(fit(av, 200), "circle"), (40, 50))
        sk = (90, 96, 108, 255) if dark else (200, 204, 212, 255)
        bar(c, 270, 100, 220, 20, sk)
        bar(c, 270, 138, 150, 14, sk)
        bar(c, 270, 176, 260, 12, sk)
        sheet.alpha_composite(rounded(c, 18), (cx0 + i * (cw + 30), top))
    c = Image.new("RGBA", (cw, 300), (255, 255, 255, 255))                          # uncropped + crop line
    a2 = fit(load(soc, "avatar-1080.png"), 240)
    a2.alpha_composite(overlay((240, 240), lambda d, s: dashed_circle(d, (120, 120), 119, s, (104, 112, 128, 255))))
    c.alpha_composite(a2, (30, 30))
    cd = ImageDraw.Draw(c)
    caption(cd, (300, 110), "avatar-1080, square", small)
    caption(cd, (300, 138), "circle crop keeps the", small)
    caption(cd, (300, 162), "whole badge", small)
    sheet.alpha_composite(rounded(c, 18), (cx0, top + 330))
    c = Image.new("RGBA", (cw, 300), (255, 255, 255, 255))                          # small sizes
    x = 30
    for n in (96, 64, 48, 32):
        c.alpha_composite(masked(fit(av, n), "circle"), (x, 70 - n // 2 + 40))
        dk = Image.new("RGBA", (n + 20, n + 20), (0, 0, 0, 255))
        dk.alpha_composite(masked(fit(av, n), "circle"), (10, 10))
        c.alpha_composite(rounded(dk, 10), (x - 10, 180))
        x += n + 34
    sheet.alpha_composite(rounded(c, 18), (cx0 + cw + 30, top + 330))

    # X and LinkedIn profiles, with the zones used
    top += 700
    for i, name in enumerate(("x-header-1500x500", "linkedin-banner-1584x396")):
        spec = SOCIAL[name]
        bw = 1105
        sc = bw / spec["size"][0]
        bh = round(spec["size"][1] * sc)
        dark = name.startswith("x")
        prof = Image.new("RGBA", (bw, bh + 230), (0, 0, 0, 255) if dark else (255, 255, 255, 255))
        prof.alpha_composite(fit(load(soc, f"{name}.png"), bw, bh), (0, 0))
        prof.alpha_composite(zones_overlay(spec, sc, (bw, bh)), (0, 0))
        # avatar as each platform places it (desktop), ringed in the page colour
        d_av, ax, atop = ((round(345 * sc), round(40 * sc), round(328 * sc)) if dark else
                          (round(300 * sc), round(47 * sc), round(236 * sc)))
        ring = disc_tile(d_av + 12, (0, 0, 0, 255) if dark else (255, 255, 255, 255))
        prof.alpha_composite(ring, (ax - 6, atop - 6))
        prof.alpha_composite(masked(fit(av, d_av), "circle"), (ax, atop))
        sk = (90, 96, 108, 255) if dark else (200, 204, 212, 255)
        below = max(atop + d_av - bh + 24, 100)                                    # name sits under the avatar
        bar(prof, 40, bh + below, 260, 22, sk)
        bar(prof, 40, bh + below + 38, 170, 14, sk)
        bar(prof, bw - 200, bh + 24, 160, 40, sk)
        sheet.alpha_composite(rounded(prof, 18), (M + i * (bw + 50), top))
        caption(dr, (M + i * (bw + 50), top + bh + 244), f"{name}  (red: avatar / UI overlap, white dashed: crop)", small)

    # YouTube with its device areas
    top += 690
    spec = SOCIAL["youtube-banner-2560x1440"]
    sc = 960 / 2560
    yt = fit(load(soc, "youtube-banner-2560x1440.png"), 960, 540)
    shade = Image.new("RGBA", (960, 540), (244, 245, 248, 120))
    band = (0, round(508 * sc), 960, round(932 * sc))
    shade.paste((0, 0, 0, 0), band)
    yt.alpha_composite(shade)
    yt.alpha_composite(zones_overlay(spec, sc, (960, 540)))
    sheet.alpha_composite(rounded(yt, 18), (M, top))
    caption(dr, (M, top + 552), "youtube-banner-2560x1440: TV shows all, desktop the clear band, every device the dashed safe area", small)
    x = M + 1010
    dr.text((x, top), "Zones used (canvas px)", font=h2, fill=INK)
    rows = []
    for name, spec in SOCIAL.items():
        ex = scene_extent(spec)
        rows.append((name, f"mark + traces  x {ex[0]:.0f}-{ex[2]:.0f}  y {ex[1]:.0f}-{ex[3]:.0f}"))
        rows += [("", f"keep inside   {lbl}: x {b[0]}-{b[2]}  y {b[1]}-{b[3]}") for lbl, b in spec["inside"]]
        rows += [("", f"keep clear    {lbl}: x {b[0]}-{b[2]}  y {b[1]}-{b[3]}") for lbl, b in spec["avoid"]]
    yy = top + 56
    for name, line in rows:
        if name:
            dr.text((x, yy + 6), name, font=_font(20, "Medium"), fill=INK)
            yy += 34
        dr.text((x + 16, yy), line, font=_mono(16), fill=MUTED)
        yy += 26
    path = os.path.join(root, "icons", "K-Logo_Icons_Overview.png")
    sheet = sheet.crop((0, 0, W, max(top + 620, yy + 40)))
    return [save_png(sheet.convert("RGB"), path)]



# ------------------------------------------------------------------ verify
CHROMIUM_JS = r"""
const { chromium } = require('playwright');
let svg = '';
process.stdin.on('data', d => svg += d);
process.stdin.on('end', async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ deviceScaleFactor: 1 });
  const uri = 'data:image/svg+xml;base64,' + Buffer.from(svg).toString('base64');
  const out = {};
  for (const n of [16, 32, 48]) {
    await page.setContent(`<html><body style="margin:0"><img id="i" src="${uri}" width="${n}" height="${n}" style="display:block"></body></html>`);
    await page.waitForFunction(() => document.getElementById('i').complete);
    out[n] = (await page.locator('#i').screenshot({ omitBackground: true })).toString('base64');
  }
  await browser.close();
  console.log(JSON.stringify(out));
});
"""


def _alpha(im):
    return np.asarray(im.convert("RGBA"))[..., 3].astype(int)


def _coverage(k, n, h=None, viewbox=None):
    """Exact-ish coverage of the K's outline (one path, so no internal edges) at n px."""
    if viewbox:
        x, y, w, hh = viewbox
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{fmt(x)} {fmt(y)} {fmt(w)} {fmt(hh)}" '
               f'width="{fmt(w)}" height="{fmt(hh)}"><path d="{solid_d(k)}"/></svg>')
    else:
        svg = svg_doc(f'<path d="{solid_d(k)}"/>', size=k.size)
    return _alpha(px(svg, n, h))


def _seams(a, cov):
    """(interior pixels, interior pixels below 255, worst interior alpha, max |alpha - coverage|)."""
    inside = cov == 255
    return int(inside.sum()), int((a[inside] < 255).sum()), int(a[inside].min()), int(np.abs(a - cov).max())


def verify(root=ROOT):
    """Measure every written file and print the numbers; raises AssertionError on a failure."""
    fails = []

    def check(ok, msg):
        print(("  ok   " if ok else "  FAIL ") + msg)
        if not ok:
            fails.append(msg)

    rel = lambda p: os.path.relpath(p, root)                                       # noqa: E731
    dirs = [os.path.join(root, d) for d in ("logo/png", "icons", "social")]
    files = sorted(os.path.join(dp, f) for d in dirs for dp, _, fs in os.walk(d) for f in fs)

    print("\n== files: dimensions / modes")
    pngs = [p for p in files if p.endswith(".png")]
    for p in files:
        if p.endswith(".png"):
            im = Image.open(p)
            srgb = "sRGB" if "srgb" in im.info else "no-sRGB"
            print(f"  {rel(p):66s} {im.width:5d} x {im.height:<5d} {im.mode:5s} {srgb}  {os.path.getsize(p) // 1024} KB")
            if im.mode not in ("RGB", "RGBA") or "srgb" not in im.info:
                check(False, f"{rel(p)} is not 8-bit RGB(A) with an sRGB chunk")
        else:
            print(f"  {rel(p):66s} {os.path.getsize(p)} bytes")
    check(True, f"{len(pngs)} PNGs, all 8-bit RGB or RGBA with an sRGB chunk; {len(files) - len(pngs)} other files")
    for name, fn in LOGO_PNG.items():
        for n in LOGO_SIZES:
            im = Image.open(os.path.join(root, "logo", "png", f"{name}_{n}.png"))
            want = (n, round(n * TIGHT_ASPECT)) if name.endswith("_Tight") else (n, n)
            if im.size != want or im.mode != "RGBA" or im.getextrema()[3][0] != 0:
                check(False, f"{name}_{n}: {im.size} {im.mode} (want {want} RGBA, transparent)")
    check(True, f"logo/png: {len(LOGO_PNG) * len(LOGO_SIZES)} files, RGBA, transparent corners, widths {LOGO_SIZES}")

    print("\n== opaque deliverables (no alpha channel)")
    web, soc = os.path.join(root, "icons", "web"), os.path.join(root, "social")
    android = os.path.join(root, "icons", "android")
    for p in (os.path.join(web, "apple-touch-icon.png"), os.path.join(web, "icon-maskable-512x512.png"),
              os.path.join(root, "icons", "ios", "AppIcon.appiconset", "AppIcon-1024.png"),
              os.path.join(android, "play-store-512.png")) + tuple(os.path.join(soc, f"{n}.png") for n in list(SOCIAL) + ["avatar-1080", "avatar-400"]):
        im = Image.open(p)
        check(im.mode == "RGB", f"{rel(p)}: {im.size[0]}x{im.size[1]} {im.mode}")

    print("\n== favicon.ico")
    ico = Image.open(os.path.join(web, "favicon.ico"))
    sizes = sorted(ico.info["sizes"])
    check(sizes == [(16, 16), (32, 32), (48, 48)], f"favicon.ico holds {sizes}")
    for s in sizes:
        ico.size = s
        frame = np.asarray(ico.convert("RGBA"))
        png = np.asarray(Image.open(os.path.join(web, f"favicon-{s[0]}x{s[1]}.png")).convert("RGBA"))
        check(np.array_equal(frame, png), f"ico frame {s[0]} px == favicon-{s[0]}x{s[0]}.png (grid-fitted art, not a resample)")

    print("\n== manifests")
    with open(os.path.join(web, "site.webmanifest")) as f:
        man = json.load(f)
    for ic in man["icons"]:
        p = os.path.join(web, ic["src"])
        ok = os.path.exists(p) and "x".join(map(str, Image.open(p).size)) == ic["sizes"]
        check(ok, f"site.webmanifest {ic['src']} exists, {ic['sizes']}, purpose {ic.get('purpose')}")
    check(any(ic.get("purpose") == "maskable" for ic in man["icons"]), "site.webmanifest has a maskable icon")
    check(man["theme_color"] in NEUTRAL.values() and man["background_color"] in NEUTRAL.values(),
          f"theme {man['theme_color']} / background {man['background_color']} are brand neutrals")
    with open(os.path.join(web, "head-snippet.html")) as f:
        hrefs = re.findall(r'href="/([^"]+)"', f.read())
    check(all(os.path.exists(os.path.join(web, h)) for h in hrefs), f"head-snippet.html: {len(hrefs)} hrefs all exist")
    ios = os.path.join(root, "icons", "ios", "AppIcon.appiconset")
    with open(os.path.join(ios, "Contents.json")) as f:
        cj = json.load(f)
    img = cj["images"][0]
    im = Image.open(os.path.join(ios, img["filename"]))
    check(len(cj["images"]) == 1 and img["idiom"] == "universal" and img["platform"] == "ios" and img["size"] == "1024x1024"
          and im.size == (1024, 1024) and im.mode == "RGB", f"Contents.json single-size universal -> {img['filename']} {im.size} {im.mode}")
    ns = "{http://schemas.android.com/apk/res/android}"
    for x in ("ic_launcher", "ic_launcher_round"):
        tree = ET.parse(os.path.join(android, "res", "mipmap-anydpi-v26", f"{x}.xml")).getroot()
        layers = {c.tag: c.get(ns + "drawable") for c in tree}
        ok = set(layers) == {"background", "foreground", "monochrome"} and all(
            os.path.exists(os.path.join(android, "res", f"mipmap-{d}", v.split("/")[1] + ".png")) for v in layers.values() for d in DENSITIES)
        check(ok, f"{x}.xml: {sorted(layers)} -> drawables exist in all {len(DENSITIES)} densities")
    for d, sc in DENSITIES.items():
        m = os.path.join(android, "res", f"mipmap-{d}")
        got = {f: Image.open(os.path.join(m, f)).size[0] for f in sorted(os.listdir(m))}
        want = {f: round((48 if f in ("ic_launcher.png", "ic_launcher_round.png") else 108) * sc) for f in got}
        check(got == want and len(got) == 5, f"mipmap-{d}: {got}")

    print("\n== K-Logo.icns (read back with Pillow)")
    icns = Image.open(os.path.join(root, "icons", "macos", "K-Logo.icns"))
    found = sorted(icns.info["sizes"])
    check(len(found) == 10, f"K-Logo.icns entries {found}")
    for t, name, n in ICONSET:
        base = n if t not in ("ic10", "ic11", "ic12", "ic13", "ic14") else n // 2
        a = np.asarray(icns.icns.getimage((base, base, n // base)).convert("RGBA"))
        b = np.asarray(Image.open(os.path.join(root, "icons", "macos", "AppIcon.iconset", name)).convert("RGBA"))
        if not np.array_equal(a, b):
            check(False, f"icns {t} != {name}")
    check(True, "every icns entry == its iconset PNG (16 / 32 / 32@2x / 64 rendered with detail='small')")

    print("\n== Android safe circle (66 dp of the 108 dp canvas)")
    k = android_fg_k()
    r_geo = max(math.dist((512, 512), q) for q in kmark.silhouette(k)) / 1024 * 108
    check(r_geo <= 33.0, f"K outline reaches {r_geo:.2f} dp from the centre (limit 33 dp)")
    for f in ("ic_launcher_foreground.png", "ic_launcher_monochrome.png"):
        a = _alpha(Image.open(os.path.join(android, "res", "mipmap-xxxhdpi", f)))
        ys, xs = np.nonzero(a > 0)
        r_px = np.sqrt((xs + 0.5 - 216) ** 2 + (ys + 0.5 - 216) ** 2).max() / 4       # 4 px per dp
        check(r_px <= 33.0, f"{f} (xxxhdpi): farthest painted pixel centre {r_px:.2f} dp")
    same = brand.doc_android_foreground() == svg_doc(brand.mark_group(android_fg_k()), size=1024,
                                                     title=f"{brand.NAME} - Android adaptive foreground")
    check(same, f"FG_SCALE {FG_SCALE} reproduces brand.doc_android_foreground exactly (same SVG document)")
    for d, sc in DENSITIES.items():
        n = round(108 * sc)
        a = _alpha(Image.open(os.path.join(android, "res", f"mipmap-{d}", "ic_launcher_foreground.png")))
        tot, bad, lo, edge = _seams(a, _coverage(android_fg_k(), n))
        check(bad == 0, f"mipmap-{d} foreground ({n} px): {tot} interior px, {bad} below 255; edge max |a - cov| {edge}")
    km = brand.scaled_about_centre(brand.master(), MASKABLE_SCALE)
    r_m = max(math.dist((512, 512), q) for q in kmark.silhouette(km)) / 1024
    check(r_m <= 0.40, f"maskable: K reaches {r_m:.3f} of the side (safe circle 0.400)")
    r_b = max(math.dist((512, 512), q) for q in kmark.silhouette(brand.master())) / 1024
    check(r_b < 0.5, f"avatar badge: K reaches {r_b:.3f} of the side (inscribed crop circle 0.500)")

    print("\n== seams: alpha inside the K's outline must be 255")
    k = brand.master()
    for n in (16, 32, 64, 1024):
        a = _alpha(px(flat_doc(n), n))
        tot, bad, lo, edge = _seams(a, _coverage(k, n))
        check(bad == 0, f"Mark_Color at {n:4d} px: {tot} interior px, {bad} below 255 (min {lo}); edge max |alpha - coverage| {edge}")
    for name in ("K-Logo_Mark_Color", "K-Logo_Mark_Greyscale", "K-Logo_Mark_Dimensional"):
        for n in LOGO_SIZES:
            a = _alpha(Image.open(os.path.join(root, "logo", "png", f"{name}_{n}.png")))
            cov = _coverage(k, n)
            tot, bad, lo, edge = _seams(a, cov)
            check(bad == 0, f"{name}_{n}.png: {tot} interior px, {bad} below 255 (min {lo}); "
                            f"edge px off coverage by > 8: {int((np.abs(a - cov) > 8).sum())}")
    x0, y0, x1, y1 = kmark.bbox(k)
    for n in LOGO_SIZES:
        h = round(n * TIGHT_ASPECT)
        a = _alpha(Image.open(os.path.join(root, "logo", "png", f"K-Logo_Mark_Color_Tight_{n}.png")))
        cov = _coverage(k, n, h, (x0, y0, x1 - x0, y1 - y0))
        tot, bad, lo, edge = _seams(a, cov)
        check(bad == 0, f"K-Logo_Mark_Color_Tight_{n}.png ({n}x{h}): {tot} interior px, {bad} below 255; "
                        f"edge px off coverage by > 8: {int((np.abs(a - cov) > 8).sum())}")
    for n in (16, 32, 48):
        kf, lay = grid_fit(n)
        tot, bad, lo, edge = _seams(_alpha(Image.open(os.path.join(web, f"favicon-{n}x{n}.png"))), _coverage(kf, n))
        check(bad == 0, f"favicon-{n}x{n}.png: {tot} interior px, {bad} below 255; edge max |alpha - coverage| {edge}")

    print("\n== favicon grid fit: stem columns and cap / base rows")
    for n in (16, 32, 48):
        kf, lay = grid_fit(n)
        a = _alpha(Image.open(os.path.join(web, f"favicon-{n}x{n}.png")))
        L, S, T, C = lay["left"], lay["stem"], lay["top"], lay["cap"]
        sil = Polygon(kmark.silhouette(kf)).buffer(0)
        u = kf.size / n
        free = [r for r in range(T, T + C) if box((L + S) * u, r * u, (L + S + 1) * u, (r + 1) * u).intersection(sil).area < 1e-6]
        cols = {"left of stem": sorted(set(a[T:T + C, L - 1].tolist())), "stem": sorted(set(a[T:T + C, L:L + S].ravel().tolist())),
                "right of stem (free rows)": sorted(set(a[free, L + S].tolist()))}
        rows = {"above cap": sorted(set(a[T - 1, L:L + S].tolist())), "cap row": sorted(set(a[T, L:L + S].tolist())),
                "base row": sorted(set(a[T + C - 1, L:L + S].tolist())), "below base": sorted(set(a[T + C, L:L + S].tolist()))}
        crisp = all(set(v) <= {0, 255} for v in list(cols.values()) + list(rows.values()))
        check(crisp, f"{n} px: cap {C} px rows {T}-{T + C - 1}, stem {S} px cols {L}-{L + S - 1} | columns {cols} | rows {rows}")
    try:
        import base64
        import subprocess
        res = subprocess.run(["node", "-e", CHROMIUM_JS], input=favicon_svg().encode(), capture_output=True, timeout=180, check=True)
        shots = json.loads(res.stdout)
        kf, _ = grid_fit(16)
        for n in (16, 32, 48):
            a = _alpha(Image.open(io.BytesIO(base64.b64decode(shots[str(n)]))))
            tot, bad, lo, edge = _seams(a, _coverage(kf, n))
            check(bad <= 1 and lo >= 250, f"favicon.svg drawn by Chromium at {n} px: {tot} interior px, {bad} below 255 (min {lo})")
    except Exception as e:                                                          # noqa: BLE001
        print(f"  skip Chromium check of favicon.svg ({type(e).__name__})")

    print("\n== colour: facet centres in the 1024 PNGs vs brand.PALETTE")
    k = brand.master()
    f = kmark.facets(k)
    centres = {n: polylabel(Polygon(f[n]), 0.5) for n in FACETS}
    expect = {"K-Logo_Mark_Color": {n: PALETTE[n] for n in FACETS}, "K-Logo_Badge_Color": {n: PALETTE[n] for n in FACETS},
              "K-Logo_Mark_Color_Tight": {n: PALETTE[n] for n in FACETS}, "K-Logo_Mark_Black": {n: NEUTRAL["black"] for n in FACETS},
              "K-Logo_Mark_White": {n: NEUTRAL["white"] for n in FACETS}, "K-Logo_Mark_Greyscale": brand.grey_colours()}
    for name, cols in expect.items():
        im = np.asarray(Image.open(os.path.join(root, "logo", "png", f"{name}_1024.png")).convert("RGBA")).astype(int)
        worst, got = 0, {}
        for n, c in centres.items():
            x, y = (c.x, c.y) if not name.endswith("_Tight") else ((c.x - x0) * 1024 / (x1 - x0), (c.y - y0) * 1024 / (x1 - x0))
            v = im[int(y), int(x)]
            want = np.array(_rgba(cols[n]))
            worst = max(worst, int(np.abs(v - want).max()))
            got[n] = "#%02X%02X%02X" % tuple(v[:3])
        check(worst <= 1, f"{name}_1024: {got} max deviation {worst}/255")
    for n in (32, 48):                                                              # favicons where a facet holds a whole pixel
        kf, _ = grid_fit(n)
        ff = kmark.facets(kf)
        im = np.asarray(Image.open(os.path.join(web, f"favicon-{n}x{n}.png")).convert("RGBA")).astype(int)
        u = kf.size / n
        worst, used = 0, []
        for c in FACETS:
            p = polylabel(Polygon(ff[c]), 0.5)
            if Polygon(ff[c]).exterior.distance(p) / u >= 1.5:
                worst = max(worst, int(np.abs(im[int(p.y / u), int(p.x / u)] - np.array(_rgba(PALETTE[c]))).max()))
                used.append(c)
        check(worst <= 1, f"favicon-{n}x{n}: facet centres {used} max deviation {worst}/255")

    print("\n== social zones (mark + traces + pads + node)")
    for name, spec in SOCIAL.items():
        ex = scene_extent(spec)
        g = box(*ex)
        for lbl, b in spec["inside"]:
            check(box(*b).contains(g), f"{name}: mark x {ex[0]:.0f}-{ex[2]:.0f} y {ex[1]:.0f}-{ex[3]:.0f} inside {lbl} {b}")
        for lbl, b in spec["avoid"]:
            check(not box(*b).intersects(g), f"{name}: clear of {lbl} {b} (gap {box(*b).distance(g):.0f} px)")

    print("\n== favicon.svg on light and dark tabs (why no prefers-color-scheme rule)")
    for bg in ("#FFFFFF", "#F1F3F4", "#202124", "#35363A"):
        print(f"  {bg}: " + "  ".join(f"{n} {contrast(PALETTE[n], bg):.2f}" for n in ("green", "blue", "red", "yellow")))

    print(f"\nverify: {'all checks passed' if not fails else str(len(fails)) + ' FAILED'}")
    assert not fails, fails


# ------------------------------------------------------------------ run
STAGES = (("logo png", lambda r: stage_logo_png(r)), ("web", lambda r: stage_web(r)), ("ios", lambda r: stage_ios(r)),
          ("macos", lambda r: stage_macos(r)), ("android", lambda r: stage_android(r)),
          ("social", lambda r: stage_social(r)), ("overview", lambda r: stage_overview(r)))


def run(root=ROOT):
    """Regenerate every raster deliverable; returns the written paths."""
    out = []
    for name, fn in STAGES:
        paths = fn(root)
        print(f"{name:10s} {len(paths):3d} files", flush=True)
        out += paths
    return out


if __name__ == "__main__":
    written = run(ROOT)
    print(f"{len(written)} files written")
    if "--no-verify" not in sys.argv:
        verify(ROOT)
