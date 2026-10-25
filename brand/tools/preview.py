"""Rendering helpers shared by every build/design script.

    svg_to_png(svg, size)            -> PIL.Image (RGBA)   [librsvg; supports SVG filters]
    contact_sheet(items, out_path)   -> writes a comparison sheet: each item at 512 px plus
                                        128/48/32/16 px thumbnails on light and dark tiles.
"""
from __future__ import annotations

import io
import subprocess

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/inter/Inter-Medium.otf"


def svg_to_png(svg: str, size: int) -> Image.Image:
    """Render with rsvg-convert (handles gradients, masks and filters)."""
    out = subprocess.run(
        ["rsvg-convert", "-w", str(size), "-h", str(size), "-f", "png"],
        input=svg.encode(), capture_output=True, check=True,
    ).stdout
    return Image.open(io.BytesIO(out)).convert("RGBA")


def _font(px):
    try:
        return ImageFont.truetype(FONT, px)
    except OSError:
        return ImageFont.load_default()


def contact_sheet(items, out_path, big=512, smalls=(128, 48, 32, 16)):
    """items: list of (label, svg_string). Each row: big render on a checker, then thumbnails
    on light (#FFFFFF) and dark (#0B0F19) tiles; 16/32 px are also shown 4x nearest-neighbour."""
    pad = 24
    thumb_w = sum(smalls) + pad * len(smalls)
    zoom_w = 4 * 32 + pad + 4 * 16
    row_h = big + 2 * pad + 36
    width = pad + big + pad + max(thumb_w, zoom_w) + pad + 4 * 32 + pad
    height = len(items) * row_h + pad
    sheet = Image.new("RGB", (width, height), (236, 238, 242))
    d = ImageDraw.Draw(sheet)
    f = _font(22)
    for r, (label, svg) in enumerate(items):
        y0 = pad + r * row_h
        d.text((pad, y0), label, fill=(20, 24, 33), font=f)
        y = y0 + 36
        # checkerboard so transparency is visible
        chk = Image.new("RGB", (big, big), (255, 255, 255))
        cd = ImageDraw.Draw(chk)
        for yy in range(0, big, 16):
            for xx in range(0, big, 16):
                if (xx // 16 + yy // 16) % 2:
                    cd.rectangle([xx, yy, xx + 15, yy + 15], fill=(228, 230, 235))
        im = svg_to_png(svg, big)
        chk.paste(im, (0, 0), im)
        sheet.paste(chk, (pad, y))
        x = pad + big + pad
        for bg, yy in (((255, 255, 255), y), ((11, 15, 25), y + 150)):
            xx = x
            for s in smalls:
                tile = Image.new("RGB", (s + 8, s + 8), bg)
                t = svg_to_png(svg, s)
                tile.paste(t, (4, 4), t)
                sheet.paste(tile, (xx, yy))
                xx += s + pad
        # nearest-neighbour zooms of the 32 and 16 px renders (pixel-level legibility)
        xx = x
        yy = y + 320
        for s in (32, 16):
            t = svg_to_png(svg, s)
            tile = Image.new("RGBA", (s, s), (255, 255, 255, 255))
            tile.alpha_composite(t)
            z = tile.resize((s * 4, s * 4), Image.NEAREST).convert("RGB")
            sheet.paste(z, (xx, yy))
            xx += s * 4 + pad
    sheet.save(out_path)
    return out_path
