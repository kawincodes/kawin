"""Client preview: before/after, the main variants, small sizes on light/dark and the palette."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageDraw  # noqa: E402

from brand import NEUTRAL, PALETTE, PALETTE_OKLCH, doc_badge, doc_mark, doc_tech, master, render  # noqa: E402
from okcolor import contrast  # noqa: E402
from preview import _font  # noqa: E402

TRACE = os.path.join(os.path.dirname(__file__), "..", "source", "K-Logo_trace.svg")
BG = (244, 245, 248)
INK = (20, 26, 41)
MUTED = (104, 112, 128)


def tile(im, bg, pad=0):
    t = Image.new("RGBA", (im.width + 2 * pad, im.height + 2 * pad), bg)
    t.alpha_composite(im, (pad, pad))
    return t


def build(out_path):
    k = master()
    W = 2000
    sheet = Image.new("RGBA", (W, 2600), BG)
    d = ImageDraw.Draw(sheet)
    h1, h2, h3, mono = _font(54), _font(30), _font(24), _font(22)
    x, y = 80, 70
    d.text((x, y), "K logo — premium redesign", fill=INK, font=h1)
    d.text((x, y + 72), "Final master  ·  sharp corners  ·  version 1.0", fill=MUTED, font=h3)

    # before / after
    y = 220
    S = 600
    for i, (label, svg) in enumerate((("Before — your original (trace)", open(TRACE).read()),
                                       ("After — redesigned master", doc_badge(k)))):
        xx = 80 + i * (S + 140)
        d.text((xx, y), label, fill=INK, font=h2)
        sheet.alpha_composite(tile(render(svg, S), (255, 255, 255, 0)), (xx, y + 50))
    d.text((80 + S + 30, y + 50 + S // 2 - 30), "→", fill=MUTED, font=_font(64))
    xx = 80 + 2 * (S + 140) - 70
    notes = ["What changed", "",
             "·  Arm and leg share one angle",
             "   (53.13°) and one weight",
             "   (96.5% of the stem)",
             "·  Right edges of arm and leg aligned",
             "·  Junction and crotch mirror the fold",
             "·  Optically centred in the circle",
             "·  Your four colours, refined: deeper,",
             "   cleaner; blue and green nudged",
             "   off Google's hues",
             "·  Every corner sharp",
             "·  Same footprint as your original"]
    for j, line in enumerate(notes):
        d.text((xx, y + 60 + j * 38), line, fill=INK if j == 0 else MUTED, font=h3 if j else h2)

    # variants
    y = 940
    S2 = 410
    d.text((80, y), "Variants", fill=INK, font=h2)
    variants = (("Flat mark", doc_mark(k, raster=True), (255, 255, 255, 255)),
                ("Premium shaded", doc_badge(k, "dimensional"), (236, 238, 242, 255)),
                ("Tech app icon", doc_tech(k), (236, 238, 242, 255)),
                ("One colour", doc_mark(k, "black-gapped", raster=True), (255, 255, 255, 255)))
    for i, (label, svg, bg) in enumerate(variants):
        xx = 80 + i * (S2 + 50)
        d.text((xx, y + 50), label, fill=MUTED, font=h3)
        sheet.alpha_composite(tile(render(svg, S2), bg), (xx, y + 90))

    # reversed + small sizes
    y = 1520
    d.text((80, y), "On dark  &  small sizes", fill=INK, font=h2)
    dark = (7, 13, 26, 255)
    panel = Image.new("RGBA", (W - 160, 330), dark)
    panel.alpha_composite(render(doc_mark(k, "white-gapped", raster=True), 270), (20, 30))
    panel.alpha_composite(render(doc_mark(k, raster=True), 270), (300, 30))
    xx = 620
    for s in (128, 64, 48, 32, 16):
        panel.alpha_composite(render(doc_badge(k, raster=True), s), (xx, 165 - s // 2))
        xx += s + 40
    for s in (128, 64, 32, 16):
        panel.alpha_composite(render(doc_tech(k, detail="small" if s <= 64 else "full"), s), (xx, 165 - s // 2))
        xx += s + 40
    sheet.alpha_composite(panel, (80, y + 50))
    light = Image.new("RGBA", (W - 160, 200), (255, 255, 255, 255))
    xx = 30
    for s in (128, 64, 48, 32, 16):
        light.alpha_composite(render(doc_mark(k, raster=True), s), (xx, 100 - s // 2))
        xx += s + 40
    for s in (128, 64, 48, 32, 16):
        light.alpha_composite(render(doc_badge(k, raster=True), s), (xx, 100 - s // 2))
        xx += s + 40
    sheet.alpha_composite(light, (80, y + 400))

    # palette
    y = 2160
    d.text((80, y), "Palette", fill=INK, font=h2)
    sw = 400
    for i, n in enumerate(("red", "yellow", "green", "blue")):
        xx = 80 + i * (sw + 40)
        d.rectangle([xx, y + 50, xx + sw, y + 210], fill=PALETTE[n])
        L, C, H = PALETTE_OKLCH[n]
        d.text((xx, y + 225), f"{n.capitalize()}  {PALETTE[n]}", fill=INK, font=h3)
        d.text((xx, y + 262), f"oklch({L} {C} {H})", fill=MUTED, font=mono)
        d.text((xx, y + 294), f"contrast on white {contrast(PALETTE[n], '#FFFFFF'):.2f}:1", fill=MUTED, font=mono)
    sheet.convert("RGB").save(out_path, quality=95)
    return out_path


def build_tech(out_path):
    """Tech app icon sheet: the tile, store and macOS versions, Android masks, the node up close
    and every size on light and dark (full detail from 120 px, the small version below)."""
    import kmark
    from brand import doc_android_background, doc_android_foreground

    k = master()
    night = tuple(int(NEUTRAL["night"][i:i + 2], 16) for i in (1, 3, 5))
    S, x0 = 940, 80 + 940 + 70
    W, H = x0 + 940 + 80, 1990
    sheet = Image.new("RGBA", (W, H), (236, 238, 243))
    d = ImageDraw.Draw(sheet)
    h1, h2, h3 = _font(52), _font(28), _font(21)

    def head(x, y, title, sub=None):
        d.text((x, y), title, fill=INK, font=h2)
        if sub:
            d.text((x, y + 36), sub, fill=MUTED, font=h3)

    d.text((80, 60), "K logo — tech app icon", fill=INK, font=h1)
    d.text((80, 130), "Dark plate · construction grid · the four colours lit from the node where they meet · "
                      "fold lines running out as traces", fill=MUTED, font=h3)
    head(80, 200, "App icon (rounded tile)")
    sheet.alpha_composite(render(doc_tech(k), S), (80, 250))
    head(x0, 200, "App Store / Play", "full-bleed — the OS applies its mask")
    sheet.alpha_composite(render(doc_tech(k, shape="square"), 420), (x0, 280))
    head(x0 + 500, 200, "macOS", "Apple's icon grid")
    sheet.alpha_composite(tile(render(doc_tech(k, shape="macos"), 420), (214, 222, 236, 255)), (x0 + 500, 280))

    # Android: the 72 dp visible area of the adaptive layers under three launcher masks
    y, m = 760, 138
    head(x0, y, "Android adaptive", "the same icon under launcher masks")
    both = render(doc_android_background(k), 324)
    both.alpha_composite(render(doc_android_foreground(k), 324))
    cut = 324 * 18 // 108
    vis = both.crop((cut, cut, 324 - cut, 324 - cut)).resize((m, m), Image.LANCZOS)
    shapes = (("circle", lambda dr: dr.ellipse([0, 0, m - 1, m - 1], fill=255)),
              ("squircle", lambda dr: dr.rounded_rectangle([0, 0, m - 1, m - 1], radius=int(m * 0.32), fill=255)),
              ("rounded", lambda dr: dr.rounded_rectangle([0, 0, m - 1, m - 1], radius=int(m * 0.16), fill=255)))
    for i, (name, draw) in enumerate(shapes):
        mask = Image.new("L", (m, m), 0)
        draw(ImageDraw.Draw(mask))
        icon = Image.new("RGBA", (m, m), (0, 0, 0, 0))
        icon.paste(vis, (0, 0), mask)
        xx = x0 + i * (m + 28)
        sheet.alpha_composite(icon, (xx, y + 100))
        d.text((xx + (m - d.textlength(name, font=h3)) / 2, y + 100 + m + 14), name, fill=MUTED, font=h3)

    # the node where the four colours meet
    head(x0 + 500, y, "The node, up close", "where all four colours meet")
    J = kmark.points(k)["J"]
    big = render(doc_tech(k), 3072)
    sheet.alpha_composite(big.crop((int(J[0] * 3) - 300, int(J[1] * 3) - 300, int(J[0] * 3) + 300,
                                    int(J[1] * 3) + 300)).resize((420, 420), Image.LANCZOS), (x0 + 500, y + 90))

    y = 1290
    head(80, y, "Every size — full detail from 120 px, the small version below it")
    for row, bg in enumerate(((255, 255, 255, 255), (*night, 255))):
        strip = Image.new("RGBA", (W - 160, 300), bg)
        sd = ImageDraw.Draw(strip)
        x = 30
        for s in (256, 180, 120, 87, 64, 40, 29, 20):
            strip.alpha_composite(render(doc_tech(k, detail="full" if s >= 120 else "small"), s), (x, 138 - s // 2))
            sd.text((x, 272), f"{s} px", fill=MUTED if row == 0 else (150, 158, 175), font=h3)
            x += max(s, 60) + 70
        sheet.alpha_composite(strip, (80, y + 50 + row * 320))
    sheet.convert("RGB").save(out_path, quality=95)
    return out_path


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "preview.png"))
