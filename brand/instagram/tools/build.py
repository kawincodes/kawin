"""Build the KRS Tech Solutions Instagram profile-picture kit.

This folder (brand/instagram/) is the one canonical set of Instagram profile
pictures; profile-navy.png is the recommended upload.

Reads the master vector logo files in brand/logo/, composes four 1080x1080
profile pictures as SVG (geometry baked in, no transforms, no fonts, flat
brand-colour backgrounds), renders them to PNG with Chromium via
brand/logo/tools/render_svg.js, then draws preview.png: every option cropped to
a circle at real pixel sizes on a white and a black page.

Usage:  python3 -B brand/instagram/tools/build.py
"""
import os
import re
import subprocess

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # brand/instagram
LOGO = os.path.normpath(os.path.join(HERE, '..', 'logo'))
RENDER = os.path.join(LOGO, 'tools', 'render_svg.js')

NAVY = '#012153'
ORANGE = '#FC7403'
WHITE = '#FFFFFF'
SIZE = 1080            # Instagram upload size (square; shown as the inscribed circle)
C = SIZE / 2           # circle centre / radius

# ---- sizing (tuned by reading preview.png at 110 px) -----------------------
MARK_WIDTH = 0.70      # monogram width as a fraction of the circle diameter
FULL_SAFE_R = 0.90     # full logo: farthest ink point sits at 90% of the radius
RING_WIDTH = 18        # orange ring stroke, px at 1080
RING_INSET = 26        # gap between the circle edge and the ring's outer edge


# ---- path helpers ----------------------------------------------------------
ARGS = {'M': 2, 'L': 2, 'C': 6, 'Q': 4, 'A': 7, 'Z': 0}
TOKEN = re.compile(r'[A-Za-z]|-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


def parse(d):
    """Absolute-command SVG path -> list of (cmd, [numbers]) segments."""
    toks = TOKEN.findall(d)
    out, i, cmd = [], 0, None
    while i < len(toks):
        if toks[i].isalpha():
            cmd = toks[i]
            if cmd not in ARGS:
                raise ValueError(f'unsupported path command {cmd!r}')
            i += 1
            if cmd == 'Z':
                out.append(('Z', []))
                continue
        n = ARGS[cmd]
        out.append((cmd, [float(t) for t in toks[i:i + n]]))
        i += n
        if cmd == 'M':
            cmd = 'L'  # implicit lineto after moveto
    return out


def fmt(v):
    s = f'{v:.2f}'.rstrip('0').rstrip('.')
    return '0' if s == '-0' else s


def transform(segs, s, tx, ty):
    """Bake a uniform scale + translate into the path coordinates."""
    out = []
    for cmd, a in segs:
        if cmd == 'Z':
            out.append('Z')
        elif cmd == 'A':
            rx, ry, rot, large, sweep, x, y = a
            out.append('A' + ' '.join([fmt(rx * s), fmt(ry * s), fmt(rot), str(int(large)),
                                       str(int(sweep)), fmt(x * s + tx), fmt(y * s + ty)]))
        else:
            pts = [fmt(v * s + (tx if k % 2 == 0 else ty)) for k, v in enumerate(a)]
            out.append(cmd + ' '.join(pts))
    return ''.join(out)


def subpaths(segs):
    cur = []
    for seg in segs:
        if seg[0] == 'M' and cur:
            yield cur
            cur = []
        cur.append(seg)
    if cur:
        yield cur


def points(segs):
    for cmd, a in segs:
        if cmd == 'A':
            yield a[5], a[6]
        else:
            for k in range(0, len(a), 2):
                yield a[k], a[k + 1]


def load_paths(svg_file):
    s = open(os.path.join(LOGO, svg_file), encoding='utf-8').read()
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', s).group(1).split()]
    paths = [(fill.upper(), parse(d)) for fill, d in re.findall(r'<path fill="([^"]+)" d="([^"]+)"', s)]
    return vb, paths


# ---- artwork ---------------------------------------------------------------
MARK_VB, MARK = load_paths('krs-mark.svg')          # KRS monogram, 538.55 x 149.8
LOGO_VB, FULL = load_paths('krs-logo.svg')          # full logo, 550 x 253

# Full logo without the tagline (BUILD - INNOVATE - SCALE is ~2.5 px tall at 110 px:
# unreadable, it only adds grey noise). The tagline is every subpath below y = 225.
FULL_NO_TAG = []
for fill, segs in FULL:
    keep = [sp for sp in subpaths(segs) if min(y for _, y in points(sp)) < 225]
    FULL_NO_TAG.append((fill, [seg for sp in keep for seg in sp]))


def ink_bbox(paths):
    xs, ys = [], []
    for _, segs in paths:
        for x, y in points(segs):
            xs.append(x)
            ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


def place(paths, colours, s, tx, ty):
    return ''.join(f'<path fill="{colours.get(fill, fill)}" d="{transform(segs, s, tx, ty)}"/>'
                   for fill, segs in paths)


def place_mark(colours):
    x0, y0, x1, y1 = ink_bbox(MARK)
    w = SIZE * MARK_WIDTH
    s = w / (x1 - x0)
    h = (y1 - y0) * s
    # Optical centring: the S is heavier than the open K, so the ink centroid sits
    # ~0.7% of the width right of the box centre. Move left by half of that.
    # Vertically the mark is already top-heavy (centroid 1.3% above centre), which
    # supplies the usual "slightly above centre" lift, so the box stays centred.
    dx = -0.0036 * w
    tx = C - w / 2 + dx - x0 * s
    ty = C - h / 2 - y0 * s
    return place(MARK, colours, s, tx, ty)


def place_full(colours):
    # Scale so the farthest ink point from the circle centre sits at FULL_SAFE_R.
    x0, y0, x1, y1 = ink_bbox(FULL_NO_TAG)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    far = max(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
              for _, segs in FULL_NO_TAG for x, y in points(segs))
    s = C * FULL_SAFE_R / far
    return place(FULL_NO_TAG, colours, s, C - cx * s, C - cy * s)


def svg(title, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" viewBox="0 0 {SIZE} {SIZE}">'
            f'<title>{title}</title>{body}</svg>\n')


# Flat brand-colour backgrounds: exact palette values, and nothing for Instagram's
# JPEG re-compression to band.
NAVY_BG = f'<rect width="{SIZE}" height="{SIZE}" fill="{NAVY}"/>'
WHITE_BG = f'<rect width="{SIZE}" height="{SIZE}" fill="{WHITE}"/>'
ON_DARK = {NAVY: WHITE}
ring_r = C - RING_INSET - RING_WIDTH / 2
RING = (f'<circle cx="{fmt(C)}" cy="{fmt(C)}" r="{fmt(ring_r)}" fill="none" '
        f'stroke="{ORANGE}" stroke-width="{RING_WIDTH}"/>')

OPTIONS = [
    ('profile-navy', 'KRS Tech Solutions - Instagram profile (navy)',
     NAVY_BG + place_mark(ON_DARK)),
    ('profile-white', 'KRS Tech Solutions - Instagram profile (white)',
     WHITE_BG + place_mark({})),
    ('profile-orange-ring', 'KRS Tech Solutions - Instagram profile (navy, orange ring)',
     NAVY_BG + RING + place_mark(ON_DARK)),
    ('profile-full-logo', 'KRS Tech Solutions - Instagram profile (full logo)',
     WHITE_BG + place_full({})),
]


def render_all():
    for name, title, body in OPTIONS:
        svg_path = os.path.join(HERE, name + '.svg')
        png_path = os.path.join(HERE, name + '.png')
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(svg(title, body))
        subprocess.run(['node', RENDER, svg_path, png_path, str(SIZE)], check=True,
                       stdout=subprocess.DEVNULL)
        Image.open(png_path).convert('RGB').save(png_path, optimize=True)  # opaque sRGB PNG
        print('wrote', png_path)


# ---- preview sheet ---------------------------------------------------------
FONT_DIR = '/usr/share/fonts/opentype/inter'


def font(weight, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, f'Inter-{weight}.otf'), size)


def circle(img, d):
    """Downscale like Instagram's server does (Lanczos) and crop to an anti-aliased circle."""
    small = img.resize((d, d), Image.LANCZOS)
    k = 4
    m = Image.new('L', (d * k, d * k), 0)
    ImageDraw.Draw(m).ellipse((0, 0, d * k - 1, d * k - 1), fill=255)
    return small, m.resize((d, d), Image.LANCZOS)


SIZES = [320, 180, 110, 40]
LABELS = {
    'profile-navy': 'profile-navy.png  (recommended)',
    'profile-white': 'profile-white.png',
    'profile-orange-ring': 'profile-orange-ring.png',
    'profile-full-logo': 'profile-full-logo.png',
}


def preview():
    pad, gap, row_gap, label_h, cap_h = 48, 44, 40, 46, 30
    panel_w = pad * 2 + sum(SIZES) + gap * (len(SIZES) - 1)
    head_h = 150
    panel_head = 64
    row_h = label_h + max(SIZES) + cap_h + row_gap
    W = panel_w * 2
    H = head_h + panel_head + row_h * len(OPTIONS) + 10
    sheet = Image.new('RGB', (W, H), '#FFFFFF')
    dr = ImageDraw.Draw(sheet)
    dr.rectangle((panel_w, 0, W, H), fill='#000000')
    # header band (spans both panels)
    dr.rectangle((0, 0, W, head_h), fill='#F1F3F7')
    dr.text((pad, 34), 'KRS Tech Solutions  -  Instagram profile picture kit', font=font('Bold', 34), fill=NAVY)
    dr.text((pad, 88), 'Circle crops at true pixel size (1x, no retina help). 110 px = phone profile, '
            '180 / 320 px = web and tap-to-zoom, 40 px = feed and comment avatar.',
            font=font('Regular', 21), fill='#4A5568')
    imgs = {name: Image.open(os.path.join(HERE, name + '.png')).convert('RGB') for name, *_ in OPTIONS}
    for p, (bg, fg, sub, title) in enumerate([('#FFFFFF', '#111111', '#6B7280', 'Light mode - white page'),
                                              ('#000000', '#F5F5F5', '#9CA3AF', 'Dark mode - black page')]):
        ox = p * panel_w
        dr.text((ox + pad, head_h + 22), title.upper(), font=font('Bold', 20), fill=sub)
        for r, (name, *_) in enumerate(OPTIONS):
            oy = head_h + panel_head + r * row_h
            dr.text((ox + pad, oy + 6), LABELS[name], font=font('SemiBold', 25), fill=fg)
            x = ox + pad
            base = oy + label_h + max(SIZES)          # circles share a bottom line
            for d in SIZES:
                small, mask = circle(imgs[name], d)
                y = oy + label_h + (max(SIZES) - d) // 2  # vertically centred in the row
                sheet.paste(small, (x, y), mask)
                cap = f'{d} px'
                tw = dr.textlength(cap, font=font('Medium', 18))
                dr.text((x + (d - tw) / 2, base + 6), cap, font=font('Medium', 18), fill=sub)
                x += d + gap
    out = os.path.join(HERE, 'preview.png')
    sheet.save(out, optimize=True)
    print('wrote', out, sheet.size)


if __name__ == '__main__':
    render_all()
    preview()
