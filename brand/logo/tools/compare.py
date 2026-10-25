"""Score a logo SVG against the reference raster, layer by layer.

The candidate SVG must use viewBox="0 0 550 253" (the reference's pixel grid).
Usage: python3 -I compare.py <candidate.svg> [diff_out.png]
Prints IoU for the navy layer, the orange layer and the combined silhouette,
and writes a diff image: grey = match, red = missing from candidate,
blue = extra in candidate.
"""
import os
import subprocess
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'reference.png')
SCALE = 2  # compare at 2x so sub-pixel edge quality counts


def masks(img):
    img = img.convert('RGBA')
    w, h = img.size
    px = img.load()
    navy = [[False] * w for _ in range(h)]
    orange = [[False] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a < 128:
                continue
            if r > 150 and r - b > 90:
                orange[y][x] = True
            else:
                navy[y][x] = True
    return navy, orange


def iou(a, b):
    inter = union = 0
    for ra, rb in zip(a, b):
        for va, vb in zip(ra, rb):
            inter += va and vb
            union += va or vb
    return inter / union if union else 1.0


def main():
    cand_svg = sys.argv[1]
    diff_out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(cand_svg)[0] + '-diff.png'
    W, H = 550 * SCALE, 253 * SCALE
    cand_png = os.path.splitext(diff_out)[0] + '-render.png'
    subprocess.run(['node', os.path.join(HERE, 'render_svg.js'), cand_svg, cand_png, str(W)], check=True,
                   stdout=subprocess.DEVNULL)
    ref = Image.open(REF).convert('RGBA').resize((W, H), Image.LANCZOS)
    cand = Image.open(cand_png).convert('RGBA')
    if cand.size != (W, H):
        cand = cand.resize((W, H), Image.LANCZOS)
    rn, ro = masks(ref)
    cn, co = masks(cand)
    rs = [[a or b for a, b in zip(x, y)] for x, y in zip(rn, ro)]
    cs = [[a or b for a, b in zip(x, y)] for x, y in zip(cn, co)]
    print(f'IoU navy    {iou(rn, cn):.4f}')
    print(f'IoU orange  {iou(ro, co):.4f}')
    print(f'IoU overall {iou(rs, cs):.4f}')
    diff = Image.new('RGB', (W, H), (255, 255, 255))
    dp = diff.load()
    for y in range(H):
        for x in range(W):
            r_on, c_on = rs[y][x], cs[y][x]
            if r_on and c_on:
                same = (rn[y][x] == cn[y][x])
                dp[x, y] = (170, 170, 170) if same else (230, 160, 0)
            elif r_on:
                dp[x, y] = (220, 30, 30)
            elif c_on:
                dp[x, y] = (30, 80, 230)
    diff.save(diff_out)
    print(f'Diff written to {diff_out} (grey=match, red=missing, blue=extra, amber=wrong colour)')


if __name__ == '__main__':
    main()
