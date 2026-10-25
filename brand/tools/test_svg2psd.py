#!/usr/bin/env python3
"""End-to-end verification of svg2psd.py (run it on the fixture, the trace, or the final artwork).

    python3 test_svg2psd.py --out DIR [--px 4096] [--pydeps DIR] [--ag-psd DIR] [extra.svg ...]

For every input SVG it writes DIR/<label>.psd and checks, independently of the converter's code:
  1. psd-tools re-opens the file; the layer tree (kinds, names, nesting, stacking order) is
     printed from the parsed file and compared with the expectation; every shape layer must be
     kind 'shape' with a vector mask holding the expected number of knots / path operations.
     The low-level structure must re-serialise byte-identically (PSD.read -> write).
  2. psd.composite() (psd-tools' own compositor) vs the rsvg reference render: mean abs
     difference (premultiplied RGBA) and the count of pixels > 10/255 away from AA edges.
     With aggdraw available (--pydeps), also composite(force=True), which ignores stored
     pixels and re-renders every shape layer from its vector mask + SoCo + vstk.
  3. ImageMagick: `convert out.psd[0]` (merged image) vs reference; `identify` lists the layers.
  4. Each shape layer's vector path, rasterised here from the parsed path records, vs that
     element's own rsvg render (IoU > 0.995) and vs the layer's stored pixels.
  5. File size; image resources 1005 (dpi) and 1039 (ICC) present and correct.
  6. Optional (--ag-psd): an independent reader (ag-psd, JavaScript) parses the file.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import svg2psd  # noqa: E402
from kmark import KParams, _is_convex, flat_group, polygon_area_centroid, silhouette, silhouette_d  # noqa: E402
from preview import svg_to_png  # noqa: E402

SVG = "http://www.w3.org/2000/svg"
TOL_MEAN = 2 / 255
TOL_REGION = 10 / 255


# =========================================================================== fixture
def _silhouette_knots(k):
    """Knots kmark.rounded_d produces for the filleted silhouette: 1 per sharp corner,
    2 per filleted corner (+1 when the fillet turns more than 90 deg: split into two cubics)."""
    poly = silhouette(k)
    area, _ = polygon_area_centroid(poly)
    orient = 1 if area > 0 else -1
    n = 0
    for i in range(len(poly)):
        prev, cur, nxt = poly[i - 1], poly[i], poly[(i + 1) % len(poly)]
        r = k.r_convex if _is_convex(prev, cur, nxt, orient) else k.r_concave
        u1 = np.subtract(prev, cur); u2 = np.subtract(nxt, cur)
        ang = math.acos(max(-1, min(1, np.dot(u1, u2) / np.linalg.norm(u1) / np.linalg.norm(u2))))
        if r <= 0 or ang < 1e-6 or abs(math.pi - ang) < 1e-6:
            n += 1
        else:
            n += 2 + (1 if (math.pi - ang) > math.pi / 2 + 1e-9 else 0)
    return n


def build_fixture():
    """Layered SVG exercising every feature of the input convention, plus the expected tree."""
    k = KParams.from_dict(json.load(open(os.path.join(HERE, "trace_params.json"))))
    kr = KParams(**{**asdict(k), "r_convex": 28.0, "r_concave": 10.0})       # rounded variant
    sil = silhouette_d(kr)
    nsil = _silhouette_knots(kr)
    from kmark import facets
    fk = {n: len(p) for n, p in facets(kr).items()}
    pentagram = []
    for i in (0, 2, 4, 1, 3):
        a = math.radians(-90 + 72 * i)
        pentagram.append((-330 + 70 * math.cos(a), 140 + 70 * math.sin(a)))
    star_d = "M" + " L".join(f"{x:.3f} {y:.3f}" for x, y in pentagram) + " Z"
    svg = f'''<svg xmlns="{SVG}" xmlns:xlink="http://www.w3.org/1999/xlink"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" viewBox="0 0 1024 1024" width="1024" height="1024">
<title>svg2psd fixture</title>
<defs>
  <linearGradient id="sheenGrad" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#FFFFFF" stop-opacity="0.95"/><stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/>
  </linearGradient>
  <radialGradient id="glowGrad" cx="150" cy="560" r="90" gradientUnits="userSpaceOnUse"
                  gradientTransform="translate(150 560) scale(1 0.6) rotate(20) translate(-150 -560)">
    <stop offset="0" stop-color="#3B81F1"/><stop offset="0.6" stop-color="#2CA53B" stop-opacity="0.6"/>
    <stop offset="1" stop-color="#2CA53B" stop-opacity="0"/>
  </radialGradient>
  <linearGradient id="fadeGrad" x1="0" x2="1"><stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#000000"/></linearGradient>
  <mask id="fade" maskContentUnits="objectBoundingBox"><rect width="1" height="1" fill="url(#fadeGrad)"/></mask>
  <filter id="shadow" x="-25%" y="-25%" width="150%" height="150%">
    <feGaussianBlur in="SourceAlpha" stdDeviation="8"/><feOffset dx="6" dy="10" result="b"/>
    <feFlood flood-color="#0B0F19" flood-opacity="0.35"/><feComposite in2="b" operator="in"/>
    <feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <clipPath id="badgeClip"><circle cx="0" cy="0" r="80"/></clipPath>
  <clipPath id="bboxClip" clipPathUnits="objectBoundingBox"><ellipse cx=".5" cy=".5" rx=".5" ry=".42"/></clipPath>
</defs>
<style>.ink {{ fill: #0B0F19 }}</style>
<rect id="backdrop" width="1024" height="1024" rx="96" fill="#F4F1EA"/>
<g id="Mark" data-name="K Mark">
  <path id="shadow-silhouette" d="{sil}" fill="#0B0F19" opacity="0.18" transform="translate(10 14)"/>
  <g id="facets" inkscape:label="Facets">{flat_group(kr, trap=False, gid="K")}</g>
  <path id="sheen" fill="url(#sheenGrad)" opacity="0.6"
        d="M300 220 h220 a40 40 0 0 1 40 40 v80 a40 40 0 0 1 -40 40 h-220 a40 40 0 0 1 -40 -40 v-80 a40 40 0 0 1 40 -40 Z"/>
</g>
<g id="Badges" transform="translate(760 800) rotate(-12)">
  <g id="ClippedBadge" clip-path="url(#badgeClip)">
    <rect id="badge-bg" x="-100" y="-100" width="200" height="200" fill="rgb(59, 129, 241)"/>
    <polygon id="badge-tri" points="-100,100 100,-100 100,100" fill="#FAC311" fill-opacity="0.8" style="mix-blend-mode:multiply"/>
  </g>
  <path id="ring" fill="#EE3F2A" fill-rule="evenodd"
        d="M-60 -170 a40 40 0 1 0 80 0 a40 40 0 1 0 -80 0 Z M-40 -170 a20 20 0 1 1 40 0 a20 20 0 1 1 -40 0 Z"/>
  <path id="donut-nz" fill="#2CA53B"
        d="M60 -170 a40 40 0 1 1 80 0 a40 40 0 1 1 -80 0 Z M80 -170 a20 20 0 1 0 40 0 a20 20 0 1 0 -40 0 Z"/>
  <path id="blob-nz" fill="#3B81F1"
        d="M160 -100 a40 40 0 1 1 80 0 a40 40 0 1 1 -80 0 Z M190 -60 a40 40 0 1 1 80 0 a40 40 0 1 1 -80 0 Z"/>
  <path id="star-nz" fill="#FAC311" d="{star_d}"/>
  <path id="curvy" class="ink"
        d="M-200 -60 C-180 -120 -120 -100 -100 -60 S-20 -30 0 -60 Q20 -110 60 -60 T120 -60 q20 30 0 60 t-60 0 s-40 40 -80 0 c-20 -10 -60 -10 -80 0 h-20 v-30 H-210 V-60 z"
        transform="translate(0 -20)"/>
  <ellipse id="oval" cx="-150" cy="60" rx="40" ry="24" fill="#3B81F1" transform="skewX(-20)"/>
  <circle id="dot" cx="140" cy="120" r="18" fill="hsl(5, 85%, 55%)" stroke="#0B0F19" stroke-width="6"/>
</g>
<g id="Tech" opacity="0.85" transform="matrix(0.9 0 0 0.9 40 30)">
  <path id="glow" d="{sil}" fill="#3B81F1" filter="url(#shadow)" transform="translate(-40 600) scale(0.3)"/>
  <rect id="masked" x="300" y="760" width="220" height="80" fill="#EE3F2A" mask="url(#fade)"/>
  <circle id="radial" cx="150" cy="560" r="90" fill="url(#glowGrad)"/>
</g>
<g id="Clips" transform="matrix(0.9 0 0 0.9 40 30)">
  <rect id="bbox-clipped" x="680" y="60" width="200" height="120" fill="#FAC311" clip-path="url(#bboxClip)"/>
  <g id="obbGroup" data-name="OBB clipped" clip-path="url(#bboxClip)">
    <rect id="obb-a" x="560" y="200" width="120" height="60" fill="#2CA53B"/>
    <circle id="obb-b" cx="690" cy="270" r="40" fill="#EE3F2A"/>
  </g>
</g>
<g id="FX" data-name="Shadowed group" filter="url(#shadow)">
  <rect id="fx-a" x="760" y="380" width="120" height="120" rx="18" fill="#2CA53B"/>
  <circle id="fx-b" cx="880" cy="390" r="50" fill="#FAC311"/>
</g>
<g transform="translate(40 40)"><polyline points="0,0 60,0 60,60" fill="#3B81F1"/></g>
<g opacity="0.5">
  <rect x="900" y="40" width="80" height="80" fill="#EE3F2A"/>
  <rect x="930" y="70" width="80" height="80" fill="royalblue"/>
</g>
<line id="guide" x1="0" y1="512" x2="1024" y2="512" stroke="#FF00FF" stroke-width="2" display="none"/>
</svg>'''
    S = lambda name, knots, ops=("combine",), **kw: ("shape", name, dict(knots=knots, ops=list(ops), **kw))  # noqa: E731
    P = lambda name, **kw: ("pixel", name, kw)  # noqa: E731
    G = lambda name, children, **kw: ("group", name, children, kw)  # noqa: E731
    clip = ("combine", "intersect")
    expected = [   # SVG paint order: bottom first
        S("backdrop", 8),
        G("K Mark", [
            S("shadow-silhouette", nsil, opacity=0.18),
            G("Facets", [G("K", [S(f"K-{n}", fk[n] + nsil, clip) for n in ("red", "yellow", "green", "blue")])]),
            P("sheen", opacity=0.6),
        ]),
        G("Badges", [
            G("ClippedBadge", [S("badge-bg", 4 + 4, clip), S("badge-tri", 3 + 4, clip, blend="multiply")]),
            S("ring", 8, ("combine", "exclude")),
            S("donut-nz", 8, ("combine", "exclude")),
            S("blob-nz", 8, ("combine", "combine")),
            S("star-nz", 10, geometry="flattened"),
            S("curvy", 13),
            S("oval", 4),
            S("dot", 4, stroke=True),
        ]),
        G("Tech", [P("glow"), P("masked"), P("radial")], opacity=0.85),
        G("Clips", [S("bbox-clipped", 8, clip), G("OBB clipped", [S("obb-a", 8, clip), S("obb-b", 8, clip)])]),
        G("Shadowed group", [S("fx-a", 8, visible=False), S("fx-b", 4, visible=False),
                             P("Shadowed group (rendered filter)")]),
        S("polyline 1", 3),
        G("Group 1", [S("rect 1", 4), S("rect 2", 4)], opacity=0.5),
        S("guide", 2, visible=False, stroke=True),
    ]
    return svg, expected


def trace_expected():
    k = KParams.from_dict(json.load(open(os.path.join(HERE, "trace_params.json"))))
    from kmark import FACETS, facets
    return [("shape", "disc", dict(knots=4, ops=["combine"])),
            ("group", "K", [("shape", n, dict(knots=len(facets(k)[n]), ops=["combine"])) for n in FACETS], {})]


# =========================================================================== helpers
def premul(a):
    a = a.astype(np.float64)
    out = a.copy()
    out[..., :3] *= a[..., 3:4] / 255.0
    return out


def compare(img, ref, label):
    """Premultiplied-RGBA comparison; anti-aliased edges are where the reference changes."""
    import cv2
    A, B = premul(img), premul(ref)
    d = np.abs(A - B) / 255.0
    dmax = d.max(axis=2)
    k = np.ones((3, 3), np.uint8)
    rb = B.astype(np.float32)
    grad = np.zeros(dmax.shape, np.float32)
    for c in range(4):
        grad = np.maximum(grad, cv2.dilate(rb[..., c], k) - cv2.erode(rb[..., c], k))
    edge = cv2.dilate((grad > 8).astype(np.uint8), k) > 0
    off_edge_bad = int(((dmax > TOL_REGION) & ~edge).sum())
    bs = 32
    h, w = dmax.shape
    blocks = dmax[: h // bs * bs, : w // bs * bs].reshape(h // bs, bs, w // bs, bs).mean(axis=(1, 3))
    res = {"mean_abs": float(d.mean()), "mean_abs_x255": float(d.mean() * 255),
           "max_abs_x255": float(dmax.max() * 255), "px_gt10_total": int((dmax > TOL_REGION).sum()),
           "px_gt10_off_edge": off_edge_bad, "edge_px": int(edge.sum()),
           "worst_32px_block_mean_x255": float(blocks.max() * 255)}
    res["pass"] = res["mean_abs"] < TOL_MEAN and off_edge_bad == 0 and blocks.max() < TOL_REGION
    print(f"    {label}: mean|d|={res['mean_abs_x255']:.4f}/255  max={res['max_abs_x255']:.1f}/255  "
          f">10/255: {res['px_gt10_total']} px (off-edge {off_edge_bad})  worst 32px block "
          f"{res['worst_32px_block_mean_x255']:.3f}/255  -> {'PASS' if res['pass'] else 'FAIL'}")
    return res


def walk_psd(group, depth=0):
    for layer in reversed(list(group)):          # psd-tools lists bottom first; panel shows top first
        yield depth, layer
        if layer.is_group():
            yield from walk_psd(layer, depth + 1)


def vm_stats(layer):
    vm = layer.vector_mask
    from psd_tools.psd.vector import Knot
    knots = sum(sum(isinstance(x, Knot) for x in sp) for sp in vm.paths)
    ops = [{0: "exclude", 1: "combine", 2: "subtract", 3: "intersect"}.get(sp.operation, str(sp.operation))
           for sp in vm.paths]
    return knots, ops


def check_tree(psd, expected):
    """Compare the parsed layer tree with the expectation (SVG order, bottom first)."""
    errors = []

    def rec(layers, exp, path):
        layers = list(layers)                      # bottom first, like `exp`
        if len(layers) != len(exp):
            errors.append(f"{path}: {len(layers)} layers, expected {len(exp)}: "
                          f"{[l.name for l in layers]} vs {[e[1] for e in exp]}")
            return
        for l, e in zip(layers, exp):
            kind, name = e[0], e[1]
            p = f"{path}/{name}"
            if l.name != name:
                errors.append(f"{p}: name {l.name!r}")
            if l.kind != kind:
                errors.append(f"{p}: kind {l.kind!r}, expected {kind!r}")
                continue
            opts = e[3] if kind == "group" else e[2]
            if "opacity" in opts and abs(l.opacity / 255 - opts["opacity"]) > 0.003:
                errors.append(f"{p}: opacity {l.opacity}")
            if l.visible != opts.get("visible", True):
                errors.append(f"{p}: visible {l.visible}")
            if kind == "group":
                rec(l, e[2], p)
            elif kind == "shape":
                if not l.has_vector_mask():
                    errors.append(f"{p}: no vector mask")
                    continue
                knots, ops = vm_stats(l)
                if knots != opts["knots"]:
                    errors.append(f"{p}: {knots} knots, expected {opts['knots']}")
                if ops != opts["ops"]:
                    errors.append(f"{p}: ops {ops}, expected {opts['ops']}")
                if "blend" in opts and opts["blend"] not in str(l.blend_mode).lower():
                    errors.append(f"{p}: blend {l.blend_mode}")
                st = l.stroke
                if bool(st is not None and st.enabled) != bool(opts.get("stroke")):
                    errors.append(f"{p}: stroke enabled={st.enabled if st else None}")
    rec(psd, expected, "")
    return errors


# --------------------------------------------------------------------------- independent rasteriser
def _bez(p0, c1, c2, p3, n):
    t = np.linspace(0, 1, n + 1)[1:, None]
    mt = 1 - t
    return mt ** 3 * p0 + 3 * mt * mt * t * c1 + 3 * mt * t * t * c2 + t ** 3 * p3


def subpath_polyline(sp, W, H):
    """Knot records -> polyline in document pixels (vertical comes first in each knot)."""
    from psd_tools.psd.vector import Knot
    ks = [x for x in sp if isinstance(x, Knot)]
    P = lambda yx: np.array([yx[1] * W, yx[0] * H])  # noqa: E731
    pts = [P(ks[0].anchor)[None]]
    pairs = list(zip(ks, ks[1:] + ks[:1])) if sp.is_closed() else list(zip(ks, ks[1:]))
    for a, b in pairs:
        p0, c1, c2, p3 = P(a.anchor), P(a.leaving), P(b.preceding), P(b.anchor)
        L = np.linalg.norm(c1 - p0) + np.linalg.norm(c2 - c1) + np.linalg.norm(p3 - c2)
        pts.append(_bez(p0, c1, c2, p3, int(min(4096, max(1, math.ceil(L / 0.05))))))
    return np.vstack(pts)


def fill_evenodd(poly, x0, y0, w, h, ss):
    """Exact point sampling (ss x ss samples per pixel, at sample centres) of a closed polygon."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    ya, yb = a[:, 1], b[:, 1]
    m = ya != yb
    a, b = a[m], b[m]
    ylo, yhi = np.minimum(a[:, 1], b[:, 1]), np.maximum(a[:, 1], b[:, 1])
    r0 = np.ceil((ylo - y0) * ss - 0.5).astype(np.int64)
    r1 = np.ceil((yhi - y0) * ss - 0.5).astype(np.int64)     # half-open [r0, r1)
    r0, r1 = np.clip(r0, 0, h * ss), np.clip(r1, 0, h * ss)
    cnt = np.maximum(r1 - r0, 0)
    e = np.repeat(np.arange(len(a)), cnt)
    rows = np.concatenate([np.arange(s, t) for s, t in zip(r0, r1) if t > s]) if cnt.sum() else np.zeros(0, np.int64)
    yc = y0 + (rows + 0.5) / ss
    xa, ya_, xb, yb_ = a[e, 0], a[e, 1], b[e, 0], b[e, 1]
    xs = xa + (yc - ya_) * (xb - xa) / (yb_ - ya_)
    order = np.lexsort((xs, rows))
    rows, xs = rows[order], xs[order]
    col = np.clip(np.ceil((xs - x0) * ss - 0.5).astype(np.int64), 0, w * ss)
    diff = np.zeros((h * ss, w * ss + 1), np.int32)
    sign = np.where(np.arange(len(rows)) % 2 == 0, 1, -1)   # each row has an even crossing count
    np.add.at(diff, (rows, col), sign)
    return np.cumsum(diff[:, :-1], axis=1) > 0


def raster_vector(layer, W, H, box):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    ss = int(max(1, min(8, math.floor(math.sqrt(48e6 / max(1, w * h))))))
    acc = None
    for sp in layer.vector_mask.paths:
        mk = fill_evenodd(subpath_polyline(sp, W, H), x0, y0, w, h, ss)
        op = sp.operation
        acc = mk if acc is None else {1: acc | mk, 0: acc ^ mk, 2: acc & ~mk, 3: acc & mk}[op]
    return acc.reshape(h, ss, w, ss).mean(axis=(1, 3)), ss


def element_only_svg(svg_text, uid):
    """Independent isolation: only element #uid paints (every other drawable gets opacity 0, which
    keeps it in its ancestors' bounding boxes); opacity/blend/filter/mask stripped from the target
    and its ancestors; stroke removed (vector mask = fill area); fill-opacity forced to 1."""
    root = ET.fromstring(svg_text)
    els = list(root.iter())
    target = els[uid]
    parent = {c: p for p in root.iter() for c in p}
    chain, p = [], parent.get(target)
    while p is not None:
        chain.append(p)
        p = parent.get(p)
    keep = set(chain) | {target}
    nonr = {"defs", "clipPath", "mask", "linearGradient", "radialGradient", "filter", "style", "title"}

    def hide(e):
        for c in list(e):
            if c in keep:
                if c is not target:
                    hide(c)
            elif c.tag.split("}")[-1] not in nonr:
                c.set("style", c.get("style", "") + ";opacity:0")
    hide(root)
    for e in chain + [target]:
        for a in ("opacity", "filter", "mask", "display"):
            e.attrib.pop(a, None)
        if "style" in e.attrib:
            e.set("style", ";".join(d for d in e.get("style").split(";")
                                     if d.split(":")[0].strip() not in ("mix-blend-mode", "opacity")))
    target.set("fill-opacity", "1")
    target.set("stroke", "none")
    return ET.tostring(root, encoding="unicode")


def run_ag_psd(path, ag_dir):
    js = r"""
const fs = require('fs'); const ag = require('ag-psd'); const {readPsd} = ag;
ag.initializeCanvas(() => { throw new Error('no canvas'); }, undefined,
  (w, h) => ({width: w, height: h, data: new Uint8ClampedArray(w * h * 4)}));
const buf = fs.readFileSync(process.argv[process.argv.length - 2]);
const psd = readPsd(buf, {skipLayerImageData: false, skipCompositeImageData: false, skipThumbnail: false, useRawThumbnail: true, useImageData: true});
function walk(ls, d, out) { for (const l of [...(ls||[])].reverse()) {
  const vm = l.vectorMask; const knots = vm ? vm.paths.reduce((s,p)=>s+p.knots.length,0) : null;
  out.push({depth:d, name:l.name, group:!!l.children, hidden:!!l.hidden, opacity:l.opacity, blend:l.blendMode,
    fill: l.vectorFill ? l.vectorFill.color : null, knots, ops: vm ? vm.paths.map(p=>p.operation) : null,
    stroke: l.vectorStroke ? {enabled: l.vectorStroke.strokeEnabled, fill: l.vectorStroke.fillEnabled, width: l.vectorStroke.lineWidth} : null,
    bbox:[l.left,l.top,l.right,l.bottom], section: l.sectionDivider ? l.sectionDivider.key : null, pixels: l.imageData ? l.imageData.width*l.imageData.height : 0});
  if (l.children) walk(l.children, d+1, out); } return out; }
const r = psd.imageResources || {};
if (psd.imageData) fs.writeFileSync(process.argv[process.argv.length - 1], Buffer.from(psd.imageData.data.buffer));
console.log(JSON.stringify({width: psd.width, height: psd.height, channels: psd.channels, bitsPerChannel: psd.bitsPerChannel,
  colorMode: psd.colorMode, resolution: r.resolutionInfo, composite: psd.imageData ? [psd.imageData.width, psd.imageData.height] : null,
  thumbnail: r.thumbnailRaw ? [r.thumbnailRaw.width, r.thumbnailRaw.height] : null, xmp: !!r.xmpMetadata, layers: walk(psd.children, 0, [])}));
"""
    raw = path + ".agpsd.rgba"
    p = subprocess.run(["node", "--max-old-space-size=8192", "-e", js, path, raw], capture_output=True, text=True,
                       env=dict(os.environ, NODE_PATH=os.path.join(ag_dir, "node_modules")))
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    out = json.loads(p.stdout)
    if os.path.exists(raw):
        out["composite_rgba"] = np.fromfile(raw, np.uint8).reshape(out["height"], out["width"], 4)
        os.unlink(raw)
    return out


# =========================================================================== verify one file
def verify(svg_text, label, out_dir, px, expected=None, dpi=300, ag_psd=None):
    from psd_tools import PSDImage
    from psd_tools.constants import Resource
    from psd_tools.psd import PSD
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{label}.svg"), "w") as f:
        f.write(svg_text)
    out = os.path.join(out_dir, f"{label}.psd")
    print(f"\n=== {label}: converting at {px}px")
    t = time.time()
    rep = svg2psd.convert(svg_text, out, px=px, dpi=dpi)
    with open(os.path.join(out_dir, f"{label}.report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=str)
    res = {"label": label, "px": px, "convert_s": round(time.time() - t, 2), "bytes": rep["bytes"],
           "MB": round(rep["bytes"] / 1e6, 2), "counts": rep["counts"], "warnings": rep["warnings"]}
    print(f"  wrote {out}: {res['MB']} MB in {res['convert_s']} s, {rep['counts']}")

    # ---------------------------------------------------------------- (1) psd-tools re-open
    psd = PSDImage.open(out)
    print(f"  (1) psd-tools: {psd.width}x{psd.height} depth={psd.depth} mode={psd.color_mode.name} "
          f"channels={psd.channels} layers={len(list(psd.descendants()))}")
    shapes = []
    for depth, l in walk_psd(psd):
        extra = ""
        if l.kind == "shape":
            kn, ops = vm_stats(l)
            soco = l.tagged_blocks.get_data(b"SoCo")
            c = soco[b"Clr "] if soco else None
            col = "#%02X%02X%02X" % tuple(int(round(c[k].value)) for k in (b"Rd  ", b"Grn ", b"Bl  ")) if c else "?"
            st = l.stroke
            extra = (f" fill={col} vmsk: {len(l.vector_mask.paths)} subpaths, {kn} knots, ops={ops}"
                     f" stroke={'on' if st and st.enabled else 'off'} flags=0x{_flags(l):02X}")
            shapes.append(l)
        elif l.kind == "group":
            extra = f" ({'open' if l._setting.kind.value == 1 else 'closed'}, blend={l.blend_mode.name})"
        vis = "" if l.visible else " [hidden]"
        op = f" opacity={l.opacity / 255:.2f}" if l.opacity < 255 else ""
        print(f"      {'  ' * depth}{l.kind:5s} {l.name!r} bbox={l.bbox}{op}{vis}{extra}")
    if expected is not None:
        errs = check_tree(psd, expected)
        res["tree_errors"] = errs
        print(f"      tree vs expectation: {'OK' if not errs else 'ERRORS'}")
        for e in errs:
            print("        !", e)
    raw = open(out, "rb").read()
    low = PSD.read(io.BytesIO(raw))
    buf = io.BytesIO()
    low.write(buf)
    res["roundtrip_identical"] = buf.getvalue() == raw
    ir = low.image_resources
    resinfo = ir.get_data(Resource.RESOLUTION_INFO)
    icc = ir.get_data(Resource.ICC_PROFILE)
    res["dpi_read"] = resinfo.horizontal / 65536.0
    res["icc_bytes"] = len(icc) if icc else 0
    from PIL import ImageCms
    res["icc_desc"] = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip()
    print(f"      low-level re-serialisation byte-identical: {res['roundtrip_identical']}; "
          f"1005 dpi={res['dpi_read']}; 1039 ICC {res['icc_bytes']} bytes ({res['icc_desc']}); "
          f"merged compression={low.image_data.compression.name}; layer channel compression="
          f"{sorted({cd.compression.name for cl in low._get_layer_info().channel_image_data for cd in cl if cd.data})}")

    # ---------------------------------------------------------------- reference
    ref = np.asarray(svg_to_png(svg_text, px))
    Image.fromarray(ref).save(os.path.join(out_dir, f"{label}.ref.png"))

    # ---------------------------------------------------------------- (2) psd-tools composite
    # NB: plain psd.composite() returns the stored merged image when one exists; ignore_preview
    # makes psd-tools composite the layer stack itself (stored layer pixels, groups, blending).
    print("  (2) psd-tools layer compositing vs rsvg reference")
    t = time.time()
    comp = np.asarray(psd.composite(ignore_preview=True).convert("RGBA"))
    Image.fromarray(comp).save(os.path.join(out_dir, f"{label}.psdtools.png"))
    res["composite"] = compare(comp, ref, f"composite(ignore_preview=True) [{time.time() - t:.1f}s]")
    try:
        import aggdraw  # noqa: F401
        t = time.time()
        psd2 = PSDImage.open(out)
        compf = np.asarray(psd2.composite(force=True, ignore_preview=True).convert("RGBA"))
        Image.fromarray(compf).save(os.path.join(out_dir, f"{label}.psdtools_force.png"))
        res["composite_force"] = compare(compf, ref, f"composite(force=True) [{time.time() - t:.1f}s] (shapes from vectors)")
    except ImportError:
        print("    composite(force=True): skipped (aggdraw not importable)")
    merged = np.asarray(psd.topil().convert("RGBA"))
    res["merged_psdtools"] = compare(merged, ref, "merged image (psd.topil)")

    # ---------------------------------------------------------------- (3) ImageMagick
    print("  (3) ImageMagick")
    flat = os.path.join(out_dir, f"{label}.im_flat.png")
    subprocess.run(["convert", out + "[0]", flat], check=True)
    im = np.asarray(Image.open(flat).convert("RGBA"))
    res["imagemagick"] = compare(im, ref, "convert out.psd[0]")
    ident = subprocess.run(["identify", "-format", "%p|%[label]|%wx%h%X%Y|%[opacity]\n", out],
                           capture_output=True, text=True)
    lines = [x for x in ident.stdout.strip().splitlines() if x]
    res["identify_frames"] = len(lines)
    print(f"    identify: {len(lines)} frames (frame 0 = merged image, then every layer record)")
    for x in lines:
        print("      " + x)

    # ---------------------------------------------------------------- (4) vector path IoU
    print("  (4) vector path (parsed records, rasterised here) vs element's own rsvg render")
    ious = []
    for l in shapes:
        e = next((x for x in rep["layers"] if x["kind"] == "shape" and x["name"] == l.name
                  and tuple(x["bbox"]) == tuple(l.bbox)), None)
        if e is None or not e["fill"]:
            print(f"      {l.name!r}: stroke-only path, no fill area to compare - skipped")
            continue
        alpha = np.asarray(svg_to_png(element_only_svg(svg_text, e["uid"]), px))[..., 3] / 255.0
        ys, xs = np.nonzero(alpha)
        vb = l.vector_mask.bbox
        bx = (min(xs.min(), int(math.floor(vb[0] * px))), min(ys.min(), int(math.floor(vb[1] * px))),
              max(xs.max() + 1, int(math.ceil(vb[2] * px))), max(ys.max() + 1, int(math.ceil(vb[3] * px))))
        bx = (max(0, bx[0]), max(0, bx[1]), min(px, bx[2]), min(px, bx[3]))
        cov, ss = raster_vector(l, px, px, bx)
        al = alpha[bx[1]:bx[3], bx[0]:bx[2]]
        iou = float(np.minimum(cov, al).sum() / np.maximum(cov, al).sum())
        stored = np.zeros((px, px))
        la = np.asarray(l.topil().convert("RGBA"))[..., 3] / 255.0
        stored[l.bbox[1]:l.bbox[3], l.bbox[0]:l.bbox[2]] = la
        st = stored[bx[1]:bx[3], bx[0]:bx[2]]
        iou_s = float(np.minimum(cov, st).sum() / np.maximum(cov, st).sum()) if not e["stroke"] else None
        ious.append((l.name, iou, iou_s))
        print(f"      {l.name!r:28s} IoU vs element render = {iou:.5f}"
              + (f"   vs stored pixels = {iou_s:.5f}" if iou_s is not None else "   (stroked: stored pixels include stroke)")
              + f"   [{ss}x{ss} samples/px]  {'PASS' if iou > 0.995 else 'FAIL'}")
    res["iou_min"] = min(i for _, i, _ in ious) if ious else None
    res["iou_stored_min"] = min(s for _, _, s in ious if s is not None) if ious else None
    res["iou"] = {n: round(i, 5) for n, i, _ in ious}

    # ---------------------------------------------------------------- (6) independent reader
    if ag_psd:
        try:
            ag = run_ag_psd(out, ag_psd)
            res["ag_psd"] = {"layers": len(ag["layers"]), "shapes_with_vectors": sum(1 for x in ag["layers"] if x["knots"]),
                             "resolution": ag.get("resolution"), "composite": ag.get("composite"), "thumbnail": ag.get("thumbnail")}
            print(f"  (6) ag-psd (independent JS reader): {ag['width']}x{ag['height']} bits={ag['bitsPerChannel']} "
                  f"colorMode={ag['colorMode']} composite={ag['composite']} xmp={ag['xmp']} thumbnail={ag['thumbnail']} "
                  f"resolution={ag['resolution']}")
            for x in ag["layers"]:
                print(f"      {'  ' * x['depth']}{'group' if x['group'] else ('shape' if x['knots'] else 'pixel')} "
                      f"{x['name']!r} op={x['opacity']:.3f} blend={x['blend']}{' lsct=' + x['section'] if x['section'] else ''}"
                      f"{' hidden' if x['hidden'] else ''}{' pixels=' + str(x['pixels']) if not x['group'] else ''}"
                      + (f" fill={x['fill']} knots={x['knots']} ops={x['ops']} stroke={x['stroke']}" if x['knots'] else ""))
            if "composite_rgba" in ag:
                res["ag_psd"]["merged"] = compare(ag["composite_rgba"], ref, "ag-psd merged image")
            # cross-check knots with psd-tools
            pt = {(l.name, tuple(l.bbox)): vm_stats(l)[0] for l in shapes}
            agk = {x["name"]: x["knots"] for x in ag["layers"] if x["knots"]}
            res["ag_psd"]["knots_agree"] = all(agk.get(n) == k for (n, _), k in pt.items())
            print(f"      ag-psd knot counts agree with psd-tools: {res['ag_psd']['knots_agree']}")
        except Exception as ex:  # pragma: no cover
            res["ag_psd"] = {"error": str(ex)}
            print("  (6) ag-psd failed:", ex)
    print(f"  (5) file size {res['MB']} MB ({res['bytes']} bytes) for {px}x{px}")
    return res


EDGE_PNG = ("iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFklEQVR42mP8z8DwnwEIGBkZGRgYGAAkCAP9q2V3UAAAAABJRU5ErkJggg==")


def edge_cases():
    """(label, svg, expectations) for inputs outside the fixture's happy path."""
    return [
        ("no_namespace", '<svg viewBox="0 0 100 100"><rect id="a" x="10" y="10" width="50" height="50" fill="#f00"/>'
         '<circle cx="60" cy="60" r="30" fill="rgba(0,0,255,0.5)"/></svg>', {"shapes": 2}),
        ("illustrator_css", f'''<svg xmlns="{SVG}" xmlns:xlink="http://www.w3.org/1999/xlink" id="Layer_1" data-name="Layer 1" viewBox="0 0 200 100">
<defs><style>.cls-1{{fill:#ee3f2a;}}.cls-2{{fill:#3b81f1;opacity:.5}}#x{{fill:#2ca53b}}</style>
<symbol id="sym"><circle cx="10" cy="10" r="10" fill="#FAC311"/></symbol></defs>
<g id="Unicode Ünïcødé ✓ 日本"><path class="cls-1" d="M10 10h80v80H10z"/><path class="cls-2" d="M50 10h80v80H50z"/><rect id="x" x="120" y="20" width="30" height="30"/></g>
<use href="#sym" x="160" y="60"/><use xlink:href="#x" x="0" y="50"/><text x="20" y="95" font-size="12" fill="#000">Hi</text>
<image x="170" y="5" width="20" height="20" href="data:image/png;base64,{EDGE_PNG}"/>
<svg x="100" y="50" width="40" height="40" viewBox="0 0 10 10"><rect width="20" height="20" fill="#0f0"/></svg></svg>''', {}),
        ("misc", f'''<svg xmlns="{SVG}" viewBox="10 20 100 50" width="400" height="200">
<defs><clipPath id="empty"></clipPath><clipPath id="obb" clipPathUnits="objectBoundingBox"><rect x="0" y="0" width=".5" height="1"/></clipPath>
<clipPath id="two"><circle cx="30" cy="40" r="10"/><circle cx="50" cy="40" r="10"/></clipPath></defs>
<g id="hiddenGroup" display="none"><rect id="hg" x="10" y="20" width="20" height="20" fill="#000"/></g>
<rect id="pct" x="10" y="20" width="10%" height="20%" fill="#123456" opacity="50%"/>
<path id="broken" d="M10 20 L 30 30 X 40 40" fill="#f0f"/>
<rect id="emptyclip" x="10" y="20" width="10" height="10" fill="#f00" clip-path="url(#empty)"/>
<g id="obbGroup" clip-path="url(#obb)"><rect id="o1" x="60" y="25" width="40" height="20" fill="#0aa"/><circle id="o2" cx="90" cy="50" r="10" fill="#a0a"/></g>
<g id="twoClip" clip-path="url(#two)"><path id="ringx" fill-rule="evenodd" fill="#fa0" d="M20 30h40v20h-40z M35 35h10v10h-10z"/></g>
<path id="dashed" d="M15 65 h40" stroke="#00f" stroke-width="2" stroke-dasharray="4 2"/>
<rect id="vis" x="15" y="45" width="10" height="10" fill="#0f0" visibility="hidden"/>
<g id="blendGrp" style="mix-blend-mode:screen"><rect id="b1" x="80" y="25" width="15" height="15" fill="#808080"/></g>
<g clip-path="url(#two)"><rect id="mulInHoisted" x="30" y="30" width="30" height="20" fill="#0f0" style="mix-blend-mode:multiply"/></g>
<g><path id="zeroArea" d="M10 22 L100 22" fill="#000" stroke="#c00" stroke-width="1"/></g>
<rect id="longname_{"x" * 300}" x="100" y="60" width="5" height="5" fill="#000"/></svg>''',
         {"flattened": ["ringx"]}),
        ("rsvg_obb_bug", f'''<svg xmlns="{SVG}" viewBox="0 0 1024 1024" width="1024" height="1024"><defs><clipPath id="c" clipPathUnits="objectBoundingBox"><rect x=".1" y=".1" width=".8" height=".8"/></clipPath></defs>
<rect id="bg" width="1024" height="1024" fill="#eee"/><g id="Faded" opacity="0.85"><rect id="clipped" x="680" y="60" width="200" height="120" fill="#FAC311" clip-path="url(#c)"/></g></svg>''',
         {"warn": "objectBoundingBox clip-path",
          # ground truth: the same clip in userSpaceOnUse units (rsvg renders that right at any scale)
          "truth": f'''<svg xmlns="{SVG}" viewBox="0 0 1024 1024" width="1024" height="1024"><defs><clipPath id="c"><rect x="700" y="72" width="160" height="96"/></clipPath></defs>
<rect id="bg" width="1024" height="1024" fill="#eee"/><g id="Faded" opacity="0.85"><rect id="clipped" x="680" y="60" width="200" height="120" fill="#FAC311" clip-path="url(#c)"/></g></svg>'''}),
    ]


def run_edge_cases(out_dir, px):
    """Convert each edge case; layer composite (stored pixels, and vectors if aggdraw) vs rsvg."""
    from psd_tools import PSDImage
    os.makedirs(out_dir, exist_ok=True)
    results = []
    print(f"\n=== edge cases at {px}px")
    for label, svg, exp in edge_cases():
        out = os.path.join(out_dir, f"edge_{label}.psd")
        rep = svg2psd.convert(svg, out, px=px)
        ref_svg = svg if "xmlns=" in svg else svg.replace("<svg ", f'<svg xmlns="{SVG}" ', 1)
        if "truth" in exp:                 # the reference renderer is wrong for this input
            print("      reference = equivalent userSpaceOnUse SVG (librsvg mis-renders the original)")
            ref_svg = exp["truth"]
        ref = np.asarray(svg_to_png(ref_svg, px))
        psd = PSDImage.open(out)
        r = {"label": label, "counts": rep["counts"], "warnings": rep["warnings"]}
        print(f"  {label}: {rep['counts']}")
        for w in rep["warnings"]:
            print(f"      warning: {w[:150]}")
        lay = np.asarray(psd.composite(ignore_preview=True).convert("RGBA"))
        r["composite"] = compare(lay, ref, "layers vs rsvg")
        try:
            import aggdraw  # noqa: F401
            r["composite_force"] = compare(np.asarray(psd.composite(force=True, ignore_preview=True).convert("RGBA")),
                                           ref, "vectors vs rsvg")
        except ImportError:
            pass
        ok = True
        if "shapes" in exp:
            ok &= rep["counts"]["shape"] == exp["shapes"]
        for n in exp.get("flattened", []):
            ok &= any(e["name"] == n and e.get("geometry") == "flattened" for e in rep["layers"])
        if "warn" in exp:
            ok &= any(exp["warn"] in w for w in rep["warnings"]) == (px != 1024)
        r["pass"] = bool(ok and r["composite"]["pass"] and r.get("composite_force", {"pass": True})["pass"])
        results.append(r)
    return results


def _flags(layer):
    f = layer._record.flags
    return (f.transparency_protected * 1 | (not f.visible) * 2 | f.photoshop_v5_later * 8 |
            f.pixel_data_irrelevant * 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("svgs", nargs="*", help="extra layered SVGs to convert + verify")
    ap.add_argument("--out", required=True)
    ap.add_argument("--px", type=int, default=4096)
    ap.add_argument("--pydeps", default=None, help="dir with aggdraw (enables composite(force=True))")
    ap.add_argument("--ag-psd", default=None, help="dir whose node_modules has ag-psd")
    ap.add_argument("--skip-fixture", action="store_true")
    ap.add_argument("--skip-trace", action="store_true")
    ap.add_argument("--edge-px", type=int, default=512, help="px for the edge-case suite (0 = skip)")
    a = ap.parse_args()
    if a.pydeps:
        sys.path.append(a.pydeps)
    results = []
    if not a.skip_fixture:
        svg, exp = build_fixture()
        results.append(verify(svg, "fixture", a.out, a.px, exp, ag_psd=a.ag_psd))
    if not a.skip_trace:
        svg = open(os.path.join(HERE, "..", "source", "K-Logo_trace.svg")).read()
        results.append(verify(svg, "trace", a.out, a.px, trace_expected(), ag_psd=a.ag_psd))
    for p in a.svgs:
        results.append(verify(open(p).read(), os.path.splitext(os.path.basename(p))[0], a.out, a.px, None,
                              ag_psd=a.ag_psd))
    edge = run_edge_cases(a.out, a.edge_px) if a.edge_px else []
    with open(os.path.join(a.out, "edge_cases.json"), "w") as f:
        json.dump(edge, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(os.path.join(a.out, "verification.json"), "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    ok = True
    print("\n=== summary")
    for r in results:
        checks = {
            "tree": not r.get("tree_errors"), "roundtrip": r["roundtrip_identical"],
            "composite": r["composite"]["pass"], "merged": r["merged_psdtools"]["pass"],
            "imagemagick": r["imagemagick"]["pass"],
            "iou>0.995": r["iou_min"] is None or r["iou_min"] > 0.995,
        }
        if "composite_force" in r:
            checks["composite(force)"] = r["composite_force"]["pass"]
        ok &= all(checks.values())
        print(f"  {r['label']}: {r['MB']} MB, {r['counts']}, composite mean {r['composite']['mean_abs_x255']:.4f}/255, "
              f"IM mean {r['imagemagick']['mean_abs_x255']:.4f}/255, min IoU {r['iou_min']}, "
              + ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in checks.items()))
    for r in edge:
        ok &= r["pass"]
        print(f"  edge/{r['label']}: layers mean {r['composite']['mean_abs_x255']:.4f}/255"
              + (f", vectors mean {r['composite_force']['mean_abs_x255']:.4f}/255" if "composite_force" in r else "")
              + f" -> {'ok' if r['pass'] else 'FAIL'}")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
