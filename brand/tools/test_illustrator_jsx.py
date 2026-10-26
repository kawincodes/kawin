#!/usr/bin/env python3
"""End-to-end verification of illustrator_jsx.py (the fixture, the logo trace, or the final artwork).

    python3 test_illustrator_jsx.py --out DIR [--acorn NODE_MODULES_DIR] [--px 4096] [final.svg ...]

Runs, for the logo trace (K Mark, K Badge on the circle disc, a squircle app icon and the trace
read back from source/K-Logo_trace.svg), for a two-artboard fixture that exercises the whole
layered-SVG input convention, and for every extra SVG given on the command line:

  1. static: `node --check` on both scripts; the ES3 lint (itself tested on known-bad snippets);
     with --acorn, a strict ECMAScript 3 parse (acorn ecmaVersion 3, reserved words banned as
     property names).
  2. Illustrator: the script runs inside jsx_mocks/ai_mock.js (a strict, recording mock of the
     DOM) twice, starting in the artboard and in the document coordinate system. Asserted from the
     recorded document: artboards (count, names, sizes, no overlap), one layer per variant and one
     sub-layer per top-level item (names, stacking order), pathItems / anchor counts, geometry
     round trip (recorded Illustrator coordinates converted back to SVG space vs the input),
     every fill a SpotColor of a GLOBAL (process) swatch named "K ..." in the "K Brand" group,
     saves (.pdf, .svg, then .ai with PDF compatibility) next to the script, and the script's own
     summary (no failure, self-check deviation, stacking order).
  3. Photoshop: jsx_mocks/ps_mock.js runs the script with new layers appearing beside or inside
     the active group, and with "move PLACEATEND into an empty group" failing. Asserted: one
     px x px document per variant, one root group per variant, every leaf a solid-colour (or
     gradient) fill layer created by the Action Manager "make contentLayer" call from the selected
     path (or a live shape), its vector-mask points vs the input, groups / clips / order, swatches,
     the script's own path and colour-sampler checks, and the saved .psd files.
  4. Geometry vs an INDEPENDENT reference: cairosvg parses the input SVG (its own path parser and
     arc code) and cairo flattens every element at 1e-4 tolerance; each recorded path must lie
     within 0.01 artwork units of it (symmetric Hausdorff distance).
  5. Pixels: the documents recorded by the mocks are turned back into SVG and rendered with
     librsvg; each artboard is compared with librsvg's render of the input.

Exit status 0 = every assertion passed. DIR receives the scripts, mock records, renders and
summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import illustrator_jsx as IJ  # noqa: E402
from kmark import FACETS, KParams, facets, flat_group, svg_doc  # noqa: E402
from preview import svg_to_png  # noqa: E402

MOCKS = os.path.join(HERE, "jsx_mocks")
SVGNS = "http://www.w3.org/2000/svg"
INK = "http://www.inkscape.org/namespaces/inkscape"
GEOM_TOL = 0.01           # artwork units
REF_ERR = 0.001           # error of the cairosvg reference itself (measured <= 0.0006 on circles, 0.0005 on cubics)
RENDER = 1024             # px, pixel comparisons


# =========================================================================== reporting
class Report:
    def __init__(self):
        self.rows = []
        self.section = ""

    def sec(self, name):
        self.section = name
        print(f"\n== {name}")

    def check(self, ok, name, detail=""):
        ok = bool(ok)
        self.rows.append({"section": self.section, "check": name, "ok": ok, "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
        return ok

    def info(self, name, detail):
        self.rows.append({"section": self.section, "check": name, "ok": None, "detail": str(detail)})
        print(f"  [info] {name}  -- {detail}")

    @property
    def failed(self):
        return [r for r in self.rows if r["ok"] is False]


R = Report()


# =========================================================================== fixture
def trace_k():
    with open(os.path.join(HERE, "trace_params.json")) as fh:
        return KParams.from_dict(json.load(fh))


def build_fixtures():
    """Two layered SVGs covering the input convention. Fixture: background disc, nested named
    groups (id / data-name / inkscape:label), a transformed group, solid K facets inside a
    clip-path group whose clip is the arc-rounded silhouette, a gradient-filled arc-rounded path
    (relative commands, 3 stops, stop-opacity, objectBoundingBox) with opacity, a clipped badge
    (rotate + clip-path circle, rgb() colour, fill-opacity), a radial gradient, a drop-shadow
    filter on a group, a white mask on a skewed (matrix) rect, an evenodd ring and a C/S/Q/T
    path. Extras: rounded-rect tile background, elliptical radial gradient (gradientTransform),
    soft luminance mask, Gaussian-blur filter, nested opacity groups."""
    k = trace_k()
    kr = KParams(**{**asdict(k), "r_convex": 26.0, "r_concave": 9.0})
    main = f'''<svg xmlns="{SVGNS}" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:inkscape="{INK}"
     viewBox="0 0 1024 1024" width="1024" height="1024">
<title>Fixture</title>
<defs>
  <linearGradient id="sheen" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#FAC311"/><stop offset="0.55" stop-color="#EE3F2A" stop-opacity="0.9"/>
    <stop offset="1" stop-color="#3B81F1"/>
  </linearGradient>
  <radialGradient id="glowR" cx="860" cy="560" r="70" gradientUnits="userSpaceOnUse">
    <stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#3B81F1"/>
  </radialGradient>
  <clipPath id="circleClip"><circle id="circleClipShape" cx="0" cy="0" r="90"/></clipPath>
  <mask id="whiteMask"><rect id="whiteMaskShape" x="600" y="640" width="300" height="140" rx="30" fill="#FFFFFF"/></mask>
  <filter id="shadow" x="-30%" y="-30%" width="160%" height="160%">
    <feGaussianBlur in="SourceAlpha" stdDeviation="6"/><feOffset dx="8" dy="12" result="b"/>
    <feFlood flood-color="#0B0F19" flood-opacity="0.4"/><feComposite in2="b" operator="in"/>
    <feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
</defs>
<circle id="disc" cx="512" cy="512" r="500" fill="#FEFDFB"/>
<g id="Logo" data-name="K Logo" transform="translate(-60 40) scale(0.85)">
  <g id="Facets" inkscape:label="K Facets">{flat_group(kr, trap=False, gid="K")}</g>
  <path id="sheen-bar" fill="url(#sheen)" opacity="0.7"
        d="M300 220 h220 a40 40 0 0 1 40 40 v80 a40 40 0 0 1 -40 40 h-220 a40 40 0 0 1 -40 -40 v-80 a40 40 0 0 1 40 -40 Z"/>
</g>
<g id="Badge" transform="translate(800 250) rotate(-15)">
  <g id="ClippedBadge" data-name="Clipped badge" clip-path="url(#circleClip)">
    <rect id="badge-bg" x="-100" y="-100" width="200" height="200" fill="rgb(59, 129, 241)"/>
    <polygon id="badge-tri" points="-100,100 100,-100 100,100" fill="#FAC311" fill-opacity="0.85"/>
  </g>
</g>
<circle id="glow-dot" cx="860" cy="560" r="70" fill="url(#glowR)"/>
<g id="Tech" data-name="Tech chip" filter="url(#shadow)">
  <rect id="chip" x="80" y="760" width="200" height="160" rx="24" fill="#2CA53B"/>
  <circle id="chip-dot" cx="250" cy="790" r="34" fill="#FAC311"/>
</g>
<rect id="masked-bar" x="560" y="660" width="380" height="100" fill="#EE3F2A" mask="url(#whiteMask)"
      transform="matrix(1 0 0.2 1 -120 0)"/>
<path id="ring" fill="#3B81F1" fill-rule="evenodd"
      d="M820 870 m-80 0 a80 80 0 1 0 160 0 a80 80 0 1 0 -160 0 Z m40 0 a40 40 0 1 1 80 0 a40 40 0 1 1 -80 0 Z"/>
<path id="wave" fill="#0B0F19" opacity="0.8"
      d="M360 900 C400 840 460 840 500 900 S600 960 640 900 Q660 870 680 900 T720 900 l0 40 h-360 z"/>
</svg>'''
    extras = f'''<svg xmlns="{SVGNS}" viewBox="0 0 512 512" width="512" height="512">
<title>Fixture Extras</title>
<defs>
  <radialGradient id="ellGrad" cx="150" cy="150" r="110" gradientUnits="userSpaceOnUse"
                  gradientTransform="translate(150 150) rotate(25) scale(1 0.55) translate(-150 -150)">
    <stop offset="0" stop-color="#FAC311"/><stop offset="1" stop-color="#EE3F2A"/>
  </radialGradient>
  <linearGradient id="fadeGrad" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#000000"/>
  </linearGradient>
  <mask id="fade" maskContentUnits="objectBoundingBox"><rect width="1" height="1" fill="url(#fadeGrad)"/></mask>
  <filter id="soft" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="6"/></filter>
</defs>
<rect id="tile" width="512" height="512" rx="72" fill="#0B0F19"/>
<ellipse id="oval" cx="150" cy="150" rx="110" ry="70" fill="url(#ellGrad)"/>
<rect id="faded" x="280" y="60" width="180" height="180" fill="#EE3F2A" mask="url(#fade)"/>
<circle id="blurred" cx="380" cy="380" r="70" fill="#FAC311" filter="url(#soft)"/>
<g id="Outer" opacity="0.8">
  <g id="Inner" opacity="0.5">
    <rect id="inner-a" x="60" y="300" width="140" height="120" fill="#3B81F1"/>
    <rect id="inner-b" x="120" y="340" width="140" height="120" fill="#2CA53B"/>
  </g>
</g>
</svg>'''
    return {"Fixture": main, "Fixture Extras": extras}


def superellipse_svg(W, H, fill, n=5.0, steps=2000):
    """True superellipse |u|^n + |v|^n = 1 as a dense polygon (independent of squircle_d)."""
    pts = []
    for i in range(steps):
        t = 2 * math.pi * i / steps
        c, s = math.cos(t), math.sin(t)
        u = math.copysign(abs(c) ** (2 / n), c)
        v = math.copysign(abs(s) ** (2 / n), s)
        pts.append(f"{W / 2 + W / 2 * u:.4f} {H / 2 + H / 2 * v:.4f}")
    return f'<path fill="{fill}" d="M{" L".join(pts)} Z"/>'


# =========================================================================== independent reference
def name_map(svg_text):
    """element id -> layer name (inkscape:label > data-name > id), written independently."""
    root = ET.fromstring(svg_text)
    out = {}
    for el in root.iter():
        i = el.get("id")
        if i:
            out[i] = (el.get(f"{{{INK}}}label") or el.get("data-name") or i).strip()
    return out


def geometry_svg(svg_text):
    """Copy of the input for the geometry reference: filters removed (cairosvg turns feOffset into
    a translation of the element) and every mask expressed as the equivalent clipPath (same id,
    clipPathUnits = maskContentUnits; cairosvg places mask content incorrectly). Neither changes
    the geometry the converter must reproduce: the clip derived from a mask is its silhouette.
    Rounded <rect>s become their SVG 2 path equivalent (cairosvg's own rounded rect is wrong)."""
    for pfx, uri in (("", SVGNS), ("xlink", "http://www.w3.org/1999/xlink"), ("inkscape", INK)):
        ET.register_namespace(pfx, uri)
    root = ET.fromstring(svg_text)
    parent = {c: p for p in root.iter() for c in p}
    ids = {el.get("id"): el for el in root.iter() if el.get("id")}
    for el in list(root.iter()):
        if el.get("filter") is not None:
            del el.attrib["filter"]
        if el.tag == f"{{{SVGNS}}}rect" and (el.get("rx") or el.get("ry")):
            # cairosvg draws rounded rects with a misplaced second handle: use the SVG 2 path equivalent
            x, y = float(el.get("x", 0)), float(el.get("y", 0))
            w, h = float(el.get("width")), float(el.get("height"))
            rx = float(el.get("rx") or el.get("ry"))
            ry = float(el.get("ry") or rx)
            rx, ry = min(rx, w / 2), min(ry, h / 2)
            a = f"A{rx} {ry} 0 0 1"
            el.tag = f"{{{SVGNS}}}path"
            for k in ("x", "y", "width", "height", "rx", "ry"):
                el.attrib.pop(k, None)
            el.set("d", f"M{x + rx} {y} H{x + w - rx} {a} {x + w} {y + ry} V{y + h - ry} {a} {x + w - rx} {y + h} "
                        f"H{x + rx} {a} {x} {y + h - ry} V{y + ry} {a} {x + rx} {y} Z")
        m = el.get("mask")
        if m and m.startswith("url(#") and el.get("clip-path") is None:
            mk = ids.get(m[5:-1])
            if mk is not None and mk.tag.endswith("mask"):
                cp = ET.Element(f"{{{SVGNS}}}clipPath", {"id": mk.get("id"),
                                "clipPathUnits": mk.get("maskContentUnits", "userSpaceOnUse")})
                cp.extend(list(mk))
                par = parent[mk]
                par.insert(list(par).index(mk), cp)
                par.remove(mk)
                ids[mk.get("id")] = cp
            del el.attrib["mask"]
            el.set("clip-path", m)
    return ET.tostring(root, encoding="unicode")


def cairo_reference(svg_text, size, scale=64):
    """Every shape cairosvg draws, flattened by cairo, in root SVG units:
    [{id, stack (ancestor ids), clip, subs: [[(x, y), ...], ...]}]. Drawn on a PDF surface at
    `scale` x so cairo's tolerance floor (1/256 device unit) is ~1e-4 artwork units: measured
    0.0006 max radial error on an r = 70 circle (0.025 at scale 1)."""
    import io
    import cairosvg
    import cairosvg.surface as CS
    captured, stack = [], []
    orig_draw = CS.Surface.draw
    orig_tags = dict(CS.TAGS)
    k = 0.75 * scale                                   # PDF: 1 px = 0.75 pt

    def draw(self, node):
        stack.append(node.get("id"))
        try:
            return orig_draw(self, node)
        finally:
            stack.pop()

    def wrap(fn):
        def inner(surface, node):
            surface.context.set_tolerance(1e-3)
            r = fn(surface, node)
            ctx = surface.context
            subs, cur = [], None
            for kind, pts in ctx.copy_path_flat():
                if kind == 0:
                    cur = [ctx.user_to_device(*pts)]
                    subs.append(cur)
                elif kind == 1:
                    cur.append(ctx.user_to_device(*pts))
            captured.append({"id": node.get("id"), "stack": [s for s in stack if s], "clip": not surface.stroke_and_fill,
                             "subs": [[(x / k, y / k) for x, y in s_] for s_ in subs if len(s_) > 1]})
            return r
        return inner

    CS.Surface.draw = draw
    for t in ("path", "rect", "circle", "ellipse", "polygon", "polyline"):
        CS.TAGS[t] = wrap(orig_tags[t])
    try:
        cairosvg.svg2pdf(bytestring=geometry_svg(svg_text).encode(), write_to=io.BytesIO(), scale=scale)
    finally:
        CS.Surface.draw = orig_draw
        CS.TAGS.clear()
        CS.TAGS.update(orig_tags)
    return captured


def flatten_rec(subs, steps=None, sag=2e-4):
    """[(closed, [[ax, ay, lx, ly, rx, ry, ...], ...]), ...] -> [[(x, y), ...], ...] (closed rings
    repeat no point). Each cubic gets n chords with n^2 >= max|B''| / (8 sag), so the polyline stays
    within `sag` of the curve (or exactly `steps` chords when given)."""
    out = []
    for closed, pts in subs:
        n = len(pts)
        ring = [(pts[0][0], pts[0][1])]
        for i in range(n if closed else n - 1):
            p, q = pts[i], pts[(i + 1) % n]
            if (p[4], p[5]) == (p[0], p[1]) and (q[2], q[3]) == (q[0], q[1]):
                ring.append((q[0], q[1]))
                continue
            if steps is None:
                d2 = 6 * max(math.hypot(p[0] - 2 * p[4] + q[2], p[1] - 2 * p[5] + q[3]),
                             math.hypot(p[4] - 2 * q[2] + q[0], p[5] - 2 * q[3] + q[1]))
                nseg = min(4096, max(8, int(math.ceil(math.sqrt(d2 / (8 * sag))))))
            else:
                nseg = steps
            for k in range(1, nseg + 1):
                t = k / nseg
                u = 1 - t
                ring.append((u ** 3 * p[0] + 3 * u * u * t * p[4] + 3 * u * t * t * q[2] + t ** 3 * q[0],
                             u ** 3 * p[1] + 3 * u * u * t * p[5] + 3 * u * t * t * q[3] + t ** 3 * q[1]))
        out.append(ring)
    return out


def _segments(rings):
    A, B = [], []
    for r in rings:
        r = list(r)
        if len(r) > 1 and r[0] != r[-1]:
            r = r + [r[0]]
        for i in range(len(r) - 1):
            A.append(r[i])
            B.append(r[i + 1])
    return np.array(A, float), np.array(B, float)


def _dist_to(P, A, B):
    out = np.empty(len(P))
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-30)
    for i in range(0, len(P), 512):
        p = P[i:i + 512, None, :]
        t = np.clip(((p - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
        q = A[None] + t[..., None] * AB[None]
        out[i:i + 512] = np.sqrt(((p - q) ** 2).sum(2)).min(1)
    return out


def hausdorff(rings_a, rings_b):
    Pa = np.array([p for r in rings_a for p in r], float)
    Pb = np.array([p for r in rings_b for p in r], float)
    Aa, Ba = _segments(rings_a)
    Ab, Bb = _segments(rings_b)
    return max(_dist_to(Pa, Ab, Bb).max(), _dist_to(Pb, Aa, Ba).max())


# =========================================================================== static checks
def node_check(path):
    tmp = path + ".check.js"
    shutil.copyfile(path, tmp)
    r = subprocess.run(["node", "--check", tmp], capture_output=True, text=True)
    os.remove(tmp)
    return r.returncode == 0, (r.stderr.strip().splitlines() or [""])[0]


def acorn_check(path, acorn_dir):
    js = ("const acorn = require(process.argv[1]); const fs = require('fs');"
          "try { const ast = acorn.parse(fs.readFileSync(process.argv[2], 'utf8'), {ecmaVersion: 3, allowReserved: 'never', sourceType: 'script'});"
          " console.log('ok ' + ast.body.length); } catch (e) { console.log('ERR ' + e.message); process.exit(1); }")
    r = subprocess.run(["node", "-e", js, os.path.join(acorn_dir, "acorn"), path], capture_output=True, text=True)
    return r.returncode == 0, (r.stdout.strip() or r.stderr.strip())


BAD_SNIPPETS = {
    "let": "let a = 1;", "const": "const a = 1;", "arrow": "var f = function () {}; var g = (a) => a;",
    "template": "var s = `x${1}`;", "class": "class A {}", "forEach": "[1].forEach(function (x) {});",
    "map": "var b = [1].map(function (x) { return x; });", "filter": "var b = [1].filter(function (x) { return x; });",
    "Object.keys": "var k = Object.keys({});", "JSON.stringify": "var s = JSON.stringify({});",
    "JSON.parse": "var o = JSON.parse('{}');", "trailing comma": "var o = {a: 1,};", "for...of": "for (var x of [1]) {}",
    "default param": "function f(a = 1) { return a; }", "spread": "var a = [...[1]];",
    "reserved key": "var o = {class: 1};", "Array.isArray": "var t = Array.isArray([]);",
}
# The lint is deliberately conservative: it cannot tell arrays from strings, so it rejects every
# .indexOf (Array.prototype.indexOf is ES5) although String.prototype.indexOf is ES3.
GOOD_SNIPPET = ("var a = [1, 2], i, o = {'class': 1, b: [3]}; for (i = 0; i < a.length; i++) { a[i] += o.b[0]; }"
                " var s = 'x'.charAt(0) + 'y'.substring(0, 1); function f(x) { return x * 2; } var r = /a+/g.test('aa');"
                " try { f(1); } catch (e) { s = String(e); } finally { i = 0; }")


def static_checks(paths, acorn_dir):
    R.sec("ES3 lint self-test")
    missed = [k for k, v in BAD_SNIPPETS.items() if not IJ.es3_lint(v)]
    R.check(not missed, f"lint rejects all {len(BAD_SNIPPETS)} banned constructs", "missed: " + ", ".join(missed) if missed else
            ", ".join(BAD_SNIPPETS))
    R.check(not IJ.es3_lint(GOOD_SNIPPET), "lint accepts a clean ES3 snippet", IJ.es3_lint(GOOD_SNIPPET))
    R.sec("generated scripts: parse and ES3 conformance")
    for p in paths:
        ok, msg = node_check(p)
        R.check(ok, f"node --check {os.path.basename(p)}", msg)
        probs = IJ.es3_lint(open(p).read())
        R.check(not probs, f"ES3 lint {os.path.basename(p)}", f"{len(probs)} problems {probs[:3]}" if probs else "0 problems")
        if acorn_dir:
            ok, msg = acorn_check(p, acorn_dir)
            R.check(ok, f"acorn ecmaVersion 3 parse {os.path.basename(p)}", msg)


# =========================================================================== converter unit checks
def converter_checks():
    R.sec("svg_path_to_mlcz")
    cases = [
        ("M10 10 h50 v20 h-50 z", "M10 10 L60 10 L60 30 L10 30 Z"),
        ("m10 10 20 0 0 20 -20 0z", "M10 10 L30 10 L30 30 L10 30 Z"),
        ("M0,0L1e1,0l0-10Z", "M0 0 L10 0 L10 -10 Z"),
        ("M0 0 C10 0 20 10 20 20 S30 40 40 40", "M0 0 C10 0 20 10 20 20 C20 30 30 40 40 40"),
        ("M0 0 Q10 0 10 10 T20 20", "M0 0 C6.6667 0 10 3.3333 10 10 C10 16.6667 13.3333 20 20 20"),
    ]
    for d, want in cases:
        got = IJ.svg_path_to_mlcz(d)
        R.check(got == want, f"{d!r} -> {want!r}", got if got != want else "")
    a = IJ.svg_path_to_mlcz("M0 0a5 5 0 1010 0")
    b = IJ.svg_path_to_mlcz("M0 0 a5 5 0 1 0 10 0")
    R.check(a == b and "A" not in a, "compact arc flags == spaced flags, arcs become C", a[:60])
    worst = 0.0
    for r in (1.0, 26.0, 512.0, 5000.0):
        d = IJ.svg_path_to_mlcz(IJ.ellipse_d(0, 0, r, r))
        for ring in flatten_rec([(True, p) for p in [[[*q[:6]] for q in IJ.subpath_points(sp)] for sp in IJ.parse_path(d)]], 256):
            worst = max(worst, max(abs(math.hypot(x, y) - r) for x, y in ring))
    R.check(worst <= GEOM_TOL, "arcs -> cubics radial error <= 0.01 for r = 1 .. 5000", f"max {worst:.5f}")
    d = IJ.svg_path_to_mlcz("M0 0 h100 v50 h-100 z", transform="rotate(30 50 25) translate(5 5)")
    m = IJ.parse_transform("rotate(30 50 25) translate(5 5)")
    want = [IJ.mat_apply(m, *p) for p in ((0, 0), (100, 0), (100, 50), (0, 50))]
    got = [(float(x), float(y)) for x, y in re.findall(r"([-\d.]+) ([-\d.]+)", d)]
    R.check(max(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(got, want)) < 1e-3, "transform argument baked in", d)
    # independent reference for curves: cairosvg
    svg = (f'<svg xmlns="{SVGNS}" viewBox="0 0 400 400" width="400" height="400"><path id="p" fill="#000" '
           f'd="M20 200 C40 120 120 120 140 200 S240 280 260 200 Q280 160 300 200 T340 200 a30 50 20 1 1 -40 60 l-20 40 h-260 z"/></svg>')
    ref = [c for c in cairo_reference(svg, 400) if c["id"] == "p"][0]["subs"]
    mine = [[[*q[:6]] for q in IJ.subpath_points(sp)] for sp in IJ.parse_path(IJ.svg_path_to_mlcz(re.search(r' d="([^"]+)"', svg).group(1)))]
    h = hausdorff(flatten_rec([(True, p) for p in mine]), ref)
    R.check(h <= GEOM_TOL, "C/S/Q/T/A path vs cairosvg reference (Hausdorff)", f"{h:.5f} units")


# =========================================================================== mocks
def run_mock(kind, script, record, *args):
    js = os.path.join(MOCKS, f"{kind}_mock.js")
    r = subprocess.run(["node", js, script, record, *args], capture_output=True, text=True, timeout=600)
    with open(record) as fh:
        rec = json.load(fh)
    return rec, (r.stdout + r.stderr).strip()


# ---------------------------------------------------------------- Illustrator record helpers
def ai_leaves(doc, layer, ab, s):
    """Paint-order list of drawn leaves / clip paths of one variant layer, in SVG units."""
    L, T = ab[0], ab[1]

    def conv(points):
        return [[(p[0] - L) / s, (T - p[1]) / s, (p[2] - L) / s, (T - p[3]) / s, (p[4] - L) / s, (T - p[5]) / s, p[6]] for p in points]

    out = []

    def visit(it, path):
        if it["cls"] == "PathItem":
            out.append({"name": it["name"], "clip": it["clipping"], "subs": [(it["closed"], conv(it["points"]))],
                        "fill": it["fill"], "item": it, "path": path, "live": it.get("live")})
        elif it["cls"] == "CompoundPathItem":
            m = it["items"]
            out.append({"name": it["name"], "clip": bool(m and m[0]["clipping"]),
                        "subs": [(x["closed"], conv(x["points"])) for x in reversed(m)],
                        "fill": m[0]["fill"] if m else None, "item": it, "path": path, "members": m})
        else:
            for c in reversed(it["items"]):
                visit(c, path + [it["name"]])
            if it.get("opacityMask"):
                visit(it["opacityMask"], path + [it["name"], "(opacity mask)"])

    def vlayer(lay, path):
        for sub in reversed(lay["layers"]):
            vlayer(sub, path + [sub["name"]])
        for it in reversed(lay["items"]):
            visit(it, path)

    vlayer(layer, [layer["name"]])
    return out


def ai_svg(doc, layer, ab, s, W, H):
    """Recorded Illustrator layer -> SVG (artboard frame) for rendering."""
    L, T = ab[0], ab[1]
    spots = {x["id"]: x for x in doc["spots"]}
    grads = {g["id"]: g for g in doc["gradients"]}
    defs, uid = [], [0]

    def nid(p):
        uid[0] += 1
        return f"{p}{uid[0]}"

    def P(x, y):
        return (x - L) / s, (T - y) / s

    def colour(c):
        if c["t"] == "rgb":
            rgb = c["rgb"]
        elif c["t"] == "spot":
            base = spots[c["spot"]]["color"]["rgb"]
            k = c["tint"] / 100
            rgb = [255 - (255 - v) * k for v in base]
        else:
            raise ValueError(c)
        return "#%02X%02X%02X" % tuple(int(round(v)) for v in rgb)

    def d_of(members):
        parts = []
        for m in members:
            pts, n = m["points"], len(m["points"])
            parts.append("M%.4f %.4f" % P(pts[0][0], pts[0][1]))
            for i in range(n if m["closed"] else n - 1):
                p, q = pts[i], pts[(i + 1) % n]
                parts.append("C%.4f %.4f %.4f %.4f %.4f %.4f" % (*P(p[4], p[5]), *P(q[2], q[3]), *P(q[0], q[1])))
            if m["closed"]:
                parts.append("Z")
        return " ".join(parts)

    def paint(f):
        if f["t"] in ("rgb", "spot"):
            return f'fill="{colour(f)}"'
        if f["t"] == "gradient":
            g = grads[f["gradient"]]
            a, b, c, d, e, ff = f["m"]
            M = (a / s, -b / s, c / s, -d / s, (e - L) / s, (T - ff) / s)
            gid = nid("g")
            stops = "".join(f'<stop offset="{st["ramp"] / 100:.5f}" stop-color="{colour(st["color"])}" stop-opacity="{st["opacity"] / 100:.5f}"/>'
                            for st in g["stops"])
            mt = "matrix(" + " ".join(f"{v:.6f}" for v in M) + ")"
            if g["type"].endswith("RADIAL"):
                defs.append(f'<radialGradient id="{gid}" gradientUnits="userSpaceOnUse" cx="0" cy="0" r="1" gradientTransform="{mt}">{stops}</radialGradient>')
            else:
                defs.append(f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="1" y2="0" gradientTransform="{mt}">{stops}</linearGradient>')
            return f'fill="url(#{gid})"'
        return 'fill="none"'

    def effects(it):
        fx = it.get("effects") or []
        if not fx:
            return ""
        prims = []
        for xml in fx:
            name = re.search(r'LiveEffect name="([^"]+)"', xml).group(1)
            kv = {m.group(2): float(m.group(3)) for m in re.finditer(r"([RIB]) (\w+) ([-\d.e]+)", xml)}
            if name == "Adobe PSL Gaussian Blur":
                prims.append(("blur", kv["blur"] / s))
                continue
            sc = re.search(r'Fill color="2 ([\d.]+) ([\d.]+) ([\d.]+)"', xml)
            col = "#%02X%02X%02X" % tuple(int(round(float(v) * 255)) for v in sc.groups()) if sc else "#000000"
            sig = kv["blur"] / 2 / s
            dx, dy = (kv.get("horz", 0) / s, kv.get("vert", 0) / s)
            prims.append(("shadow", sig, dx, dy, col, kv["opac"]))
        fid = nid("f")
        body, ins = [], []
        for i, p in enumerate(prims):
            if p[0] == "blur":
                body.append(f'<feGaussianBlur in="SourceGraphic" stdDeviation="{p[1]:.5f}"/>')
            else:
                body.append(f'<feGaussianBlur in="SourceAlpha" stdDeviation="{p[1]:.5f}"/><feOffset dx="{p[2]:.5f}" dy="{p[3]:.5f}" result="o{i}"/>'
                            f'<feFlood flood-color="{p[4]}" flood-opacity="{p[5]:.5f}"/><feComposite in2="o{i}" operator="in" result="s{i}"/>')
                ins.append(f"s{i}")
        if ins:
            body.append("<feMerge>" + "".join(f'<feMergeNode in="{x}"/>' for x in ins) + '<feMergeNode in="SourceGraphic"/></feMerge>')
        defs.append(f'<filter id="{fid}" x="-50%" y="-50%" width="200%" height="200%">{"".join(body)}</filter>')
        return f' filter="url(#{fid})"'

    def op(it):
        o = it.get("opacity", 100)
        return f' opacity="{o / 100:.5f}"' if o != 100 else ""

    def emit(it):
        if it.get("hidden"):
            return ""
        if it["cls"] == "PathItem":
            if it["clipping"] or not it["filled"]:
                return ""
            rule = ' fill-rule="evenodd"' if it["evenodd"] else ""
            return f'<path {paint(it["fill"])}{rule}{op(it)}{effects(it)} d="{d_of([it])}"/>'
        if it["cls"] == "CompoundPathItem":
            m = it["items"]
            if not m or m[0]["clipping"] or not m[0]["filled"]:
                return ""
            rule = ' fill-rule="evenodd"' if m[0]["evenodd"] else ""
            return f'<path {paint(m[0]["fill"])}{rule}{op(it)}{effects(it)} d="{d_of(list(reversed(m)))}"/>'
        items = it["items"]
        attrs = op(it) + effects(it)
        if it.get("clipped"):
            cl = items[0]
            mem = cl["items"] if cl["cls"] == "CompoundPathItem" else [cl]
            rule = "evenodd" if mem[0]["evenodd"] else "nonzero"
            cid = nid("c")
            defs.append(f'<clipPath id="{cid}"><path clip-rule="{rule}" d="{d_of(list(reversed(mem)))}"/></clipPath>')
            inner = "".join(emit(c) for c in reversed(items[1:]))
            return f'<g{attrs}><g clip-path="url(#{cid})">{inner}</g></g>'
        inner = "".join(emit(c) for c in reversed(items))
        if it.get("opacityMask"):
            mid = nid("m")
            defs.append(f'<mask id="{mid}" maskUnits="userSpaceOnUse" x="-10000" y="-10000" width="20000" height="20000">{emit(it["opacityMask"])}</mask>')
            inner = f'<g mask="url(#{mid})">{inner}</g>'
        return f"<g{attrs}>{inner}</g>"

    def elayer(lay):
        return "".join(elayer(x) for x in reversed(lay["layers"])) + "".join(emit(i) for i in reversed(lay["items"]))

    body = elayer(layer)
    return (f'<svg xmlns="{SVGNS}" viewBox="0 0 {W} {H}" width="{W}" height="{H}"><defs>{"".join(defs)}</defs>{body}</svg>')


# ---------------------------------------------------------------- Photoshop record helpers
def ps_region(vmask, W, H):
    """vector mask -> shapely geometry in px."""
    from shapely.geometry import Point, Polygon, box
    from shapely import affinity
    if vmask is None:
        return box(0, 0, W, H)
    if "shape" in vmask:
        s = vmask["shape"]
        if s["type"] == "ellipse":
            cx, cy = (s["left"] + s["right"]) / 2, (s["top"] + s["bottom"]) / 2
            return affinity.scale(Point(cx, cy).buffer(1, 512), (s["right"] - s["left"]) / 2, (s["bottom"] - s["top"]) / 2)
        r = s.get("radius", 0)
        b = box(s["left"] + r, s["top"] + r, s["right"] - r, s["bottom"] - r)
        return b.buffer(r, 128) if r > 0 else box(s["left"], s["top"], s["right"], s["bottom"])
    g = None
    for i, c in enumerate(vmask["components"]):
        ring = flatten_rec([(c["closed"], c["points"])], 48)[0]
        poly = Polygon(ring)
        if not poly.is_valid:
            import shapely
            poly = shapely.make_valid(poly)          # keeps every lobe of a self-intersecting ring
        if g is None:
            g = poly if c["op"] not in ("SHAPESUBTRACT", "SHAPEINTERSECT") else Polygon()
        elif c["op"] == "SHAPEADD":
            g = g.union(poly)
        elif c["op"] == "SHAPESUBTRACT":
            g = g.difference(poly)
        elif c["op"] == "SHAPEXOR":
            g = g.symmetric_difference(poly)
        else:
            g = g.intersection(poly)
    return g


def geom_d(g):
    parts = []
    for poly in getattr(g, "geoms", [g]):
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        for ring in [poly.exterior, *poly.interiors]:
            c = list(ring.coords)
            parts.append("M" + " L".join(f"{x:.3f} {y:.3f}" for x, y in c) + " Z")
    return " ".join(parts) or "M0 0 Z"


def ps_svg(snap):
    W, H = snap["width"], snap["height"]
    defs, uid = [], [0]

    def nid(p):
        uid[0] += 1
        return f"{p}{uid[0]}"

    def hexc(rgb):
        return "#%02X%02X%02X" % tuple(int(round(v)) for v in rgb)

    def fill_of(f):
        if f["t"] == "solid":
            return f'fill="{hexc(f["rgb"])}"'
        a = math.radians(f["angle"])
        span = (W * abs(math.cos(a)) + H * abs(math.sin(a))) * f["scale"] / 100
        mx, my = W / 2 + f["ox"] / 100 * W, H / 2 + f["oy"] / 100 * H
        cs = "".join(f'<stop offset="{c["loc"]:.5f}" stop-color="{hexc(c["rgb"])}"/>' for c in f["colors"])
        gid = nid("g")
        # colour and opacity ramps are separate in Photoshop; merge onto the colour stop positions
        al = f["alphas"]

        def alpha_at(t):
            if t <= al[0]["loc"]:
                return al[0]["op"] / 100
            for i in range(1, len(al)):
                if t <= al[i]["loc"]:
                    u = (t - al[i - 1]["loc"]) / max(al[i]["loc"] - al[i - 1]["loc"], 1e-9)
                    return (al[i - 1]["op"] + (al[i]["op"] - al[i - 1]["op"]) * u) / 100
            return al[-1]["op"] / 100
        locs = sorted({c["loc"] for c in f["colors"]} | {x["loc"] for x in al})

        def col_at(t):
            cc = f["colors"]
            if t <= cc[0]["loc"]:
                return cc[0]["rgb"]
            for i in range(1, len(cc)):
                if t <= cc[i]["loc"]:
                    u = (t - cc[i - 1]["loc"]) / max(cc[i]["loc"] - cc[i - 1]["loc"], 1e-9)
                    return [a0 + (b0 - a0) * u for a0, b0 in zip(cc[i - 1]["rgb"], cc[i]["rgb"])]
            return cc[-1]["rgb"]
        cs = "".join(f'<stop offset="{t:.5f}" stop-color="{hexc(col_at(t))}" stop-opacity="{alpha_at(t):.5f}"/>' for t in locs)
        if f["type"] == "radial":
            defs.append(f'<radialGradient id="{gid}" gradientUnits="userSpaceOnUse" cx="{mx:.4f}" cy="{my:.4f}" r="{span / 2:.4f}">{cs}</radialGradient>')
        else:
            ux, uy = math.cos(a), -math.sin(a)
            defs.append(f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{mx - ux * span / 2:.4f}" y1="{my - uy * span / 2:.4f}" '
                        f'x2="{mx + ux * span / 2:.4f}" y2="{my + uy * span / 2:.4f}">{cs}</linearGradient>')
        return f'fill="url(#{gid})"'

    def fx_of(L):
        prims = []
        for k, v in (L.get("styles") or {}).items():
            if not isinstance(v, dict) or v.get("obj") not in ("dropShadow", "outerGlow"):
                continue
            e = v["v"]
            col = hexc([e["color"]["v"]["red"], e["color"]["v"]["grain"], e["color"]["v"]["blue"]])
            sig = e["blur"]["v"] / 2
            dx = dy = 0.0
            if v["obj"] == "dropShadow":
                a = math.radians(e["localLightingAngle"]["v"])
                dx, dy = -e["distance"]["v"] * math.cos(a), e["distance"]["v"] * math.sin(a)
            prims.append(("shadow", sig, dx, dy, col, e["opacity"]["v"] / 100))
        for f in L.get("filters") or []:
            prims.append(("blur", f["blur"]))
        if not prims:
            return ""
        fid = nid("f")
        body, ins = [], []
        for i, p in enumerate(prims):
            if p[0] == "blur":
                body.append(f'<feGaussianBlur in="SourceGraphic" stdDeviation="{p[1]:.4f}"/>')
            else:
                body.append(f'<feGaussianBlur in="SourceAlpha" stdDeviation="{p[1]:.4f}"/><feOffset dx="{p[2]:.4f}" dy="{p[3]:.4f}" result="o{i}"/>'
                            f'<feFlood flood-color="{p[4]}" flood-opacity="{p[5]:.5f}"/><feComposite in2="o{i}" operator="in" result="s{i}"/>')
                ins.append(f"s{i}")
        if ins:
            body.append("<feMerge>" + "".join(f'<feMergeNode in="{x}"/>' for x in ins) + '<feMergeNode in="SourceGraphic"/></feMerge>')
        defs.append(f'<filter id="{fid}" x="-50%" y="-50%" width="200%" height="200%">{"".join(body)}</filter>')
        return f' filter="url(#{fid})"'

    def emit(L):
        if not L["visible"]:
            return ""
        o = L["opacity"] / 100
        if L["cls"] == "LayerSet":
            inner = "".join(emit(c) for c in reversed(L["layers"]))
            if L.get("vmask"):
                cid = nid("c")
                defs.append(f'<clipPath id="{cid}"><path clip-rule="evenodd" d="{geom_d(ps_region(L["vmask"], W, H))}"/></clipPath>')
                inner = f'<g clip-path="url(#{cid})">{inner}</g>'
            return f'<g opacity="{o:.5f}"{fx_of(L)}>{inner}</g>'
        if not L.get("fill"):
            return ""
        o *= L["fillOpacity"] / 100
        return f'<path fill-rule="evenodd" {fill_of(L["fill"])} opacity="{o:.5f}"{fx_of(L)} d="{geom_d(ps_region(L.get("vmask"), W, H))}"/>'

    body = "".join(emit(L) for L in reversed(snap["layers"]))
    return f'<svg xmlns="{SVGNS}" viewBox="0 0 {W} {H}" width="{W}" height="{H}"><defs>{"".join(defs)}</defs>{body}</svg>'


def ps_leaves(layers, path=()):
    out = []
    for L in reversed(layers):
        if L["cls"] == "LayerSet":
            out += ps_leaves(L["layers"], path + (L["name"],))
        else:
            out.append(dict(L, path=list(path)))
    return out


def ps_groups(layers, path=()):
    out = []
    for L in reversed(layers):
        if L["cls"] == "LayerSet":
            out.append(dict(L, path=list(path)))
            out += ps_groups(L["layers"], path + (L["name"],))
    return out


# ---------------------------------------------------------------- pixels
def premul(im):
    a = np.asarray(im.convert("RGBA")).astype(np.float64)
    a[..., :3] *= a[..., 3:4] / 255.0
    return a


def compare(img, ref):
    import cv2
    A, B = premul(img), premul(ref)
    d = np.abs(A - B)
    lum = B.max(axis=2).astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    edge = (cv2.dilate(lum, k) - cv2.erode(lum, k)) > 4
    edge = cv2.dilate(edge.astype(np.uint8), k).astype(bool)
    pix = d.max(axis=2)
    bad = (pix > 10) & ~edge
    return {"mean": float(d.mean()), "max": float(pix.max()), "bad": int(bad.sum()), "bad_pct": 100.0 * bad.sum() / bad.size}


def save_pair(out_dir, label, img, ref):
    w, h = ref.size
    sheet = Image.new("RGBA", (w * 3, h), (255, 255, 255, 255))
    sheet.alpha_composite(ref, (0, 0))
    sheet.alpha_composite(img, (w, 0))
    d = np.abs(premul(img) - premul(ref)).max(axis=2)
    dm = Image.fromarray(np.clip(d * 4, 0, 255).astype(np.uint8)).convert("RGBA")
    sheet.alpha_composite(dm, (2 * w, 0))
    p = os.path.join(out_dir, f"{label}.png")
    sheet.convert("RGB").save(p)
    return p


# =========================================================================== one run
def check_run(label, variants, palette, refs, out_dir, acorn_dir, px, independent=None, approximations=None, expect=None):
    """expect: {variant name: {"paths": n, "points": n, "fills": {leaf name: palette key}}} computed
    independently of the converter (used for the trace)."""
    """refs: variant name -> reference SVG text (rendered with librsvg).
    independent: variant name -> (svg_text) for the cairosvg geometry reference, or polygons."""
    run_dir = os.path.join(out_dir, label)
    os.makedirs(run_dir, exist_ok=True)
    ai_path = os.path.join(run_dir, f"{label}_Illustrator.jsx")
    ps_path = os.path.join(run_dir, f"{label}_Photoshop.jsx")
    ai = IJ.illustrator_script(variants, palette, ai_path, base=label)
    ps = IJ.photoshop_script(variants, palette, ps_path, px=px, base=label)
    R.sec(f"[{label}] generation")
    R.info("variants", ", ".join(v["name"] for v in variants))
    R.info("Illustrator script", f"{ai['bytes']} bytes, counts {dict((k, v) for k, v in ai['counts'].items() if v)}")
    R.info("Photoshop script", f"{ps['bytes']} bytes, counts {dict((k, v) for k, v in ps['counts'].items() if v)}")
    for w in sorted(set(ai["warnings"]) | set(ps["warnings"])):
        R.info("converter warning", w)
    static_checks([ai_path, ps_path], acorn_dir)
    model = ai["model"]
    approximations = approximations or {}
    N = len(variants)
    names = [v["name"] for v in model.variants]
    result = {"label": label, "ai": {}, "ps": {}}

    # ------------------------------------------------------------- Illustrator
    for coord in ("artboard", "document"):
        R.sec(f"[{label}] Illustrator mock (script starts in {coord} coordinates)")
        rec, log = run_mock("ai", ai_path, os.path.join(run_dir, f"ai_{coord}.json"), "--coord", coord)
        R.check(not rec["crashed"], "script ran to completion", rec["crashed"] or log)
        R.check(not rec["errors"], "no DOM misuse (unknown members, bad enums / types / ranges)", rec["errors"][:3])
        alert = rec["alerts"][0] if rec["alerts"] else ""
        R.check(len(rec["alerts"]) == 1 and "build failed" not in alert, "one summary alert, no failure", alert.splitlines()[0] if alert else "none")
        m = re.search(r"max anchor/handle deviation ([\d.]+)", alert)
        R.check(m and float(m.group(1)) <= GEOM_TOL and "could not be checked" not in alert,
                "script's own read-back geometry check", m.group(0) if m else alert[:200])
        R.check("Stacking order: as designed" in alert, "script's own stacking-order check", re.search(r"Stacking order: .*", alert).group(0) if "Stacking order" in alert else "missing")
        doc = rec["document"]
        if not doc:
            continue
        R.check(doc["space"].endswith("RGB"), "RGB document", doc["space"])
        R.check(rec["finalCoord"].endswith(coord.upper() + "COORDINATESYSTEM"), "coordinate system restored", rec["finalCoord"])
        abs_ = doc["artboards"]
        R.check(len(abs_) == N and [a["name"] for a in abs_] == names, f"{N} artboards named after the variants", [a["name"] for a in abs_])
        sizes_ok = all(abs(a["rect"][2] - a["rect"][0] - v["W"]) < 1e-6 and abs(a["rect"][1] - a["rect"][3] - v["H"]) < 1e-6
                       for a, v in zip(abs_, model.variants))
        R.check(sizes_ok, "artboard sizes = artboard_size (1 pt per unit)", [(round(a["rect"][2] - a["rect"][0], 3), round(a["rect"][1] - a["rect"][3], 3)) for a in abs_])
        overl = [(i, j) for i in range(N) for j in range(i + 1, N)
                 if min(abs_[i]["rect"][2], abs_[j]["rect"][2]) > max(abs_[i]["rect"][0], abs_[j]["rect"][0])
                 and min(abs_[i]["rect"][1], abs_[j]["rect"][1]) > max(abs_[i]["rect"][3], abs_[j]["rect"][3])]
        R.check(not overl, "artboards laid out in a grid without overlap", f"grid {ai['grid']}")
        lays = doc["layers"]
        R.check(len(lays) == N and [x["name"] for x in lays] == names, f"{N} layers (default 'Layer 1' removed), variant order top to bottom", [x["name"] for x in lays])
        # swatches
        spots = {x["id"]: x for x in doc["spots"]}
        grp = [g for g in doc["swatchGroups"] if g["name"] == "K Brand"]
        shade = [g for g in doc["swatchGroups"] if g["name"] == IJ.SHADE_GROUP]
        brand_ids = {x["id"] for x in doc["spots"] if x["name"] in {IJ.swatch_name(k) for k in palette}}
        shade_ids = {x["id"] for x in doc["spots"]} - brand_ids
        want_sw = [IJ.swatch_name(k) for k in palette]
        R.check(all(x["colorType"] == "ColorModel.PROCESS" for x in doc["spots"]), "all swatches are GLOBAL process colours (Spot, ColorModel.PROCESS)",
                [(x["name"], x["colorType"]) for x in doc["spots"]][:8])
        R.check([x["name"] for x in doc["spots"]][:len(want_sw)] == want_sw, "palette swatches named 'K <Key>'", [x["name"] for x in doc["spots"]])
        R.check(all(x["color"]["t"] == "rgb" and "#%02X%02X%02X" % tuple(x["color"]["rgb"]) == IJ.parse_color(palette[k])[0]
                    for x, k in zip(doc["spots"], palette)), "swatch colours = palette hex", "")
        spot_ids = {x["id"] for x in doc["spots"]}
        R.check(len(grp) == 1 and set(grp[0]["members"]) == brand_ids,
                "palette swatches are exactly the swatch group 'K Brand'", f"{len(grp[0]['members']) if grp else 0} members")
        R.check((not shade_ids and not shade) or (len(shade) == 1 and set(shade[0]["members"]) & spot_ids == shade_ids),
                f"gradient-only tones are exactly the swatch group '{IJ.SHADE_GROUP}'",
                f"{len(shade[0]['members']) if shade else 0} members")
        global_ids = set(grp[0]["members"]) | (set(shade[0]["members"]) if shade else set()) if grp else set()
        tot_paths = tot_pts = n_filled = 0
        all_fill_ok, fill_detail = True, []
        geo_rows, live_rows = [], []
        recon_stats = []
        for vi, (v, lay, ab) in enumerate(zip(model.variants, lays, abs_)):
            sub_names = [x["name"] for x in reversed(lay["layers"])]
            want_sub = [it["n"] for it in v["items"]]
            R.check(sub_names == want_sub, f"'{v['name']}': one sub-layer per top-level item, paint order", sub_names)
            leaves = ai_leaves(doc, lay, ab["rect"], 1.0)
            ex = (expect or {}).get(v["name"])
            if ex:
                n_p = sum(len(lf["subs"]) for lf in leaves)
                n_a = sum(len(p) for lf in leaves for _, p in lf["subs"])
                R.check((n_p, n_a) == (ex["paths"], ex["points"]), f"'{v['name']}': pathItems / anchors = independent count from kmark",
                        f"{n_p} / {n_a} (expected {ex['paths']} / {ex['points']})")
                got = {lf["name"]: spots[lf["fill"]["spot"]]["name"] for lf in leaves if lf["fill"] and lf["fill"]["t"] == "spot"}
                want = {n: IJ.swatch_name(key) for n, key in ex["fills"].items()}
                R.check(all(got.get(n) == w for n, w in want.items()), f"'{v['name']}': each facet filled with its own swatch",
                        ", ".join(f"{n} -> {got.get(n)}" for n in want))
            for lf in leaves:
                tot_paths += len(lf["subs"])
                tot_pts += sum(len(p) for _, p in lf["subs"])
                f = lf["fill"]
                if lf["clip"] or f is None:
                    continue
                n_filled += 1
                if f["t"] == "spot":
                    sp = spots[f["spot"]]
                    ok = sp["colorType"] == "ColorModel.PROCESS" and f["tint"] == 100 and sp["id"] in global_ids
                elif f["t"] == "gradient":
                    g = [x for x in doc["gradients"] if x["id"] == f["gradient"]][0]
                    ok = all(st["color"]["t"] == "spot" for st in g["stops"])
                else:
                    ok = False
                if not ok:
                    all_fill_ok = False
                    fill_detail.append(lf["name"])
            # geometry vs input
            ref = (independent or {}).get(v["name"])
            if ref is not None:
                rows = geometry_vs_reference(leaves, ref, v["name"])
                for r in rows:
                    if r["live"]:
                        live_rows.append(r)
                    else:
                        geo_rows.append(r)
            # pixels
            W, H = v["W"], v["H"]
            svg = ai_svg(doc, lay, ab["rect"], 1.0, W, H)
            with open(os.path.join(run_dir, f"ai_{coord}_{IJ._slug(v['name'])}.svg"), "w") as fh:
                fh.write(svg)
            if v["name"] in refs:
                rs = int(RENDER * max(W, H) / max(W, H))
                img = svg_to_png(svg, rs)
                refimg = svg_to_png(refs[v["name"]], rs)
                c = compare(img, refimg)
                save_pair(run_dir, f"ai_{coord}_{IJ._slug(v['name'])}_ref-mock-diff", img, refimg)
                recon_stats.append((v["name"], c))
        R.check(tot_paths == ai["counts"]["paths"] and tot_pts == ai["counts"]["points"],
                "pathItems / anchor points recorded = expected", f"{tot_paths} paths, {tot_pts} points (expected {ai['counts']['paths']} / {ai['counts']['points']})")
        R.check(all_fill_ok, "every fill references a global swatch in 'K Brand' / 'K Shading' (solid: SpotColor tint 100; gradient: stops are SpotColors)",
                fill_detail or f"{n_filled} filled paths")
        report_geometry("Illustrator", geo_rows, live_rows)
        for name, c in recon_stats:
            pixel_row(name, c, approximations.get("ai", {}), "recorded document")
        # saves
        folder = os.path.dirname(rec["scriptPath"])
        saved = [x["path"] for x in doc["saved"]]
        R.check(saved and saved[-1] == f"{folder}/{label}.ai" and doc["saved"][-1]["opts"]["pdfCompatible"] is True,
                "saved last as .ai next to the script, IllustratorSaveOptions.pdfCompatible = true", saved)
        R.check(f"{folder}/{label}.pdf" in saved, "PDF saved next to the script", "")
        R.check([e["path"] for e in doc["exports"]] == [f"{folder}/{label}.svg"] and doc["exports"][0]["opts"]["saveMultipleArtboards"],
                "SVG exported (one file per artboard)", [e["path"] for e in doc["exports"]])
        R.check(f"{folder}/{label}_illustrator-build.txt" in rec["files"], "build log written next to the script", "")
        result["ai"][coord] = {"alert": alert, "paths": tot_paths, "points": tot_pts, "pixels": recon_stats,
                               "geometry_max": max([r["h"] for r in geo_rows] or [0])}

    # ------------------------------------------------------------- Photoshop
    for mode, extra in (("sibling", ["--new-layer", "sibling"]), ("inside", ["--new-layer", "inside"]),
                        ("quirk", ["--new-layer", "inside", "--quirk-empty-set"])):
        R.sec(f"[{label}] Photoshop mock (new layers {'inside the active group' if mode != 'sibling' else 'beside the active layer'}"
              f"{', PLACEATEND into empty groups fails' if mode == 'quirk' else ''})")
        rec, log = run_mock("ps", ps_path, os.path.join(run_dir, f"ps_{mode}.json"), *extra)
        R.check(not rec["crashed"], "script ran to completion", rec["crashed"] or log)
        errs = rec["errors"]
        if mode == "quirk":
            unexpected = [e for e in errs if "General Photoshop error" not in e and "relative to itself" not in e]
            R.check(not unexpected, "only the simulated move failures were raised (and handled)", f"{len(errs)} simulated, unexpected: {unexpected[:2]}")
        else:
            R.check(not errs, "no DOM / Action Manager misuse", errs[:3])
        alert = rec["alerts"][0] if rec["alerts"] else ""
        R.check(len(rec["alerts"]) == 1 and "build failed" not in alert, "one summary alert, no failure", alert.splitlines()[0] if alert else "none")
        m = re.search(r"Path check: (\d+) points, max deviation ([\d.]+)", alert)
        R.check(m and float(m.group(2)) <= GEOM_TOL and "not checked" not in alert, "script's own path read-back check", m.group(0) if m else alert[:200])
        m2 = re.search(r"Colour check: (\d+) samples, max difference ([\d.]+) levels, (\d+) failed", alert)
        R.check(m2 and int(m2.group(3)) == 0 and int(m2.group(1)) == ps["counts"]["samples"], "script's own colour-sampler check", m2.group(0) if m2 else "missing")
        R.check("Stacking order: as designed" in alert, "script's own stacking-order check", re.search(r"Stacking order: .*", alert).group(0) if "Stacking order" in alert else "missing")
        docs = rec["documents"]
        R.check(len(docs) == N, f"{N} documents", [d["name"] for d in docs])
        folder = os.path.dirname(rec["scriptPath"])
        calls = rec["calls"]
        makes = [c for c in calls if c["op"] == "executeAction" and c["event"] == "make" and c["desc"]["null"]["ref"][0]["cls"] == "contentLayer"]
        n_leaf = ps["counts"]["layers"]
        R.check(len(makes) == n_leaf, "one Action Manager 'make contentLayer' per leaf", f"{len(makes)} calls, {n_leaf} leaves")
        solid_ok = all(c["desc"]["using"]["obj"] == "contentLayer" and c["desc"]["using"]["v"]["type"]["obj"] in ("solidColorLayer", "gradientLayer") for c in makes)
        R.check(solid_ok, "descriptors: Mk / null=class contentLayer / Usng contentLayer{Type solidColorLayer{Clr RGBC}}", "")
        sel_ok = True
        for i, c in enumerate(calls):
            if c in makes and "shape" not in c["desc"]["using"]["v"]:
                prev = [x for x in calls[:i] if x["op"] in ("path.select", "path.remove")]
                sel_ok &= bool(prev) and prev[-1]["op"] == "path.select"
        R.check(sel_ok, "every path-based fill layer is made while its path is selected", "")
        R.check(len(rec["swatches"]) == len(model.swatches) and [x["name"] for x in rec["swatches"]] == [x["name"] for x in model.swatches],
                "palette added to the Swatches panel", [x["name"] for x in rec["swatches"]])
        geo_rows, live_rows, pix = [], [], []
        for vi, (v, d) in enumerate(zip(model.variants, docs)):
            snap = d["saved"][-1]["snapshot"] if d["saved"] else None
            want_px = (int(round(v["W"] * px / max(v["W"], v["H"]))), int(round(v["H"] * px / max(v["W"], v["H"]))))
            R.check((d["width"], d["height"]) == want_px and d["mode"].endswith("RGB") and d["bits"].endswith("EIGHT") and d["profile"].startswith("sRGB"),
                    f"'{v['name']}': {want_px[0]} x {want_px[1]} px RGB 8-bit sRGB document", (d["width"], d["height"], d["profile"]))
            want_file = f"{folder}/{label}_{IJ._slug(v['name'])}.psd"
            R.check(snap is not None and [x["path"] for x in d["saved"]] == [want_file] and d["saved"][0]["opts"]["layers"],
                    f"'{v['name']}': saved as {os.path.basename(want_file)} with layers", [x["path"] for x in d["saved"]])
            if snap is None:
                continue
            R.check(snap["resolution"] == 300 and snap["resizes"] == [300], "resolution set to 300 ppi without resampling", snap["resizes"])
            R.check(len(snap["layers"]) == 1 and snap["layers"][0]["cls"] == "LayerSet" and snap["layers"][0]["name"] == v["name"],
                    "one root layer group named after the variant ('Layer 1' removed)", [L["name"] for L in snap["layers"]])
            R.check(not snap["paths"], "no build paths left in the Paths panel", snap["paths"])
            leaves = ps_leaves(snap["layers"])
            ex = (expect or {}).get(v["name"])
            if ex:
                n_l = len(leaves)
                n_a = sum(len(c["points"]) for L in leaves if "components" in (L.get("vmask") or {}) for c in L["vmask"]["components"])
                n_live = sum(1 for L in leaves if "shape" in (L.get("vmask") or {}))
                R.check((n_l, n_a + 4 * n_live) == (ex["paths"], ex["points"]),
                        f"'{v['name']}': shape layers / vector-mask anchors = independent count from kmark",
                        f"{n_l} layers, {n_a} anchors + {n_live} live shape(s) (expected {ex['paths']} layers / {ex['points']} incl. 4 per live ellipse)")
                cols = {L["name"]: "#%02X%02X%02X" % tuple(int(round(c)) for c in L["fill"]["rgb"]) for L in leaves if L["fill"]["t"] == "solid"}
                R.check(all(cols.get(n) == IJ.parse_color(palette[key])[0] for n, key in ex["fills"].items()),
                        f"'{v['name']}': each facet layer has its palette colour", ", ".join(f"{n} {cols.get(n)}" for n in ex["fills"]))
            kinds_ok = all(L.get("kind") in ("LayerKind.SOLIDFILL", "LayerKind.GRADIENTFILL", "LayerKind.SMARTOBJECT") and L.get("vmask") for L in leaves)
            R.check(kinds_ok, "every leaf is a fill layer with a vector mask (shape layer)", f"{len(leaves)} layers")
            s = ps["documents"][vi]["scale"]          # px per artwork unit (the canvas W x H is rounded to whole px)
            rows = []
            ref = (independent or {}).get(v["name"])
            if ref is not None:
                conv = []
                for L in leaves:
                    vm = L["vmask"]
                    if "shape" in vm:
                        conv.append({"name": L["name"], "clip": False, "live": vm["shape"], "subs": None, "ps_shape": vm["shape"], "s": s})
                    else:
                        conv.append({"name": L["name"], "clip": False, "live": None,
                                     "subs": [(c["closed"], [[q / s if i < 6 else q for i, q in enumerate(p)] for p in c["points"]]) for c in vm["components"]]})
                for G in ps_groups(snap["layers"]):
                    if G.get("vmask") and "components" in G["vmask"]:
                        conv.append({"name": G["name"] + " (clip)", "clip": True, "live": None,
                                     "subs": [(c["closed"], [[q / s if i < 6 else q for i, q in enumerate(p)] for p in c["points"]]) for c in G["vmask"]["components"]]})
                rows = geometry_vs_reference(conv, ref, v["name"])
            for r in rows:
                (live_rows if r["live"] else geo_rows).append(r)
            svg = ps_svg(snap)
            with open(os.path.join(run_dir, f"ps_{mode}_{IJ._slug(v['name'])}.svg"), "w") as fh:
                fh.write(svg)
            if v["name"] in refs and mode == "sibling":
                img = svg_to_png(svg, RENDER)
                refimg = svg_to_png(refs[v["name"]], RENDER)
                c = compare(img, refimg)
                save_pair(run_dir, f"ps_{IJ._slug(v['name'])}_ref-mock-diff", img, refimg)
                pix.append((v["name"], c))
            if mode != "sibling":
                base = rec_sibling_tree.get(v["name"])
                R.check(base is None or json.dumps(strip_names(snap["layers"])) == base, "layer tree identical to the 'beside' run", "")
            else:
                rec_sibling_tree[v["name"]] = json.dumps(strip_names(snap["layers"]))
        report_geometry("Photoshop", geo_rows, live_rows)
        for name, c in pix:
            pixel_row(name, c, approximations.get("ps", {}), "recorded documents")
        result["ps"][mode] = {"alert": alert, "pixels": pix, "geometry_max": max([r["h"] for r in geo_rows] or [0])}
    return result


rec_sibling_tree = {}
PIX_MEAN, PIX_BAD = 0.5, 0.05      # mean |diff| in levels (0..255), % of pixels off by > 10 away from AA edges


def pixel_row(name, c, approx, what):
    stat = (f"mean |diff| {c['mean']:.3f}/255, max {c['max']:.0f}, {c['bad']} px ({c['bad_pct']:.3f}%) "
            f"off by >10 away from edges")
    if name in approx:
        R.info(f"pixels '{name}': {what} vs librsvg render of the input ({RENDER} px)",
               stat + f"  [known approximation: {approx[name]}]")
        return
    R.check(c["mean"] <= PIX_MEAN and c["bad_pct"] <= PIX_BAD,
            f"pixels '{name}': {what} vs librsvg render of the input ({RENDER} px)", stat)


def strip_names(layers):
    return [{k: (strip_names(v) if k == "layers" else v) for k, v in L.items()} for L in layers]


def report_geometry(app, rows, live_rows):
    if rows:
        worst = max(rows, key=lambda r: r["h"])
        bad = [r for r in rows if r["h"] > r["tol"] or not r["count_ok"]]
        exact = all(r["tol"] == GEOM_TOL for r in rows)
        R.check(not bad, f"{app} geometry round trip: {len(rows)} paths within {GEOM_TOL} of the input"
                + ("" if exact else f" (+{REF_ERR} reference error where the reference is cairosvg)"),
                f"max deviation {worst['h']:.5f} ({worst['name']}); " + (f"FAILED: {[(r['name'], round(r['h'], 4), r['count_ok']) for r in bad[:4]]}" if bad else
                                                                        f"{sum(r['n'] for r in rows)} anchors compared"))
    for r in live_rows:
        R.check(r["h"] <= r.get("tol", GEOM_TOL), f"{app} live shape '{r['name']}' matches the input shape", r["detail"])


def geometry_vs_reference(recorded, ref, vname):
    """recorded: [{name, clip, subs: [(closed, pts in SVG units)], live}]
    ref: ('polys', {name: [poly, ...]}, background) | ('cairo', captured, id->name, clip map, background);
    background = {box, rings, r} or None (checked against live shapes)."""
    rows = []
    kind = ref[0]
    bg = ref[-1]
    for lf in recorded:
        if lf.get("live") is not None or lf.get("ps_shape"):
            rows.append(live_check(lf, bg))
            continue
        if kind == "polys":
            polys = ref[1]
            if lf["name"] not in polys:
                continue
            want = polys[lf["name"]]
            got = lf["subs"]
            n_ok = len(got) == len(want) and all(len(p) == len(w) for (_, p), w in zip(got, want))
            dev = 0.0
            n_ok = n_ok and all(closed and all(p[6] == 0 for p in pts) for closed, pts in got)   # closed, corner points
            if n_ok:
                for (closed, pts), w in zip(got, want):
                    for p, q in zip(pts, w):
                        dev = max(dev, abs(p[0] - q[0]), abs(p[1] - q[1]), abs(p[2] - q[0]), abs(p[3] - q[1]),
                                  abs(p[4] - q[0]), abs(p[5] - q[1]))
            rows.append({"name": f"{vname}/{lf['name']}", "h": dev if n_ok else float("inf"), "count_ok": n_ok,
                         "n": sum(len(p) for _, p in got), "live": None, "tol": GEOM_TOL})
            continue
        _, captured, idname, clip_of, _bg = ref
        if lf["clip"]:
            owner = lf["name"][:-len(" (clip)")] if lf["name"].endswith(" (clip)") else None
            spec = clip_of.get(owner)
            if not spec:
                continue
            owner_id, ref_id = spec
            want = [sub for c in captured if ref_id in c["stack"] and (not owner_id or owner_id in c["stack"])
                    and c["id"] != owner_id for sub in c["subs"]]
        else:
            ids = {i for i, n in idname.items() if n == lf["name"]}
            want = [sub for c in captured if c["id"] in ids and not c["clip"] for sub in c["subs"]]
        if not want:
            continue
        got = flatten_rec(lf["subs"])
        rows.append({"name": f"{vname}/{lf['name']}", "h": hausdorff(got, want), "count_ok": len(got) == len(want),
                     "n": sum(len(p) for _, p in lf["subs"]), "live": None, "tol": GEOM_TOL + REF_ERR})
    return rows


def live_check(lf, want):
    """Live ellipse / (rounded) rectangle: its bounds (and corner radius) must match the input;
    the native shape's own curve deviation from the true circle is reported."""
    if lf.get("ps_shape"):
        sh, s = lf["ps_shape"], lf["s"]
        box = (sh["left"] / s, sh["top"] / s, sh["right"] / s, sh["bottom"] / s)
        rad = sh.get("radius", 0) / s
    else:
        xs = [p[0] for _, pts in lf["subs"] for p in pts]
        ys = [p[1] for _, pts in lf["subs"] for p in pts]
        box = (min(xs), min(ys), max(xs), max(ys))
        rad = None
    if want is None:
        return {"name": lf["name"], "h": float("inf"), "live": True, "detail": "no reference for this live shape", "tol": GEOM_TOL}
    dev = max(abs(a - b) for a, b in zip(box, want["box"]))
    det = f"bounds {tuple(round(v, 4) for v in box)} vs input {tuple(round(v, 4) for v in want['box'])} (max {dev:.5f})"
    if lf.get("subs") and want.get("rings"):
        h = hausdorff(flatten_rec(lf["subs"]), want["rings"])
        det += f"; native Bezier curve vs true shape {h:.4f} units"
    if rad is not None and want.get("r") is not None:
        dev = max(dev, abs(rad - want["r"]))
        det += f"; corner radius {rad:.4f} vs {want['r']:.4f}"
    return {"name": lf["name"], "h": dev, "live": True, "detail": det, "tol": GEOM_TOL}


def svg_reference(svg_text, size):
    """('cairo', captured, id->name, clip map, background) for geometry_vs_reference."""
    captured = cairo_reference(svg_text, size)
    idname = name_map(svg_text)
    root = ET.fromstring(svg_text)
    clip_of = {}
    for el in root.iter():
        for attr in ("clip-path", "mask"):
            v = el.get(attr)
            if v and v.startswith("url(#"):
                owner = idname.get(el.get("id")) if el.get("id") else "Group"
                clip_of[owner] = (el.get("id") or "", v[5:-1])
    bg = None
    first = next((el for el in root if el.tag.split("}")[-1] in ("circle", "rect", "ellipse", "path")), None)
    if first is not None and (first.get("id") or "").lower() in IJ.BACKGROUND_NAMES:
        rings = [sub for c in captured if c["id"] == first.get("id") and not c["clip"] for sub in c["subs"]]
        xs = [p[0] for r in rings for p in r]
        ys = [p[1] for r in rings for p in r]
        bg = {"box": (min(xs), min(ys), max(xs), max(ys)), "rings": rings,
              "r": float(first.get("rx")) if first.get("rx") is not None else None}
    return ("cairo", captured, idname, clip_of, bg)


# =========================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--acorn", default=os.environ.get("ACORN_DIR"), help="node_modules dir that holds acorn (optional)")
    ap.add_argument("--px", type=int, default=4096)
    ap.add_argument("svgs", nargs="*", help="extra layered SVGs (e.g. the final artwork), one variant each")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    acorn = a.acorn if a.acorn and os.path.isdir(os.path.join(a.acorn, "acorn")) else None
    if not acorn:
        print("note: acorn not available (--acorn); the strict ES3 parse is skipped")
    converter_checks()
    results = []

    # ---------------------------------------------------------------- trace
    k = trace_k()
    variants, palette = IJ.trace_variants(k)
    pal = dict(k.palette)
    variants.append({"name": "K App Icon", "artboard_size": 1024, "background": {"type": "squircle", "fill": "#0B0F19"},
                     "facets": [dict(f) for f in variants[0]["facets"]]})
    palette = dict(palette, ink="#0B0F19")
    with open(os.path.join(HERE, "..", "source", "K-Logo_trace.svg")) as fh:
        trace_svg = fh.read()
    variants.append(IJ.svg_to_variant(trace_svg, "K Trace SVG"))
    mark = flat_group(k, trap=False)
    refs = {
        "K Mark": svg_doc(mark),
        "K Badge": svg_doc(f'<circle cx="512" cy="512" r="512" fill="{pal["disc"]}"/>' + mark),
        "K App Icon": svg_doc(superellipse_svg(1024, 1024, "#0B0F19") + mark),
        "K Trace SVG": trace_svg,
    }
    exact = {n: [list(facets(k)[n])] for n in FACETS}           # exact construction points (kmark)
    circle_ref = {"box": (0.0, 0.0, 1024.0, 1024.0), "r": None,
                  "rings": [[(512 + 512 * math.cos(2 * math.pi * i / 4096), 512 + 512 * math.sin(2 * math.pi * i / 4096)) for i in range(4096)]]}
    indep = {"K Mark": ("polys", exact, None), "K Badge": ("polys", exact, circle_ref), "K App Icon": ("polys", exact, None),
             "K Trace SVG": svg_reference(trace_svg, 1024)}
    R.sec("[trace] input")
    rounded = max(abs(float(x) - y) for n in FACETS for (x, _), (y, _) in
                  zip(re.findall(r"([-\d.]+) ([-\d.]+)", variants[0]["facets"][FACETS.index(n)]["d"]), facets(k)[n]))
    R.info("trace facet d strings vs exact kmark construction points", f"max {rounded:.5f} (poly_d rounds to 0.001)")
    se = squircle_error()
    R.info("default squircle background (one cubic per quadrant) vs true superellipse n=5", f"max radial deviation {se:.3f} units at 1024")
    sq = "default squircle = one cubic per quadrant, compared with the true superellipse"
    n_facet_pts = sum(len(facets(k)[n]) for n in FACETS)
    fills = {n: n for n in FACETS}
    expect = {"K Mark": {"paths": 4, "points": n_facet_pts, "fills": fills},
              "K Badge": {"paths": 5, "points": n_facet_pts + 4, "fills": dict(fills, Background="disc")},
              "K App Icon": {"paths": 5, "points": n_facet_pts + 4, "fills": dict(fills, Background="ink")},
              "K Trace SVG": {"paths": 5, "points": n_facet_pts + 4, "fills": dict(fills, disc="disc")}}
    results.append(check_run("K-Logo", variants, palette, refs, out, acorn, a.px, indep,
                             approximations={"ai": {"K App Icon": sq}, "ps": {"K App Icon": sq}}, expect=expect))

    # ---------------------------------------------------------------- fixture
    fx = build_fixtures()
    fvars, frefs, findep = [], {}, {}
    for name, svg in fx.items():
        p = os.path.join(out, f"{IJ._slug(name)}.svg")
        with open(p, "w") as fh:
            fh.write(svg)
        fvars.append(IJ.svg_to_variant(svg, name))
        frefs[name] = svg
        size = int(re.search(r'width="(\d+)"', svg).group(1))
        findep[name] = svg_reference(svg, size)
    R.sec("[fixture] input")
    R.info("fixture files", ", ".join(os.path.join(out, f"{IJ._slug(n)}.svg") for n in fx))
    for v in fvars:
        R.info(f"svg_to_variant('{v['name']}') warnings", v["warnings"] or "none")
    fpal = dict(pal)
    fpal.update(ink="#0B0F19", white="#FFFFFF")
    fpal.pop("disc")
    fpal["paper"] = pal["disc"]
    results.append(check_run("Fixture", fvars, fpal, frefs, out, acorn, a.px, findep, approximations={"ps": {
        "Fixture Extras": "Photoshop: soft mask -> its silhouette as a vector mask, elliptical radial gradient -> circular (both warned)"}}))
    # the soft mask and the elliptical radial gradient are Photoshop approximations (warned); quantify the rest
    # ---------------------------------------------------------------- extra SVGs (final artwork)
    if a.svgs:
        evars, erefs, eind = [], {}, {}
        for spec in a.svgs:
            pth, _, nmn = spec.partition("=")
            with open(pth) as fh:
                svg = fh.read()
            v = IJ.svg_to_variant(svg, nmn or os.path.splitext(os.path.basename(pth))[0])
            evars.append(v)
            erefs[v["name"]] = svg
            eind[v["name"]] = svg_reference(svg, int(round(v["artboard_size"][0])))
        results.append(check_run("Artwork", evars, dict(pal), erefs, out, acorn, a.px, eind))

    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump({"rows": R.rows, "failed": len(R.failed)}, fh, indent=1, default=str)
    n_ok = sum(1 for r in R.rows if r["ok"])
    print(f"\n{n_ok} passed, {len(R.failed)} failed ({len(R.rows)} rows incl. info) -> {out}")
    for r in R.failed:
        print(f"  FAILED [{r['section']}] {r['check']}: {r['detail'][:300]}")
    return 1 if R.failed else 0


def squircle_error():
    d = IJ.squircle_d(0, 0, 1024, 1024)
    subs = [[[*q[:6]] for q in IJ.subpath_points(sp)] for sp in IJ.parse_path(d)]
    worst = 0.0
    for ring in flatten_rec([(True, p) for p in subs], 512):
        for x, y in ring:
            u, v = abs(x - 512) / 512, abs(y - 512) / 512
            rr = (u ** 5 + v ** 5) ** 0.2
            worst = max(worst, abs(rr - 1) * 512)
    return worst


if __name__ == "__main__":
    sys.exit(main())
