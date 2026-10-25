#!/usr/bin/env python3
"""End-to-end verification of vector_export.py.

    python3 test_vector_export.py [OUT_DIR]

Builds a fixture that exercises the whole "layered SVG" convention (nested named groups,
inkscape:label / data-name, solid facets, linear + radial gradients with stop-opacity,
gradientUnits / gradientTransform / href / reflect, arc-rounded paths, rect rx, circle, ellipse,
polygon, relative path commands incl. s/q/t/a, evenodd, style="" and <style> CSS, group and fill
opacity, clip-path (userSpace and objectBoundingBox, multi-child), mask, translate/scale/rotate/
skewX/matrix transforms, <use>, strokes incl. dashes and stroke-opacity, a solid alpha overlay on
gradient facets, and two filters), runs export_rgb / export_cmyk on it and
on the logo trace, renders every output with Ghostscript and compares against the librsvg
reference, checks PDF structure (pypdf, qpdf via pikepdf, pdfinfo, gs -dPDFSTOPONWARNING), counts
image XObjects / shadings, checks EPS DSC + bounding boxes, hairline gaps and the CMYK values.
Exit status 0 = every check passed.
"""
from __future__ import annotations


import io
import json
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vector_export as vx                                       # noqa: E402
from kmark import (FACETS, KParams, facets, flat_group, poly_d,  # noqa: E402
                   silhouette, silhouette_d, svg_doc)
from preview import svg_to_png                                   # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/tmp/claude-0/-home-user-kawin/112bef45-5b8b-536e-9b0e-4c4d94d8b9e3/scratchpad/infra/vector_export/run"
os.makedirs(OUT, exist_ok=True)
K = KParams.from_dict(json.load(open(os.path.join(HERE, "trace_params.json"))))
PAL = K.palette
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


# =========================================================================== fixture
def fixture_svg():
    kr = KParams.from_dict(dict(json.load(open(os.path.join(HERE, "trace_params.json"))), r_convex=30.0, r_concave=12.0))
    sil_round = silhouette_d(kr)                       # arc-rounded silhouette (A commands)
    f = facets(K)
    place = "translate(256 250) scale(0.42) translate(-512 -512)"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" viewBox="0 0 1024 1024" width="1024" height="1024">
<title>vector_export fixture</title>
<style>.panel {{ fill: #F4F1EA }}  #Tech .panel {{ fill: #0B0F19 }}</style>
<defs>
  <linearGradient id="gRed" gradientUnits="userSpaceOnUse" x1="267" y1="211" x2="405" y2="443" gradientTransform="rotate(12 336 327)">
    <stop offset="0" stop-color="#FF7A5C"/><stop offset="1" stop-color="#C42A15"/></linearGradient>
  <linearGradient id="gYellow" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#FFE680"/><stop offset="55%" style="stop-color:#FAC311"/><stop offset="100%" stop-color="#D99A00"/></linearGradient>
  <linearGradient id="gBlueBase"><stop offset="0" stop-color="#7FB0FF"/><stop offset="1" stop-color="#1F5FD0"/></linearGradient>
  <linearGradient id="gBlue" xlink:href="#gBlueBase" x1="0.1" y1="0" x2="0.9" y2="1"/>
  <radialGradient id="gGreen" cx="0.35" cy="0.3" r="0.85" fx="0.25" fy="0.2">
    <stop offset="0" stop-color="#7BE38A"/><stop offset="0.7" stop-color="#2CA53B" stop-opacity="0.9"/><stop offset="1" stop-color="#17722A" stop-opacity="0.55"/></radialGradient>
  <radialGradient id="gHi" gradientUnits="userSpaceOnUse" cx="420" cy="330" r="260">
    <stop offset="0" stop-color="#FFFFFF" stop-opacity="0.55"/><stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/></radialGradient>
  <linearGradient id="gRep" x1="0" y1="0" x2="0.2" y2="0" spreadMethod="reflect">
    <stop offset="0" stop-color="#3B81F1"/><stop offset="1" stop-color="#FAC311"/></linearGradient>
  <linearGradient id="gMask" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#fff"/><stop offset="1" stop-color="#000"/></linearGradient>
  <clipPath id="cRound"><path d="{sil_round}"/></clipPath>
  <clipPath id="cTwo" clipPathUnits="objectBoundingBox"><circle cx="0.3" cy="0.5" r="0.3"/><rect x="0.5" y="0.15" width="0.45" height="0.7"/></clipPath>
  <mask id="mFade" maskUnits="userSpaceOnUse" x="40" y="380" width="420" height="96">
    <rect x="40" y="380" width="420" height="96" fill="url(#gMask)"/></mask>
  <filter id="fGlow" x="-30%" y="-30%" width="160%" height="160%">
    <feGaussianBlur in="SourceGraphic" stdDeviation="22" result="blur"/>
    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <filter id="fShadow" x="-50%" y="-50%" width="200%" height="200%">
    <feDropShadow dx="6" dy="8" stdDeviation="6" flood-color="#000000" flood-opacity="0.5"/></filter>
  <path id="star" d="M0 -26 L7.6 -8 L26 -8 L11 3 L16 22 L0 11 L-16 22 L-11 3 L-26 -8 L-7.6 -8 Z"/>
</defs>
<g id="Flat" inkscape:label="Tile A / flat facets">
  <rect class="panel" x="16" y="16" width="480" height="480" rx="28"/>
  <circle id="FlatRing" cx="256" cy="256" r="232" fill="none" stroke="#141A29" stroke-opacity="0.08" stroke-width="6"/>
  <g id="FlatMark" data-name="Mark (flat)" transform="{place}">{flat_group(K, trap=False, gid="KA")}</g>
</g>
<g id="Gradient" data-name="Tile B / gradients" transform="translate(512 0)">
  <rect class="panel" x="16" y="16" width="480" height="480" rx="28" ry="28"/>
  <g id="GradMark" transform="{place}">
    <g id="GradFacets" clip-path="url(#cRound)">
      <path id="g-red" fill="url(#gRed)" d="{poly_d(f['red'])}"/>
      <path id="g-yellow" fill="url(#gYellow)" d="{poly_d(f['yellow'])}"/>
      <path id="g-green" fill="url(#gGreen)" d="{poly_d(f['green'])}"/>
      <path id="g-blue" fill="url(#gBlue)" d="{poly_d(f['blue'])}"/>
      <path id="g-edge-lit" fill="#FFFFFF" fill-opacity="0.22" d="M267.51 211.435 L299.51 211.435 L299.51 812.589 L267.51 812.589 Z M405.578 443.258 L521.922 514.665 L515 524 L405.578 456 Z"/>
      <path id="g-highlight" fill="url(#gHi)" d="{poly_d(silhouette(K))}"/>
    </g>
    <path id="g-outline" fill="none" stroke="#0B0F19" stroke-width="10" stroke-linejoin="round" d="{sil_round}"/>
  </g>
</g>
<g id="Effects" data-name="Tile C / opacity, clip, mask" transform="translate(0 512)">
  <rect class="panel" x="16" y="16" width="480" height="480" rx="28"/>
  <g id="Overlap" opacity="0.6">
    <circle cx="130" cy="140" r="80" fill="#EE3F2A"/>
    <ellipse cx="210" cy="150" rx="95" ry="55" fill="rgb(59,129,241)" transform="rotate(-25 210 150)"/>
    <polygon points="90,200 250,200 170,90" style="fill:#FAC311"/>
  </g>
  <path id="Ring" fill="#2CA53B" fill-rule="evenodd" d="m330 60 h140 v140 h-140 z m70 20 a50 50 0 0 1 0 100 a50 50 0 0 1 0-100 z"/>
  <path id="Wave" fill="#3B81F1" fill-opacity="0.8" d="M40 330 q40 -60 80 0 t80 0 t80 0 s60 60 100 0 c20 -40 40 -40 60 0 l0 30 H40 z"/>
  <g id="ClipTwo" clip-path="url(#cTwo)">
    <rect x="300" y="215" width="180" height="110" fill="#EE3F2A"/>
    <rect x="300" y="215" width="60" height="110" fill="#0B0F19" transform="matrix(1 0 0.3 1 -60 0)"/>
  </g>
  <path id="Dash" d="M40 362 H460" fill="none" stroke="#0B0F19" stroke-width="3" stroke-dasharray="12 6" stroke-linecap="round"/>
  <g id="Masked" mask="url(#mFade)"><rect x="40" y="380" width="420" height="96" fill="url(#gRep)"/></g>
</g>
<g id="Tech" data-name="Tile D / filters" transform="translate(512 512)">
  <rect class="panel" x="16" y="16" width="480" height="480" rx="28"/>
  <g id="TechMark" filter="url(#fGlow)" transform="translate(240 230) scale(0.36) translate(-512 -512)">{flat_group(K, trap=False, gid="KD")}</g>
  <circle id="Dot" cx="410" cy="410" r="36" fill="#FAC311" filter="url(#fShadow)"/>
  <use id="Star" xlink:href="#star" x="90" y="420" fill="#FEFDFB" transform="skewX(-8)"/>
</g>
<circle id="Center" cx="512" cy="512" r="9" fill="#EE3F2A" fill-opacity="0.5"/>
</svg>"""


def edge_svg():
    """Second fixture: the less common paths (nested <svg> + preserveAspectRatio, <use> of a
    <symbol> with currentColor, CSS class / descendant / child selectors, hsl() and rgba() colours,
    repeat + reflect gradients, clip-path on a <clipPath>, evenodd clip-rule, alpha mask,
    visibility:hidden / display:none, dashes with offset, paint-order, gradient stroke with
    stop-opacity)."""
    return """<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 400 400" width="400" height="400">
<style>.warm { fill: hsl(20, 90%, 55%) } #Edge .cool { fill: rgba(59,129,241,0.7) } g > .outline { stroke: #222 }</style>
<defs>
  <linearGradient id="rep" x1="0" y1="0" x2="0.15" y2="0.1" spreadMethod="repeat"><stop offset="0" stop-color="#EE3F2A"/><stop offset="1" stop-color="#FAC311"/></linearGradient>
  <radialGradient id="rrefl" r="0.2" spreadMethod="reflect"><stop offset="0" stop-color="#2CA53B"/><stop offset="1" stop-color="#FEFDFB"/></radialGradient>
  <linearGradient id="sg"><stop offset="0" stop-color="#3B81F1" stop-opacity="1"/><stop offset="1" stop-color="#EE3F2A" stop-opacity="0.2"/></linearGradient>
  <clipPath id="inner"><circle cx="300" cy="100" r="70"/></clipPath>
  <clipPath id="outer" clip-path="url(#inner)"><rect x="250" y="40" width="140" height="70"/></clipPath>
  <clipPath id="eo"><path clip-rule="evenodd" d="M220 220 h160 v160 h-160z M260 260 h80 v80 h-80z"/></clipPath>
  <mask id="am" mask-type="alpha" maskUnits="userSpaceOnUse" x="20" y="220" width="180" height="160"><circle cx="110" cy="300" r="70" fill="#000" fill-opacity="0.6"/></mask>
  <symbol id="sym"><rect width="40" height="40" rx="8" fill="currentColor"/></symbol>
</defs>
<g id="Edge" style="color:#2CA53B">
  <rect x="10" y="10" width="180" height="180" fill="url(#rep)"/>
  <rect x="20" y="20" width="60" height="60" class="warm"/>
  <rect x="100" y="20" width="60" height="60" class="cool"/>
  <rect x="20" y="100" width="60" height="60" fill="#000" visibility="hidden"/>
  <rect x="20" y="100" width="60" height="60" fill="#000" style="display:none"/>
  <use xlink:href="#sym" x="110" y="110"/>
  <g><path class="outline" d="M20 180 L180 180" stroke-width="6" stroke-dasharray="10 4 2 4" stroke-dashoffset="5" fill="none"/></g>
  <rect x="200" y="10" width="190" height="190" fill="url(#rrefl)" clip-path="url(#outer)"/>
  <path d="M210 150 C260 120 340 200 390 150" fill="none" stroke="url(#sg)" stroke-width="14" stroke-linecap="round"/>
  <circle cx="300" cy="170" r="20" fill="#FAC311" stroke="#0B0F19" stroke-width="8" style="paint-order:stroke"/>
  <rect x="200" y="200" width="200" height="200" fill="#3B81F1" clip-path="url(#eo)"/>
  <rect x="20" y="220" width="180" height="160" fill="#EE3F2A" mask="url(#am)"/>
  <svg x="40" y="240" width="120" height="60" viewBox="0 0 10 10" preserveAspectRatio="xMaxYMid meet"><circle cx="5" cy="5" r="6" fill="#0B0F19"/></svg>
</g>
</svg>"""


# =========================================================================== render helpers
def gs_render(path, px, page_pt, eps=False, aa=True, extra=()):
    res = 72.0 * px / page_pt
    out = os.path.join(OUT, "_gs.png")
    cmd = ["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", f"-r{res:.6f}", "-o", out]
    cmd += ["-dGraphicsAlphaBits=4", "-dTextAlphaBits=4"] if aa else ["-dGraphicsAlphaBits=1"]
    if eps:
        cmd.append("-dEPSCrop")
    cmd += list(extra) + [path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr)
    return np.asarray(Image.open(out).convert("RGB")).astype(np.int16), p.stderr.strip()


def ref_render(svg, px):
    im = svg_to_png(svg, px)
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    return np.asarray(bg.convert("RGB")).astype(np.int16)


def diff_stats(a, b):
    if a.shape != b.shape:
        return {"shape": (a.shape, b.shape), "mean": 999}
    d = np.abs(a - b)
    return {"mean": float(d.mean()), "p99": float(np.percentile(d.max(axis=2), 99)), "max": int(d.max()),
            "px_gt16_pct": float((d.max(axis=2) > 16).mean() * 100)}


# =========================================================================== structural checks
def pdf_structure(path, expect_pt):
    from pypdf import PdfReader
    info = {}
    raw = open(path, "rb").read()
    m = re.match(rb"%PDF-(\d)\.(\d)", raw)
    info["version"] = f"{m.group(1).decode()}.{m.group(2).decode()}"
    r = PdfReader(path, strict=True)
    info["pages"] = len(r.pages)
    pg = r.pages[0]
    boxes = {}
    for b in ("/MediaBox", "/CropBox", "/BleedBox", "/TrimBox", "/ArtBox"):
        boxes[b[1:]] = [float(v) for v in pg[b]] if b in pg else None
    info["boxes"] = boxes
    meta = r.metadata or {}
    info["Info"] = {k: str(meta.get(k)) for k in ("/Title", "/Creator", "/Producer")}
    counts = {"image_xobjects": 0, "form_xobjects": 0, "shadings": 0, "smasks": 0, "inline_images": 0,
              "fill_ops": 0, "stroke_ops": 0, "transparency_groups": 0}
    seen = set()

    def walk_res(res):
        if res is None:
            return
        res = res.get_object()
        for name, x in (res.get("/XObject") or {}).items():
            ref = x.indirect_reference if hasattr(x, "indirect_reference") else None
            xo = x.get_object()
            key = (ref.idnum if ref else id(xo))
            if key in seen:
                continue
            seen.add(key)
            if xo.get("/Subtype") == "/Image":
                counts["image_xobjects"] += 1
                if "/SMask" in xo:
                    counts["image_xobjects_smask"] = counts.get("image_xobjects_smask", 0) + 1
            else:
                counts["form_xobjects"] += 1
                if "/Group" in xo:
                    counts["transparency_groups"] += 1
                walk_stream(xo)
                walk_res(xo.get("/Resources"))
        for name, p in (res.get("/Pattern") or {}).items():
            po = p.get_object()
            if po.get("/PatternType") == 2:
                counts["shadings"] += 1
        counts["shadings"] += len(res.get("/Shading") or {})
        for name, g in (res.get("/ExtGState") or {}).items():
            go = g.get_object()
            sm = go.get("/SMask")
            if sm is not None and sm != "/None":
                counts["smasks"] += 1
                smo = sm.get_object()
                gform = smo["/G"].get_object()
                walk_stream(gform)
                walk_res(gform.get("/Resources"))

    def walk_stream(obj):
        data = obj.get_data()
        counts["inline_images"] += len(re.findall(rb"(^|\s)BI(\s)", data))
        counts["fill_ops"] += len(re.findall(rb"(?m)(^|\s)(f\*?|B\*?|b\*?)(\s|$)", data))
        counts["stroke_ops"] += len(re.findall(rb"(?m)(^|\s)S(\s|$)", data))

    walk_stream(pg.get_contents())
    walk_res(pg.get("/Resources"))
    info["counts"] = counts
    # qpdf (pikepdf) structural check
    try:
        import pikepdf
        with pikepdf.open(path) as pdf:
            probs = pdf.check_pdf_syntax()
        info["qpdf_problems"] = [str(p) for p in probs]
    except ImportError:
        info["qpdf_problems"] = ["pikepdf not installed"]
    p = subprocess.run(["pdfinfo", "-box", path], capture_output=True, text=True)
    info["pdfinfo_stderr"] = p.stderr.strip()
    p = subprocess.run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-dPDFSTOPONERROR", "-dPDFSTOPONWARNING",
                        "-sDEVICE=nullpage", path], capture_output=True, text=True)
    info["gs_strict"] = (p.returncode, (p.stdout + p.stderr).strip())
    ok = (float(info["version"]) >= 1.5 and info["pages"] == 1
          and all(v == [0.0, 0.0, expect_pt, expect_pt] for v in boxes.values())
          and not info["qpdf_problems"] and not info["pdfinfo_stderr"] and info["gs_strict"][0] == 0
          and not info["gs_strict"][1] and counts["inline_images"] == 0
          and all(info["Info"][k] not in ("None", "") for k in info["Info"])
          and "adobe" not in (info["Info"]["/Creator"] + info["Info"]["/Producer"]).lower())
    return ok, info


def eps_structure(path, expect_pt):
    txt = open(path, "rb").read().decode("latin-1")
    lines = txt.splitlines()
    hdr = {}
    for ln in lines:
        if ln.startswith("%%EndComments"):
            break
        if ln.startswith("%%") and ":" in ln:
            k, v = ln[2:].split(":", 1)
            hdr[k] = v.strip()
    bb = [float(v) for v in hdr.get("BoundingBox", "").split()]
    hbb = [float(v) for v in hdr.get("HiResBoundingBox", "").split()]
    body = re.sub(r"(?m)^ .*$", "", txt)            # drop ASCII85 data lines
    body = re.sub(r"(?m)%.*$", "", body)             # drop comments
    toks = body.split()
    info = {"first_line": lines[0], "BoundingBox": bb, "HiResBoundingBox": hbb,
            "LanguageLevel": hdr.get("LanguageLevel"), "showpage": toks.count("showpage"),
            "save/restore": (toks.count("save"), toks.count("restore")),
            "gsave/grestore": (toks.count("q") + toks.count("gsave"), toks.count("Q") + toks.count("grestore")),
            "images": toks.count("image"), "shfill": toks.count("shfill"), "ends_EOF": txt.rstrip().endswith("%%EOF")}
    ok = (lines[0] == "%!PS-Adobe-3.0 EPSF-3.0" and bb == [0, 0, round(expect_pt), round(expect_pt)]
          and hbb == [0.0, 0.0, expect_pt, expect_pt] and info["showpage"] == 1
          and info["save/restore"] == (1, 1) and info["ends_EOF"]
          and info["gsave/grestore"][0] == info["gsave/grestore"][1])
    return ok, info


# =========================================================================== tests
def run_rgb(label, svg, base, px_list=(1024, 2048), expect_images=0, expect_fills=None):
    print(f"\n== RGB: {label}")
    rep = vx.export_rgb(svg, base)
    print(f"   pdf stats: {rep['pdf_stats']}")
    print(f"   eps: level {rep['eps_level']}, flattened {rep['eps_flattened']}, vector-blended {rep['eps_vector_blended']}")
    if rep["warnings"]:
        print(f"   warnings: {rep['warnings']}")
    pt = rep["page_pt"][0]
    out = {"report": rep}
    for kind in ("pdf", "ai"):
        ok, info = pdf_structure(rep[kind], pt)
        out[kind + "_structure"] = info
        check(f"{label} .{kind} structure (PDF {info['version']}, boxes, Info, qpdf/pdfinfo/gs strict)", ok,
              f"counts={info['counts']} qpdf={info['qpdf_problems'][:2]}")
    c = out["pdf_structure"]["counts"]
    check(f"{label} .pdf image XObjects == {expect_images} (filters only)", c["image_xobjects"] == expect_images,
          f"images={c['image_xobjects']}, shadings={c['shadings']}, smasks={c['smasks']}")
    if expect_fills is not None:
        check(f"{label} .pdf fill operators == {expect_fills} (one per leaf)", c["fill_ops"] == expect_fills,
              f"fill_ops={c['fill_ops']}")
    ok, info = eps_structure(rep["eps"], pt)
    out["eps_structure"] = info
    check(f"{label} .eps DSC/BoundingBox/showpage", ok, json.dumps(info))
    out["diff"] = {}
    for px in px_list:
        ref = ref_render(svg, px)
        for kind in ("pdf", "ai", "eps"):
            img, err = gs_render(rep[kind], px, pt, eps=(kind == "eps"))
            st = diff_stats(img, ref)
            out["diff"][f"{kind}@{px}"] = st
            check(f"{label} .{kind} vs rsvg @{px}px: mean |d| < 2/255", st["mean"] < 2.0 and not err,
                  f"mean={st['mean']:.3f}/255 p99={st.get('p99')} max={st.get('max')} >16:{st.get('px_gt16_pct', 0):.3f}%"
                  + (f" gs-stderr={err[:120]}" if err else ""))
            if px == px_list[0]:
                Image.fromarray(np.abs(img - ref).clip(0, 255).astype(np.uint8) * 4).save(
                    os.path.join(OUT, f"diff_{os.path.basename(base)}_{kind}.png"))
                Image.fromarray(img.astype(np.uint8)).save(os.path.join(OUT, f"gs_{os.path.basename(base)}_{kind}.png"))
        # independent PDF engines (pdfium = Chrome's, poppler) on the same file
        for eng, img in (("pdfium", pdfium_render(rep["pdf"], px, pt)), ("poppler", poppler_render(rep["pdf"], px, pt))):
            st = diff_stats(img, ref)
            out["diff"][f"pdf[{eng}]@{px}"] = st
            check(f"{label} .pdf rendered by {eng} vs rsvg @{px}px: mean |d| < 2/255", st["mean"] < 2.0,
                  f"mean={st['mean']:.3f}/255 p99={st.get('p99')} >16:{st.get('px_gt16_pct', 0):.3f}%")
    return out


def pdfium_render(path, px, page_pt):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(path)
    im = doc[0].render(scale=px / page_pt, fill_color=(255, 255, 255, 255)).to_pil().convert("RGB")
    doc.close()
    return np.asarray(im).astype(np.int16)


def poppler_render(path, px, page_pt):
    stem = os.path.join(OUT, "_pp")
    subprocess.run(["pdftoppm", "-r", f"{72.0 * px / page_pt:.6f}", "-png", "-singlefile", path, stem], check=True,
                   capture_output=True)
    return np.asarray(Image.open(stem + ".png").convert("RGB")).astype(np.int16)


def gap_check(label, path, page_pt, eps=False, px=4096):
    """No-AA render: every pixel well inside the K silhouette must be painted (no hairline gaps)."""
    img, _ = gs_render(path, px, page_pt, eps=eps, aa=False)
    s = px / K.size
    m = Image.new("L", (px, px), 0)
    ImageDraw.Draw(m).polygon([(x * s, y * s) for x, y in silhouette(K)], fill=255)
    m = np.asarray(m.filter(ImageFilter.MinFilter(5))) > 0
    white = (img.min(axis=2) >= 250) & m
    return check(f"{label}: no hairline gaps (no-AA {px}px, interior px unpainted = {int(white.sum())})",
                 white.sum() == 0, f"interior px checked={int(m.sum())}")


def vertex_identity(label, pdf_path):
    """Vertices shared by adjacent facets must be byte-identical in the content streams."""
    import pikepdf
    pts = []
    with pikepdf.open(pdf_path) as pdf:
        streams = [pdf.pages[0].Contents.read_bytes()]
        for _, xo in (pdf.pages[0].Resources.get("/XObject") or {}).items():
            streams.append(xo.read_bytes())
    for s in streams:
        pts += re.findall(rb"(-?[\d.]+) (-?[\d.]+) [ml]\b", s)
    near_mismatch = 0
    vals = [(float(a), float(b), a + b" " + b) for a, b in pts]
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if abs(vals[i][0] - vals[j][0]) < 0.01 and abs(vals[i][1] - vals[j][1]) < 0.01 and vals[i][2] != vals[j][2]:
                near_mismatch += 1
    return check(f"{label}: shared vertices byte-identical ({len(vals)} vertices)", near_mismatch == 0,
                 f"near-but-different pairs={near_mismatch}")


def shapes_svg(shapes, size, colors, clip_d=None):
    """Reference SVG of an export_cmyk shapes list, each shape painted with colors[name]."""
    defs, body = [], []

    def clip_wrap(el, levels, tag):
        levels = levels if isinstance(levels, (list, tuple)) else [levels]
        for i, lv in enumerate(levels):
            d, rule = (lv, "nonzero") if isinstance(lv, str) else lv
            cid = f"c{len(defs)}"
            defs.append(f'<clipPath id="{cid}"><path clip-rule="{rule}" d="{d}"/></clipPath>')
            el = f'<g clip-path="url(#{cid})">{el}</g>'
        return el
    for i, sh in enumerate(shapes):
        name, d = sh[0], sh[1]
        opt = sh[3] if len(sh) > 3 else {}
        col = colors[name]
        if opt.get("stroke"):
            dash = f' stroke-dasharray="{" ".join(map(str, opt["dash"]))}" stroke-dashoffset="{opt.get("dashoffset", 0)}"' if opt.get("dash") else ""
            el = (f'<path fill="none" stroke="{col}" stroke-width="{opt["stroke"]}" stroke-linecap="{opt.get("linecap", "butt")}" '
                  f'stroke-linejoin="{opt.get("linejoin", "miter")}" stroke-miterlimit="{opt.get("miterlimit", 4)}"{dash} d="{d}"/>')
        else:
            el = f'<path fill="{col}" fill-rule="{opt.get("fill_rule", "nonzero")}" d="{d}"/>'
        if opt.get("clip"):
            el = clip_wrap(el, opt["clip"], i)
        body.append(el)
    b = "".join(body)
    if clip_d:
        b = clip_wrap(b, clip_d, "all")
    return svg_doc(b, "".join(defs), size=size if not isinstance(size, (tuple, list)) else size[0])


def run_cmyk(label, shapes, size, base, title, clip_d=None):
    """shapes: export_cmyk shapes; a 5th tuple item (original sRGB hex) enables the hue-region IoU."""
    print(f"\n== CMYK: {label}")
    rep = vx.export_cmyk([sh[:4] for sh in shapes], size, base, title, clip_d=clip_d)
    print(f"   ink totals %: {rep['ink_total_pct']}  inks: {rep['process_colors']}  warnings: {rep['warnings']}")
    pt = rep["page_pt"][0]
    out = {"report": rep}
    for kind in ("pdf", "ai"):
        ok, info = pdf_structure(rep[kind], pt)
        out[kind + "_structure"] = info
        check(f"{label} .{kind} structure", ok, f"counts={info['counts']}")
    c = out["pdf_structure"]["counts"]
    n_stroke = sum(1 for sh in shapes if len(sh) > 3 and sh[3].get("stroke"))
    check(f"{label} .pdf: 0 images, {len(shapes) - n_stroke} fills + {n_stroke} strokes (one per shape)",
          c["image_xobjects"] == 0 and c["fill_ops"] == len(shapes) - n_stroke and c["stroke_ops"] == n_stroke, str(c))
    import pikepdf
    with pikepdf.open(rep["pdf"]) as pdf:
        data = pdf.pages[0].Contents.read_bytes().decode()
        oi = pdf.Root.get("/OutputIntents")
    rx = r"([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) [kK]\n"
    ks = [tuple(float(v) for v in m) for m in re.findall(rx, data)]
    want = [tuple(round(v, 4) for v in sh[2]) for sh in shapes]
    check(f"{label} .pdf 'k'/'K' operands == ICC CMYK values", ks == want, f"{ks}")
    check(f"{label} .pdf OutputIntent (embedded {vx.icc_profile_name()})", oi is not None and len(oi) == 1)
    eps_txt = open(rep["eps"]).read()
    ks_eps = [tuple(float(v) for v in m) for m in re.findall(rx, eps_txt)]
    check(f"{label} .eps setcmykcolor operands == ICC CMYK values", ks_eps == want)
    ok, info = eps_structure(rep["eps"], pt)
    check(f"{label} .eps DSC/BoundingBox/showpage (+%%DocumentProcessColors)", ok and "%%DocumentProcessColors" in eps_txt,
          json.dumps(info))
    # colour: gs renders DeviceCMYK through default_cmyk.icc -> sRGB (relative colorimetric + BPC);
    # reference = rsvg render of the same shapes painted with the ICC proof of each CMYK value
    proof = {sh[0]: "#%02X%02X%02X" % vx.cmyk_to_srgb(sh[2]) for sh in shapes}
    for px in (1024, 2048):
        ref = ref_render(shapes_svg(shapes, size, proof, clip_d), px)
        for kind in ("pdf", "eps"):
            img, err = gs_render(rep[kind], px, pt, eps=(kind == "eps"), extra=("-dRenderIntent=1", "-dBlackPtComp=1"))
            st = diff_stats(img, ref)
            out[f"{kind}@{px}"] = st
            check(f"{label} .{kind} vs ICC-proofed rsvg reference @{px}px: mean |d| < 2/255", st["mean"] < 2.0,
                  f"mean={st['mean']:.3f} p99={st['p99']} max={st['max']}")
            if px == 1024 and all(len(sh) > 4 for sh in shapes):
                orig = {sh[0]: sh[4] for sh in shapes}
                iou = facet_iou(img, ref_render(shapes_svg(shapes, size, orig, clip_d), px), shapes, proof)
                out[f"{kind}_iou"] = iou
                check(f"{label} .{kind} hue regions IoU per shape > 0.995", min(iou.values()) > 0.995,
                      " ".join(f"{k}={v:.5f}" for k, v in iou.items()))
    return out


def facet_iou(img, ref_orig, shapes, proof, tol=30):
    """Hue regions: each pixel is labelled with the nearest palette colour (ICC proof colours for the
    CMYK render, the original sRGB colours for the reference) when it lies within `tol` of it, else it
    is an anti-aliased edge pixel. IoU per shape is taken over pixels labelled in both renders, so the
    two renderers' different edge coverage (gs 4-bit AA vs librsvg exact coverage) does not count,
    while any misplaced or mis-coloured region does."""
    names = [s[0] for s in shapes]
    orig = {s[0]: vx.hex_to_rgb(s[4]) for s in shapes}
    prf = {n: vx.hex_to_rgb(proof[n]) for n in names}

    def classify(a, pal):
        cols = np.array([pal[n] for n in names] + [(255, 255, 255)], dtype=np.int32)
        d = ((a[:, :, None, :].astype(np.int32) - cols[None, None]) ** 2).sum(-1)
        lab = d.argmin(-1)
        lab[d.min(-1) > tol * tol] = -1
        return lab
    ca, cb = classify(img, prf), classify(ref_orig, orig)
    both = (ca >= 0) & (cb >= 0)
    out = {}
    for i, n in enumerate(names):
        A, Bm = (ca == i) & both, (cb == i) & both
        u = (A | Bm).sum()
        out[n] = float((A & Bm).sum() / u) if u else 1.0
    return out


def main():
    summary = {}
    print(f"output dir: {OUT}")
    trace_src = open(os.path.join(os.path.dirname(HERE), "source", "K-Logo_trace.svg")).read()
    # ---------------------------------------------------------------- sRGB -> CMYK
    print(f"\n== sRGB -> CMYK via ICC: {vx.icc_profile_name()} ({vx.CMYK_PROFILE}), relative colorimetric + BPC")
    conv = {}
    for hx in ("#EE3F2A", "#FAC311", "#2CA53B", "#3B81F1"):
        raw = vx.srgb_to_cmyk(hx, snap_pct=None)
        v = vx.srgb_to_cmyk(hx)
        conv[hx] = {"cmyk_pct": [round(x * 100, 1) for x in v], "raw_pct": [round(x * 100, 2) for x in raw],
                    "ink_pct": round(sum(v) * 100, 1), "proof_srgb": vx.cmyk_to_srgb(v), "dE76": round(vx.delta_e76(hx, v), 2)}
        print(f"   {hx}: C {v[0]*100:.0f} M {v[1]*100:.0f} Y {v[2]*100:.0f} K {v[3]*100:.0f}  (raw {conv[hx]['raw_pct']}, "
              f"ink {conv[hx]['ink_pct']}%, proof {conv[hx]['proof_srgb']}, dE76 {conv[hx]['dE76']})")
    for hx in ("#000000", "#808080", "#FEFDFB"):
        print(f"   {hx}: neutrals->K {vx.srgb_to_cmyk(hx)}   icc {vx.srgb_to_cmyk(hx, neutrals='icc')}")
    summary["srgb_to_cmyk"] = conv

    # ---------------------------------------------------------------- fixture
    fx = fixture_svg()
    open(os.path.join(OUT, "fixture.svg"), "w").write(fx)
    r = run_rgb("fixture", fx, os.path.join(OUT, "fixture"), expect_images=2)
    summary["fixture"] = r
    rp = r["report"]
    check("fixture .eps: alpha paints over solid/gradient backdrops stay vector (FlatRing, g-edge-lit, g-green, Wave, Center)",
          all(x in rp["eps_vector_blended"] for x in ("FlatRing", "g-edge-lit", "g-green", "Wave", "Center")),
          f"vector-blended={rp['eps_vector_blended']}")
    check("fixture .eps: only non-representable transparency rasterised (alpha-gradient over gradients, overlapping "
          "group opacity, mask, 2 filters)", len(rp["eps_flattened"]) == 5, f"{rp['eps_flattened']}")
    st = r["pdf_structure"]["counts"]
    check("fixture: gradients are shading patterns (>= 7) and stop-opacity uses soft masks",
          st["shadings"] >= 7 and st["smasks"] >= 3, str(st))
    check("fixture: named groups kept as Form XObjects", st["form_xobjects"] >= 10, str(st))

    # ---------------------------------------------------------------- options: flat groups, scale, viewBox origin
    rf = vx.export_rgb(fx, os.path.join(OUT, "fixture_flat_half"), group_mode="flat", scale=0.5)
    ok, info = pdf_structure(rf["pdf"], 512.0)
    ref = ref_render(fx, 1024)
    stf = [diff_stats(gs_render(rf[k], 1024, 512.0, eps=(k == "eps"))[0], ref)["mean"] for k in ("pdf", "eps")]
    check("fixture group_mode='flat', scale=0.5 (512 pt page): valid, no Form groups except compositing, "
          "pdf/eps vs rsvg mean < 2/255", ok and info["counts"]["form_xobjects"] == info["counts"]["transparency_groups"]
          and max(stf) < 2.0, f"boxes={info['boxes']['TrimBox']} forms={info['counts']['form_xobjects']} "
          f"mean pdf={stf[0]:.3f} eps={stf[1]:.3f}")
    vb = trace_src.replace('viewBox="0 0 1024 1024" width="1024" height="1024"', 'viewBox="-100 -50 1224 1124" width="1224" height="1124"')
    rv = vx.export_rgb(vb, os.path.join(OUT, "trace_vb_offset"))
    refv = np.asarray(Image.alpha_composite(Image.new("RGBA", (1224, 1124), (255, 255, 255, 255)),
                                            Image.open(io.BytesIO(subprocess.run(["rsvg-convert", "-f", "png"], input=vb.encode(),
                                                                                 capture_output=True, check=True).stdout)).convert("RGBA")).convert("RGB")).astype(np.int16)
    stv = [diff_stats(gs_render(rv[k], 1224, 1224.0, eps=(k == "eps"))[0][:1124], refv)["mean"] for k in ("pdf", "eps")]
    check("viewBox with non-zero origin (-100 -50 1224 1124): page 1224x1124 pt, pdf/eps vs rsvg mean < 2/255",
          rv["page_pt"] == (1224.0, 1124.0) and max(stv) < 2.0, f"mean pdf={stv[0]:.3f} eps={stv[1]:.3f}")

    # ---------------------------------------------------------------- edge-case fixture
    eg = edge_svg()
    open(os.path.join(OUT, "edge.svg"), "w").write(eg)
    re_ = run_rgb("edge fixture", eg, os.path.join(OUT, "edge"), px_list=(800,))
    summary["edge"] = re_
    check("edge fixture: clip-path on <clipPath> ignored as librsvg does, with a warning",
          any("ignored, as librsvg does" in w for w in re_["report"]["warnings"]), str(re_["report"]["warnings"]))
    sp = vx.export_rgb(eg, os.path.join(OUT, "edge_spec"), compat="spec")
    g1, _ = gs_render(sp["pdf"], 800, sp["page_pt"][0])
    p1 = pdfium_render(sp["pdf"], 800, sp["page_pt"][0])
    st = diff_stats(g1, p1)
    check("edge fixture compat='spec' (nested clip applied): gs vs pdfium agree, mean |d| < 2/255", st["mean"] < 2.0,
          f"mean={st['mean']:.3f}/255")

    # ---------------------------------------------------------------- trace (as delivered SVG, with disc)
    summary["trace_svg"] = run_rgb("trace", trace_src, os.path.join(OUT, "K-Logo_trace"), expect_fills=5)
    # trace K only (kmark exact polygons), for gap checks
    kk = svg_doc(flat_group(K, trap=False), size=K.size, title="K logo - trace (mark only)")
    summary["trace_mark"] = run_rgb("trace mark", kk, os.path.join(OUT, "K-mark"), px_list=(1024,), expect_fills=4)
    gap_check("trace mark .pdf", os.path.join(OUT, "K-mark.pdf"), K.size)
    gap_check("trace mark .eps", os.path.join(OUT, "K-mark.eps"), K.size, eps=True)
    vertex_identity("trace mark .pdf", os.path.join(OUT, "K-mark.pdf"))

    # ---------------------------------------------------------------- CMYK
    shapes = [(n, poly_d(facets(K)[n]), vx.srgb_to_cmyk(PAL[n]), {}, PAL[n]) for n in FACETS]
    summary["cmyk_mark"] = run_cmyk("CMYK mark", shapes, K.size, os.path.join(OUT, "K-mark"), "K mark (CMYK)")
    gap_check("CMYK mark .pdf", os.path.join(OUT, "K-mark_CMYK.pdf"), K.size)
    gap_check("CMYK mark .eps", os.path.join(OUT, "K-mark_CMYK.eps"), K.size, eps=True)
    kr = KParams.from_dict(dict(json.load(open(os.path.join(HERE, "trace_params.json"))), r_convex=30.0, r_concave=12.0))
    summary["cmyk_rounded"] = run_cmyk("CMYK rounded (clip_d with arcs)", shapes, K.size, os.path.join(OUT, "K-rounded"),
                                       "K mark rounded (CMYK)", clip_d=silhouette_d(kr))
    # flat SVG straight through the CLI path (svg_to_cmyk_shapes incl. disc circle)
    shp, size = vx.svg_to_cmyk_shapes(trace_src)
    check("svg_to_cmyk_shapes(trace SVG): 5 shapes incl. disc", len(shp) == 5, f"{[(sh[0], sh[2]) for sh in shp]}")
    summary["cmyk_trace"] = run_cmyk("CMYK trace via svg_to_cmyk_shapes", shp, size, os.path.join(OUT, "K-Logo_trace"),
                                     "K logo - trace (CMYK)")

    # ---------------------------------------------------------------- badge: ring stroke at 8 % opacity
    badge = svg_doc('<circle id="disc" cx="512" cy="512" r="512" fill="#FEFDFB"/>'
                    '<circle id="ring" cx="512" cy="512" r="510" fill="none" stroke="#141A29" stroke-opacity="0.08" stroke-width="4"/>'
                    + flat_group(K, trap=False), size=K.size, title="K badge (flat)")
    open(os.path.join(OUT, "badge.svg"), "w").write(badge)
    rb = run_rgb("badge", badge, os.path.join(OUT, "badge"), px_list=(1024,), expect_fills=5)
    check("badge .eps: ring stroke with stroke-opacity stays vector (nothing rasterised)",
          not rb["report"]["eps_flattened"] and "ring" in rb["report"]["eps_vector_blended"],
          f"flattened={rb['report']['eps_flattened']} vector-blended={rb['report']['eps_vector_blended']}")
    shp, size = vx.svg_to_cmyk_shapes(badge)
    ring = [sh for sh in shp if sh[0] == "ring"]
    check("svg_to_cmyk_shapes(badge): ring -> K-only stroke pre-blended over the disc",
          len(ring) == 1 and ring[0][3].get("stroke") == 4.0 and ring[0][2][:3] == (0.0, 0.0, 0.0),
          f"{[(sh[0], sh[2], sh[3].get('stroke')) for sh in shp]}")
    summary["cmyk_badge"] = run_cmyk("CMYK badge (stroke)", shp, size, os.path.join(OUT, "badge"), "K badge (CMYK)")

    # ---------------------------------------------------------------- cairo/rsvg cross-check (informational)
    print("\n== cairo (rsvg-convert -f pdf/eps) on the fixture, for comparison")
    cpdf, ceps = os.path.join(OUT, "cairo_fixture.pdf"), os.path.join(OUT, "cairo_fixture.eps")
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", cpdf, os.path.join(OUT, "fixture.svg")], check=True)
    subprocess.run(["rsvg-convert", "-f", "eps", "-o", ceps, os.path.join(OUT, "fixture.svg")], check=True)
    _, ci = pdf_structure(cpdf, 768.0)
    et = open(ceps, "rb").read().decode("latin-1")
    cinfo = {"pdf_counts": ci["counts"], "pdf_boxes": ci["boxes"], "pdf_Info": ci["Info"],
             "eps_BoundingBox": re.search(r"%%BoundingBox: (.*)", et).group(1),
             "eps_HiResBoundingBox": (re.search(r"%%HiResBoundingBox: (.*)", et) or [None, None])[1] if re.search(r"%%HiResBoundingBox: (.*)", et) else None,
             "eps_image_ops": len(re.findall(r"\bimage\b", et)), "eps_bytes": len(et)}
    print(f"   {json.dumps(cinfo)}")
    summary["cairo_crosscheck"] = cinfo

    # ---------------------------------------------------------------- summary
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    summary["checks"] = [{"name": n, "ok": ok, "detail": d} for n, ok, d in RESULTS]
    with open(os.path.join(OUT, "verification.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
