"""Reproducible build of the brand kit.

    python3 brand/tools/build.py svg        # editable master SVGs  -> brand/logo/svg/
    python3 brand/tools/build.py all        # every stage that is wired in

Every stage regenerates its outputs from brand.py, so a change to the master geometry or the
palette is one edit plus one re-run.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import brand  # noqa: E402

# name -> (builder, description). The SVGs are the editable masters: exact facet polygons,
# named groups (id + data-name) that become layers in the PSD and named groups in the AI.
VARIANTS = {
    "K-Logo_Mark_Color": (lambda: brand.doc_mark(), "Primary mark, full colour, 1024 square artboard"),
    "K-Logo_Mark_Color_Tight": (lambda: brand.doc_mark(artboard="tight"), "Primary mark cropped to its bounds"),
    "K-Logo_Mark_Color_ClearSpace": (lambda: brand.doc_mark(artboard="clearspace"), "Primary mark with minimum clear space (1 stem width)"),
    "K-Logo_Badge_Color": (lambda: brand.doc_badge(), "Avatar badge: mark on the white disc"),
    "K-Logo_Mark_Dimensional": (lambda: brand.doc_mark(style="dimensional"), "Premium shaded mark for light backgrounds"),
    "K-Logo_Badge_Dimensional": (lambda: brand.doc_badge(style="dimensional"), "Premium shaded avatar badge"),
    "K-Logo_Mark_Black": (lambda: brand.doc_mark(style="black-gapped"), "One colour, black, construction gaps"),
    "K-Logo_Mark_White": (lambda: brand.doc_mark(style="white-gapped"), "One colour, white (reversed), construction gaps"),
    "K-Logo_Mark_Black_Solid": (lambda: brand.doc_mark(style="black"), "One colour, black, solid (below 128 px / 20 mm mark height, stamps, embroidery)"),
    "K-Logo_Mark_White_Solid": (lambda: brand.doc_mark(style="white"), "One colour, white, solid"),
    "K-Logo_Mark_Greyscale": (lambda: brand.doc_mark(style="grey"), "Greyscale tints for black-only print"),
    "K-Logo_Tech_Icon": (lambda: brand.doc_tech(), "Tech app icon, rounded tile"),
    "K-Logo_Tech_Icon_Small": (lambda: brand.doc_tech(detail="small"), "Tech app icon simplified for sizes below 120 px"),
    "K-Logo_Tech_Icon_Square": (lambda: brand.doc_tech(shape="square"), "Tech app icon, full-bleed opaque square (App Store / Play)"),
    "K-Logo_Tech_Icon_macOS": (lambda: brand.doc_tech(shape="macos"), "Tech app icon on Apple's macOS grid (824 tile on 1024)"),
    "K-Logo_Android_Foreground": (lambda: brand.doc_android_foreground(), "Android adaptive icon foreground"),
    "K-Logo_Android_Background": (lambda: brand.doc_android_background(), "Android adaptive icon background"),
}


def stage_svg():
    out = os.path.join(ROOT, "logo", "svg")
    os.makedirs(out, exist_ok=True)
    for name, (fn, _) in VARIANTS.items():
        with open(os.path.join(out, f"{name}.svg"), "w") as f:
            f.write(fn())
    with open(os.path.join(ROOT, "logo", "K-Logo_master.json"), "w") as f:
        f.write(brand.params_json())
    return out


# Variants exported as vector files. Gradient-heavy variants skip EPS: EPS has no transparency,
# so their soft lighting would be flattened to pixels there; PDF/AI keep it vector.
VECTOR_RGB = [
    ("K-Logo_Mark_Color", True), ("K-Logo_Mark_Color_Tight", True), ("K-Logo_Mark_Color_ClearSpace", True),
    ("K-Logo_Badge_Color", True), ("K-Logo_Mark_Black", True), ("K-Logo_Mark_White", True),
    ("K-Logo_Mark_Black_Solid", True), ("K-Logo_Mark_White_Solid", True), ("K-Logo_Mark_Greyscale", True),
    ("K-Logo_Mark_Dimensional", False), ("K-Logo_Badge_Dimensional", False),
    ("K-Logo_Tech_Icon", False), ("K-Logo_Tech_Icon_Square", False),
]
# Print masters in device CMYK (flat art only).
VECTOR_CMYK = ["K-Logo_Mark_Color", "K-Logo_Mark_Color_Tight", "K-Logo_Badge_Color",
               "K-Logo_Mark_Black", "K-Logo_Mark_Black_Solid", "K-Logo_Mark_Greyscale"]
# Layered Photoshop files (flat facets become editable vector shape layers).
PSD = ["K-Logo_Mark_Color", "K-Logo_Badge_Color", "K-Logo_Mark_Dimensional", "K-Logo_Badge_Dimensional",
       "K-Logo_Tech_Icon", "K-Logo_Tech_Icon_Square", "K-Logo_Mark_Black", "K-Logo_Mark_White",
       "K-Logo_Mark_Black_Solid", "K-Logo_Mark_Greyscale"]
# Photoshop composites every shape layer with its own anti-aliasing, so facets that only touch
# would show a hairline seam; the PSDs therefore use the trapped facets (each grows a hair under
# the facets painted after it). The visible outline is unchanged. The vector masters keep the
# exact, abutting facets.
PSD_SOURCES = {
    "K-Logo_Mark_Color": lambda: brand.doc_mark(raster=True),
    "K-Logo_Badge_Color": lambda: brand.doc_badge(raster=True),
    "K-Logo_Mark_Greyscale": lambda: brand.doc_mark(style="grey", raster=True),
}
# One master PSD holding the main variants as layer groups; only the first is visible.
PSD_MASTER = [("Badge - colour", "K-Logo_Badge_Color"), ("Mark - colour", "K-Logo_Mark_Color"),
              ("Badge - premium shaded", "K-Logo_Badge_Dimensional"), ("Tech icon", "K-Logo_Tech_Icon"),
              ("Mark - one colour black", "K-Logo_Mark_Black"), ("Mark - one colour white", "K-Logo_Mark_White")]
PSD_PX = 4096


def _svg(name):
    return VARIANTS[name][0]()


def _psd_svg(name):
    return PSD_SOURCES[name]() if name in PSD_SOURCES else _svg(name)


def _inner(svg):
    """(defs, body) of one of our single-root SVG documents."""
    import re
    defs = "".join(re.findall(r"<defs>(.*?)</defs>", svg, re.S))
    body = re.sub(r"^.*?<svg[^>]*>", "", svg, count=1, flags=re.S)
    body = re.sub(r"</svg>\s*$", "", body)
    body = re.sub(r"<title>.*?</title>", "", body, flags=re.S)
    body = re.sub(r"<defs>.*?</defs>", "", body, flags=re.S)
    return defs, body


def _prefix_ids(text, p):
    """Make ids unique when several variant documents are merged into one."""
    import re
    text = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{p}{m.group(1)}"', text)
    text = re.sub(r"url\(#([^)]+)\)", lambda m: f"url(#{p}{m.group(1)})", text)
    return re.sub(r'href="#([^"]+)"', lambda m: f'href="#{p}{m.group(1)}"', text)


def master_psd_svg():
    defs, groups = [], []
    for i, (label, name) in enumerate(PSD_MASTER):
        d, b = (_prefix_ids(t, f"v{i}-") for t in _inner(_psd_svg(name)))
        defs.append(d)
        hidden = "" if i == 0 else ' display="none"'
        groups.append(f'<g id="variant-{i}" data-name="{label}"{hidden}>{b}</g>')
    return brand.svg_doc("".join(reversed(groups)), "".join(defs), size=brand.SIZE, title="K Logo - all variants")


def stage_formats():
    import shutil
    import tempfile

    import svg2psd
    import vector_export as vx

    logo = os.path.join(ROOT, "logo")
    dirs = {ext: os.path.join(logo, ext) for ext in ("ai", "pdf", "eps", "psd")}
    for d in dirs.values():
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    written = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, eps in VECTOR_RGB:
            vx.export_rgb(_svg(name), os.path.join(tmp, name), title=f"{name.replace('_', ' ')}")
            for ext in ("ai", "pdf") + (("eps",) if eps else ()):
                dst = os.path.join(dirs[ext], f"{name}.{ext}")
                shutil.move(os.path.join(tmp, f"{name}.{ext}"), dst)
                written.append(dst)
        for name in VECTOR_CMYK:
            shapes, size = vx.svg_to_cmyk_shapes(_svg(name))
            vx.export_cmyk(shapes, size, os.path.join(tmp, name), f"{name.replace('_', ' ')} (CMYK)")
            for ext in ("ai", "pdf", "eps"):
                dst = os.path.join(dirs[ext], f"{name}_CMYK.{ext}")
                shutil.move(os.path.join(tmp, f"{name}_CMYK.{ext}"), dst)
                written.append(dst)
    for name in PSD:
        dst = os.path.join(dirs["psd"], f"{name}.psd")
        svg2psd.convert(_psd_svg(name), dst, px=PSD_PX, dpi=300, title=name.replace("_", " "))
        written.append(dst)
    dst = os.path.join(dirs["psd"], "K-Logo_All_Variants.psd")
    svg2psd.convert(master_psd_svg(), dst, px=PSD_PX, dpi=300, title="K Logo - all variants")
    written.append(dst)
    return f"{len(written)} files"


# Variants rebuilt natively by the Illustrator / Photoshop scripts (real layers, global swatches).
SCRIPT_VARIANTS = [("K Mark", "K-Logo_Mark_Color"), ("K Badge", "K-Logo_Badge_Color"),
                   ("K Premium", "K-Logo_Badge_Dimensional"), ("K Tech Icon", "K-Logo_Tech_Icon"),
                   ("K One Colour", "K-Logo_Mark_Black")]
# Photoshop's gradient fill layers cannot reproduce the shaded/tech lighting faithfully (scale
# limits, no elliptical radials), so its script rebuilds the flat variants only; the PSDs in
# logo/psd cover the rest.
PHOTOSHOP_SCRIPT_VARIANTS = ("K Mark", "K Badge", "K One Colour")


def stage_scripts():
    import illustrator_jsx as ij

    out = os.path.join(ROOT, "logo", "scripts")
    os.makedirs(out, exist_ok=True)
    variants = [ij.svg_to_variant(_svg(name), label) for label, name in SCRIPT_VARIANTS]
    palette = dict(brand.PALETTE, **brand.NEUTRAL)
    reports = [
        ij.illustrator_script(variants, palette, os.path.join(out, "K-Logo_Illustrator.jsx"), base="K-Logo"),
        ij.photoshop_script([v for v in variants if v["name"] in PHOTOSHOP_SCRIPT_VARIANTS], palette,
                            os.path.join(out, "K-Logo_Photoshop.jsx"), px=PSD_PX, base="K-Logo"),
    ]
    for rep in reports:
        warnings = rep.get("warnings", [])
        shades = [w for w in warnings if "is not in the palette" in w]
        if shades:   # expected: gradient tones of the shaded/tech variants, filed under "K Shading"
            print(f"  {os.path.basename(rep['path'])}: {len(shades)} shading tones in the 'K Shading' swatch group")
        for w in warnings:
            if w not in shades:
                print(f"  warning ({os.path.basename(rep['path'])}): {w}")
    return out


def stage_trace():
    """The exact trace of the original in every format, next to the trace SVG in source/."""
    import json
    import shutil
    import tempfile

    import svg2psd
    import vector_export as vx

    src = os.path.join(ROOT, "source")
    with open(os.path.join(HERE, "trace_params.json")) as f:
        k = brand.KParams.from_dict(json.load(f))
    exact = open(os.path.join(src, "K-Logo_trace.svg")).read()
    trapped = brand.svg_doc(brand.disc(k) + brand.mark_group(k, raster=True), size=k.size,
                            title="K logo - trace of the original")
    with tempfile.TemporaryDirectory() as tmp:
        vx.export_rgb(exact, os.path.join(tmp, "t"), title="K Logo - trace of the original")
        for ext in ("ai", "pdf", "eps"):
            shutil.move(os.path.join(tmp, f"t.{ext}"), os.path.join(src, f"K-Logo_Trace.{ext}"))
    svg2psd.convert(trapped, os.path.join(src, "K-Logo_Trace.psd"), px=PSD_PX, dpi=300,
                    title="K Logo - trace of the original")
    for px in (1024, 4096):
        brand.render(trapped, px).save(os.path.join(src, f"K-Logo_Trace_{px}.png"), optimize=True)
    return src


def stage_preview():
    import preview_sheet
    return preview_sheet.build(os.path.join(ROOT, "K-Logo_Preview.png"))


def _module_stage(module):
    def run():
        mod = __import__(module)
        return f"{len(mod.run(ROOT))} files"
    return run


# Order matters: later stages read the SVG masters written by the first one.
STAGES = {
    "svg": stage_svg,
    "formats": stage_formats,
    "scripts": stage_scripts,
    "trace": stage_trace,
    "raster": _module_stage("build_raster"),
    "web": _module_stage("build_web"),
    "guidelines": _module_stage("build_guidelines"),
    "preview": stage_preview,
}

if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    for st in (STAGES if which == ["all"] else which):
        print(st, "->", STAGES[st]())
