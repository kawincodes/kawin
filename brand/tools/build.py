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
    "K-Logo_Mark_Black_Solid": (lambda: brand.doc_mark(style="black"), "One colour, black, solid (stamps, embroidery, < 32 px)"),
    "K-Logo_Mark_White_Solid": (lambda: brand.doc_mark(style="white"), "One colour, white, solid"),
    "K-Logo_Mark_Greyscale": (lambda: brand.doc_mark(style="grey"), "Greyscale tints for black-only print"),
    "K-Logo_Tech_Icon": (lambda: brand.doc_tech(), "Tech app icon, rounded tile"),
    "K-Logo_Tech_Icon_Small": (lambda: brand.doc_tech(detail="small"), "Tech app icon simplified for <= 64 px"),
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


STAGES = {"svg": stage_svg}

if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    for st in (STAGES if which == ["all"] else which):
        print(st, "->", STAGES[st]())
