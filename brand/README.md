# K Logo — brand kit

The client's four-colour "K", traced exactly from the original and redesigned in premium
quality. Every file here is generated from one master (`tools/brand.py`), so a change to the
geometry or a colour is one edit plus one re-run.

![Preview](K-Logo_Preview.png)

**Start with the brand guidelines:** [`guidelines/K-Logo_Brand_Guidelines.pdf`](guidelines/K-Logo_Brand_Guidelines.pdf)
(11 pages: the idea, construction, refinement, clear space and sizes, variants, colour,
backgrounds, misuse, icons, file guide). An HTML version sits next to it.

## What changed in the redesign

| | Original | Redesign |
| --- | --- | --- |
| Arm / leg angle | 52.75° / 51.66° (unequal) | one 4:3 slope, 53.13°, for both |
| Stroke weights | arm 98.4 %, leg 96.7 % of the stem | arm = leg = 96.5 % (optical compensation for diagonals) |
| Right edges | arm and leg off by 5 units | aligned exactly |
| Junction / crotch | arbitrary | mirror each other about the fold point P |
| Centring | bounding box | optical (+14 / −6 units on the 1024 grid) |
| Corners | sharp | sharp, everywhere (client's choice) |
| Colours | `#EE3F2A #FAC311 #2CA53B #3B81F1` | your own four colours, refined — deeper and cleaner, with the blue and green nudged off Google's hues; still the same four-colour family |
| Footprint | cap height 58.7 % of the disc | unchanged |

The trace matches the original JPEG on 99.8 % of pixels; every mismatch is a 1-pixel
anti-aliased edge. It is delivered in every format in `source/` (`K-Logo_Trace.ai/.pdf/.eps/.psd`,
`K-Logo_Trace_1024.png`, `K-Logo_Trace_4096.png`, `K-Logo_trace.svg`) next to the original JPEG.

## Which file do I use?

| Need | Use |
| --- | --- |
| Rules and usage | `guidelines/K-Logo_Brand_Guidelines.pdf` |
| Edit in **Adobe Illustrator** | `logo/ai/*.ai` — PDF-compatible Illustrator files (the same file as the matching PDF). Illustrator opens them as editable vector artwork on one layer. For **native Illustrator layers and global swatches**, open Illustrator and run `logo/scripts/K-Logo_Illustrator.jsx` (File › Scripts › Other Script…): it rebuilds every variant on its own artboard and layer, with the brand colours as global swatches in a "K Brand" group (gradient tones go in a separate "K Shading" group), and saves `.ai`, `.pdf` and `.svg` next to the script. |
| Edit in **Adobe Photoshop** | `logo/psd/*.psd` — 4096 px, 300 dpi, sRGB. In the flat variants every facet and the disc is an **editable vector shape layer** (facets overlap by a hair under each other, so joins never show a seam). In the shaded and tech variants the lit facets and lighting are pixel layers at 4096 px; their exact vector paths are in the `.ai`/`.pdf`/`.svg`. `K-Logo_All_Variants.psd` holds the main variants as layer groups — switch them on and off. `logo/scripts/K-Logo_Photoshop.jsx` rebuilds the flat variants natively in Photoshop. |
| **Print** (offset, signage, merch) | `logo/ai/*_CMYK.ai`, `logo/pdf/*_CMYK.pdf`, `logo/eps/*_CMYK.eps` — device CMYK |
| Other vector apps (Figma, Affinity, Inkscape, CorelDRAW) | `logo/svg/*.svg` (layered, named groups) or `logo/eps/*.eps` |
| Websites, documents, slides | `logo/png/*` (256–4096 px wide, transparent: colour, tight, clear-space, badge, shaded, gapped and solid one-colour, greyscale, tech) or `web/svg/*` |
| Favicon / PWA | `icons/web/` + paste `icons/web/head-snippet.html` into `<head>`; set `name` / `short_name` in `icons/web/site.webmanifest` to your product name (it says "K" until there is one) |
| iOS / macOS / Android apps | `icons/ios/AppIcon.appiconset/` (Xcode "All Sizes", 13 opaque PNGs), `icons/macos/` (iconset + `.icns`), `icons/android/res/` (adaptive foreground/background/themed + legacy, all densities) — all overviewed in `icons/K-Logo_Icons_Overview.png` |
| Profile pictures, link previews, banners | `social/` |
| Code (React, Web Component, design tokens) | `web/` — see `web/README.md` |

The vector masters (`.svg`, `.ai`, `.pdf`, `.eps`) keep the exact construction: facets meet edge
to edge on shared vertices, so the numbers in the guidelines can be measured in them directly.

### Variants

| Variant | File stem | When |
| --- | --- | --- |
| Primary mark | `K-Logo_Mark_Color` (`_Tight`, `_ClearSpace`) | default, on white or light backgrounds |
| Avatar badge | `K-Logo_Badge_Color` | profile pictures, round crops, photos |
| Premium shaded | `K-Logo_Mark_Dimensional`, `K-Logo_Badge_Dimensional` | hero use, splash screens, presentations (light backgrounds) |
| Tech app icon | `K-Logo_Tech_Icon` (`_Small` below 120 px, `_Square` for app stores, `_macOS`) | app icons, dark product UI |
| One colour, gapped | `K-Logo_Mark_Black`, `K-Logo_Mark_White` | one-colour print, engraving, reversed on dark — mark at least 128 px / 20 mm tall |
| One colour, solid | `K-Logo_Mark_Black_Solid`, `K-Logo_Mark_White_Solid` | smaller than that, stamps, embroidery |
| Greyscale | `K-Logo_Mark_Greyscale` | black-only print where tints are possible |

## Colour

| | HEX | RGB | OKLCH | CMYK (approx.) | on white | on night |
| --- | --- | --- | --- | --- | --- | --- |
| Red | `#EA3A23` | 234 58 35 | 0.615 0.215 31.0 | 2 93 99 0 | 4.10:1 | 4.74:1 |
| Yellow | `#F6C105` | 246 193 5 | 0.835 0.170 88.5 | 4 24 100 0 | 1.67:1 | 11.60:1 |
| Green | `#2CA235` | 44 162 53 | 0.625 0.180 144.0 | 80 10 100 1 | 3.32:1 | 5.85:1 |
| Blue | `#4074F4` | 64 116 244 | 0.595 0.200 264.5 | 75 58 0 0 | 4.18:1 | 4.64:1 |
| Ink | `#141A29` | 20 26 41 | — | 84 77 55 70 | 17.36:1 | — |
| Night | `#070D1A` | 7 13 26 | — | 80 73 59 79 | 19.42:1 | — |

CMYK values come from an ICC conversion (sRGB → Ghostscript's *Artifex CMYK SWOP Profile*,
relative colorimetric with black-point compensation). The **blue** is well outside the CMYK
gamut (ΔE76 ≈ 32): where it matters, have the printer match `#4074F4` with a spot ink chosen
against a physical swatch book. The **green** is just outside (ΔE76 ≈ 9) — check it on the proof.
Use your printer's own profile for press work.

## Rules

- Every corner of the K is **sharp**. Only containers (the disc, the app tile) are curved.
- Never recolour, rearrange, rotate, stretch, outline or add effects to the flat mark.
- Brand yellow never stands alone on white (1.67:1) — it only appears inside the mark.
- Clear space: one stem width on every side (0.232 × the mark height).
- Minimum size: colour mark, badge and solid mark 16 px (the favicons are pixel-hinted) / 8 mm in
  print. The gapped one-colour mark needs 128 px / 20 mm so its construction gaps stay visible;
  below that use the solid one. The tech icon uses full detail from 120 px and its small version below
  that (the app-icon sets already follow this: iOS 20–87 px, macOS up to 128 px and Android 48–96 px
  legacy icons are the small version).
- No name or wordmark is defined yet, so none is drawn anywhere in the kit.

## Rebuild

```bash
python3 brand/tools/build.py all                   # everything below, in order
python3 brand/tools/build.py svg formats           # or individual stages:
#   svg formats scripts trace raster web guidelines preview
```

Needs Python 3.10+ with `pillow numpy shapely psd-tools fonttools pypdf`, librsvg
(`rsvg-convert`), Ghostscript, Node 22 with Playwright's Chromium (guidelines PDF), and the
Inter and JetBrains Mono fonts. Vector and PSD outputs embed build timestamps, so a rebuild
differs from the committed files only in those dates.

| Tool | Job |
| --- | --- |
| `tools/kmark.py` | Parametric construction of the K (stem, arm, leg, junction J, fold point P, crotch C) |
| `tools/brand.py` | Final master: geometry, palette, every variant as layered SVG, supersampled rendering |
| `tools/build.py` | The build: master SVGs, AI/PDF/EPS/PSD, scripts, trace formats, preview, and the stages below |
| `tools/svg2psd.py` | SVG → layered PSD with editable vector shape layers |
| `tools/vector_export.py` | SVG → PDF / AI / EPS (RGB, vector gradients) and device-CMYK print files |
| `tools/illustrator_jsx.py` | Generates the Illustrator / Photoshop rebuild scripts |
| `tools/build_raster.py`, `tools/build_web.py`, `tools/build_guidelines.py` | PNG / icon / social, developer-kit and guidelines stages |
| `tools/trace_svg.py`, `tools/trace_params.json` | The exact trace of the original |
| `tools/test_*.py` | Verification suites for the converters |
