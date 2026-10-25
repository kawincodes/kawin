"""Writes the faithful vector trace of the original logo (geometry fitted to the source pixels)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from kmark import KParams, facets, poly_d, fmt, FACETS

k = KParams.from_dict(json.load(open(os.path.join(os.path.dirname(__file__), "trace_params.json"))))


def trace_svg(with_disc=True):
    pal = k.palette
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {fmt(k.size)} {fmt(k.size)}" width="{fmt(k.size)}" height="{fmt(k.size)}">',
         '<title>K logo - faithful vector trace of the original</title>']
    if with_disc:
        s.append(f'<circle id="disc" cx="{fmt(k.size/2)}" cy="{fmt(k.size/2)}" r="{fmt(k.size/2)}" fill="{pal["disc"]}"/>')
    s.append('<g id="K">')
    for n in FACETS:
        s.append(f'  <path id="{n}" fill="{pal[n]}" d="{poly_d(facets(k)[n])}"/>')
    s.append('</g></svg>')
    return "\n".join(s)


if __name__ == "__main__":
    print(trace_svg())
