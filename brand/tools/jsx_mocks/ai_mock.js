#!/usr/bin/env node
// Recording mock of the Adobe Illustrator ExtendScript DOM: exactly the members that the scripts
// written by illustrator_jsx.py use, implemented STRICTLY. Unknown properties, unknown enum
// members, wrong argument types and out-of-range values throw (and are logged in `errors`, so a
// failure swallowed by a try/catch inside the script is still reported).
//
//   node ai_mock.js SCRIPT.jsx RECORD.json [--coord artboard|document] [--script-path PATH]
//
// What the mock ASSUMES about Illustrator (documented behaviour as far as known; these are the
// things a mock run cannot prove):
//  * Scripting coordinates are points with y UP. DOCUMENTCOORDINATESYSTEM measures from the ruler
//    origin, ARTBOARDCOORDINATESYSTEM from the active artboard's top-left corner. The mock keeps
//    all geometry in a private canvas frame that is neither of the two, and starts with the ruler
//    origin away from the canvas origin, so a script that mixes frames or relies on defaults fails.
//  * artboardRect is [left, top, right, bottom] with top > bottom; artboards must stay inside the
//    16383 pt canvas centred on the first artboard.
//  * pageItems / layers collection.add() puts the new object in FRONT (index 0 = top of stack).
//  * PathPoint.rightDirection is the handle leaving the anchor (towards the next point),
//    leftDirection the handle arriving from the previous point.
//  * Assigning a GradientColor fits the gradient to the object's bounds: the angle is kept, the
//    origin and length are ignored (as several Illustrator versions do); rotate / resize /
//    translate with the "fill gradients" flag then transform the gradient only.
//  * Gradient stops are kept sorted by rampPoint (Illustrator re-sorts the ramp).
//  * A Spot whose colorType is ColorModel.PROCESS is a global process swatch.
'use strict';
const fs = require('fs');
const vm = require('vm');
const path = require('path');

const argv = process.argv.slice(2);
if (argv.length < 2) { console.error('usage: ai_mock.js SCRIPT.jsx RECORD.json [--coord artboard|document] [--script-path PATH]'); process.exit(2); }
const SCRIPT = argv[0], OUT = argv[1];
const opt = (name, dflt) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : dflt; };
const START_CS = opt('--coord', 'artboard');
const SCRIPT_PATH = opt('--script-path', '/Users/client/K Brand Kit/' + path.basename(SCRIPT));

const errors = [];
const calls = [];
function fail(msg) { errors.push(msg); throw new Error(msg); }
function rec(op, data) { calls.push(Object.assign({ op }, data || {})); }
const isNum = (v) => typeof v === 'number' && isFinite(v);
function needNum(v, what) { if (!isNum(v)) fail(`${what}: number expected, got ${JSON.stringify(v)}`); return v; }
function needPt(v, what) {
  if (!Array.isArray(v) || v.length !== 2 || !isNum(v[0]) || !isNum(v[1])) fail(`${what}: [x, y] expected, got ${JSON.stringify(v)}`);
  return [v[0], v[1]];
}
function needBool(v, what) { if (typeof v !== 'boolean') fail(`${what}: boolean expected, got ${JSON.stringify(v)}`); return v; }

// ------------------------------------------------------------------ strict objects
let NEXT_ID = 1;
function strict(obj, cls) {
  obj._id = NEXT_ID++;
  obj._cls = cls;
  const px = new Proxy(obj, {
    get(t, k) {
      if (typeof k === 'symbol' || k in t || k[0] === '_') return t[k];
      if (k === 'toJSON' || k === 'then' || k === 'inspect' || k === 'constructor' || k === 'valueOf' || k === 'toString') return undefined;
      return fail(`${cls} has no property "${k}"`);
    },
    set(t, k, v) {
      if (typeof k === 'string' && k[0] === '_') { t[k] = v; return true; }
      if (!(k in t)) fail(`${cls}.${k} is not a property`);
      const d = Object.getOwnPropertyDescriptor(t, k) || Object.getOwnPropertyDescriptor(Object.getPrototypeOf(t), k);
      if (d && d.get && !d.set) fail(`${cls}.${k} is read-only`);
      t[k] = v;
      return true;
    },
  });
  obj._self = px;
  return px;
}
function enumType(name, members) {
  const o = {};
  for (const m of members) o[m] = `${name}.${m}`;
  return new Proxy(o, { get(t, k) { if (typeof k === 'symbol' || k in t) return t[k]; return fail(`${name}.${String(k)} does not exist`); },
                         set() { return fail(`${name} is read-only`); } });
}
function needEnum(v, E, what) { if (typeof v !== 'string' || !Object.values(E).includes(v)) fail(`${what}: enum value expected, got ${JSON.stringify(v)}`); return v; }

const DocumentColorSpace = enumType('DocumentColorSpace', ['RGB', 'CMYK']);
const CoordinateSystem = enumType('CoordinateSystem', ['DOCUMENTCOORDINATESYSTEM', 'ARTBOARDCOORDINATESYSTEM']);
const ColorModel = enumType('ColorModel', ['PROCESS', 'SPOT', 'REGISTRATION']);
const PointType = enumType('PointType', ['SMOOTH', 'CORNER']);
const GradientType = enumType('GradientType', ['LINEAR', 'RADIAL']);
const Transformation = enumType('Transformation', ['CENTER', 'TOPLEFT', 'TOP', 'TOPRIGHT', 'LEFT', 'RIGHT', 'BOTTOMLEFT', 'BOTTOM', 'BOTTOMRIGHT', 'DOCUMENTORIGIN']);
const UserInteractionLevel = enumType('UserInteractionLevel', ['DISPLAYALERTS', 'DONTDISPLAYALERTS']);
const PDFCompatibility = enumType('PDFCompatibility', ['ACROBAT4', 'ACROBAT5', 'ACROBAT6', 'ACROBAT7', 'ACROBAT8']);
const SVGCSSPropertyLocation = enumType('SVGCSSPropertyLocation', ['ENTITIES', 'STYLEATTRIBUTES', 'STYLEELEMENTS', 'PRESENTATIONATTRIBUTES']);
const SVGDocumentEncoding = enumType('SVGDocumentEncoding', ['ASCII', 'UTF8', 'UTF16']);
const ExportType = enumType('ExportType', ['SVG', 'PNG24', 'PNG8', 'JPEG', 'GIF', 'PHOTOSHOP', 'AUTOCAD', 'FLASH', 'TIFF', 'WOSVG']);

// ------------------------------------------------------------------ geometry helpers
const ID = [1, 0, 0, 1, 0, 0];
const mmul = (m, n) => [m[0] * n[0] + m[2] * n[1], m[1] * n[0] + m[3] * n[1], m[0] * n[2] + m[2] * n[3],
  m[1] * n[2] + m[3] * n[3], m[0] * n[4] + m[2] * n[5] + m[4], m[1] * n[4] + m[3] * n[5] + m[5]];
const about = (m, p) => mmul([1, 0, 0, 1, p[0], p[1]], mmul(m, [1, 0, 0, 1, -p[0], -p[1]]));
const apply = (m, p) => [m[0] * p[0] + m[2] * p[1] + m[4], m[1] * p[0] + m[3] * p[1] + m[5]];
function cubicExt(p0, p1, p2, p3) {
  const a = -p0 + 3 * p1 - 3 * p2 + p3, b = 2 * (p0 - 2 * p1 + p2), c = p1 - p0, out = [];
  let ts = [];
  if (Math.abs(a) < 1e-12) { if (Math.abs(b) > 1e-12) ts = [-c / b]; } else {
    const d = b * b - 4 * a * c;
    if (d >= 0) ts = [(-b + Math.sqrt(d)) / (2 * a), (-b - Math.sqrt(d)) / (2 * a)];
  }
  for (const t of ts) if (t > 0 && t < 1) { const u = 1 - t; out.push(u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3); }
  return out;
}
function ptsBounds(pts, closed) {      // exact Bezier bounds, internal coords (y up) -> [L, T, R, B]
  const xs = [], ys = [];
  const n = pts.length;
  for (let i = 0; i < n; i++) {
    xs.push(pts[i].a[0]); ys.push(pts[i].a[1]);
    if (i === n - 1 && !closed) break;
    const p = pts[i], q = pts[(i + 1) % n];
    xs.push(...cubicExt(p.a[0], p.r[0], q.l[0], q.a[0])); ys.push(...cubicExt(p.a[1], p.r[1], q.l[1], q.a[1]));
  }
  return xs.length ? [Math.min(...xs), Math.max(...ys), Math.max(...xs), Math.min(...ys)] : null;
}
const unionB = (a, b) => (!a ? b : !b ? a : [Math.min(a[0], b[0]), Math.max(a[1], b[1]), Math.max(a[2], b[2]), Math.min(a[3], b[3])]);

// ------------------------------------------------------------------ app / coordinate frames
const CANVAS_HALF = 16383 / 2;
const app = {};
let activeDoc = null;
let coordSys = CoordinateSystem[START_CS === 'document' ? 'DOCUMENTCOORDINATESYSTEM' : 'ARTBOARDCOORDINATESYSTEM'];
function frame() {                     // offset of the current scripting frame's origin, canvas coords
  const d = activeDoc;
  if (coordSys === CoordinateSystem.DOCUMENTCOORDINATESYSTEM) return d._ruler;
  const r = d._artboards[d._activeAB]._rect;
  return [r[0], r[1]];
}
const toInt = (p, what) => { const q = needPt(p, what), o = frame(); return [q[0] + o[0], q[1] + o[1]]; };
const fromInt = (p) => { const o = frame(); return [p[0] - o[0], p[1] - o[1]]; };

// ------------------------------------------------------------------ colours
function RGBColor() { const o = { red: 0, green: 0, blue: 0, typename: 'RGBColor' }; return strict(o, 'RGBColor'); }
function CMYKColor() { const o = { cyan: 0, magenta: 0, yellow: 0, black: 0, typename: 'CMYKColor' }; return strict(o, 'CMYKColor'); }
function SpotColor() { const o = { spot: null, tint: 100, typename: 'SpotColor' }; return strict(o, 'SpotColor'); }
function NoColor() { return strict({ typename: 'NoColor' }, 'NoColor'); }
function GradientColor() {
  const o = { gradient: null, origin: [0, 0], angle: 0, length: 0, hiliteAngle: 0, hiliteLength: 0, matrix: null, typename: 'GradientColor' };
  return strict(o, 'GradientColor');
}
function snapColor(c, what) {           // value copy, as Illustrator copies colours on assignment
  if (!c || !c._cls) fail(`${what}: colour object expected, got ${JSON.stringify(c)}`);
  if (c._cls === 'RGBColor') {
    for (const k of ['red', 'green', 'blue']) if (!isNum(c[k]) || c[k] < 0 || c[k] > 255) fail(`${what}: RGBColor.${k} out of range: ${c[k]}`);
    return { t: 'rgb', rgb: [c.red, c.green, c.blue] };
  }
  if (c._cls === 'CMYKColor') return { t: 'cmyk', cmyk: [c.cyan, c.magenta, c.yellow, c.black] };
  if (c._cls === 'SpotColor') {
    if (!c.spot || c.spot._cls !== 'Spot') fail(`${what}: SpotColor.spot must be a Spot`);
    if (!isNum(c.tint) || c.tint < 0 || c.tint > 100) fail(`${what}: SpotColor.tint out of range`);
    return { t: 'spot', spot: c.spot._id, tint: c.tint };
  }
  if (c._cls === 'GradientColor') {
    if (!c.gradient || c.gradient._cls !== 'Gradient') fail(`${what}: GradientColor.gradient must be a Gradient`);
    return { t: 'gradient', gradient: c.gradient._id, angle: c.angle, hiliteAngle: c.hiliteAngle, hiliteLength: c.hiliteLength };
  }
  if (c._cls === 'NoColor') return { t: 'none' };
  return fail(`${what}: unsupported colour ${c._cls}`);
}

// ------------------------------------------------------------------ collections
function collection(name, list, methods) {
  const base = Object.assign({}, methods || {});
  return new Proxy(base, {
    get(t, k) {
      if (typeof k === 'symbol') return undefined;
      if (k === 'length') return list().length;
      if (/^\d+$/.test(k)) { const L = list(), i = +k; if (i >= L.length) fail(`${name}[${i}]: no such element (length ${L.length})`); return L[i]; }
      if (k in t) return t[k];
      if (k === 'typename') return name;
      if (k === 'toJSON' || k === 'then') return undefined;
      return fail(`${name} has no property "${k}"`);
    },
    set(t, k) { return fail(`${name}.${String(k)} cannot be set`); },
  });
}
function getByName(list, what) {
  return function (n) { const x = list().find((o) => o.name === n); if (!x) fail(`${what}.getByName("${n}"): no such element`); return x; };
}

// ------------------------------------------------------------------ page items
class Item {
  constructor(doc, parent, typename) {
    this._doc = doc; this._parent = parent; this.typename = typename;
    this.name = ''; this._opacity = 100; this.hidden = false; this.note = ''; this._selected = false; this.locked = false;
    this._effects = [];
  }
  get opacity() { return this._opacity; }
  set opacity(v) { needNum(v, 'opacity'); if (v < 0 || v > 100) fail(`opacity ${v} outside 0..100`); this._opacity = v; }
  get selected() { return this._selected; }
  set selected(v) { needBool(v, 'selected'); this._selected = v; }
  get parent() { return this._parent._self; }
  get geometricBounds() { const b = this._bounds(); if (!b) fail('geometricBounds of empty item'); const p = fromInt([b[0], b[1]]), q = fromInt([b[2], b[3]]); return [p[0], p[1], q[0], q[1]]; }
  get position() { const g = this.geometricBounds; return [g[0], g[1]]; }
  remove() { const L = this._parent._items; const i = L.indexOf(this._self); if (i < 0) fail('remove: not in parent'); L.splice(i, 1); this._removed = true; rec('remove', { id: this._id }); }
  applyEffect(xml) {
    if (typeof xml !== 'string' || !/^<LiveEffect name="[^"]+">[\s\S]*<\/LiveEffect>$/.test(xml)) fail(`applyEffect: LiveEffect XML expected, got ${String(xml).slice(0, 60)}`);
    this._effects.push(xml); rec('applyEffect', { id: this._id, xml });
  }
  _xform(m, geometry, gradients) {
    if (geometry) this._eachPath((p) => { for (const q of p._pts) { q.a = apply(m, q.a); q.l = apply(m, q.l); q.r = apply(m, q.r); } });
    if (gradients) this._eachPath((p) => { if (p._fill && p._fill.t === 'gradient') p._fill.m = mmul(m, p._fill.m); });
  }
  _pivot(about_) {
    needEnum(about_, Transformation, 'transform pivot');
    if (about_ !== Transformation.CENTER) fail('mock implements Transformation.CENTER only');
    const b = this._bounds(); return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
  }
  rotate(angle, changePositions, changeFillPatterns, changeFillGradients, changeStrokePattern, rotateAbout) {
    needNum(angle, 'rotate angle'); [changePositions, changeFillPatterns, changeFillGradients, changeStrokePattern].forEach((b, i) => needBool(b, `rotate arg ${i + 2}`));
    const a = angle * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
    const m = about([c, s, -s, c, 0, 0], this._pivot(rotateAbout === undefined ? Transformation.CENTER : rotateAbout));
    this._xform(m, changePositions, changeFillGradients);
    rec('rotate', { id: this._id, angle, changePositions, changeFillGradients });
  }
  resize(sx, sy, changePositions, changeFillPatterns, changeFillGradients, changeStrokePattern, changeLineWidths, scaleAbout) {
    needNum(sx, 'resize x'); needNum(sy, 'resize y'); [changePositions, changeFillPatterns, changeFillGradients, changeStrokePattern].forEach((b, i) => needBool(b, `resize arg ${i + 3}`));
    const m = about([sx / 100, 0, 0, sy / 100, 0, 0], this._pivot(scaleAbout === undefined ? Transformation.CENTER : scaleAbout));
    this._xform(m, changePositions, changeFillGradients);
    rec('resize', { id: this._id, sx, sy, changePositions, changeFillGradients });
  }
  translate(dx, dy, transformObjects, transformFillPatterns, transformFillGradients, transformStrokePatterns) {
    needNum(dx, 'translate dx'); needNum(dy, 'translate dy'); [transformObjects, transformFillPatterns, transformFillGradients, transformStrokePatterns].forEach((b, i) => needBool(b, `translate arg ${i + 3}`));
    this._xform([1, 0, 0, 1, dx, dy], transformObjects, transformFillGradients);
    rec('translate', { id: this._id, dx, dy, transformObjects, transformFillGradients });
  }
}

class PathItem extends Item {
  constructor(doc, parent) {
    super(doc, parent, 'PathItem');
    this._pts = []; this._closed = false; this.filled = true; this.stroked = true; this._fill = { t: 'none' };
    this.evenodd = false; this.clipping = false; this.pixelAligned = true; this.guides = false; this._live = null;
  }
  get closed() { return this._closed; }
  set closed(v) { needBool(v, 'closed'); this._closed = v; }
  setEntirePath(arr) {
    if (!Array.isArray(arr) || arr.length < 1) fail('setEntirePath: array of points expected');
    this._pts = arr.map((p, i) => { const q = toInt(p, `setEntirePath[${i}]`); return { a: q, l: q.slice(), r: q.slice(), t: PointType.CORNER }; });
    rec('setEntirePath', { id: this._id, n: arr.length });
  }
  get pathPoints() {
    const self = this;
    return collection('PathPoints', () => self._pts.map((_, i) => pathPoint(self, i)), {
      add() { self._pts.push({ a: [0, 0], l: [0, 0], r: [0, 0], t: PointType.CORNER }); return pathPoint(self, self._pts.length - 1); },
    });
  }
  get fillColor() {
    const f = this._fill;
    if (f.t === 'spot') { const c = SpotColor(); c.spot = this._doc._spots.find((s) => s._id === f.spot); c.tint = f.tint; return c; }
    if (f.t === 'rgb') { const c = RGBColor(); c.red = f.rgb[0]; c.green = f.rgb[1]; c.blue = f.rgb[2]; return c; }
    if (f.t === 'gradient') {
      const c = GradientColor(), m = f.m;
      c.gradient = this._doc._gradients.find((g) => g._id === f.gradient);
      c.origin = fromInt([m[4], m[5]]);
      c.angle = Math.atan2(m[1], m[0]) * 180 / Math.PI;
      c.length = Math.hypot(m[0], m[1]);
      c.matrix = m.slice();
      return c;
    }
    return NoColor();
  }
  set fillColor(c) {
    const f = snapColor(c, `fillColor of "${this.name}"`);
    if (f.t === 'gradient') {
      // Illustrator fits the gradient to the object (the compound path when inside one); the angle is kept
      const b = (this._parent._cls === 'CompoundPathItem' ? this._parent : this)._bounds();
      const cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2, w = b[2] - b[0], h = b[1] - b[3];
      const a = (f.angle || 0) * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a);
      const gt = this._doc._gradients.find((g) => g._id === f.gradient);
      let L, o;
      if (gt._type === GradientType.RADIAL) { L = Math.max(w, h) / 2; o = [cx, cy]; } else {
        L = w * Math.abs(ca) + h * Math.abs(sa); o = [cx - ca * L / 2, cy - sa * L / 2];
      }
      f.m = [L * ca, L * sa, -L * sa, L * ca, o[0], o[1]];
    }
    this._fill = f;
    rec('fillColor', { id: this._id, fill: f.t });
  }
  _bounds() { return this._pts.length ? ptsBounds(this._pts, this._closed) : null; }
  _eachPath(fn) { fn(this); }
}
function pathPoint(item, i) {
  const o = {
    typename: 'PathPoint',
    get anchor() { return fromInt(item._pts[i].a); },
    set anchor(v) { item._pts[i].a = toInt(v, 'anchor'); },
    get leftDirection() { return fromInt(item._pts[i].l); },
    set leftDirection(v) { item._pts[i].l = toInt(v, 'leftDirection'); },
    get rightDirection() { return fromInt(item._pts[i].r); },
    set rightDirection(v) { item._pts[i].r = toInt(v, 'rightDirection'); },
    get pointType() { return item._pts[i].t; },
    set pointType(v) { item._pts[i].t = needEnum(v, PointType, 'pointType'); },
    get selected() { return false; },
    set selected(v) {},
  };
  return strict(o, 'PathPoint');
}

function itemsHost(self, doc, kinds) {      // pathItems / compoundPathItems / groupItems / pageItems views
  const out = {};
  const front = (it) => { self._items.unshift(it); return it; };
  out.pathItems = collection('PathItems', () => self._items.filter((x) => x._cls === 'PathItem'), {
    add() { rec('pathItems.add', { parent: self._id }); return front(strict(new PathItem(doc, self), 'PathItem')); },
    ellipse(top, left, width, height, reversed, inscribed) {
      [top, left, width, height].forEach((v, i) => needNum(v, `ellipse arg ${i + 1}`));
      if (width <= 0 || height <= 0) fail('ellipse: width/height must be > 0');
      const p = front(strict(new PathItem(doc, self), 'PathItem')), tl = toInt([left, top], 'ellipse');
      const rx = width / 2, ry = height / 2, cx = tl[0] + rx, cy = tl[1] - ry, k = 0.5522847498307936;
      const P = (x, y, lx, ly, rx_, ry_) => ({ a: [x, y], l: [lx, ly], r: [rx_, ry_], t: PointType.SMOOTH });
      // anticlockwise in y-up from the left point (Illustrator's default, not reversed)
      p._pts = [P(cx - rx, cy, cx - rx, cy + k * ry, cx - rx, cy - k * ry), P(cx, cy - ry, cx - k * rx, cy - ry, cx + k * rx, cy - ry),
        P(cx + rx, cy, cx + rx, cy - k * ry, cx + rx, cy + k * ry), P(cx, cy + ry, cx + k * rx, cy + ry, cx - k * rx, cy + ry)];
      if (reversed) p._pts.reverse().forEach((q) => { const t = q.l; q.l = q.r; q.r = t; });
      p._closed = true; p._live = { type: 'ellipse', left, top, width, height };
      rec('pathItems.ellipse', { id: p._id, top, left, width, height });
      return p;
    },
    rectangle(top, left, width, height, reversed) {
      [top, left, width, height].forEach((v, i) => needNum(v, `rectangle arg ${i + 1}`));
      const p = front(strict(new PathItem(doc, self), 'PathItem')), tl = toInt([left, top], 'rectangle');
      const C = (x, y) => ({ a: [x, y], l: [x, y], r: [x, y], t: PointType.CORNER });
      p._pts = [C(tl[0], tl[1]), C(tl[0], tl[1] - height), C(tl[0] + width, tl[1] - height), C(tl[0] + width, tl[1])];
      p._closed = true; p._live = { type: 'rect', left, top, width, height };
      rec('pathItems.rectangle', { id: p._id, top, left, width, height });
      return p;
    },
    roundedRectangle(top, left, width, height, hr, vr, reversed) {
      [top, left, width, height, hr, vr].forEach((v, i) => needNum(v, `roundedRectangle arg ${i + 1}`));
      const p = front(strict(new PathItem(doc, self), 'PathItem')), tl = toInt([left, top], 'roundedRectangle');
      const k = 0.5522847498307936, L = tl[0], T = tl[1], R = L + width, B = T - height;
      const P = (x, y, lx, ly, rx_, ry_) => ({ a: [x, y], l: [lx, ly], r: [rx_, ry_], t: PointType.SMOOTH });
      p._pts = [
        P(L, T - vr, L, T - vr + k * vr, L, T - vr), P(L, B + vr, L, B + vr, L, B + vr - k * vr),
        P(L + hr, B, L + hr - k * hr, B, L + hr, B), P(R - hr, B, R - hr, B, R - hr + k * hr, B),
        P(R, B + vr, R, B + vr - k * vr, R, B + vr), P(R, T - vr, R, T - vr, R, T - vr + k * vr),
        P(R - hr, T, R - hr + k * hr, T, R - hr, T), P(L + hr, T, L + hr, T, L + hr - k * hr, T)];
      p._closed = true; p._live = { type: 'roundrect', left, top, width, height, hr, vr };
      rec('pathItems.roundedRectangle', { id: p._id, top, left, width, height, hr, vr });
      return p;
    },
    getByName: getByName(() => self._items.filter((x) => x._cls === 'PathItem'), 'PathItems'),
  });
  if (kinds.includes('compound')) {
    out.compoundPathItems = collection('CompoundPathItems', () => self._items.filter((x) => x._cls === 'CompoundPathItem'), {
      add() { rec('compoundPathItems.add', { parent: self._id }); return front(strict(new CompoundPathItem(doc, self), 'CompoundPathItem')); },
    });
  }
  if (kinds.includes('group')) {
    out.groupItems = collection('GroupItems', () => self._items.filter((x) => x._cls === 'GroupItem'), {
      add() { rec('groupItems.add', { parent: self._id }); return front(strict(new GroupItem(doc, self), 'GroupItem')); },
      getByName: getByName(() => self._items.filter((x) => x._cls === 'GroupItem'), 'GroupItems'),
    });
  }
  out.pageItems = collection('PageItems', () => self._items.slice(), {});
  return out;
}

class CompoundPathItem extends Item {
  constructor(doc, parent) { super(doc, parent, 'CompoundPathItem'); this._items = []; this._h = itemsHost(this, doc, []); }
  get pathItems() { return this._h.pathItems; }
  get pageItems() { return this._h.pageItems; }
  _bounds() { return this._items.reduce((b, p) => unionB(b, p._bounds()), null); }
  _eachPath(fn) { this._items.forEach(fn); }
}
class GroupItem extends Item {
  constructor(doc, parent) { super(doc, parent, 'GroupItem'); this._items = []; this._clipped = false; this._h = itemsHost(this, doc, ['compound', 'group']); this._opacityMask = null; }
  get pathItems() { return this._h.pathItems; }
  get compoundPathItems() { return this._h.compoundPathItems; }
  get groupItems() { return this._h.groupItems; }
  get pageItems() { return this._h.pageItems; }
  get clipped() { return this._clipped; }
  set clipped(v) {
    needBool(v, 'clipped');
    if (v) {           // Illustrator: the topmost item must be a clipping path
      const top = this._items[0];
      const ok = top && ((top._cls === 'PathItem' && top.clipping) || (top._cls === 'CompoundPathItem' && top._items.length && top._items[0].clipping));
      if (!ok) fail(`group "${this.name}": clipped = true but the topmost item is not a clipping path`);
    }
    this._clipped = v;
  }
  _bounds() { const L = this._clipped ? [this._items[0]] : this._items; return L.reduce((b, p) => unionB(b, p._bounds()), null); }
  _eachPath(fn) { for (const it of this._items) it._eachPath(fn); }
}

class Layer {
  constructor(doc, parent) {
    this._doc = doc; this._parent = parent; this.typename = 'Layer'; this.name = ''; this.visible = true; this.locked = false;
    this._sub = []; this._items = []; this._opacity = 100; this._h = itemsHost(this, doc, ['compound', 'group']);
  }
  get opacity() { return this._opacity; }
  set opacity(v) { needNum(v, 'layer opacity'); this._opacity = v; }
  get layers() {
    const self = this;
    return collection('Layers', () => self._sub.slice(), {
      add() { const L = strict(new Layer(self._doc, self), 'Layer'); self._sub.unshift(L); rec('layers.add', { parent: self._id }); return L; },
      getByName: getByName(() => self._sub, 'Layers'),
    });
  }
  get pathItems() { return this._h.pathItems; }
  get compoundPathItems() { return this._h.compoundPathItems; }
  get groupItems() { return this._h.groupItems; }
  get pageItems() { return this._h.pageItems; }
  remove() {
    const L = this._parent._sub; const i = L.indexOf(this._self);
    if (this._parent._cls === 'Document' && L.length < 2) fail('cannot remove the last layer');
    L.splice(i, 1); rec('layer.remove', { id: this._id, name: this.name });
  }
}

// ------------------------------------------------------------------ swatches / gradients
class Spot {
  constructor(doc) { this._doc = doc; this.typename = 'Spot'; this.name = ''; this._colorType = ColorModel.SPOT; this._color = null; this.tint = 100; }
  get colorType() { return this._colorType; }
  set colorType(v) { this._colorType = needEnum(v, ColorModel, 'Spot.colorType'); }
  get color() { return this._color; }
  set color(c) { this._color = snapColor(c, `Spot "${this.name}".color`); if (this._color.t === 'spot' || this._color.t === 'gradient') fail('Spot.color must be a process colour'); }
}
class Swatch {
  constructor(doc, ref) { this._doc = doc; this.typename = 'Swatch'; this._ref = ref; }
  get name() { return this._ref.name; }
  set name(v) { this._ref.name = v; }
  get color() { return this._ref._cls === 'Spot' ? 'SpotColor' : 'GradientColor'; }
}
class SwatchGroup {
  constructor(doc) { this._doc = doc; this.typename = 'SwatchGroup'; this.name = ''; this._members = []; }
  addSpot(s) { if (!s || s._cls !== 'Spot') fail('addSpot: Spot expected'); this._members.push(s._id); rec('swatchGroup.addSpot', { group: this.name, spot: s.name }); }
  addSwatch(sw) { if (!sw || sw._cls !== 'Swatch') fail('addSwatch: Swatch expected'); this._members.push(sw._ref._id); rec('swatchGroup.addSwatch', { group: this.name, swatch: sw.name }); }
  getAllSwatches() { return this._members.map((id) => this._doc._swatchFor(id)); }
}
class GradientStop {
  constructor(g, ramp) { this._g = g; this.typename = 'GradientStop'; this._ramp = ramp; this._mid = 50; this._color = null; this._opacity = 100; }
  get rampPoint() { return this._ramp; }
  set rampPoint(v) { needNum(v, 'rampPoint'); if (v < 0 || v > 100) fail(`rampPoint ${v} outside 0..100`); this._ramp = v; this._g._sort(); }
  get midPoint() { return this._mid; }
  set midPoint(v) { needNum(v, 'midPoint'); if (v < 13 || v > 87) fail(`midPoint ${v} outside 13..87`); this._mid = v; }
  get color() { return this._color; }
  set color(c) { this._color = snapColor(c, 'GradientStop.color'); }
  get opacity() { return this._opacity; }
  set opacity(v) { needNum(v, 'stop opacity'); if (v < 0 || v > 100) fail(`stop opacity ${v} outside 0..100`); this._opacity = v; }
  remove() { const L = this._g._stops; if (L.length <= 2) fail('a gradient needs at least two stops'); L.splice(L.indexOf(this._self), 1); }
}
class Gradient {
  constructor(doc) {
    this._doc = doc; this.typename = 'Gradient'; this.name = ''; this._type = GradientType.LINEAR;
    this._stops = []; this._stops.push(strict(new GradientStop(this, 0), 'GradientStop')); this._stops.push(strict(new GradientStop(this, 100), 'GradientStop'));
    const k = RGBColor(); this._stops[1]._color = snapColor(k, 'init'); this._stops[0]._color = { t: 'rgb', rgb: [255, 255, 255] };
  }
  get type() { return this._type; }
  set type(v) { this._type = needEnum(v, GradientType, 'Gradient.type'); }
  _sort() { this._stops.sort((a, b) => a._ramp - b._ramp); }
  get gradientStops() {
    const self = this;
    return collection('GradientStops', () => self._stops.slice(), {
      add() { const s = strict(new GradientStop(self, 50), 'GradientStop'); s._color = { t: 'rgb', rgb: [128, 128, 128] }; self._stops.push(s); self._sort(); return s; },
    });
  }
}

// ------------------------------------------------------------------ artboards / document
class Artboard {
  constructor(doc, rect) { this._doc = doc; this.typename = 'Artboard'; this.name = ''; this._rect = rect; }
  get artboardRect() { const p = fromInt([this._rect[0], this._rect[1]]), q = fromInt([this._rect[2], this._rect[3]]); return [p[0], p[1], q[0], q[1]]; }
  set artboardRect(v) { this._rect = this._doc._checkRect(v); rec('artboardRect', { name: this.name, rect: this._rect }); }
}

class Document {
  constructor(space, w, h) {
    this.typename = 'Document'; this.name = 'Untitled-1'; this._space = space;
    this._center = [5000.25, -3000.75];                 // canvas centre (arbitrary private frame)
    const L = this._center[0] - w / 2, T = this._center[1] + h / 2;
    this._artboards = [new Artboard(this, [L, T, L + w, T - h])];
    this._activeAB = 0;
    this._ruler = [L, T];                              // initial ruler origin: artboard 1 top-left
    this._sub = []; this._spots = []; this._gradients = []; this._groups = []; this._saved = []; this._exports = [];
    this._sel = [];
    this._dfltSwatches = ['[None]', '[Registration]', 'White', 'Black'];
  }
  _init() { const l = strict(new Layer(this, this), 'Layer'); l.name = 'Layer 1'; this._sub.push(l); this._artboards[0] = strict(this._artboards[0], 'Artboard'); this._artboards[0].name = 'Artboard 1'; }
  _checkRect(v) {
    if (!Array.isArray(v) || v.length !== 4 || !v.every(isNum)) fail(`artboardRect: [l, t, r, b] expected, got ${JSON.stringify(v)}`);
    if (!(v[2] > v[0])) fail(`artboardRect: right must be > left: ${JSON.stringify(v)}`);
    if (!(v[1] > v[3])) fail(`artboardRect: top must be > bottom (y up): ${JSON.stringify(v)}`);
    const p = toInt([v[0], v[1]], 'artboardRect'), q = toInt([v[2], v[3]], 'artboardRect');
    for (const z of [p, q]) {
      if (Math.abs(z[0] - this._center[0]) > CANVAS_HALF || Math.abs(z[1] - this._center[1]) > CANVAS_HALF) fail(`artboard ${JSON.stringify(v)} is outside the canvas`);
    }
    return [p[0], p[1], q[0], q[1]];
  }
  // Modelled as a point of the private canvas frame (get and set alike): rulerOrigin = [0, 0] moves the
  // document origin far away from the artboards, so only artboardRect-relative placement survives.
  get rulerOrigin() { return this._ruler.slice(); }
  set rulerOrigin(v) { needPt(v, 'rulerOrigin'); this._ruler = [v[0], v[1]]; rec('rulerOrigin', { v }); }
  get artboards() {
    const self = this;
    return collection('Artboards', () => self._artboards.slice(), {
      add(rect) { const a = strict(new Artboard(self, self._checkRect(rect)), 'Artboard'); a.name = `Artboard ${self._artboards.length + 1}`; self._artboards.push(a); self._activeAB = self._artboards.length - 1; rec('artboards.add', { rect: a._rect }); return a; },
      setActiveArtboardIndex(i) { if (!(i >= 0 && i < self._artboards.length)) fail('setActiveArtboardIndex: bad index'); self._activeAB = i; },
      getActiveArtboardIndex() { return self._activeAB; },
      getByName: getByName(() => self._artboards, 'Artboards'),
    });
  }
  get layers() {
    const self = this;
    return collection('Layers', () => self._sub.slice(), {
      add() { const L = strict(new Layer(self, self), 'Layer'); L.name = `Layer ${self._sub.length + 1}`; self._sub.unshift(L); rec('layers.add', { parent: 'doc' }); return L; },
      getByName: getByName(() => self._sub, 'Layers'),
    });
  }
  get spots() {
    const self = this;
    return collection('Spots', () => self._spots.slice(), {
      add() { const s = strict(new Spot(self), 'Spot'); self._spots.push(s); rec('spots.add', {}); return s; },
      getByName: getByName(() => self._spots, 'Spots'),
    });
  }
  _swatchFor(id) {
    const ref = this._spots.find((s) => s._id === id) || this._gradients.find((g) => g._id === id);
    return strict(new Swatch(this, ref), 'Swatch');
  }
  get swatches() {
    const self = this;
    const list = () => self._spots.concat(self._gradients).map((r) => self._swatchFor(r._id));
    return collection('Swatches', list, {
      getByName(n) { const r = self._spots.concat(self._gradients).find((x) => x.name === n); if (!r) fail(`swatches.getByName("${n}"): no such swatch`); return self._swatchFor(r._id); },
    });
  }
  get swatchGroups() {
    const self = this;
    return collection('SwatchGroups', () => self._groups.slice(), {
      add() { const g = strict(new SwatchGroup(self), 'SwatchGroup'); self._groups.push(g); rec('swatchGroups.add', {}); return g; },
    });
  }
  get gradients() {
    const self = this;
    return collection('Gradients', () => self._gradients.slice(), {
      add() { const g = strict(new Gradient(self), 'Gradient'); g.name = `New Gradient Swatch ${self._gradients.length + 1}`; self._gradients.push(g); rec('gradients.add', {}); return g; },
      getByName: getByName(() => self._gradients, 'Gradients'),
    });
  }
  get selection() { const out = []; walkItems(this, (it) => { if (it._selected) out.push(it._self); }); return out; }
  set selection(v) {
    if (v !== null && !(Array.isArray(v) && v.length === 0)) fail('mock supports selection = null only');
    walkItems(this, (it) => { it._selected = false; });
  }
  saveAs(file, opts) {
    if (!file || file._cls !== 'File') fail('saveAs: File expected');
    if (!opts || !['IllustratorSaveOptions', 'PDFSaveOptions'].includes(opts._cls)) fail('saveAs: IllustratorSaveOptions or PDFSaveOptions expected');
    const ext = file.fsName.split('.').pop().toLowerCase();
    if ((opts._cls === 'IllustratorSaveOptions') !== (ext === 'ai')) fail(`saveAs: ${opts._cls} with a .${ext} file`);
    const o = plain(opts);
    this._saved.push({ path: file.fsName, options: opts._cls, opts: o });
    this.name = path.basename(file.fsName);
    rec('saveAs', { path: file.fsName, options: opts._cls });
  }
  exportFile(file, type, opts) {
    if (!file || file._cls !== 'File') fail('exportFile: File expected');
    needEnum(type, ExportType, 'exportFile type');
    if (type === ExportType.SVG && (!opts || opts._cls !== 'ExportOptionsSVG')) fail('exportFile(SVG): ExportOptionsSVG expected');
    this._exports.push({ path: file.fsName, type, opts: plain(opts) });
    rec('exportFile', { path: file.fsName, type });
  }
}
function walkItems(doc, fn) {
  const visit = (c) => { for (const it of c._items || []) { fn(it); if (it._items) visit(it); } for (const l of c._sub || []) visit(l); };
  visit(doc);
}
function plain(o) { const out = {}; for (const k of Object.keys(o)) if (k[0] !== '_') out[k] = o[k]; return out; }

function options(cls, props) { return function () { return strict(Object.assign({ typename: cls }, props), cls); }; }
const IllustratorSaveOptions = options('IllustratorSaveOptions', { pdfCompatible: true, embedICCProfile: false, compressed: true, embedLinkedFiles: false, fontSubsetThreshold: 100, saveMultipleArtboards: false, artboardRange: '', compatibility: null });
const PDFSaveOptions = options('PDFSaveOptions', { compatibility: null, preserveEditability: false, generateThumbnails: false, viewAfterSaving: false, acrobatLayers: false, pDFPreset: '', artboardRange: '' });
const ExportOptionsSVG = options('ExportOptionsSVG', { embedRasterImages: false, cssProperties: null, coordinatePrecision: 3, documentEncoding: null, saveMultipleArtboards: false, artboardRange: '', fontSubsetting: null, embedAllFonts: false, preserveEditability: false, includeFileInfo: false });

// ------------------------------------------------------------------ files
const files = {};
function File(p) {
  if (typeof p !== 'string') fail('File(path): string expected');
  const o = {
    typename: 'File', fsName: p, name: path.basename(p), encoding: 'ASCII', lineFeed: 'Macintosh', _mode: null,
    get parent() { return Folder(path.dirname(p)); },
    get exists() { return p === SCRIPT_PATH || p in files; },
    open(mode) { this._mode = mode; if (mode === 'w') files[p] = ''; return true; },
    write(s) { if (this._mode !== 'w') fail('File.write: not open for writing'); files[p] += String(s); return true; },
    writeln(s) { return this.write(String(s) + '\n'); },
    close() { this._mode = null; return true; },
  };
  return strict(o, 'File');
}
function Folder(p) { return strict({ typename: 'Folder', fsName: p, name: path.basename(p), exists: true }, 'Folder'); }
Object.defineProperty(Folder, 'desktop', { get: () => Folder('/Users/client/Desktop') });

// ------------------------------------------------------------------ app
const alerts = [];
const appObj = {
  typename: 'Application', version: '28.0.0',
  get coordinateSystem() { return coordSys; },
  set coordinateSystem(v) { coordSys = needEnum(v, CoordinateSystem, 'coordinateSystem'); rec('coordinateSystem', { v }); },
  _uil: UserInteractionLevel.DISPLAYALERTS,
  get userInteractionLevel() { return this._uil; },
  set userInteractionLevel(v) { this._uil = needEnum(v, UserInteractionLevel, 'userInteractionLevel'); },
  get activeDocument() { if (!activeDoc) fail('no active document'); return activeDoc._self; },
  documents: null,
  executeMenuCommand(cmd) {
    rec('executeMenuCommand', { cmd });
    if (cmd === 'fitall') return;
    if (cmd === 'makeMask') {        // Transparency > Make Opacity Mask: topmost selected = mask art
      const sel = activeDoc.selection.map((x) => x);
      if (sel.length !== 2) fail(`makeMask: two selected objects expected, got ${sel.length}`);
      if (sel[0]._parent !== sel[1]._parent) fail('makeMask: selection must share a parent');
      const L = sel[0]._parent._items;
      sel.sort((a, b) => L.indexOf(a) - L.indexOf(b));
      const [mask, art] = sel;
      L.splice(L.indexOf(mask), 1);
      art._opacityMask = mask;
      return;
    }
    fail(`executeMenuCommand("${cmd}") not modelled`);
  },
};
appObj.documents = collection('Documents', () => (activeDoc ? [activeDoc._self] : []), {
  add(space, w, h) {
    needEnum(space, DocumentColorSpace, 'documents.add colour space'); needNum(w, 'documents.add width'); needNum(h, 'documents.add height');
    const d = strict(new Document(space, w, h), 'Document'); activeDoc = d; d._init();
    rec('documents.add', { space, w, h }); return d;
  },
});
const appProxy = strict(appObj, 'Application');

// ------------------------------------------------------------------ run
const ctx = {
  app: appProxy, $: { fileName: SCRIPT_PATH, writeln() {} }, alert(m) { alerts.push(String(m)); rec('alert', {}); },
  File, Folder, RGBColor, CMYKColor, SpotColor, GradientColor, NoColor, IllustratorSaveOptions, PDFSaveOptions, ExportOptionsSVG,
  DocumentColorSpace, CoordinateSystem, ColorModel, PointType, GradientType, Transformation, UserInteractionLevel,
  PDFCompatibility, SVGCSSPropertyLocation, SVGDocumentEncoding, ExportType,
};
vm.createContext(ctx);
let crashed = null;
const src = fs.readFileSync(SCRIPT, 'utf8');
try { vm.runInContext(src, ctx, { filename: path.basename(SCRIPT), timeout: 60000 }); } catch (e) { crashed = String(e && e.stack || e); }

// ------------------------------------------------------------------ dump (canvas frame, y up)
function dumpItem(it) {
  const o = { cls: it._cls, name: it.name, id: it._id };
  if (it._opacity !== 100) o.opacity = it._opacity;
  if (it._effects.length) o.effects = it._effects;
  if (it.note) o.note = it.note;
  if (it.hidden) o.hidden = true;
  if (it._cls === 'PathItem') {
    Object.assign(o, { closed: it._closed, filled: it.filled, stroked: it.stroked, evenodd: it.evenodd, clipping: it.clipping,
      pixelAligned: it.pixelAligned, fill: it._fill, live: it._live,
      points: it._pts.map((q) => [q.a[0], q.a[1], q.l[0], q.l[1], q.r[0], q.r[1], q.t === PointType.SMOOTH ? 1 : 0]) });
  } else {
    o.items = it._items.map(dumpItem);
    if (it._cls === 'GroupItem') { o.clipped = it._clipped; if (it._opacityMask) o.opacityMask = dumpItem(it._opacityMask); }
  }
  return o;
}
function dumpLayer(l) { return { cls: 'Layer', name: l.name, id: l._id, layers: l._sub.map(dumpLayer), items: l._items.map(dumpItem) }; }
const out = { script: SCRIPT, scriptPath: SCRIPT_PATH, startCoord: START_CS, crashed, errors, alerts, files, calls, finalCoord: coordSys, document: null };
if (activeDoc) {
  const d = activeDoc;
  out.document = {
    space: d._space, name: d.name, canvasCenter: d._center, ruler: d._ruler,
    artboards: d._artboards.map((a) => ({ name: a.name, rect: a._rect })),
    layers: d._sub.map(dumpLayer),
    spots: d._spots.map((s) => ({ id: s._id, name: s.name, colorType: s._colorType, color: s._color })),
    swatchGroups: d._groups.map((g) => ({ name: g.name, members: g._members })),
    gradients: d._gradients.map((g) => ({ id: g._id, name: g.name, type: g._type,
      stops: g._stops.map((s) => ({ ramp: s._ramp, mid: s._mid, color: s._color, opacity: s._opacity })) })),
    saved: d._saved, exports: d._exports,
  };
}
fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
console.log(`ai_mock: ${calls.length} calls, ${errors.length} errors, ${alerts.length} alerts${crashed ? ', CRASHED' : ''}`);
process.exit(crashed ? 1 : 0);
