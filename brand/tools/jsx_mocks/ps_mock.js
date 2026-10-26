#!/usr/bin/env node
// Recording mock of the Adobe Photoshop ExtendScript DOM + Action Manager: exactly the members the
// scripts written by illustrator_jsx.py use, implemented STRICTLY (unknown properties, enum values,
// 3-character charIDs, non-integer putInteger values ... throw and are logged in `errors`).
//
//   node ps_mock.js SCRIPT.jsx RECORD.json [--new-layer sibling|inside] [--quirk-empty-set]
//                                          [--script-path PATH]
//
// Every executeAction call is decoded (type IDs -> names) and recorded, and the mock then does
// what Photoshop does with it, so the final layer tree, vector masks and fills can be checked.
// What the mock ASSUMES about Photoshop (the things a mock run cannot prove):
//  * make contentLayer while a path is selected -> a fill layer whose vector mask is that path;
//    with a "shape" (Elps / Rctn) in the descriptor -> a live-shape layer; with neither, a
//    full-canvas fill layer. Layer > Vector Mask > Current Path copies the selected path.
//  * PathPointInfo coordinates are POINTS (px = pt * resolution / 72).
//  * A new layer appears directly above the active layer (--new-layer sibling) or, when the
//    active layer is a group, at the top inside it (--new-layer inside); --quirk-empty-set makes
//    move(emptyGroup, PLACEATEND) fail, as reported for some versions.
//  * Sub-path operations combine sequentially (add / subtract / xor / intersect), each component
//    filled with the nonzero rule.
//  * Gradient fill layers (Align off): the ramp is centred on the canvas centre + offset (% of
//    W, H), at `angle` (counter-clockwise), and spans (W|cos a| + H|sin a|) * scale at 100 %;
//    a radial ramp's radius is half that span.
'use strict';
const fs = require('fs');
const vm = require('vm');
const path = require('path');

const argv = process.argv.slice(2);
if (argv.length < 2) { console.error('usage: ps_mock.js SCRIPT.jsx RECORD.json [--new-layer sibling|inside] [--quirk-empty-set]'); process.exit(2); }
const SCRIPT = argv[0], OUT = argv[1];
const opt = (name, dflt) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : dflt; };
const NEW_LAYER = opt('--new-layer', 'sibling');
const QUIRK_EMPTY = argv.includes('--quirk-empty-set');
const SCRIPT_PATH = opt('--script-path', '/Users/client/K Brand Kit/' + path.basename(SCRIPT));

const errors = [];
const calls = [];
function fail(msg) { errors.push(msg); throw new Error(msg); }
function rec(op, data) { calls.push(Object.assign({ op }, data || {})); }
const isNum = (v) => typeof v === 'number' && isFinite(v);
function needNum(v, what) { if (!isNum(v)) fail(`${what}: number expected, got ${JSON.stringify(v)}`); return v; }
function needPt(v, what) { if (!Array.isArray(v) || v.length !== 2 || !isNum(v[0]) || !isNum(v[1])) fail(`${what}: [x, y] expected, got ${JSON.stringify(v)}`); return [v[0], v[1]]; }
function needBool(v, what) { if (typeof v !== 'boolean') fail(`${what}: boolean expected, got ${JSON.stringify(v)}`); return v; }

let NEXT_ID = 1;
const S = (o) => (o && o._self) || o;          // canonical (proxy) identity of a mock object
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

const Units = enumType('Units', ['PIXELS', 'POINTS', 'INCHES', 'CM', 'MM', 'PERCENT', 'PICAS']);
const DialogModes = enumType('DialogModes', ['NO', 'ALL', 'ERROR']);
const NewDocumentMode = enumType('NewDocumentMode', ['RGB', 'CMYK', 'GRAYSCALE', 'LAB', 'BITMAP']);
const DocumentFill = enumType('DocumentFill', ['TRANSPARENT', 'WHITE', 'BACKGROUNDCOLOR']);
const BitsPerChannelType = enumType('BitsPerChannelType', ['ONE', 'EIGHT', 'SIXTEEN', 'THIRTYTWO']);
const QueryStateType = enumType('QueryStateType', ['ALWAYS', 'ASK', 'NEVER']);
const ShapeOperation = enumType('ShapeOperation', ['SHAPEADD', 'SHAPESUBTRACT', 'SHAPEXOR', 'SHAPEINTERSECT']);
const PointKind = enumType('PointKind', ['SMOOTHPOINT', 'CORNERPOINT']);
const ElementPlacement = enumType('ElementPlacement', ['PLACEATEND', 'PLACEATBEGINNING', 'INSIDE', 'PLACEBEFORE', 'PLACEAFTER']);
const ResampleMethod = enumType('ResampleMethod', ['NONE', 'BICUBIC', 'BILINEAR', 'NEARESTNEIGHBOR', 'BICUBICSHARPER', 'BICUBICSMOOTHER', 'AUTOMATIC', 'PRESERVEDETAILS']);
const Extension = enumType('Extension', ['LOWERCASE', 'UPPERCASE', 'NONE']);
const LayerKind = enumType('LayerKind', ['NORMAL', 'SOLIDFILL', 'GRADIENTFILL', 'SMARTOBJECT', 'TEXT']);

// ------------------------------------------------------------------ type IDs
// charIDs that have a stringID twin resolve to the same name (Photoshop maps both to one typeID)
const CHAR2STR = {
  'Mk  ': 'make', 'null': 'null', 'Usng': 'using', 'Type': 'type', 'Clr ': 'color', 'RGBC': 'RGBColor',
  'Rd  ': 'red', 'Grn ': 'grain', 'Bl  ': 'blue', 'Shp ': 'shape', 'Elps': 'ellipse', 'Rctn': 'rectangle',
  'Top ': 'top', 'Left': 'left', 'Btom': 'bottom', 'Rght': 'right', '#Pxl': 'pixelsUnit', '#Prc': 'percentUnit',
  '#Ang': 'angleUnit', 'Clrs': 'colors', 'Nm  ': 'name', 'Path': 'path', 'At  ': 'at', 'Ordn': 'ordinal',
  'Trgt': 'targetEnum', 'setd': 'set', 'Prpr': 'property', 'Lefx': 'layerEffects', 'Lyr ': 'layer', 'T   ': 'to',
  'Scl ': 'scale', 'GrdT': 'gradientType', 'Rdl ': 'radial', 'Lnr ': 'linear', 'Angl': 'angle', 'Algn': 'align',
  'Ofst': 'offset', 'Hrzn': 'horizontal', 'Vrtc': 'vertical', 'Pnt ': 'paint', 'Grad': 'gradient',
  'Grdn': 'gradientClassEvent', 'GrdF': 'gradientForm', 'CstS': 'customStops', 'Intr': 'interfaceIconFrameDimmed',
  'Clrt': 'colorStop', 'Clry': 'colorStopType', 'UsrS': 'userStop', 'Lctn': 'location', 'Mdpn': 'midpoint',
  'Trns': 'transparency', 'TrnS': 'transferSpec', 'Opct': 'opacity', 'enab': 'enabled', 'Md  ': 'mode',
  'BlnM': 'blendMode', 'Nrml': 'normal', 'uglg': 'useGlobalAngle', 'lagl': 'localLightingAngle', 'Dstn': 'distance',
  'GlwT': 'glowTechnique', 'BETE': 'bevelEmbossTechnique', 'SfBL': 'softMatte', 'Ckmt': 'chokeMatte', 'blur': 'blur',
  'Nose': 'noise', 'AntA': 'antiAlias', 'OrGl': 'outerGlow', 'DrSh': 'dropShadow', 'GsnB': 'gaussianBlur', 'Rds ': 'radius',
};
const STR2CHAR = {};
for (const [c, s] of Object.entries(CHAR2STR)) STR2CHAR[s] = c;
const id2name = new Map();
function charIDToTypeID(s) {
  if (typeof s !== 'string' || s.length !== 4) fail(`charIDToTypeID("${s}"): exactly 4 characters required`);
  const id = ((s.charCodeAt(0) << 24) | (s.charCodeAt(1) << 16) | (s.charCodeAt(2) << 8) | s.charCodeAt(3)) >>> 0;
  id2name.set(id, CHAR2STR[s] || s);
  return id;
}
let nextSid = 1000;
const sids = new Map();
function stringIDToTypeID(s) {
  if (typeof s !== 'string' || !s.length) fail('stringIDToTypeID: string expected');
  if (STR2CHAR[s]) return charIDToTypeID(STR2CHAR[s]);
  if (!sids.has(s)) { sids.set(s, nextSid); id2name.set(nextSid, s); nextSid++; }
  return sids.get(s);
}
function nm(id) { if (!id2name.has(id)) fail(`unknown typeID ${id} (not made by charIDToTypeID / stringIDToTypeID)`); return id2name.get(id); }

class ActionDescriptor {
  constructor() { this._m = new Map(); this.typename = 'ActionDescriptor'; }
  get count() { return this._m.size; }
  _put(k, v) { nm(k); this._m.set(k, v); }
  putBoolean(k, v) { this._put(k, { t: 'bool', v: needBool(v, 'putBoolean') }); }
  putDouble(k, v) { this._put(k, { t: 'double', v: needNum(v, 'putDouble') }); }
  putInteger(k, v) { needNum(v, 'putInteger'); if (Math.round(v) !== v) fail(`putInteger(${nm(k)}): integer expected, got ${v}`); this._put(k, { t: 'int', v }); }
  putString(k, v) { if (typeof v !== 'string') fail('putString: string expected'); this._put(k, { t: 'str', v }); }
  putUnitDouble(k, unit, v) { nm(unit); this._put(k, { t: 'unit', unit: nm(unit), v: needNum(v, `putUnitDouble(${nm(k)})`) }); }
  putEnumerated(k, type, v) { this._put(k, { t: 'enum', type: nm(type), v: nm(v) }); }
  putObject(k, cls, d) { if (!(d instanceof ActionDescriptor) && !(d && d._cls === 'ActionDescriptor')) fail(`putObject(${nm(k)}): ActionDescriptor expected`); this._put(k, { t: 'obj', cls: nm(cls), v: d }); }
  putReference(k, r) { if (!r || r._cls !== 'ActionReference') fail(`putReference(${nm(k)}): ActionReference expected`); this._put(k, { t: 'ref', v: r }); }
  putList(k, l) { if (!l || l._cls !== 'ActionList') fail(`putList(${nm(k)}): ActionList expected`); this._put(k, { t: 'list', v: l }); }
  putClass(k, cls) { this._put(k, { t: 'class', v: nm(cls) }); }
  hasKey(k) { return this._m.has(k); }
}
class ActionReference {
  constructor() { this._parts = []; this.typename = 'ActionReference'; }
  putClass(cls) { this._parts.push({ form: 'class', cls: nm(cls) }); }
  putEnumerated(cls, type, v) { this._parts.push({ form: 'enum', cls: nm(cls), type: nm(type), v: nm(v) }); }
  putProperty(cls, v) { this._parts.push({ form: 'prop', cls: nm(cls), v: nm(v) }); }
  putName(cls, v) { this._parts.push({ form: 'name', cls: nm(cls), v: String(v) }); }
  putIndex(cls, v) { this._parts.push({ form: 'index', cls: nm(cls), v: needNum(v, 'putIndex') }); }
  putIdentifier(cls, v) { this._parts.push({ form: 'id', cls: nm(cls), v: needNum(v, 'putIdentifier') }); }
}
class ActionList {
  constructor() { this._l = []; this.typename = 'ActionList'; }
  get count() { return this._l.length; }
  putObject(cls, d) { if (!d || d._cls !== 'ActionDescriptor') fail('ActionList.putObject: ActionDescriptor expected'); this._l.push({ t: 'obj', cls: nm(cls), v: d }); }
  putDouble(v) { this._l.push({ t: 'double', v: needNum(v, 'ActionList.putDouble') }); }
  putInteger(v) { if (Math.round(v) !== v) fail('ActionList.putInteger: integer expected'); this._l.push({ t: 'int', v }); }
  putString(v) { this._l.push({ t: 'str', v: String(v) }); }
  putUnitDouble(unit, v) { this._l.push({ t: 'unit', unit: nm(unit), v: needNum(v, 'ActionList.putUnitDouble') }); }
}
const mkAD = () => strict(new ActionDescriptor(), 'ActionDescriptor');
const mkAR = () => strict(new ActionReference(), 'ActionReference');
const mkAL = () => strict(new ActionList(), 'ActionList');
function decodeV(x) {
  if (x.t === 'obj') return { obj: x.cls, v: decode(x.v) };
  if (x.t === 'ref') return { ref: x.v._parts };
  if (x.t === 'list') return { list: x.v._l.map(decodeV) };
  if (x.t === 'unit') return { unit: x.unit, v: x.v };
  if (x.t === 'enum') return { enum: x.type, v: x.v };
  return x.v;
}
function decode(d) { const o = {}; for (const [k, v] of d._m) o[nm(k)] = decodeV(v); return o; }

// ------------------------------------------------------------------ geometry
function flatten(comp, steps = 24) {     // component (px) -> polygon
  const P = comp.points, n = P.length, out = [];
  const segs = comp.closed ? n : n - 1;
  out.push(P[0].a);
  for (let i = 0; i < segs; i++) {
    const p = P[i], q = P[(i + 1) % n];
    const line = p.r[0] === p.a[0] && p.r[1] === p.a[1] && q.l[0] === q.a[0] && q.l[1] === q.a[1];
    if (line) { out.push(q.a); continue; }
    for (let k = 1; k <= steps; k++) {
      const t = k / steps, u = 1 - t;
      out.push([u * u * u * p.a[0] + 3 * u * u * t * p.r[0] + 3 * u * t * t * q.l[0] + t * t * t * q.a[0],
        u * u * u * p.a[1] + 3 * u * u * t * p.r[1] + 3 * u * t * t * q.l[1] + t * t * t * q.a[1]]);
    }
  }
  return out;
}
function winding(poly, x, y) {
  let w = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    if (a[1] <= y) { if (b[1] > y && (b[0] - a[0]) * (y - a[1]) - (x - a[0]) * (b[1] - a[1]) > 0) w++; }
    else if (b[1] <= y && (b[0] - a[0]) * (y - a[1]) - (x - a[0]) * (b[1] - a[1]) < 0) w--;
  }
  return w;
}
function insideMask(m, x, y) {
  if (!m) return true;
  if (m.shape) {
    const s = m.shape;
    if (s.type === 'ellipse') { const cx = (s.left + s.right) / 2, cy = (s.top + s.bottom) / 2, rx = (s.right - s.left) / 2, ry = (s.bottom - s.top) / 2; return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1; }
    if (x < s.left || x > s.right || y < s.top || y > s.bottom) return false;
    const r = s.radius || 0;
    const cx = Math.min(Math.max(x, s.left + r), s.right - r), cy = Math.min(Math.max(y, s.top + r), s.bottom - r);
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r + 1e-9;
  }
  let inside = false;
  m.components.forEach((c, i) => {
    const w = winding(c._poly || (c._poly = flatten(c)), x, y) !== 0;
    if (i === 0 && c.op !== 'SHAPESUBTRACT' && c.op !== 'SHAPEINTERSECT') { inside = w; return; }
    if (c.op === 'SHAPEADD') inside = inside || w;
    else if (c.op === 'SHAPESUBTRACT') inside = inside && !w;
    else if (c.op === 'SHAPEXOR') inside = inside !== w;
    else inside = inside && w;
  });
  return inside;
}
function gradientAt(doc, g, x, y) {
  const W = doc._w, H = doc._h, a = g.angle * Math.PI / 180;
  const span = (W * Math.abs(Math.cos(a)) + H * Math.abs(Math.sin(a))) * g.scale / 100;
  const mx = W / 2 + g.ox / 100 * W, my = H / 2 + g.oy / 100 * H;
  let t;
  if (g.type === 'linear') t = ((x - mx) * Math.cos(a) - (y - my) * Math.sin(a)) / span + 0.5;
  else t = Math.hypot(x - mx, y - my) / (span / 2);
  t = Math.min(1, Math.max(0, t));
  const lerp = (stops, f) => {
    if (t <= stops[0].loc) return f(stops[0]);
    for (let i = 1; i < stops.length; i++) if (t <= stops[i].loc) {
      const s0 = stops[i - 1], s1 = stops[i], u = s1.loc > s0.loc ? (t - s0.loc) / (s1.loc - s0.loc) : 1;
      const A = f(s0), B = f(s1); return A.map((v, k) => v + (B[k] - v) * u);
    }
    return f(stops[stops.length - 1]);
  };
  const rgb = lerp(g.colors, (s) => s.rgb), op = lerp(g.alphas, (s) => [s.op])[0];
  return [rgb[0], rgb[1], rgb[2], op / 100];
}
function composite(doc, layers, x, y) {     // premultiplied [r, g, b, a], layers top -> bottom
  let acc = [0, 0, 0, 0];
  for (let i = layers.length - 1; i >= 0; i--) {
    const L = layers[i];
    if (!L.visible) continue;
    let src = [0, 0, 0, 0];
    if (L._cls === 'LayerSet') src = composite(doc, L._layers, x, y);
    else if (L._fill) {
      const c = L._fill.t === 'solid' ? [L._fill.rgb[0], L._fill.rgb[1], L._fill.rgb[2], 1] : gradientAt(doc, L._fill, x, y);
      const a = c[3] * L.fillOpacity / 100;
      src = [c[0] * a, c[1] * a, c[2] * a, a];
    }
    const k = (L.opacity / 100) * (insideMask(L._vmask, x, y) ? 1 : 0);
    src = src.map((v) => v * k);
    acc = acc.map((v, j) => src[j] + v * (1 - src[3]));
  }
  return acc;
}

// ------------------------------------------------------------------ layers
class LayerBase {
  constructor(doc, cls) { this._doc = S(doc); this.typename = cls; this.name = ''; this._opacity = 100; this.visible = true; this._vmask = null; this._parent = null; this.allLocked = false; }
  get opacity() { return this._opacity; }
  set opacity(v) { needNum(v, 'opacity'); if (v < 0 || v > 100) fail(`opacity ${v} outside 0..100`); this._opacity = v; }
  get parent() { return this._parent._self; }
  move(rel, where) {
    needEnum(where, ElementPlacement, 'move placement');
    if (!rel || !['Document', 'LayerSet', 'ArtLayer'].includes(rel._cls)) fail('move: layer, group or document expected');
    const me = this._self, R = rel;
    if (S(R) === me) fail('move: relative to itself');
    for (let p = R._parent; p && p._cls === 'LayerSet'; p = p._parent) if (S(p) === me) fail('move: cannot move a group into itself');
    let dest, idx;
    if (where === ElementPlacement.PLACEATEND || where === ElementPlacement.PLACEATBEGINNING || where === ElementPlacement.INSIDE) {
      if (R._cls === 'ArtLayer') fail(`move(${where}): relative object must be a group or the document`);
      if (QUIRK_EMPTY && where === ElementPlacement.PLACEATEND && R._cls === 'LayerSet' && R._layers.filter((x) => x !== me).length === 0) fail('General Photoshop error occurred. (move PLACEATEND into an empty group)');
      dest = R; idx = where === ElementPlacement.PLACEATEND ? -1 : 0;
    } else {
      if (R._cls === 'Document') fail(`move(${where}): relative object must be a layer`);
      dest = R._parent; idx = null;
    }
    detach(me);
    if (idx === null) { const j = dest._layers.indexOf(R._self || R); dest._layers.splice(where === ElementPlacement.PLACEBEFORE ? j : j + 1, 0, me); }
    else if (idx === -1) dest._layers.push(me); else dest._layers.unshift(me);
    me._parent = dest._cls === 'Document' ? dest : dest;
    rec('move', { layer: this.name, rel: rel.name, where });
  }
  remove() {
    const me = this._self; detach(me);
    if (S(this._doc._active) === me) this._doc._active = this._doc._layers[0] || null;
    rec('remove', { layer: this.name });
  }
}
function detach(me) { const L = me._parent._layers; const i = L.indexOf(me); if (i < 0) fail('layer not found in its parent'); L.splice(i, 1); }
class ArtLayer extends LayerBase {
  constructor(doc) { super(doc, 'ArtLayer'); this._fillOpacity = 100; this._fill = null; this._styles = null; this._smart = false; this._filters = []; this._kind = LayerKind.NORMAL; }
  get fillOpacity() { return this._fillOpacity; }
  set fillOpacity(v) { needNum(v, 'fillOpacity'); if (v < 0 || v > 100) fail(`fillOpacity ${v} outside 0..100`); this._fillOpacity = v; }
  get kind() { return this._kind; }
}
class LayerSet extends LayerBase {
  constructor(doc) { super(doc, 'LayerSet'); this._layers = []; }
  get layers() { const s = this; return collection('Layers', () => s._layers.slice(), {}); }
  get artLayers() { const s = this; return collection('ArtLayers', () => s._layers.filter((x) => x._cls === 'ArtLayer'), {}); }
  get layerSets() {
    const s = this;
    return collection('LayerSets', () => s._layers.filter((x) => x._cls === 'LayerSet'), {
      add() { const g = newLayer(s._doc, 'LayerSet'); g._parent = s; s._layers.unshift(g); g.name = `Group ${++s._doc._groupNo}`; s._doc._active = g; rec('layerSets.add', { parent: s.name }); return g; },
    });
  }
}
function newLayer(doc, cls) { return strict(cls === 'LayerSet' ? new LayerSet(doc) : new ArtLayer(doc), cls); }
function insertAboveActive(doc, L) {
  const A = doc._active;
  if (!A) { L._parent = doc; doc._layers.unshift(L); return; }
  if (NEW_LAYER === 'inside' && A._cls === 'LayerSet') { L._parent = A; A._layers.unshift(L); return; }
  const P = A._parent, i = P._layers.indexOf(A);
  L._parent = P; P._layers.splice(i, 0, L);
}

// ------------------------------------------------------------------ paths
function PathPointInfo() { return strict({ typename: 'PathPointInfo', kind: PointKind.CORNERPOINT, anchor: null, leftDirection: null, rightDirection: null }, 'PathPointInfo'); }
function SubPathInfo() { return strict({ typename: 'SubPathInfo', closed: false, operation: ShapeOperation.SHAPEADD, entireSubPath: null }, 'SubPathInfo'); }
function makePathItem(doc, name, spis) {
  if (typeof name !== 'string') fail('pathItems.add: name string expected');
  if (!Array.isArray(spis) || !spis.length) fail('pathItems.add: array of SubPathInfo expected');
  const k = doc._res / 72;                   // points -> pixels
  const comps = spis.map((s, i) => {
    if (!s || s._cls !== 'SubPathInfo') fail(`pathItems.add: element ${i} is not a SubPathInfo`);
    needBool(s.closed, 'SubPathInfo.closed'); needEnum(s.operation, ShapeOperation, 'SubPathInfo.operation');
    if (!Array.isArray(s.entireSubPath) || s.entireSubPath.length < 1) fail('SubPathInfo.entireSubPath: array of PathPointInfo expected');
    const pts = s.entireSubPath.map((p, j) => {
      if (!p || p._cls !== 'PathPointInfo') fail(`entireSubPath[${j}] is not a PathPointInfo`);
      needEnum(p.kind, PointKind, 'PathPointInfo.kind');
      const sc = (v, w) => { const q = needPt(v, w); return [q[0] * k, q[1] * k]; };
      return { a: sc(p.anchor, 'anchor'), l: sc(p.leftDirection, 'leftDirection'), r: sc(p.rightDirection, 'rightDirection'), kind: p.kind };
    });
    return { op: s.operation.split('.')[1], closed: s.closed, points: pts };
  });
  const o = {
    typename: 'PathItem', name, kind: 'PathKind.NORMALPATH', _comps: comps, _doc: doc,
    get subPathItems() {
      const self = this;
      return collection('SubPathItems', () => self._comps.map((c) => strict({
        typename: 'SubPathItem', closed: c.closed, operation: 'ShapeOperation.' + c.op,
        get pathPoints() {
          return collection('PathPoints', () => c.points.map((q) => strict({ typename: 'PathPoint', anchor: [q.a[0] / k, q.a[1] / k],
            leftDirection: [q.l[0] / k, q.l[1] / k], rightDirection: [q.r[0] / k, q.r[1] / k], kind: q.kind }, 'PathPoint')), {});
        },
      }, 'SubPathItem')), {});
    },
    select() { doc._selPath = this._self; rec('path.select', { name }); },
    deselect() { if (doc._selPath === this._self) doc._selPath = null; },
    remove() { const i = doc._paths.indexOf(this._self); if (i < 0) fail('path.remove: not in document'); doc._paths.splice(i, 1); if (doc._selPath === this._self) doc._selPath = null; rec('path.remove', { name }); },
  };
  const p = strict(o, 'PathItem');
  doc._paths.push(p);
  rec('pathItems.add', { name, components: comps.map((c) => ({ op: c.op, closed: c.closed, n: c.points.length })) });
  return p;
}
const copyMask = (p) => ({ components: p._comps.map((c) => ({ op: c.op, closed: c.closed, points: c.points.map((q) => ({ a: q.a.slice(), l: q.l.slice(), r: q.r.slice(), kind: q.kind })) })) });

// ------------------------------------------------------------------ Action Manager behaviour
function rgbOf(o, what) {
  if (!o || o.obj !== 'RGBColor') fail(`${what}: RGBColor object expected`);
  const v = o.v, c = [v.red, v.grain, v.blue];
  if (!c.every((x) => isNum(x) && x >= 0 && x <= 255)) fail(`${what}: red / grain (Grn ) / blue doubles 0..255 expected, got ${JSON.stringify(v)}`);
  return c;
}
function px(o, what) { if (!o || o.unit !== 'pixelsUnit' || !isNum(o.v)) fail(`${what}: #Pxl unit double expected`); return o.v; }
function pct(o, what) { if (!o || o.unit !== 'percentUnit' || !isNum(o.v)) fail(`${what}: #Prc unit double expected`); return o.v; }
function parseGradient(v) {
  const g = v.gradient;
  if (!g || g.obj !== 'gradientClassEvent') fail('gradientLayer: Grad / Grdn object expected');
  if (!v.type || v.type.enum !== 'gradientType' || !['linear', 'radial'].includes(v.type.v)) fail('gradientLayer: Type GrdT Lnr / Rdl expected');
  if (!v.angle || v.angle.unit !== 'angleUnit') fail('gradientLayer: Angl #Ang expected');
  if (v.align !== false) fail('gradientLayer: Algn false expected (the mock models "align with layer" off)');
  const off = v.offset && v.offset.obj === 'paint' ? v.offset.v : fail('gradientLayer: Ofst Pnt object expected');
  const colors = g.v.colors.list.map((s, i) => {
    if (s.obj !== 'colorStop') fail('gradient Clrs: Clrt objects expected');
    if (!Number.isInteger(s.v.location) || s.v.location < 0 || s.v.location > 4096) fail(`colour stop ${i}: Lctn integer 0..4096 expected`);
    return { loc: s.v.location / 4096, rgb: rgbOf(s.v.color, `colour stop ${i}`) };
  });
  const alphas = g.v.transparency.list.map((s, i) => {
    if (s.obj !== 'transferSpec') fail('gradient Trns: TrnS objects expected');
    return { loc: s.v.location / 4096, op: pct(s.v.opacity, `transparency stop ${i}`) };
  });
  return { t: 'gradient', type: v.type.v, angle: v.angle.v, scale: pct(v.scale, 'gradient Scl'),
    ox: pct(off.horizontal, 'Ofst Hrzn'), oy: pct(off.vertical, 'Ofst Vrtc'), name: g.v.name, colors, alphas };
}
function executeAction(eventID, desc, mode) {
  const ev = nm(eventID);
  if (mode !== DialogModes.NO) fail(`executeAction(${ev}): DialogModes.NO expected`);
  if (desc !== undefined && (!desc || desc._cls !== 'ActionDescriptor')) fail(`executeAction(${ev}): ActionDescriptor or undefined expected`);
  const D = desc ? decode(desc) : null;
  rec('executeAction', { event: ev, desc: D });
  const doc = activeDoc;
  if (!doc) fail('executeAction: no document');
  if (ev === 'make') {
    const r = D.null && D.null.ref;
    if (!r || r.length !== 1 || r[0].form !== 'class') fail('make: null = class reference expected');
    if (r[0].cls === 'contentLayer') {
      const u = D.using;
      if (!u || u.obj !== 'contentLayer') fail('make contentLayer: Usng contentLayer object expected');
      const t = u.v.type;
      if (!t || !['solidColorLayer', 'gradientLayer'].includes(t.obj)) fail('make contentLayer: Type solidColorLayer / gradientLayer expected');
      const L = newLayer(doc, 'ArtLayer');
      if (t.obj === 'solidColorLayer') { L._fill = { t: 'solid', rgb: rgbOf(t.v.color, 'solidColorLayer Clr ') }; L._kind = LayerKind.SOLIDFILL; L.name = `Color Fill ${++doc._fillNo}`; }
      else { L._fill = parseGradient(t.v); L._kind = LayerKind.GRADIENTFILL; L.name = `Gradient Fill ${++doc._fillNo}`; }
      if (u.v.shape) {
        const s = u.v.shape;
        if (!['ellipse', 'rectangle'].includes(s.obj)) fail('make contentLayer: Shp Elps / Rctn expected');
        if (s.v.unitValueQuadVersion !== 1) fail('live shape: unitValueQuadVersion 1 expected');
        const sh = { type: s.obj, top: px(s.v.top, 'Top '), left: px(s.v.left, 'Left'), bottom: px(s.v.bottom, 'Btom'), right: px(s.v.right, 'Rght') };
        if (s.v.topLeft) sh.radius = px(s.v.topLeft, 'topLeft');
        if (!(sh.right > sh.left && sh.bottom > sh.top)) fail('live shape: empty bounds');
        L._vmask = { shape: sh };
      } else if (doc._selPath) L._vmask = copyMask(doc._selPath);
      insertAboveActive(doc, L);
      doc._active = L;
      return;
    }
    if (r[0].cls === 'colors') {
      const u = D.using;
      if (!u || u.obj !== 'colors' || typeof u.v.name !== 'string') fail('make swatch: Usng Clrs object with a name expected');
      doc._app_swatches.push({ name: u.v.name, rgb: rgbOf(u.v.color, 'swatch colour') });
      return;
    }
    if (r[0].cls === 'path') {
      const at = D.at && D.at.ref, us = D.using && D.using.ref;
      if (!at || at[0].form !== 'enum' || at[0].v !== 'vectorMask') fail('make path: At = vectorMask expected');
      if (!us || us[0].form !== 'enum' || us[0].v !== 'targetEnum') fail('make path: Usng = target path expected');
      if (!doc._selPath) fail('make vector mask: no path selected');
      const A = doc._active;
      if (!A) fail('make vector mask: no active layer');
      if (A._vmask) fail(`make vector mask: "${A.name}" already has one`);
      A._vmask = copyMask(doc._selPath);
      return;
    }
    fail(`make ${r[0].cls}: not modelled`);
  }
  if (ev === 'set') {
    const r = D.null && D.null.ref;
    if (!r || r[0].form !== 'prop' || r[0].v !== 'layerEffects' || r[1].v !== 'targetEnum') fail('set: target layer effects expected');
    if (!D.to || D.to.obj !== 'layerEffects') fail('set layerEffects: T Lefx object expected');
    doc._active._styles = D.to.v;
    return;
  }
  if (ev === 'newPlacedLayer') { doc._active._smart = true; doc._active._kind = LayerKind.SMARTOBJECT; return; }
  if (ev === 'gaussianBlur') {
    if (!doc._active._smart) fail('Gaussian blur on a non-smart layer would rasterise it');
    doc._active._filters.push({ blur: px(D.radius, 'Rds ') }); return;
  }
  fail(`executeAction("${ev}") not modelled`);
}

// ------------------------------------------------------------------ document / app
const files = {};
function File(p) {
  if (typeof p !== 'string') fail('File(path): string expected');
  const o = {
    typename: 'File', fsName: p, name: path.basename(p), encoding: 'ASCII', lineFeed: 'Macintosh', _mode: null,
    get parent() { return Folder(path.dirname(p)); },
    get exists() { return p === SCRIPT_PATH || p in files; },
    open(mode) { this._mode = mode; if (mode === 'w') files[p] = ''; return true; },
    write(s) { if (this._mode !== 'w') fail('File.write: not open'); files[p] += String(s); return true; },
    writeln(s) { return this.write(String(s) + '\n'); },
    close() { this._mode = null; return true; },
  };
  return strict(o, 'File');
}
function Folder(p) { return strict({ typename: 'Folder', fsName: p, name: path.basename(p), exists: true }, 'Folder'); }
Object.defineProperty(Folder, 'desktop', { get: () => Folder('/Users/client/Desktop') });
const PhotoshopSaveOptions = function () { return strict({ typename: 'PhotoshopSaveOptions', layers: true, embedColorProfile: false, alphaChannels: true, annotations: true, spotColors: false }, 'PhotoshopSaveOptions'); };

const app_swatches = [];
const docs = [];
let activeDoc = null;
const prefs = strict({ typename: 'Preferences', _ru: Units.INCHES, _mc: QueryStateType.ASK, typeUnits: 'TypeUnits.POINTS',
  get rulerUnits() { return this._ru; }, set rulerUnits(v) { this._ru = needEnum(v, Units, 'rulerUnits'); },
  get maximizeCompatibility() { return this._mc; }, set maximizeCompatibility(v) { this._mc = needEnum(v, QueryStateType, 'maximizeCompatibility'); } }, 'Preferences');

class Document {
  constructor(w, h, res, name, mode, fill, par, bits, profile) {
    this.typename = 'Document'; this.name = name; this._w = w; this._h = h; this._res = res; this._mode = mode; this._fillMode = fill;
    this._bits = bits; this._profile = profile; this._layers = []; this._paths = []; this._selPath = null; this._active = null;
    this._samplers = []; this._saved = []; this._fillNo = 0; this._groupNo = 0; this._app_swatches = app_swatches; this._resizes = [];
  }
  get width() { return this._w; }
  get height() { return this._h; }
  get resolution() { return this._res; }
  get layers() { const d = this; return collection('Layers', () => d._layers.slice(), {}); }
  get artLayers() { const d = this; return collection('ArtLayers', () => d._layers.filter((x) => x._cls === 'ArtLayer'), {}); }
  get layerSets() {
    const d = this;
    return collection('LayerSets', () => d._layers.filter((x) => x._cls === 'LayerSet'), {
      add() {
        const g = newLayer(d, 'LayerSet'); g.name = `Group ${++d._groupNo}`;
        if (d._active && S(d._active._parent) === S(d)) insertAboveActive(d, g); else { g._parent = d; d._layers.unshift(g); }
        d._active = g; rec('layerSets.add', { parent: 'document' }); return g;
      },
    });
  }
  get activeLayer() { if (!this._active) fail('no active layer'); return this._active; }
  set activeLayer(L) {
    if (!L || !['ArtLayer', 'LayerSet'].includes(L._cls) || S(L._doc) !== S(this)) fail('activeLayer: a layer of this document expected');
    this._active = L._self;
  }
  get pathItems() { const d = this; return collection('PathItems', () => d._paths.slice(), { add(name, spis) { return makePathItem(d, name, spis); } }); }
  get colorSamplers() {
    const d = this;
    return collection('ColorSamplers', () => d._samplers.slice(), {
      add(pos) {
        if (prefs.rulerUnits !== Units.PIXELS) fail('colorSamplers.add: mock models pixel ruler units only');
        const p = needPt(pos, 'colorSamplers.add');
        if (d._samplers.length >= 10) fail('colorSamplers: at most 10');
        if (p[0] < 0 || p[1] < 0 || p[0] > d._w || p[1] > d._h) fail(`colour sampler ${JSON.stringify(p)} outside the canvas`);
        const c = composite(d, d._layers, p[0], p[1]);
        const a = c[3], rgb = a > 0 ? [c[0] / a, c[1] / a, c[2] / a] : [255, 255, 255];
        const s = strict({ typename: 'ColorSampler', position: p,
          color: strict({ typename: 'SolidColor', rgb: strict({ typename: 'RGBColor', red: rgb[0], green: rgb[1], blue: rgb[2], hexValue: '' }, 'RGBColor') }, 'SolidColor'),
          remove() { const i = d._samplers.indexOf(s); if (i >= 0) d._samplers.splice(i, 1); }, move() {} }, 'ColorSampler');
        d._samplers.push(s); rec('colorSampler', { at: p, rgb, a }); return s;
      },
      removeAll() { d._samplers.length = 0; },
    });
  }
  resizeImage(w, h, res, method) {
    if (w !== undefined || h !== undefined) fail('resizeImage: the mock models resolution-only changes (width/height undefined)');
    needNum(res, 'resizeImage resolution'); needEnum(method, ResampleMethod, 'resizeImage method');
    if (method !== ResampleMethod.NONE) fail('resizeImage: ResampleMethod.NONE expected (resampling would change pixels)');
    this._resizes.push(res); this._res = res; rec('resizeImage', { res });
  }
  saveAs(file, opts, asCopy, ext) {
    if (!file || file._cls !== 'File') fail('saveAs: File expected');
    if (!opts || opts._cls !== 'PhotoshopSaveOptions') fail('saveAs: PhotoshopSaveOptions expected');
    if (!/\.psd$/i.test(file.fsName)) fail('saveAs: .psd path expected');
    if (ext !== undefined) needEnum(ext, Extension, 'saveAs extension');
    this._saved.push({ path: file.fsName, opts: { layers: opts.layers, embedColorProfile: opts.embedColorProfile, alphaChannels: opts.alphaChannels, annotations: opts.annotations, spotColors: opts.spotColors }, asCopy: !!asCopy, snapshot: dumpDoc(this) });
    rec('saveAs', { path: file.fsName });
  }
}

const appObj = {
  typename: 'Application', version: '25.0',
  preferences: prefs,
  _dd: DialogModes.ALL,
  get displayDialogs() { return this._dd; }, set displayDialogs(v) { this._dd = needEnum(v, DialogModes, 'displayDialogs'); },
  get activeDocument() { if (!activeDoc) fail('no active document'); return activeDoc._self; },
  set activeDocument(d) { if (!d || d._cls !== 'Document') fail('activeDocument: Document expected'); activeDoc = d; },
  documents: null,
};
appObj.documents = collection('Documents', () => docs.map((d) => d._self), {
  add(w, h, res, name, mode, fill, par, bits, profile) {
    if (prefs.rulerUnits !== Units.PIXELS) fail('documents.add: set rulerUnits to PIXELS first (width/height would be read in other units)');
    needNum(w, 'width'); needNum(h, 'height'); needNum(res, 'resolution');
    if (Math.round(w) !== w || Math.round(h) !== h || w < 1 || h < 1 || w > 300000 || h > 300000) fail(`documents.add: bad size ${w} x ${h}`);
    if (typeof name !== 'string') fail('documents.add: name string expected');
    needEnum(mode, NewDocumentMode, 'mode'); needEnum(fill, DocumentFill, 'initialFill'); needNum(par, 'pixelAspectRatio');
    needEnum(bits, BitsPerChannelType, 'bitsPerChannel');
    if (typeof profile !== 'string' || !profile) fail('documents.add: colour profile name expected');
    const d = strict(new Document(w, h, res, name, mode, fill, par, bits, profile), 'Document');
    const L = newLayer(d, 'ArtLayer'); L.name = 'Layer 1'; L._parent = d; d._layers.push(L); d._active = L;
    docs.push(d); activeDoc = d;
    rec('documents.add', { w, h, res, name, mode, fill, bits, profile });
    return d;
  },
});
const appProxy = strict(appObj, 'Application');

function dumpLayer(L) {
  const o = { cls: L._cls, name: L.name, opacity: L._opacity, visible: L.visible };
  if (L._vmask) o.vmask = L._vmask.shape ? { shape: L._vmask.shape } : { components: L._vmask.components.map((c) => ({ op: c.op, closed: c.closed, points: c.points.map((q) => [q.a[0], q.a[1], q.l[0], q.l[1], q.r[0], q.r[1], q.kind === PointKind.SMOOTHPOINT ? 1 : 0]) })) };
  if (L._styles) o.styles = L._styles;
  if (L._cls === 'LayerSet') o.layers = L._layers.map(dumpLayer);
  else { o.fillOpacity = L._fillOpacity; o.kind = L._kind; o.fill = L._fill; if (L._smart) o.smart = true; if (L._filters.length) o.filters = L._filters; }
  return o;
}
function dumpDoc(d) {
  return { name: d.name, width: d._w, height: d._h, resolution: d._res, mode: d._mode, fill: d._fillMode, bits: d._bits, profile: d._profile,
    layers: d._layers.map(dumpLayer), paths: d._paths.map((p) => p.name), resizes: d._resizes };
}

const alerts = [];
const ctx = {
  app: appProxy, $: { fileName: SCRIPT_PATH, writeln() {} }, alert(m) { alerts.push(String(m)); rec('alert', {}); },
  File, Folder, PhotoshopSaveOptions, PathPointInfo, SubPathInfo,
  ActionDescriptor: function () { return mkAD(); }, ActionReference: function () { return mkAR(); }, ActionList: function () { return mkAL(); },
  charIDToTypeID, stringIDToTypeID, executeAction,
  Units, DialogModes, NewDocumentMode, DocumentFill, BitsPerChannelType, QueryStateType, ShapeOperation, PointKind,
  ElementPlacement, ResampleMethod, Extension, LayerKind,
};
vm.createContext(ctx);
let crashed = null;
try { vm.runInContext(fs.readFileSync(SCRIPT, 'utf8'), ctx, { filename: path.basename(SCRIPT), timeout: 120000 }); } catch (e) { crashed = String(e && e.stack || e); }

const out = { script: SCRIPT, scriptPath: SCRIPT_PATH, newLayer: NEW_LAYER, quirkEmptySet: QUIRK_EMPTY, crashed, errors, alerts, files,
  calls, swatches: app_swatches, prefsAfter: { rulerUnits: prefs._ru, maximizeCompatibility: prefs._mc, displayDialogs: appObj._dd },
  documents: docs.map((d) => Object.assign(dumpDoc(d), { saved: d._saved })) };
fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
console.log(`ps_mock: ${calls.length} calls, ${errors.length} errors, ${alerts.length} alerts${crashed ? ', CRASHED' : ''}`);
process.exit(crashed ? 1 : 0);
