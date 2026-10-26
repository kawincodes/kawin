#!/usr/bin/env python3
"""mix.py - re-recording mixer for the KRS Tech Solutions "WE'RE HIRING - VIDEO EDITOR"
30 s Instagram Story (1080x1920, 30 fps, 120 BPM grid).

    python3 -I mix.py --cues <cues.json> --video <silent.mp4> --music techhouse|cinematic \
                      --out <out.mp4> [--wav <mix.wav>] [--report <dir>] [--seed 1] \
                      [--cache-dir <dir>] [--whoosh-sync window|peak]

What it does (all deterministic, 48 kHz stereo float internally):
  1. Reads the cue sheet.  'anchor' cues give the music anchors (drop1 / breakdown / drop2 /
     end_hit; note text is matched loosely: "drop 1", "DROP1", "end hit", "end_hit", ...).
     The video's duration (ffprobe) sets the mix length; music_<style>.render_story() renders
     the bed for exactly that duration and those anchors.
  2. Renders every SFX cue with sfx.py and places it so its perceptual sync point lands on
     the cue (see SYNC RULES below), at sfx.SUGGESTED_GAIN_DB plus the offsets in this file.
  3. Mixes: music bus (fx stem ducked under duplicating SFX transitions with
     sfx.fx_duck_gain(), then a two-band sidechain duck keyed by the impacts / rejects /
     check), SFX bus (shared short convolution room, light glue compression, a 4x-oversampled
     transient soft clip on the slam peaks), master (gentle EQ, HP150-keyed bus compressor,
     -14 LUFS integrated, look-ahead true-peak limiter at -1.5 dBTP, 0.3 s raised-cosine fade
     to digital zero at the end).
  4. Muxes: the video stream is copied untouched (-c:v copy), audio is AAC-LC 256 kb/s
     48 kHz stereo of exactly the video's duration, -movflags +faststart.  The MP4's audio is
     decoded and measured; if the AAC overshoots -1.15 dBTP (8x) somewhere, the limiter ceiling
     is lowered only around those spots and the master is redone (up to 4 passes).
  5. --report <dir>: cue_alignment.csv, anchor_alignment.csv, loudness.txt,
     master_spectrogram.png, master_waveform.png, ffprobe_output.json, video_md5.txt and
     mix_report.json (every check with PASS/FAIL).  Exit code 1 if a check fails.

SYNC RULES (cue t = the time in the cue sheet; "target" = where the perceptual hit lands)
  impact, reject, pop, tick, check, downlifter : onset at t (sfx.anchor = 0)
  slash                                       : the cut (sfx.anchor = 0.065 s) at t
  scrub, shimmer                              : array starts at t, length = dur
  riser, reverse_swell                        : t is the START, the array ENDS on the hit
                                                at t + dur (the notes say "Hit = 2.0" etc.)
  whoosh (--whoosh-sync window, default)      : the cue's (t, dur) is the on-screen motion
                                                window ("17.7-18.3 brand-slash band ... crossing the
                                                frame centre exactly on 18.0"), so the array spans
                                                [t, t+dur] and its envelope peak lands mid-motion
                                                (0.55-0.68 x dur).  The peak is measured as the
                                                centre of the -3 dB region of the 20 ms envelope;
                                                sfx.anchor() is the argmax at the trailing edge of
                                                that ~50 ms plateau (0-19 ms later), which any gain
                                                change in the bus can move.
  whoosh (--whoosh-sync peak)                 : envelope peak at t (start = t - sfx.anchor())
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:                      # python3 -I does not add the script directory
    sys.path.insert(0, HERE)

import numpy as np                            # noqa: E402
import soundfile as sf                        # noqa: E402
from scipy import signal                      # noqa: E402
from scipy.ndimage import uniform_filter1d    # noqa: E402

import sfx                                    # noqa: E402

SR = 48000
ANCHOR_KEYS = ("drop1", "breakdown", "drop2", "end_hit")
HIT_ANCHORS = ("drop1", "drop2", "end_hit")          # music hits that carry a big SFX slam
DEFAULT_ANCHORS = dict(drop1=2.0, breakdown=14.0, drop2=18.0, end_hit=26.0)
SFX_TYPES = ("slash", "shimmer", "tick", "scrub", "riser", "reverse_swell", "impact", "whoosh",
             "pop", "reject", "check", "downlifter")

# ------------------------------------------------------------------------------------------------
# mix settings (data)
# ------------------------------------------------------------------------------------------------
MUSIC_BUS_LUFS = -20.0      # music stems summed and set to this integrated loudness before the SFX
SFX_BUS_TRIM_DB = 0.0       # global SFX trim on top of SUGGESTED_GAIN_DB (SFX are -3 dBTP sounds)
HERO_SLAM_DB = +2.5         # big slam on drop1 / drop2 / end_hit: gives back the ~2.5 LU full-band loss of
                            # over_music_hit(), so the three hero slams are the loudest SFX full band AND on HP150
OTHER_BIG_DB = 0.0          # a big slam that is not on an anchor (e.g. EDITOR at 2.5) stays at its suggested level
# per-type trims on top of SUGGESTED_GAIN_DB (musical judgement, measured in the story mix): the UI pops sit
# ~3 dB under the bed's momentary level, the breakdown stamps stay ~9 dB over the sparse breakdown bed but
# under the hero slams, the glassy shimmers do not out-shine the hook slash
TYPE_TRIM_DB = dict(pop=-2.0, check=-2.5, reject=-1.5, shimmer=-2.0)
# variable-length sounds: SUGGESTED_GAIN_DB was measured on one reference length (riser(1.5), shimmer(1.2),
# scrub(1.0), whoosh(0.45,'lr'), downlifter(1.0), reverse_swell(0.8)); a longer / louder render is pulled down
# to its ROLE_LUFS_M target (never boosted: a short whoosh is meant to be smaller), at most LEN_COMP_MAX_DB
LEN_COMP_TYPES = ("riser", "reverse_swell", "shimmer", "scrub", "whoosh", "downlifter", "pop", "slash")
LEN_COMP_MAX_DB = 3.0
STACK_WINDOW_S = 0.06       # a sound whose sync point is this close to a hero slam ...
STACK_ATTEN_DB = -3.0       # ... is pulled down so the slam stays the hero (slash / whoosh / shimmer ...)
STACK_TYPES = ("slash", "whoosh", "shimmer", "pop", "tick", "check")
LIGHT_DB = -3.0             # cue note starting with "(light)"
LIGHT_INTENSITY = 0.7
LOW_PRESENT_DB = -12.0      # music LP150 peak (re its 99.5th pct) above which a slam uses over_music_hit
POP_IN_KEY = True           # snap out-of-key pop offsets to the nearest F-minor note (ties go up)

# shared SFX room: synthetic IR from sfx._make_ir (decorrelated noise, 3-band decay)
ROOM = dict(rt=0.55, hf=0.45, er=0.2, predelay=0.012, hp=300.0, lp=7000.0, ret_db=-4.0)
SEND = dict(tick=0.35, pop=0.30, check=0.20, scrub=0.25, whoosh=0.12, slash=0.20, impact=0.12,
            reject=0.18, shimmer=0.10, downlifter=0.15, riser=0.0, reverse_swell=0.0)
# SFX bus glue: detector HP80, 10 ms RMS, threshold relative to the bus's loudest moment
GLUE = dict(range_db=6.0, ratio=1.5, knee_db=4.0, attack=0.010, release=0.15, sc_hp=80.0)
# SFX bus transient stage: a 4x-oversampled soft clipper (no gain riding, so slam bodies keep their loudness)
# whose asymptote sits rel_db above the level the master make-up would put at the master ceiling.  Linear
# below knee x ceiling; above it the first-cycle peaks of the slams (click / crack / body onset) are rounded
# with tanh, so the master limiter does not have to pull the whole mix, music included, down for them.
SFX_PEAK = dict(rel_db=1.0, knee=0.6, oversample=4)

# sidechain duck of the music bus, two bands split at 150 Hz (zero-phase, perfectly complementary).
# key: (hi-band depth dB, lo-band depth on a music hit, lo-band depth otherwise, attack s, hold s, release s)
DUCK_SPLIT_HZ = 150.0
DUCK = {
    "impact_big":    (4.5, 1.5, 5.0, 0.008, 0.06, 0.25),
    "impact_medium": (4.0, 1.0, 4.0, 0.008, 0.04, 0.20),
    "impact_small":  (3.0, 0.5, 3.0, 0.006, 0.03, 0.15),
    "reject":        (4.0, 4.0, 4.0, 0.006, 0.05, 0.20),
    "check":         (2.0, 0.0, 0.0, 0.008, 0.03, 0.15),
}
# music fx-stem transitions an SFX duplicates (README "one source per transition")
FX_DUP_KINDS = {"riser": ("riser", "reverse_cymbal"), "reverse_swell": ("reverse_cymbal", "riser"),
                "downlifter": ("downlifter",)}

# master
EQ = (("highpass", 25.0, 0.0, 0.707), ("peak", 280.0, -1.0, 0.9), ("highshelf", 9000.0, 1.0, 0.7))
BUS_COMP = dict(thr_rel_db=3.0, ratio=1.5, attack=0.02, release=0.2, sc_hp=150.0)
TARGET_LUFS = -14.0
CEILING_DBTP = -1.5          # first limiter ceiling; leaves room for AAC (spec: <= -1.0 dBTP after encode)
AAC_TP_MARGIN = 0.15         # the decoded AAC must measure <= SPEC tp_max - this (8x true peak), else re-master
MAX_MASTER_PASSES = 5        # with the limiter ceiling lowered only around the spots where the AAC overshoots
AAC_LOCAL_ROUNDS = 2         # (two local rounds; AAC rate control is global, so after that the drop is global)
AAC_FIX_PAD_S = 0.06         # (+- this around each overshooting cluster, 20 ms raised-cosine edges)
LIMITER = dict(lookahead=0.005, release=0.08, oversample=4)
FADE_IN_S = 0.002            # click guard only: the hook is audible from frame 0
FADE_OUT_S = 0.3             # raised-cosine fade to digital zero at the video end
AAC = ["-c:a", "aac", "-profile:a", "aac_low", "-b:a", "256k", "-ar", "48000", "-ac", "2"]

# spec limits for the checks
SPEC = dict(lufs=(-14.5, -13.5), tp_max=-1.0, lra_max=8.0, frame_tol_s=1.0 / 30.0, dc_max=1e-4)


def log(*a):
    print("[mix]", *a, file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------
# small DSP / measurement helpers
# ------------------------------------------------------------------------------------------------
def db(v):
    return 20.0 * math.log10(max(float(v), 1e-20))


def g_db(d):
    return 10.0 ** (float(d) / 20.0)


def butter(kind, fc, order=2):
    return signal.butter(order, fc, kind, fs=SR, output="sos")


def kweight(x):
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    return signal.lfilter(b2, a2, signal.lfilter(b1, a1, np.asarray(x, float), axis=0), axis=0)


def _block_power(x, win, hop):
    y = kweight(x)
    p = np.sum(y * y, axis=1)
    c = np.concatenate([[0.0], np.cumsum(p)])
    if len(p) < win:
        return np.array([c[-1] / max(len(p), 1)])
    idx = np.arange(0, len(p) - win + 1, hop)
    return (c[idx + win] - c[idx]) / win


def lufs(x):
    """BS.1770-4 integrated loudness (400 ms blocks, 75 % overlap, -70 / -10 gates)."""
    ms = _block_power(x, int(0.4 * SR), int(0.1 * SR))
    lv = -0.691 + 10 * np.log10(ms + 1e-30)
    a = ms[lv > -70.0]
    if not len(a):
        return -120.0
    rel = -0.691 + 10 * np.log10(np.mean(a)) - 10.0
    b = ms[(lv > -70.0) & (lv > rel)]
    return float(-0.691 + 10 * np.log10(np.mean(b) + 1e-30)) if len(b) else -120.0


def loudness_curve(x, win_s=0.4, hop_s=0.01):
    """(centre times, LUFS) of a sliding K-weighted window (0.4 s = momentary, 3 s = short-term)."""
    w, h = int(win_s * SR), int(hop_s * SR)
    pad = np.zeros((w // 2, x.shape[1]))
    ms = _block_power(np.concatenate([pad, x, pad]), w, h)
    return np.arange(len(ms)) * hop_s, -0.691 + 10 * np.log10(ms + 1e-30)


def true_peak(x):
    return sfx.true_peak(x)              # 8x oversampled


def env_db(x, win_s):
    """Centred moving-average power (mean over channels) in dB."""
    p = np.mean(np.asarray(x, float) ** 2, axis=1)
    e = uniform_filter1d(p, max(1, int(round(win_s * SR))), mode="constant")
    return 10 * np.log10(np.maximum(e, 1e-20))


def add_at(dst, src, start):
    """dst[start:start+len(src)] += src, clipped to dst's bounds (start may be negative)."""
    a, b = max(start, 0), min(start + len(src), len(dst))
    if b > a:
        dst[a:b] += src[a - start:b - start]


def biquad(kind, f0, gain_db, q):
    """RBJ cookbook biquad as one SOS row."""
    A = 10 ** (gain_db / 40.0)
    w = 2 * math.pi * f0 / SR
    cw, sw = math.cos(w), math.sin(w)
    al = sw / (2 * q)
    if kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + al, -2 * cw, 1 - al]
    elif kind == "peak":
        b = [1 + al * A, -2 * cw, 1 - al * A]
        a = [1 + al / A, -2 * cw, 1 - al / A]
    elif kind == "highshelf":
        sa = 2 * math.sqrt(A) * al
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    else:
        raise ValueError(kind)
    return np.array([b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]])


def smooth_gain(target_db, attack, release, hop):
    """One-pole attack/release smoothing of a gain-reduction curve (dB, <= 0) on a hop grid."""
    a_att, a_rel = math.exp(-hop / (attack * SR)), math.exp(-hop / (release * SR))
    out = np.empty_like(target_db)
    cur = 0.0
    for i, v in enumerate(target_db.tolist()):
        a = a_att if v < cur else a_rel
        cur = a * cur + (1.0 - a) * v
        out[i] = cur
    return out


def compressor_gain(x, thr_db, ratio, attack, release, sc_hp, knee_db=0.0, rms_s=0.010, hop=48):
    """Feed-forward compressor gain (linear, per sample) from a high-passed 10 ms RMS detector."""
    sc = signal.sosfilt(butter("highpass", sc_hp, 2), np.asarray(x, float), axis=0)
    p = uniform_filter1d(np.mean(sc ** 2, axis=1), int(rms_s * SR), mode="constant")
    lv = 10 * np.log10(np.maximum(p[::hop], 0.0) + 1e-20)     # the running mean can dip below 0 by round-off
    over = lv - thr_db
    slope = 1.0 - 1.0 / ratio
    if knee_db > 0:
        gr = np.where(over <= -knee_db / 2, 0.0,
                      np.where(over >= knee_db / 2, -slope * over, -slope * (over + knee_db / 2) ** 2 / (2 * knee_db)))
    else:
        gr = np.minimum(0.0, -slope * over)
    gs = smooth_gain(gr, attack, release, hop)
    full = np.interp(np.arange(len(x)) / hop, np.arange(len(gs)), gs)
    return 10 ** (full / 20.0), float(-gs.min())


def tp_limiter(x, ceiling_dbtp, lookahead, release, oversample=4):
    """Offline look-ahead true-peak limiter.  need[n] = ceiling / (oversampled peak at n);
    g1 = forward-looking minimum over the look-ahead, exponential release towards 1, then a
    box average of the look-ahead length.  The average only spans samples whose g1 already
    covers the peak, so g <= need everywhere (no overshoot by construction)."""
    n = len(x)
    ceil = 10.0 ** (np.asarray(ceiling_dbtp, float) / 20.0)      # scalar or per-sample ceiling
    up = np.abs(signal.resample_poly(x, oversample, 1, axis=0)).max(axis=1)
    up = np.concatenate([up, np.zeros(n * oversample - len(up))])[: n * oversample]
    pk = np.maximum(up.reshape(n, oversample).max(axis=1), np.abs(x).max(axis=1))
    need = np.minimum(1.0, ceil / np.maximum(pk, 1e-12))
    la = max(1, int(lookahead * SR))
    # g1[m] = min(need[m .. m+la-1])
    padded = np.concatenate([need, np.ones(la)])
    from scipy.ndimage import minimum_filter1d
    g1 = minimum_filter1d(padded, size=la, origin=-(la // 2))[:n]
    # release: r[m] = min(g1[m], 1 - (1 - r[m-1]) * a)
    a = math.exp(-1.0 / (release * SR))
    r = np.empty(n)
    cur = 1.0
    g1l = g1.tolist()
    for i in range(n):
        cur = 1.0 - (1.0 - cur) * a
        v = g1l[i]
        if v < cur:
            cur = v
        r[i] = cur
    c = np.concatenate([np.ones(la - 1), r])
    k = np.ones(la) / la
    g = np.convolve(c, k, mode="valid")[:n]
    g = np.minimum(g, need)            # only bites in the first look-ahead window (padded with ones)
    return x * g[:, None], g


def edge_fades(n, fin_s, fout_s):
    g = np.ones(n)
    ni, no = int(round(fin_s * SR)), int(round(fout_s * SR))
    if ni > 0:
        g[:ni] = np.sin(0.5 * np.pi * np.arange(ni) / ni) ** 2
    if no > 0:
        g[n - no:] = np.cos(0.5 * np.pi * (np.arange(1, no + 1)) / no) ** 2     # last sample exactly 0
    return g


# ------------------------------------------------------------------------------------------------
# video / cues
# ------------------------------------------------------------------------------------------------
def ffprobe_json(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
                       capture_output=True, text=True, check=True)
    return json.loads(p.stdout)


def probe_video(path):
    j = ffprobe_json(path)
    vs = [s for s in j["streams"] if s.get("codec_type") == "video"]
    if not vs:
        raise SystemExit(f"{path}: no video stream")
    v = vs[0]
    num, den = (int(a) for a in v.get("r_frame_rate", "30/1").split("/"))
    fps = num / den if den else 30.0
    dur = None
    for cand in (v.get("duration"), j.get("format", {}).get("duration")):
        try:
            dur = float(cand)
            break
        except (TypeError, ValueError):
            pass
    if dur is None and v.get("nb_frames"):
        dur = int(v["nb_frames"]) / fps
    if dur is None:
        raise SystemExit(f"{path}: cannot determine the duration")
    return dict(duration=dur, fps=fps, nb_frames=int(v.get("nb_frames", 0) or 0), width=v.get("width"),
                height=v.get("height"), codec=v.get("codec_name"), probe=j)


def anchor_key(cue):
    text = " ".join(str(cue.get(k, "")) for k in ("note", "name", "anchor", "label", "id"))
    s = re.sub(r"[^a-z0-9]+", "", text.lower())
    if re.search(r"endhit|endslam|finalhit", s):
        return "end_hit"
    if re.search(r"drop0?1|dropone|firstdrop", s):
        return "drop1"
    if re.search(r"drop0?2|droptwo|seconddrop", s):
        return "drop2"
    if re.search(r"breakdown|breakdwn", s):
        return "breakdown"
    if re.match(r"(end|videoend|endofvideo|out)", s):
        return "end"
    if s.startswith("drop"):
        return "drop"
    return None


def load_cues(path):
    with open(path) as f:
        data = json.load(f)
    cues = data.get("cues", data.get("events")) if isinstance(data, dict) else data
    if not isinstance(cues, list):
        raise SystemExit(f"{path}: expected a list of cues (or {{'cues': [...]}})")
    anchors, items, issues, end_t = {}, [], [], None
    plain_drops = []
    for i, c in enumerate(cues):
        if not isinstance(c, dict) or "t" not in c or "type" not in c:
            issues.append(f"cue #{i} ignored (needs 't' and 'type'): {c!r}"[:200])
            continue
        typ = str(c["type"]).strip().lower().replace("-", "_").replace(" ", "_")
        try:
            t = float(c["t"])
        except (TypeError, ValueError):
            issues.append(f"cue #{i} ignored: bad t {c['t']!r}")
            continue
        c = dict(c, type=typ, t=t, index=i)
        if typ == "anchor":
            k = anchor_key(c)
            if k == "end":
                end_t = t
            elif k == "drop":
                plain_drops.append(t)
            elif k in ANCHOR_KEYS:
                if k in anchors and abs(anchors[k] - t) > 1e-6:
                    issues.append(f"anchor {k} given twice ({anchors[k]} and {t}); using the first")
                else:
                    anchors[k] = t
            else:
                issues.append(f"anchor cue #{i} at {t}: note {c.get('note')!r} not recognised")
            continue
        if typ == "reverseswell":
            typ = c["type"] = "reverse_swell"
        if typ not in SFX_TYPES:
            issues.append(f"cue #{i} at {t}: unknown type {typ!r}, skipped")
            continue
        items.append(c)
    for t in sorted(plain_drops):                    # an unnamed "drop" fills drop1, then drop2
        for k in ("drop1", "drop2"):
            if k not in anchors:
                anchors[k] = t
                break
    for k in ANCHOR_KEYS:
        if k not in anchors:
            anchors[k] = DEFAULT_ANCHORS[k]
            issues.append(f"anchor {k} missing from the cue sheet; using the default {DEFAULT_ANCHORS[k]}")
    a = [anchors[k] for k in ANCHOR_KEYS]
    if a != sorted(a):
        issues.append(f"anchors are out of order: {anchors}")
    items.sort(key=lambda c: (c["t"], c["index"]))
    return anchors, items, end_t, issues


# ------------------------------------------------------------------------------------------------
# music
# ------------------------------------------------------------------------------------------------
def render_music(style, duration, anchors, seed, cache_dir=None):
    mod = importlib.import_module(f"music_{style}")
    args = dict(duration=duration, seed=seed, **{k: anchors[k] for k in ANCHOR_KEYS})
    key = None
    if cache_dir:
        h = hashlib.sha256()
        with open(mod.__file__, "rb") as f:
            h.update(f.read())
        h.update(json.dumps(dict(style=style, **args), sort_keys=True).encode())
        key = os.path.join(cache_dir, f"{style}-{h.hexdigest()[:16]}")
        if os.path.exists(key + ".npz") and os.path.exists(key + ".json"):
            z = np.load(key + ".npz")
            st = {k: z[k] for k in ("drums", "bass", "music", "fx")}
            with open(key + ".json") as f:
                st["meta"] = json.load(f)
            log(f"music: {style} stems from cache {key}.npz")
            return st
    t0 = time.time()
    st = mod.render_story(**args)
    log(f"music: rendered {style} render_story({args}) in {time.time() - t0:.1f} s")
    if key:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(key + ".npz", **{k: st[k] for k in ("drums", "bass", "music", "fx")})
        with open(key + ".json", "w") as f:
            json.dump(st["meta"], f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        with open(key + ".json") as f:
            st["meta"] = json.load(f)               # same types as a cache hit
    return st


def fit_len(x, n):
    x = np.asarray(x, np.float64)
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros((n - len(x), x.shape[1]))])


class MusicLow:
    """Low-end presence of the music (drums + bass + fx below 150 Hz) around a time: decides
    whether a slam lands ON a music hit / kick (-> sfx.over_music_hit) or in a low-end gap."""

    def __init__(self, st, n):
        low = sum(fit_len(st[k], n) for k in ("drums", "bass", "fx"))
        low = signal.sosfiltfilt(butter("lowpass", 150.0, 4), low, axis=0)
        self.e = env_db(low, 0.002)
        self.ref = float(np.percentile(self.e, 99.5))

    def rel_db(self, t):
        i0, i1 = int(round((t - 0.005) * SR)), int(round((t + 0.06) * SR))
        seg = self.e[max(i0, 0):max(i1, 1)]
        return float(seg.max() - self.ref) if len(seg) else -120.0


def kick_onset(drums, t, pre=0.03, post=0.05):
    """Broadband onset (30 % of the local 1 ms-envelope peak) of the drums stem near t."""
    i0, i1 = int(round((t - pre) * SR)), int(round((t + post) * SR))
    i0 = max(i0, 0)
    seg = drums[i0:i1]
    if not len(seg):
        return None
    e = np.sqrt(10 ** (env_db(seg, 0.001) / 10))
    j = int(np.argmax(e > 0.3 * e.max()))
    return (i0 + j) / SR


# ------------------------------------------------------------------------------------------------
# SFX plan
# ------------------------------------------------------------------------------------------------
class Event:
    pass


def snap_pop_pitch(p):
    s = int(round(float(p)))
    base = sfx.note_midi(sfx.POP_BASE)
    if sfx.in_key(base + s) or not POP_IN_KEY:
        return s, None
    for d in (1, -1, 2, -2):
        if sfx.in_key(base + s + d):
            return s + d, f"pop pitch {s} ({sfx.midi_name(base + s)}) is outside F minor -> {s + d} ({sfx.midi_name(base + s + d)})"
    return s, None


def norm_direction(d):
    d = str(d or "lr").strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    return {"lr": "lr", "l2r": "lr", "lefttoright": "lr", "right": "lr", "rl": "rl", "r2l": "rl",
            "righttoleft": "rl", "left": "rl", "up": "up", "u": "up", "down": "down", "d": "down"}.get(d, "lr")


def norm_size(s):
    s = str(s or "medium").strip().lower()
    return {"big": "big", "large": "big", "l": "big", "huge": "big", "medium": "medium", "med": "medium",
            "m": "medium", "small": "small", "s": "small", "light": "small"}.get(s, "medium")


def plan_events(items, anchors, mlow, duration, whoosh_sync, issues):
    hero_t = [anchors[k] for k in HIT_ANCHORS]
    events = []
    for c in items:
        typ, t = c["type"], c["t"]
        note = str(c.get("note", ""))
        e = Event()
        e.cue, e.typ, e.t, e.note = c, typ, t, note
        e.flags, e.params = [], {}
        gain_key = typ
        light = note.strip().lower().startswith("(light)")
        dur = c.get("dur")
        if typ == "impact":
            size = norm_size(c.get("size"))
            e.params["size"] = size
            x = sfx.impact(size)
            gain_key = f"impact_{size}"
            e.sync_off, e.rule, e.detect = 0.0, "onset at t", "onset"
            rel = mlow.rel_db(t)
            on_anchor = any(abs(t - h) <= 0.010 for h in hero_t) or abs(t - anchors["breakdown"]) <= 0.010
            e.music_low_db = rel
            e.on_hit = on_anchor or rel >= LOW_PRESENT_DB
            if e.on_hit:
                x = sfx.over_music_hit(x)
                e.flags.append(f"over_music_hit (music LP150 {rel:+.1f} dB re its kick level)")
            else:
                e.flags.append(f"plain impact with sub (music LP150 {rel:+.1f} dB: low-end gap)")
            e.hero = size == "big" and any(abs(t - h) <= 0.010 for h in hero_t)
        elif typ == "reject":
            x = sfx.reject()
            e.sync_off, e.rule, e.detect = 0.0, "onset at t", "onset"
        elif typ == "check":
            x = sfx.check()
            e.sync_off, e.rule, e.detect = 0.0, "onset at t", "onset"
        elif typ == "tick":
            x = sfx.tick()
            e.sync_off, e.rule, e.detect = 0.0, "onset at t", "onset"
        elif typ == "pop":
            p, msg = snap_pop_pitch(c.get("pitch", 0))
            if msg:
                issues.append(f"cue #{c['index']} at {t}: {msg}")
                e.flags.append(msg)
            e.params["pitch"] = p
            x = sfx.pop(p)
            e.sync_off, e.rule, e.detect = 0.0, "onset at t", "onset"
        elif typ == "downlifter":
            d = float(dur) if dur is not None else 1.0
            e.params["dur"] = d
            x = sfx.downlifter(d)
            e.sync_off, e.rule, e.detect = 0.0, "onset at t (falls over dur)", "onset"
        elif typ == "slash":
            d = float(dur) if dur is not None else 0.35
            e.params["dur"] = d
            x = sfx.slash(d)
            e.sync_off, e.rule, e.detect = sfx.anchor("slash"), "cut (anchor 0.065 s) at t", "peak20"
        elif typ in ("scrub", "shimmer"):
            d = float(dur) if dur is not None else (0.5 if typ == "scrub" else 1.2)
            e.params["dur"] = d
            x = sfx.scrub(d) if typ == "scrub" else sfx.shimmer(d)
            e.sync_off, e.rule, e.detect = 0.0, "starts at t, lasts dur", "start"
        elif typ in ("riser", "reverse_swell"):
            m = re.search(r"hit\s*=\s*(\d+(?:\.\d+)?)", note, re.I)
            if dur is None:
                if m:
                    d = float(m.group(1)) - t
                else:
                    d = 1.5 if typ == "riser" else 0.8
                    issues.append(f"cue #{c['index']} {typ} at {t} has no dur; using {d} s ending at {t + d}")
            else:
                d = float(dur)
                if m and abs(t + d - float(m.group(1))) > 0.002:
                    issues.append(f"cue #{c['index']} {typ}: t + dur = {t + d:.3f} but the note says Hit = "
                                  f"{m.group(1)}; using t + dur")
            e.params["dur"] = d
            x = sfx.riser(d) if typ == "riser" else sfx.reverse_swell(d)
            e.sync_off = sfx.anchor(typ, dur=d)                       # = end of the array
            e.rule, e.detect = "t = start, ENDS on the hit at t + dur", "end"
            e.start_override = t
        elif typ == "whoosh":
            d = float(dur) if dur is not None else 0.45
            direction = norm_direction(c.get("direction"))
            inten = float(c.get("intensity", LIGHT_INTENSITY if light else 1.0))
            e.params.update(dur=d, direction=direction, intensity=inten)
            x = sfx.whoosh(d, direction, inten)
            e.readme_anchor = sfx.anchor("whoosh", dur=d, direction=direction, intensity=inten)
            e.sync_off = e.readme_anchor
            e.detect = "peak3"
            if whoosh_sync == "window":
                e.rule = "spans the motion window [t, t+dur]; envelope peak (-3 dB centre) mid-motion"
                e.start_override = t
            else:
                e.rule = "envelope peak at t"
        else:                                              # pragma: no cover (filtered in load_cues)
            continue
        e.x = np.asarray(x, np.float64)
        start_t = getattr(e, "start_override", t - e.sync_off)
        e.start = int(round(start_t * SR))
        e.start_t = e.start / SR
        e.end_t = (e.start + len(e.x)) / SR
        if e.detect == "peak3" and hasattr(e, "start_override"):
            # window mode: the expected peak is this render's own -3 dB peak-region centre (robust; the
            # README anchor is the argmax at the trailing edge of a ~50 ms plateau, 0-19 ms later)
            e.target = e.start_t + detect(e.x, 0.0, "peak3")
        else:
            e.target = e.start_t + e.sync_off
        # gain staging
        gdb = sfx.SUGGESTED_GAIN_DB[gain_key] + SFX_BUS_TRIM_DB + TYPE_TRIM_DB.get(typ, 0.0)
        if TYPE_TRIM_DB.get(typ):
            e.flags.append(f"{typ} trim {TYPE_TRIM_DB[typ]:+.1f} dB")
        if typ in LEN_COMP_TYPES and gain_key in sfx.ROLE_LUFS_M:
            dev = sfx.ROLE_LUFS_M[gain_key] - sfx.lufs_momentary_max(e.x) - sfx.SUGGESTED_GAIN_DB[gain_key]
            comp = float(np.clip(dev, -LEN_COMP_MAX_DB, 0.0))
            if comp < -0.25:
                gdb += comp
                e.flags.append(f"length comp {comp:+.1f} dB (to ROLE_LUFS_M)")
        if typ == "impact" and e.params["size"] == "big":
            if e.hero:
                gdb += HERO_SLAM_DB
                e.flags.append(f"hero slam {HERO_SLAM_DB:+.1f} dB")
            else:
                gdb += OTHER_BIG_DB
                e.flags.append(f"big slam off the anchors {OTHER_BIG_DB:+.1f} dB")
        if light:
            gdb += LIGHT_DB
            e.flags.append(f"(light) {LIGHT_DB:+.1f} dB")
        if typ in STACK_TYPES and any(abs(e.target - h) <= STACK_WINDOW_S for h in hero_t):
            gdb += STACK_ATTEN_DB
            e.flags.append(f"stacked on a hero slam {STACK_ATTEN_DB:+.1f} dB")
        e.gain_db = gdb
        e.send = SEND.get(typ, 0.0)
        if e.start < 0:
            issues.append(f"cue #{c['index']} {typ} at {t}: starts {e.start_t:.3f} s, the part before 0 is cut")
        if e.end_t > duration:
            issues.append(f"cue #{c['index']} {typ} at {t}: runs to {e.end_t:.3f} s, cut at {duration:.3f}")
        if not (0.0 <= e.target <= duration):
            issues.append(f"cue #{c['index']} {typ} at {t}: sync point {e.target:.3f} is outside the video; skipped")
            continue
        events.append(e)
    return events


# ------------------------------------------------------------------------------------------------
# buses
# ------------------------------------------------------------------------------------------------
_IR = None


def room_ir():
    global _IR
    if _IR is None:
        ir = sfx._make_ir(ROOM["rt"], ROOM["hf"], ROOM["er"])     # synthetic, unit energy per channel
        pre = np.zeros((int(ROOM["predelay"] * SR), 2))
        _IR = np.concatenate([pre, ir], axis=0)
    return _IR


def room(send):
    ir = room_ir()
    w = np.stack([signal.oaconvolve(send[:, ch], ir[:, ch])[: len(send)] for ch in range(2)], axis=1)
    w = signal.sosfilt(butter("highpass", ROOM["hp"], 2), w, axis=0)
    w = signal.sosfilt(butter("lowpass", ROOM["lp"], 2), w, axis=0)
    return w * g_db(ROOM["ret_db"])


def build_sfx_bus(events, n):
    dry = np.zeros((n, 2))
    snd = np.zeros((n, 2))
    for e in events:
        y = e.x * g_db(e.gain_db)
        add_at(dry, y, e.start)
        if e.send > 0:
            add_at(snd, y * e.send, e.start)
    wet = room(snd)
    pre = dry + wet
    det = signal.sosfilt(butter("highpass", GLUE["sc_hp"], 2), pre, axis=0)
    pk = float(np.max(uniform_filter1d(np.mean(det ** 2, axis=1), int(0.01 * SR), mode="constant")))
    pk = max(pk, 1e-12)
    thr = 10 * math.log10(pk + 1e-20) - GLUE["range_db"]
    glue, gr_max = compressor_gain(pre, thr, GLUE["ratio"], GLUE["attack"], GLUE["release"], GLUE["sc_hp"],
                                   knee_db=GLUE["knee_db"])
    return pre * glue[:, None], dict(dry=dry, wet=wet, glue=glue, glue_thr_db=thr, glue_gr_max_db=gr_max)


def soft_clip(x, ceiling, knee, os_=4):
    """Odd, C1-smooth soft clipper: identity below knee*ceiling, tanh shoulder approaching `ceiling`."""
    up = signal.resample_poly(x, os_, 1, axis=0)
    k = knee * ceiling
    a = np.abs(up)
    sh = k + (ceiling - k) * np.tanh((a - k) / (ceiling - k))
    up = np.where(a > k, np.sign(up) * sh, up)
    return signal.resample_poly(up, 1, os_, axis=0)[: len(x)]


def sfx_peak_stage(sfx_bus, music_bus):
    """Transient soft clip on the SFX bus, ceiling relative to the estimated master make-up.
    Returns the processed bus, the effective per-sample gain (for per-event analysis) and stats."""
    est = TARGET_LUFS - lufs(music_bus + sfx_bus)
    ceil_db = CEILING_DBTP - est + SFX_PEAK["rel_db"]
    c = g_db(ceil_db)
    k = SFX_PEAK["knee"] * c
    if np.abs(sfx_bus).max() <= k * 0.98:
        return sfx_bus, np.ones(len(sfx_bus)), dict(ceiling_dbfs=round(ceil_db, 2), makeup_estimate_db=round(est, 2),
                                                     peak_reduction_db=0.0, samples_shaped=0)
    y = soft_clip(sfx_bus, c, SFX_PEAK["knee"], SFX_PEAK["oversample"])
    touched = np.abs(sfx_bus).max(axis=1) > k
    # effective gain: |y|/|x| where shaped (used only to view each event's own share of the bus)
    num = np.sqrt(np.sum(y ** 2, axis=1))
    den = np.sqrt(np.sum(sfx_bus ** 2, axis=1))
    g = np.where(touched & (den > 1e-12), np.minimum(num / np.maximum(den, 1e-12), 1.0), 1.0)
    red = db(true_peak(sfx_bus)) - db(true_peak(y))
    return y, g, dict(ceiling_dbfs=round(ceil_db, 2), makeup_estimate_db=round(est, 2),
                      peak_reduction_db=round(red, 2), samples_shaped=int(touched.sum()),
                      time_shaped_s=round(float(touched.sum()) / SR, 4))


def event_in_bus(e, glue, n, tail_s=0.7):
    """This event's own contribution to the SFX bus (dry + its room send, times the bus glue gain)."""
    y = e.x * g_db(e.gain_db)
    a = max(e.start, 0)
    b = min(e.start + len(y) + int(tail_s * SR), n)
    seg = np.zeros((b - a, 2))
    add_at(seg, y, e.start - a)
    if e.send > 0:
        snd = np.zeros_like(seg)
        add_at(snd, y * e.send, e.start - a)
        seg += room(snd)
    return seg * glue[a:b, None], a / SR


def duck_curve(n, keys):
    """Gain (linear) from duck keys [(t, depth_db, attack, hold, release)], raised-cosine ramps in dB;
    overlapping keys take the deeper value."""
    d = np.zeros(n)
    for t, depth, att, hold, rel in keys:
        if depth <= 0:
            continue
        i0, i1 = int(round((t - att) * SR)), int(round(t * SR))
        i2, i3 = int(round((t + hold) * SR)), int(round((t + hold + rel) * SR))
        idx = np.arange(max(i0, 0), min(i3, n))
        if not len(idx):
            continue
        v = np.full(len(idx), float(depth))
        up = idx < i1
        v[up] = depth * np.sin(0.5 * np.pi * (idx[up] - i0) / max(i1 - i0, 1)) ** 2
        dn = idx >= i2
        v[dn] = depth * np.cos(0.5 * np.pi * (idx[dn] - i2) / max(i3 - i2, 1)) ** 2
        d[idx] = np.maximum(d[idx], v)
    return 10 ** (-d / 20.0), d


def build_music_bus(st, n, events, anchors, meta):
    stems = {k: fit_len(st[k], n) for k in ("drums", "bass", "music", "fx")}
    # 1. one source per transition: duck the music fx stem under duplicating SFX
    spans, ducked = [], []
    trans = meta.get("fx_transitions", []) or []
    hits = [anchors[k] for k in ANCHOR_KEYS]
    for e in events:
        if e.typ not in FX_DUP_KINDS:
            continue
        s, t_end = e.start_t, e.end_t
        if e.typ == "downlifter":
            for h in hits:                                  # keep the music boom on the hit whole
                if h - 0.02 <= s < h + 0.15:
                    s = h + 0.15
        over = [tr for tr in trans if tr.get("kind") in FX_DUP_KINDS[e.typ]
                and min(t_end, float(tr["t1"])) - max(s, float(tr["t0"])) > 0.05]
        if over and t_end - s > 0.05:
            spans.append((max(s, 0.0), min(t_end, n / SR)))
            ducked.append(dict(sfx=f"{e.typ}@{e.t}", span=[round(max(s, 0.0), 4), round(min(t_end, n / SR), 4)],
                               music_fx=[f"{tr['kind']} {tr['t0']}-{tr['t1']}" for tr in over]))
            e.flags.append(f"music fx stem ducked {sfx.MUSIC_FX_DUCK_DB:+.0f} dB over {s:.2f}-{t_end:.2f}")
    if spans:
        stems["fx"] = stems["fx"] * sfx.fx_duck_gain(n, spans).astype(np.float64)
    mus = stems["drums"] + stems["bass"] + stems["music"] + stems["fx"]
    raw_lufs = lufs(mus)
    mg = MUSIC_BUS_LUFS - raw_lufs
    mus = mus * g_db(mg)
    # 2. sidechain duck (two bands, zero-phase complementary split at 150 Hz)
    hi_keys, lo_keys = [], []
    for e in events:
        k = f"impact_{e.params['size']}" if e.typ == "impact" else e.typ
        if k not in DUCK:
            continue
        hi, lo_hit, lo_plain, att, hold, rel = DUCK[k]
        lo = lo_hit if getattr(e, "on_hit", False) else lo_plain
        hi_keys.append((e.target, hi, att, hold, rel))
        lo_keys.append((e.target, lo, att, hold, rel))
    g_hi, d_hi = duck_curve(n, hi_keys)
    g_lo, d_lo = duck_curve(n, lo_keys)
    low = signal.sosfiltfilt(butter("lowpass", DUCK_SPLIT_HZ, 2), mus, axis=0)
    bus = low * g_lo[:, None] + (mus - low) * g_hi[:, None]
    return bus, dict(raw_lufs=raw_lufs, gain_db=mg, fx_duck=ducked, duck_hi_db=d_hi, duck_lo_db=d_lo,
                     undocked=mus)


# ------------------------------------------------------------------------------------------------
# master
# ------------------------------------------------------------------------------------------------
def tp_curve(x, os_=4):
    """Per-sample true-peak envelope (max over channels and the os_ inter-sample positions)."""
    m = len(x)
    up = np.abs(signal.resample_poly(x, os_, 1, axis=0)).max(axis=1)
    up = np.concatenate([up, np.zeros(max(0, m * os_ - len(up)))])[: m * os_]
    return np.maximum(up.reshape(m, os_).max(axis=1), np.abs(x).max(axis=1))


def local_ceiling_dips(dec, limit_db, pad_s=AAC_FIX_PAD_S):
    """dB reduction curve that lowers the limiter ceiling only around the spots where the decoded AAC
    exceeds limit_db (each cluster by its own excess + 0.05 dB)."""
    n = len(dec)
    curve = 20 * np.log10(np.maximum(tp_curve(dec), 1e-12))
    idx = np.nonzero(curve > limit_db - 0.05)[0]
    dip = np.zeros(n)
    if not len(idx):
        return dip, []
    groups, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > int(0.1 * SR):
            groups.append((start, a))
            start = b
    groups.append((start, idx[-1]))
    spots = []
    pad, ramp = int(pad_s * SR), int(0.02 * SR)
    for g0, g1 in groups:
        ex = float(curve[g0:g1 + 1].max() - limit_db + 0.05)
        a, b = max(g0 - pad - ramp, 0), min(g1 + pad + ramp, n)
        w = np.ones(b - a)
        r = min(ramp, (b - a) // 2)
        w[:r] = np.sin(0.5 * np.pi * np.arange(r) / r) ** 2
        w[len(w) - r:] = w[:r][::-1]
        dip[a:b] = np.maximum(dip[a:b], ex * w)
        spots.append(dict(t=round(g0 / SR, 3), t_end=round(g1 / SR, 3), dip_db=round(ex, 2)))
    return dip, spots


def master(mix, ceiling=CEILING_DBTP):
    sos = np.stack([biquad(k, f, gdb, q) for k, f, gdb, q in EQ])
    y = signal.sosfilt(sos, mix, axis=0)
    thr = lufs(signal.sosfilt(butter("highpass", BUS_COMP["sc_hp"], 2), y, axis=0)) + BUS_COMP["thr_rel_db"]
    cg, comp_gr = compressor_gain(y, thr, BUS_COMP["ratio"], BUS_COMP["attack"], BUS_COMP["release"],
                                  BUS_COMP["sc_hp"])
    y = y * cg[:, None]
    fades = edge_fades(len(y), FADE_IN_S, FADE_OUT_S)
    gain = TARGET_LUFS - lufs(y)
    for it in range(6):
        z, lg = tp_limiter(y * g_db(gain), ceiling, LIMITER["lookahead"], LIMITER["release"],
                           LIMITER["oversample"])
        z = z * fades[:, None]
        L = lufs(z)
        if abs(L - TARGET_LUFS) < 0.03:
            break
        gain += TARGET_LUFS - L
    tp = true_peak(z)
    cmax = float(np.max(ceiling))
    if db(tp) > cmax + 0.2:                          # 8x estimate above the 4x-limited ceiling: trim
        z = z * g_db(cmax + 0.2 - db(tp))
    w = np.hanning(len(z)) * fades                  # exact DC removal with a zero-ended window
    w *= len(z) / w.sum()
    z = z - np.mean(z, axis=0)[None, :] * w[:, None]
    z[-1] = 0.0
    lim_gr = -20 * np.log10(np.maximum(lg, 1e-9))
    info = dict(ceiling_dbtp=round(cmax, 3), ceiling_min_dbtp=round(float(np.min(ceiling)), 3), eq=[list(e) for e in EQ], bus_comp_thr_db=round(thr, 2), bus_comp_gr_max_db=round(comp_gr, 2),
                makeup_gain_db=round(gain, 2), limiter_gr_max_db=round(float(lim_gr.max()), 2),
                limiter_time_over_1db_s=round(float(np.sum(lim_gr > 1.0)) / SR, 3),
                limiter_time_over_3db_s=round(float(np.sum(lim_gr > 3.0)) / SR, 3), iterations=it + 1,
                lufs_py=round(lufs(z), 3), true_peak_8x_dbtp=round(db(true_peak(z)), 3), comp_gain=cg, lim_gain=lg)
    return z.astype(np.float32), info


# ------------------------------------------------------------------------------------------------
# mux
# ------------------------------------------------------------------------------------------------
def mux(video, wav, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    tmp = out + ".partial.mp4"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video, "-i", wav,
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *AAC, "-map_metadata", "0",
           "-movflags", "+faststart", tmp]
    subprocess.run(cmd, check=True)
    os.replace(tmp, out)
    return cmd


def video_md5(path):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0", "-c", "copy", "-f", "md5", "-"],
                       capture_output=True, text=True, check=True)
    return p.stdout.strip().replace("MD5=", "")


def ebur128(path, af=None):
    flt = (af + "," if af else "") + "ebur128=peak=true:framelog=quiet"
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-filter_complex", flt, "-f", "null", "-"],
                       capture_output=True, text=True)
    txt = p.stderr[p.stderr.rfind("Summary:"):]

    def grab(pat):
        m = re.search(pat, txt, re.S)
        return float(m.group(1)) if m and m.group(1) not in ("-inf", "inf") else None
    return dict(I=grab(r"I:\s+(-?[\d.]+|-inf) LUFS"), LRA=grab(r"LRA:\s+(-?[\d.]+) LU"),
                LRA_low=grab(r"LRA low:\s+(-?[\d.]+|-inf) LUFS"), LRA_high=grab(r"LRA high:\s+(-?[\d.]+|-inf) LUFS"),
                TP=grab(r"True peak:\s+Peak:\s+(-?[\d.]+|-inf) dBFS"), summary=txt.strip())


def astats(path):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af",
                        "astats=measure_perchannel=DC_offset+Peak_level+Flat_factor+Peak_count:measure_overall="
                        "DC_offset+Peak_level+RMS_level+Number_of_samples", "-f", "null", "-"],
                       capture_output=True, text=True)
    lines = [ln.split("] ", 1)[-1] for ln in p.stderr.splitlines() if "Parsed_astats" in ln]
    return "\n".join(lines)


def decode_audio(path, n):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-f", "f32le", "-ac", "2",
                        "-ar", str(SR), "-"], capture_output=True, check=True)
    a = np.frombuffer(p.stdout, dtype="<f4").reshape(-1, 2).astype(np.float64)
    return a


# ------------------------------------------------------------------------------------------------
# detection / report
# ------------------------------------------------------------------------------------------------
def detect(seg, t0, mode):
    if mode == "onset":
        e = env_db(seg, 0.001)
        i = int(np.argmax(e >= e.max() - 20.0))
    elif mode == "start":
        e = env_db(seg, 0.005)
        i = int(np.argmax(e >= e.max() - 30.0))
    elif mode == "peak20":
        e = env_db(seg, 0.020)
        i = int(np.argmax(e))
    elif mode == "peak3":                      # centre of the contiguous -3 dB region around the 20 ms peak
        e = env_db(seg, 0.020)
        k = int(np.argmax(e))
        above = e >= e[k] - 3.0
        lo = k - int(np.argmin(above[k::-1])) + 1 if not above[:k + 1].all() else 0
        hi = k + int(np.argmin(above[k:])) - 1 if not above[k:].all() else len(e) - 1
        i = (lo + hi) // 2
    elif mode == "end":
        e = env_db(seg, 0.001)
        i = int(np.nonzero(e >= e.max() - 20.0)[0][-1]) + 1
    else:
        raise ValueError(mode)
    return t0 + i / SR


def bus_onset(bus, target, win=0.04):
    """Strongest level rise in the summed SFX bus within +-win of target: (time, rise dB)."""
    a = max(int((target - win - 0.02) * SR), 0)
    b = min(int((target + win + 0.005) * SR), len(bus))
    e = env_db(bus[a:b], 0.001)
    best, bt = -1e9, None
    lo, hi = int(0.012 * SR), int(0.003 * SR)
    for j in range(int(0.02 * SR), len(e), 12):
        if j - lo < 0:
            continue
        r = e[j] - e[j - lo:j - hi].max()
        if r > best:
            best, bt = r, j
    if bt is None:
        return None, None
    # refine: first sample in the 0.25 ms step where the rise exceeds half of the best rise
    j0 = max(bt - 12, lo)
    for j in range(j0, bt + 1):
        if e[j] - e[j - lo:j - hi].max() >= 0.5 * best:
            bt = j
            break
    return (a + bt) / SR, float(best)


def section_table(x, meta, n):
    st = meta.get("section_times", {})
    d1, bd, d2, hit = (float(st.get(k, DEFAULT_ANCHORS[k])) for k in ANCHOR_KEYS)
    dur = n / SR
    spans = [("intro", 0.0, d1), ("grooveA", d1, bd), ("breakdown", bd, d2), ("drop2", d2, hit), ("end", hit, dur)]
    if d2 - bd >= 4.0 - 1e-6:                       # last bar of the breakdown is the build, the rest is dark
        spans[3:3] = [(" bd.dark", bd, d2 - 2.0), (" bd.build", d2 - 2.0, d2)]
    hp = signal.sosfilt(butter("highpass", 150.0, 4), x, axis=0)
    rows = []
    for name, a, b in spans:
        i, j = int(a * SR), int(b * SR)
        if j - i < int(0.4 * SR):
            continue
        rows.append(dict(section=name, t0=a, t1=b, lufs=round(lufs(x[i:j]), 2), lufs_hp150=round(lufs(hp[i:j]), 2)))
    return rows


def draw_reports(rep_dir, y, sfx_bus, music_bus, events, anchors, meta, master_gain_db):
    from PIL import Image, ImageDraw, ImageFont
    n = len(y)
    dur = n / SR
    W, LM, RM, TM = 2400, 70, 20, 96
    pw = W - LM - RM

    def font(sz):
        try:
            return ImageFont.load_default(size=sz)
        except TypeError:
            return ImageFont.load_default()
    f_s, f_m = font(13), font(16)

    def xpos(t):
        return LM + int(round(t / dur * (pw - 1)))
    colors = dict(impact=(255, 80, 60), reject=(255, 40, 160), check=(80, 230, 120), pop=(90, 200, 255),
                  tick=(200, 200, 200), whoosh=(170, 140, 255), slash=(255, 170, 40), scrub=(140, 220, 220),
                  shimmer=(255, 240, 120), riser=(255, 120, 200), reverse_swell=(255, 120, 200),
                  downlifter=(120, 160, 255))
    abbrev = dict(impact="IMP", reject="REJ", check="CHK", pop="POP", tick="TCK", whoosh="WSH", slash="SLS",
                  scrub="SCR", shimmer="SHM", riser="RIS", reverse_swell="RVS", downlifter="DWN")

    def markers(img, top, bottom):
        dr = ImageDraw.Draw(img, "RGBA")
        lanes = [0] * 4
        for e in sorted(events, key=lambda e: e.target):
            c = colors.get(e.typ, (255, 255, 255))
            x0, x1, xt = xpos(max(e.start_t, 0)), xpos(min(e.end_t, dur)), xpos(e.target)
            lane = min(range(4), key=lambda k: lanes[k])
            for k in range(4):
                if lanes[k] <= xt - 2:
                    lane = k
                    break
            ly = 8 + lane * 20
            label = abbrev.get(e.typ, e.typ[:3].upper())
            if e.typ == "impact":
                label += e.params["size"][0].upper()
            label += f" {e.t:g}"
            tw = dr.textlength(label, font=f_s)
            lanes[lane] = xt + tw + 6
            dr.rectangle([x0, top + 2, max(x1, x0 + 1), top + 6], fill=c + (150,))
            dr.line([xt, ly + 16, xt, bottom], fill=c + (170,), width=1)
            dr.text((xt + 2, ly), label, fill=c + (255,), font=f_s)
        for k in ANCHOR_KEYS:
            xa = xpos(anchors[k])
            for yy in range(top, bottom, 8):
                dr.line([xa, yy, xa, min(yy + 4, bottom)], fill=(255, 255, 255, 230), width=2)
            dr.text((xa + 3, top + 8), k.upper().replace("_", " ") + f" {anchors[k]:g}", fill=(255, 255, 255, 255),
                    font=f_m)

    def time_axis(dr, y0):
        for s in range(0, int(dur) + 1):
            xx = xpos(s)
            dr.line([xx, y0, xx, y0 + (8 if s % 2 == 0 else 4)], fill=(200, 200, 200), width=1)
            if s % 2 == 0:
                dr.text((xx - 6, y0 + 10), f"{s}s", fill=(200, 200, 200), font=f_s)

    # ---- spectrogram (own STFT, log frequency 30 Hz - 20 kHz) ----
    H = 640
    mono = np.mean(np.asarray(y, float), axis=1)
    nfft, hop = 4096, max(1, n // pw)
    win = np.hanning(nfft)
    pad = np.concatenate([np.zeros(nfft // 2), mono, np.zeros(nfft)])
    cols = []
    scale = 2.0 / win.sum()
    for c0 in range(0, pw * hop, hop * 256):
        idx = (np.arange(c0, min(c0 + hop * 256, pw * hop), hop)[:, None] + np.arange(nfft)[None, :])
        fr = pad[idx] * win[None, :]
        cols.append(20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)) * scale + 1e-10))
    S = np.concatenate(cols, axis=0)[:pw]                         # (pw, nfft/2+1)
    fbin = SR / nfft
    rows_f = np.geomspace(20000.0, 30.0, H)
    bi = rows_f / fbin
    i0 = np.floor(bi).astype(int)
    fr_ = bi - i0
    img_db = S[:, i0] * (1 - fr_) + S[:, np.minimum(i0 + 1, S.shape[1] - 1)] * fr_
    # max-pool the bins that fall between rows at high frequency
    edges = np.geomspace(20000.0, 30.0, H + 1) / fbin
    for r in range(H):
        lo_b, hi_b = int(math.ceil(edges[r + 1])), int(math.floor(edges[r]))
        if hi_b - lo_b >= 1:
            img_db[:, r] = S[:, lo_b:hi_b + 1].max(axis=1)
    v = np.clip((img_db.T + 100.0) / 95.0, 0, 1)                 # -100 .. -5 dBFS
    stops = np.array([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21],
                      [250, 193, 39], [252, 255, 164]], float)
    pos = np.linspace(0, 1, len(stops))
    rgb = np.stack([np.interp(v, pos, stops[:, k]) for k in range(3)], axis=-1).astype(np.uint8)
    img = Image.new("RGB", (W, H + TM + 40), (12, 12, 20))
    img.paste(Image.fromarray(rgb), (LM, TM))
    dr = ImageDraw.Draw(img)
    for f in (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000):
        r = int(round((math.log(20000.0) - math.log(f)) / (math.log(20000.0) - math.log(30.0)) * (H - 1)))
        dr.line([LM - 6, TM + r, LM, TM + r], fill=(200, 200, 200))
        dr.text((4, TM + r - 7), f"{f // 1000}k" if f >= 1000 else str(f), fill=(200, 200, 200), font=f_s)
    time_axis(dr, TM + H + 2)
    markers(img, TM, TM + H)
    dr.text((LM, H + TM + 24), "master spectrogram, log frequency 30 Hz-20 kHz, -100..-5 dBFS (inferno). "
            "Dashed white = music anchors; coloured = SFX sync points, bars = SFX array spans.",
            fill=(220, 220, 220), font=f_s)
    spec_png = os.path.join(rep_dir, "master_spectrogram.png")
    img.save(spec_png)

    # ---- waveform + loudness panel ----
    WH, LH = 360, 260
    img = Image.new("RGB", (W, TM + WH + 30 + LH + 60), (12, 12, 20))
    dr = ImageDraw.Draw(img, "RGBA")
    mid = TM + WH // 2
    a = np.asarray(y, float)
    cols = np.array_split(np.arange(n), pw)
    mx = np.array([a[c].max() for c in cols])
    mn = np.array([a[c].min() for c in cols])
    sb = np.array([np.sqrt(np.mean(sfx_bus[c] ** 2)) for c in cols]) * g_db(master_gain_db)
    for i in range(pw):
        dr.line([LM + i, mid - mx[i] * WH / 2, LM + i, mid - mn[i] * WH / 2], fill=(110, 170, 255))
    pts = [(LM + i, mid - min(sb[i], 1.0) * WH / 2) for i in range(pw)]
    dr.line(pts, fill=(255, 150, 40, 230), width=2)
    for lv in (1.0, g_db(-1.0)):
        for sgn in (1, -1):
            yy = mid - sgn * lv * WH / 2
            dr.line([LM, yy, LM + pw, yy], fill=(255, 255, 255, 60 if lv == 1.0 else 140), width=1)
    dr.line([LM, mid, LM + pw, mid], fill=(255, 255, 255, 60))
    dr.text((4, mid - 7), "0", fill=(200, 200, 200), font=f_s)
    dr.text((4, TM - 7), "+1", fill=(200, 200, 200), font=f_s)
    dr.text((4, TM + WH - 7), "-1", fill=(200, 200, 200), font=f_s)
    # loudness panel
    ly0 = TM + WH + 30
    tm, lm = loudness_curve(a, 0.4, 0.01)
    ts, ls = loudness_curve(a, 3.0, 0.05)
    lo_db, hi_db = -40.0, -4.0

    def ly(v):
        return ly0 + (hi_db - np.clip(v, lo_db, hi_db)) / (hi_db - lo_db) * LH
    for v in (-10, -14, -20, -30, -40):
        dr.line([LM, ly(v), LM + pw, ly(v)], fill=(255, 255, 255, 140 if v == -14 else 40))
        dr.text((4, ly(v) - 7), f"{v}", fill=(200, 200, 200), font=f_s)
    dr.line([(xpos(t), ly(v)) for t, v in zip(tm, lm) if t <= dur], fill=(120, 220, 140, 220), width=1)
    dr.line([(xpos(t), ly(v)) for t, v in zip(ts, ls) if t <= dur], fill=(255, 255, 255, 230), width=2)
    time_axis(dr, ly0 + LH + 2)
    markers(img, TM, ly0 + LH)
    dr.text((LM, ly0 + LH + 30), "top: master waveform (blue, dashed +-1 dBFS lines) with the SFX bus RMS at master "
            "gain (orange). bottom: momentary (green) and short-term (white) loudness, LUFS; -14 line highlighted.",
            fill=(220, 220, 220), font=f_s)
    wave_png = os.path.join(rep_dir, "master_waveform.png")
    img.save(wave_png)
    return spec_png, wave_png


def run(args):
    t_start = time.time()
    issues = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        vid = probe_video(args.video)
        D = vid["duration"]
        n = int(round(D * SR))
        anchors, items, end_t, cue_issues = load_cues(args.cues)
        issues += cue_issues
        if end_t is not None and abs(end_t - D) > 0.5 / vid["fps"]:
            issues.append(f"cue sheet 'end' anchor {end_t} differs from the video duration {D:.3f}")
        log(f"video {D:.3f} s ({vid['nb_frames']} frames @ {vid['fps']:g} fps) -> {n} samples; anchors {anchors}; "
            f"{len(items)} SFX cues; music {args.music}")
        st = render_music(args.music, D, anchors, args.seed, args.cache_dir)
        meta = st["meta"]
        mlow = MusicLow(st, n)
        events = plan_events(items, anchors, mlow, D, args.whoosh_sync, issues)
        music_bus, minfo = build_music_bus(st, n, events, anchors, meta)
        sfx_bus, sinfo = build_sfx_bus(events, n)
        sfx_bus, sinfo["peak_gain"], sinfo["peak"] = sfx_peak_stage(sfx_bus, music_bus)
        sinfo["glue"] = sinfo["glue"] * sinfo["peak_gain"]          # per-event view = glue x transient stage
        mix = music_bus + sfx_bus
        caught = [f"{w.category.__name__}: {w.message}" for w in wlist]
    issues += caught
    for msg in issues:
        log("note:", msg)

    tmpdir = tempfile.mkdtemp(prefix="mix-")
    try:
        wav = args.wav or os.path.join(tmpdir, "mix.wav")
        if args.wav:
            os.makedirs(os.path.dirname(os.path.abspath(wav)) or ".", exist_ok=True)
        # master -> WAV -> mux, then measure the decoded AAC; if its 8x true peak is above the spec minus a
        # margin (AAC overshoots the PCM by ~0.3-0.6 dB), lower the limiter ceiling by the excess and redo
        limit = SPEC["tp_max"] - AAC_TP_MARGIN
        dips, passes = np.zeros(n), []
        for attempt in range(MAX_MASTER_PASSES):
            ceiling = CEILING_DBTP - dips
            y, mst = master(mix, ceiling)
            if not np.isfinite(y).all():
                raise RuntimeError("non-finite samples in the master; nothing written")
            sf.write(wav, y, SR, subtype="FLOAT")
            cmd = mux(args.video, wav, args.out)
            dec = decode_audio(args.out, n)[:n]
            tp_aac = db(true_peak(dec))
            p = dict(ceiling_dbtp=CEILING_DBTP, ceiling_min_dbtp=round(float(ceiling.min()), 3),
                     master_tp_8x=mst["true_peak_8x_dbtp"], aac_tp_8x=round(tp_aac, 3), lufs=mst["lufs_py"])
            passes.append(p)
            if tp_aac <= limit:
                break
            dip, spots = local_ceiling_dips(dec, limit)
            if attempt >= AAC_LOCAL_ROUNDS:                   # local fixes can move an overshoot elsewhere
                spots = []
            if not spots:                                     # (also when the 4x curve misses an 8x overshoot)
                dip = np.full(n, tp_aac - limit + 0.05)
                spots = [dict(t="all", dip_db=round(tp_aac - limit + 0.05, 2))]
            p["aac_overshoot_spots"] = spots
            dips = dips + dip
            log(f"decoded AAC true peak {tp_aac:.2f} dBTP > {limit:.2f}: lowering the limiter ceiling "
                + ("everywhere by " + f"{spots[0]['dip_db']:.2f} dB" if spots[0]["t"] == "all" else
                   "locally: " + ", ".join(f"{sp['t']} s by {sp['dip_db']:.2f} dB" for sp in spots)))
        mst["passes"] = passes
        log(f"wrote {args.out} (master {mst['lufs_py']:.2f} LUFS, {mst['true_peak_8x_dbtp']:.2f} dBTP, decoded AAC "
            f"{passes[-1]['aac_tp_8x']:.2f} dBTP) in {time.time() - t_start:.1f} s")
        if not args.report:
            return 0
        return report(args, vid, D, n, anchors, events, st, meta, music_bus, sfx_bus, sinfo, minfo, y, mst,
                      wav, cmd, issues, tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def report(args, vid, D, n, anchors, events, st, meta, music_bus, sfx_bus, sinfo, minfo, y, mst, wav, cmd,
           issues, tmpdir):
    rep = args.report
    os.makedirs(rep, exist_ok=True)
    checks = {}
    tol = SPEC["frame_tol_s"]
    gm = g_db(mst["makeup_gain_db"])

    # ---- cue alignment ----
    rows = []
    iso = {}
    for e in events:
        seg, t0 = event_in_bus(e, sinfo["glue"], n)
        iso[id(e)] = (seg, t0)
        det = detect(seg, t0, e.detect)
        err = (det - e.target) * 1000
        bt, rise = (bus_onset(sfx_bus, e.target) if e.detect == "onset" else (None, None))
        rows.append(dict(
            cue_index=e.cue["index"], cue_t=f"{e.t:.3f}", type=e.typ,
            params=" ".join(f"{k}={v}" for k, v in e.params.items()),
            sync_rule=e.rule, array_start_t=f"{e.start_t:.4f}", array_end_t=f"{e.end_t:.4f}",
            target_t=f"{e.target:.4f}", detector=e.detect,
            readme_anchor_t=f"{e.start_t + e.readme_anchor:.4f}" if hasattr(e, "readme_anchor") else "", detected_t=f"{det:.4f}", error_ms=f"{err:+.2f}",
            within_1_frame="PASS" if abs(err) <= tol * 1000 else "FAIL",
            error_vs_cue_t_ms=f"{(det - e.t) * 1000:+.2f}",
            bus_onset_t=f"{bt:.4f}" if bt is not None else "",
            bus_onset_error_ms=f"{(bt - e.target) * 1000:+.2f}" if bt is not None else "",
            bus_onset_rise_db=f"{rise:.1f}" if rise is not None else "",
            gain_db=f"{e.gain_db:+.1f}", send=f"{e.send:.2f}", processing="; ".join(e.flags)))
    with open(os.path.join(rep, "cue_alignment.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["cue_t"])
        w.writeheader()
        w.writerows(rows)
    errs = [abs(float(r["error_ms"])) for r in rows]
    checks["sfx_within_1_frame"] = dict(ok=all(r["within_1_frame"] == "PASS" for r in rows), n=len(rows),
                                       max_abs_error_ms=round(max(errs), 2) if errs else None)

    # ---- anchors ----
    drums = fit_len(st["drums"], n)
    db_set = [float(b) for b in meta.get("downbeat_times", [])]
    sect = meta.get("section_times", {})
    arows = []
    for k in ANCHOR_KEYS:
        t = anchors[k]
        mt = float(sect.get(k, float("nan")))
        on = kick_onset(drums, mt if np.isfinite(mt) else t)
        arows.append(dict(anchor=k, cue_t=f"{t:.3f}", music_section_t=f"{mt:.3f}",
                          music_vs_cue_ms=f"{(mt - t) * 1000:+.2f}",
                          is_music_downbeat=any(abs(b - t) < 1e-6 for b in db_set),
                          drums_onset_t=f"{on:.4f}" if on is not None else "",
                          drums_onset_vs_cue_ms=f"{(on - t) * 1000:+.2f}" if on is not None else ""))
    with open(os.path.join(rep, "anchor_alignment.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(arows[0].keys()))
        w.writeheader()
        w.writerows(arows)
    checks["anchors_on_music_downbeats"] = dict(
        ok=all(r["is_music_downbeat"] and abs(float(r["music_vs_cue_ms"])) <= 1.0
               and r["drums_onset_vs_cue_ms"] != "" and abs(float(r["drums_onset_vs_cue_ms"])) <= tol * 1000
               for r in arows),
        detail={r["anchor"]: f"music {r['music_section_t']} downbeat={r['is_music_downbeat']} kick onset "
                             f"{r['drums_onset_vs_cue_ms']} ms" for r in arows})

    # ---- loudness / peaks ----
    hp_wav = os.path.join(tmpdir, "master_hp150.wav")
    sf.write(hp_wav, signal.sosfilt(butter("highpass", 150.0, 4), y.astype(np.float64), axis=0).astype(np.float32),
             SR, subtype="FLOAT")
    eb_m = ebur128(wav)
    eb_hp = ebur128(hp_wav)
    eb_mp4 = ebur128(args.out)
    ast = astats(wav)
    yd = y.astype(np.float64)
    first_frame = yd[: int(round(SR / vid["fps"]))]
    ff_rms = db(np.sqrt(np.mean(first_frame ** 2)))
    tm, lm = loudness_curve(yd, 0.4, 0.01)
    m_first = float(lm[int(0.2 / 0.01)])                           # momentary over 0-400 ms
    last10 = db(np.sqrt(np.mean(yd[-int(0.01 * SR):] ** 2)))
    fo = yd[-int(FADE_OUT_S * SR):]
    q = np.array_split(fo, 3)
    fo_db = [round(db(np.sqrt(np.mean(c ** 2))), 1) for c in q]
    dc = np.abs(np.mean(yd, axis=0)).max()
    clip_n = int(np.sum(np.abs(yd) >= 0.999))
    checks["integrated_loudness"] = dict(ok=eb_m["I"] is not None and SPEC["lufs"][0] <= eb_m["I"] <= SPEC["lufs"][1]
                                         and eb_mp4["I"] is not None and SPEC["lufs"][0] <= eb_mp4["I"] <= SPEC["lufs"][1],
                                         master_wav=eb_m["I"], mp4=eb_mp4["I"], target=TARGET_LUFS)
    tp8 = db(true_peak(yd))
    dec = decode_audio(args.out, n)
    tp8_aac = db(true_peak(dec[:n]))
    checks["true_peak"] = dict(ok=eb_m["TP"] is not None and eb_m["TP"] <= SPEC["tp_max"] and tp8 <= SPEC["tp_max"]
                               and eb_mp4["TP"] is not None and eb_mp4["TP"] <= SPEC["tp_max"]
                               and tp8_aac <= SPEC["tp_max"],
                               master_wav_ebur128=eb_m["TP"], master_wav_8x=round(tp8, 2), mp4_ebur128=eb_mp4["TP"],
                               mp4_decoded_8x=round(tp8_aac, 2))
    checks["lra"] = dict(ok=eb_m["LRA"] is not None and eb_m["LRA"] <= SPEC["lra_max"], master_wav=eb_m["LRA"],
                         mp4=eb_mp4["LRA"])
    checks["no_clipping_no_dc"] = dict(ok=clip_n == 0 and dc < SPEC["dc_max"] and bool(np.isfinite(yd).all()),
                                       samples_ge_0dBFS=clip_n, sample_peak_dbfs=round(db(np.abs(yd).max()), 2),
                                       dc=float(dc))
    checks["audible_from_frame0"] = dict(ok=ff_rms > -45.0 and m_first > -40.0,
                                         frame0_rms_dbfs=round(ff_rms, 1), momentary_0_400ms_lufs=round(m_first, 1),
                                         first_sample=float(yd[0, 0]))
    checks["fade_to_silence"] = dict(ok=float(np.abs(yd[-1]).max()) == 0.0 and last10 < -60.0
                                     and fo_db[0] > fo_db[1] > fo_db[2],
                                     last_sample=float(np.abs(yd[-1]).max()), last_10ms_rms_dbfs=round(last10, 1),
                                     fade_thirds_rms_dbfs=fo_db)
    # SFX hierarchy: hero slams loudest (full band and phone model)
    hp_sos = butter("highpass", 150.0, 4)
    lv = []
    for e in events:
        seg, _ = iso[id(e)]
        segm = seg * gm
        lv.append((e, sfx.lufs_momentary_max(segm), sfx.lufs_momentary_max(signal.sosfilt(hp_sos, segm, axis=0))))
    heroes = [r for r in lv if getattr(r[0], "hero", False)]
    others = [r for r in lv if not getattr(r[0], "hero", False)]
    if heroes and others:
        hmin = (min(r[1] for r in heroes), min(r[2] for r in heroes))
        omax = max(others, key=lambda r: r[1])
        omax_hp = max(others, key=lambda r: r[2])
        checks["hero_slams_loudest_sfx"] = dict(
            ok=hmin[0] > omax[1] and hmin[1] > omax_hp[2],
            hero_min_lufs_m=round(hmin[0], 1), hero_min_lufs_m_hp150=round(hmin[1], 1),
            loudest_other=f"{omax[0].typ}@{omax[0].t} {omax[1]:.1f} LUFS-M",
            loudest_other_hp150=f"{omax_hp[0].typ}@{omax_hp[0].t} {omax_hp[2]:.1f} LUFS-M")
    # SFX over music at the keyed moments
    bal = []
    mb = music_bus * gm
    for e, l_full, l_hp in lv:
        a, b = max(int((e.target - 0.05) * SR), 0), min(int((e.target + 0.35) * SR), n)
        if b - a < int(0.4 * SR):
            a = max(b - int(0.4 * SR), 0)
        lmus = sfx.lufs_momentary_max(mb[a:b]) if b - a >= int(0.4 * SR) else float("nan")
        bal.append((e, l_full, l_hp, lmus))

    # ---- video / container ----
    md5_in, md5_out = video_md5(args.video), video_md5(args.out)
    pin, pout = ffprobe_json(args.video), ffprobe_json(args.out)
    with open(os.path.join(rep, "ffprobe_output.json"), "w") as f:
        json.dump(pout, f, indent=1)
    vin = [s for s in pin["streams"] if s["codec_type"] == "video"][0]
    vout = [s for s in pout["streams"] if s["codec_type"] == "video"][0]
    aout = [s for s in pout["streams"] if s["codec_type"] == "audio"]
    keys = ("codec_name", "profile", "width", "height", "pix_fmt", "r_frame_rate", "avg_frame_rate", "time_base",
            "duration_ts", "nb_frames", "bit_rate")
    vdiff = {k: (vin.get(k), vout.get(k)) for k in keys if vin.get(k) != vout.get(k)}
    with open(os.path.join(rep, "video_md5.txt"), "w") as f:
        f.write(f"input  {args.video}\n  video stream md5 (ffmpeg -map 0:v -c copy -f md5) {md5_in}\n")
        f.write(f"output {args.out}\n  video stream md5 (ffmpeg -map 0:v -c copy -f md5) {md5_out}\n")
        f.write(f"identical: {md5_in == md5_out}\nffprobe video-stream fields that differ: {vdiff or 'none'}\n")
    checks["video_stream_bit_identical"] = dict(ok=md5_in == md5_out and not vdiff, md5_in=md5_in, md5_out=md5_out)
    a0 = aout[0] if aout else {}
    fdur = float(pout["format"]["duration"])
    adur = float(a0.get("duration", "nan"))
    vdur = float(vout.get("duration", "nan"))
    checks["durations_equal"] = dict(
        ok=abs(fdur - D) < 1e-3 and abs(adur - D) < 1e-3 and abs(vdur - D) < 1e-3 and len(y) == n,
        video_in=D, video_out=vdur, audio_out=adur, container=fdur, wav_samples=len(y), expected_samples=n)
    checks["audio_format"] = dict(
        ok=a0.get("codec_name") == "aac" and a0.get("profile") == "LC" and int(a0.get("sample_rate", 0)) == SR
        and int(a0.get("channels", 0)) == 2 and abs(int(a0.get("bit_rate", 0)) - 256000) < 20000
        and "faststart" in " ".join(cmd),
        codec=a0.get("codec_name"), profile=a0.get("profile"), sample_rate=a0.get("sample_rate"),
        channels=a0.get("channels"), bit_rate=a0.get("bit_rate"))
    with open(args.out, "rb") as f:
        head = f.read(1 << 16)
    moov, mdat = head.find(b"moov"), head.find(b"mdat")
    checks["faststart_moov_first"] = dict(ok=moov != -1 and (mdat == -1 or moov < mdat), moov_offset=moov,
                                          mdat_offset=mdat)
    seg = dec[:n]
    lags = np.arange(-2048, 2049)
    ref = yd[int(2 * SR):int(6 * SR), 0]
    cc = [float(np.dot(ref, dec[int(2 * SR) + l:int(6 * SR) + l, 0])) for l in lags]
    lag = int(lags[int(np.argmax(cc))])
    resid = db(np.sqrt(np.mean((seg[:len(yd)] - yd[:len(seg)]) ** 2)) / (np.sqrt(np.mean(yd ** 2)) + 1e-20))
    checks["mp4_audio_in_sync"] = dict(ok=lag == 0 and len(dec) >= n, lag_samples=lag, decoded_samples=len(dec),
                                       aac_residual_db=round(resid, 1))

    # ---- determinism signature ----
    sig = hashlib.sha256(y.tobytes()).hexdigest()

    # ---- loudness.txt ----
    sec = section_table(yd, meta, n)
    with open(os.path.join(rep, "loudness.txt"), "w") as f:
        f.write(f"mix.py report - {os.path.basename(args.out)}  (music: {args.music}, seed {args.seed})\n")
        f.write(f"cues {args.cues}\nvideo {args.video}\n\n")
        f.write(f"== ffmpeg ebur128, master WAV ==\n{eb_m['summary']}\n\n")
        f.write(f"== ffmpeg ebur128, master through the phone model (4th-order 150 Hz high-pass) ==\n{eb_hp['summary']}\n\n")
        f.write(f"== ffmpeg ebur128, output MP4 (decoded AAC) ==\n{eb_mp4['summary']}\n\n")
        f.write(f"== ffmpeg astats, master WAV ==\n{ast}\n\n")
        f.write("== python checks ==\n")
        f.write(f"8x true peak {tp8:.2f} dBTP, integrated (python BS.1770) {mst['lufs_py']:.2f} LUFS, |DC| {dc:.2e}, "
                f"samples >= -0.01 dBFS: {clip_n}\n")
        f.write(f"frame 0 RMS {ff_rms:.1f} dBFS, momentary 0-400 ms {m_first:.1f} LUFS, last 10 ms {last10:.1f} dBFS, "
                f"fade-out thirds {fo_db} dBFS, last sample {float(np.abs(yd[-1]).max())}\n\n")
        f.write("== master chain ==\n")
        f.write(f"music bus: stems {minfo['raw_lufs']:.2f} LUFS -> {MUSIC_BUS_LUFS} LUFS ({minfo['gain_db']:+.2f} dB); "
                f"sidechain duck max hi {minfo['duck_hi_db'].max():.1f} dB / lo {minfo['duck_lo_db'].max():.1f} dB\n")
        for d in minfo["fx_duck"]:
            f.write(f"  fx stem {sfx.MUSIC_FX_DUCK_DB:+.0f} dB under {d['sfx']} span {d['span']} (music {', '.join(d['music_fx'])})\n")
        f.write(f"SFX bus: glue threshold {sinfo['glue_thr_db']:.1f} dB, max GR {sinfo['glue_gr_max_db']:.2f} dB, "
                f"room rt {ROOM['rt']} s return {ROOM['ret_db']} dB; transient stage ceiling "
                f"{sinfo['peak']['ceiling_dbfs']} dBFS soft clip (make-up estimate {sinfo['peak']['makeup_estimate_db']:+.2f} dB), "
                f"bus true-peak reduction {sinfo['peak']['peak_reduction_db']} dB, "
                f"{sinfo['peak']['samples_shaped']} samples above the knee\n")
        f.write(f"master: EQ {EQ}; bus comp threshold {mst['bus_comp_thr_db']} dB max GR {mst['bus_comp_gr_max_db']} dB; "
                f"make-up {mst['makeup_gain_db']:+.2f} dB; limiter ceiling {mst['ceiling_dbtp']} dBTP, max GR "
                f"{mst['limiter_gr_max_db']} dB, time > 1 dB GR {mst['limiter_time_over_1db_s']} s, > 3 dB "
                f"{mst['limiter_time_over_3db_s']} s\n")
        f.write("master passes (limiter ceiling, lowest local ceiling -> 8x true peak of the WAV / of the decoded AAC): "
                + "; ".join(f"{p['ceiling_dbtp']}, {p['ceiling_min_dbtp']} -> {p['master_tp_8x']} / {p['aac_tp_8x']}"
                            + (f" (AAC overshoot at {', '.join(str(sp['t']) for sp in p['aac_overshoot_spots'])} s)"
                               if p.get('aac_overshoot_spots') else "") for p in mst["passes"]) + "\n\n")
        f.write("== sections (master, BS.1770 gated over each window) ==\n")
        f.write(f"{'section':10s} {'window':>13s} {'full LUFS':>10s} {'HP150 LUFS':>11s}\n")
        for r in sec:
            f.write(f"{r['section']:10s} {r['t0']:6.2f}-{r['t1']:5.2f} {r['lufs']:10.2f} {r['lufs_hp150']:11.2f}\n")
        f.write("\n== SFX levels at master gain (max momentary of each SFX's own contribution; music bus = max momentary "
                "of the ducked music in the 0.4 s from 50 ms before the sync point) ==\n")
        f.write(f"{'t':>6s} {'sfx':14s} {'LUFS-M':>7s} {'HP150':>7s} {'music':>7s} {'sfx-mus':>8s}  processing\n")
        for e, l1, l2, lmu in bal:
            name = e.typ + (f"({e.params['size']})" if e.typ == "impact" else "")
            f.write(f"{e.t:6.2f} {name:14s} {l1:7.1f} {l2:7.1f} {lmu:7.1f} {l1 - lmu:+8.1f}  {'; '.join(e.flags)}\n")
        f.write("\nloudest SFX, full band: " + ", ".join(f"{e.typ}@{e.t:g} {l1:.1f}" for e, l1, l2, _ in
                                                       sorted(bal, key=lambda r: -r[1])[:6]) + "\n")
        f.write("loudest SFX, HP150:     " + ", ".join(f"{e.typ}@{e.t:g} {l2:.1f}" for e, l1, l2, _ in
                                                       sorted(bal, key=lambda r: -r[2])[:6]) + "\n")
        f.write("\n== checks ==\n")
        for k, v in checks.items():
            f.write(f"{'PASS' if v['ok'] else 'FAIL'}  {k}: " + ", ".join(f"{a}={b}" for a, b in v.items() if a != "ok") + "\n")
        f.write(f"\nmaster sha256 {sig}\n")

    spec_png, wave_png = draw_reports(rep, y, sfx_bus, music_bus, events, anchors, meta, mst["makeup_gain_db"])
    ok = all(v["ok"] for v in checks.values())
    out = dict(
        all_checks_pass=ok, checks=checks, out=args.out, wav=args.wav, music=args.music, seed=args.seed,
        whoosh_sync=args.whoosh_sync, cues=args.cues, video=args.video, duration=D, samples=n, anchors=anchors,
        music_meta=dict(section_times=meta.get("section_times"), fx_transitions=meta.get("fx_transitions")),
        sections=sec, master=dict((k, v) for k, v in mst.items() if k not in ("comp_gain", "lim_gain")),
        music_bus=dict(raw_lufs=round(minfo["raw_lufs"], 2), gain_db=round(minfo["gain_db"], 2), fx_duck=minfo["fx_duck"]),
        sfx_bus=dict(glue_thr_db=round(sinfo["glue_thr_db"], 2), glue_gr_max_db=round(sinfo["glue_gr_max_db"], 2),
                     transient_stage=sinfo["peak"]),
        ffmpeg_mux=cmd, notes=issues, master_sha256=sig,
        files=dict(cue_alignment="cue_alignment.csv", anchor_alignment="anchor_alignment.csv", loudness="loudness.txt",
                   spectrogram=os.path.basename(spec_png), waveform=os.path.basename(wave_png),
                   ffprobe="ffprobe_output.json", video_md5="video_md5.txt"))
    with open(os.path.join(rep, "mix_report.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    for k, v in checks.items():
        log(f"{'PASS' if v['ok'] else 'FAIL'} {k}")
    log(f"report -> {rep}  ({'all checks pass' if ok else 'SOME CHECKS FAIL'})")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cues", required=True, help="cue sheet JSON (list of {t, type, ...})")
    ap.add_argument("--video", required=True, help="silent video; its stream is copied untouched")
    ap.add_argument("--music", required=True, choices=("techhouse", "cinematic"))
    ap.add_argument("--out", required=True, help="output MP4")
    ap.add_argument("--wav", help="also keep the 48 kHz float32 master WAV here")
    ap.add_argument("--report", help="write the QC report into this directory")
    ap.add_argument("--seed", type=int, default=1, help="music seed (default 1)")
    ap.add_argument("--cache-dir", help="cache rendered music stems here (keyed by module source + args)")
    ap.add_argument("--whoosh-sync", choices=("window", "peak"), default="window",
                    help="window (default): a whoosh cue's (t, dur) is the motion window, the array spans it; "
                         "peak: the whoosh envelope peak lands on t")
    args = ap.parse_args(argv)
    for p in (args.cues, args.video):
        if not os.path.exists(p):
            ap.error(f"not found: {p}")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
