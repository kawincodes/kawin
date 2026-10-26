#!/usr/bin/env python3
"""mix.py - re-recording mixer for the KRS Tech Solutions "WE'RE HIRING - VIDEO EDITOR"
30 s Instagram Story (1080x1920, 30 fps, 120 BPM grid).

    python3 -I mix.py --cues <cues.json> --video <silent.mp4> --music techhouse|cinematic \
                      --out <out.mp4> [--wav <mix.wav>] [--report <dir>] [--seed 1] \
                      [--cache-dir <dir>] [--whoosh-sync window|peak] [--lenient]

What it does (all deterministic, 48 kHz stereo float internally):
  1. Reads and validates the cue sheet.  'anchor' cues give the music anchors (drop1 / breakdown /
     drop2 / end_hit).  An explicit anchor/name/id/label/key field wins; otherwise the EARLIEST anchor
     keyword in the note is used, matched as a whole token ("drop 1", "DROP #2", "1st drop", "Break-Down",
     "END HIT", ...; "breakdown (after drop 1)" is the breakdown, "drop @ 18" is an unnamed drop).  A
     duplicate / conflicting anchor, anchors out of order, an anchor more than one video frame off the
     2 s bar grid, a section under one bar or an end hit leaving < 1 s of tail is an error.  The grid
     offset is the least-squares fit over all four anchors; the music renders on that grid and the SFX
     authored on an anchor move with it (<= 1 frame), so slam and kick never flam.
     Strict by default: a dropped cue (no/invalid t or type, unknown type), an invalid optional value
     (dur / pitch / intensity / size / direction) or a missing anchor stops the run before anything is
     rendered (exit 2).  --lenient turns those into notes (defaults 2/14/18/26).
  2. Renders every SFX cue with sfx.py and places it so its perceptual sync point lands on the cue
     (SYNC RULES below), at sfx.SUGGESTED_GAIN_DB plus the offsets in this file.
  3. Mixes: music bus (fx stem ducked under duplicating SFX transitions with sfx.fx_duck_gain(), or the
     SFX riser replaced by the music's own build into drop 1; section gains -1.5 dB over the intro after
     the hook and +1 dB over drop 2; a two-band sidechain duck keyed by the impacts / rejects / check /
     scrubs / shimmers; octave-band ducks under cues that would be masked), SFX bus (per-section trims, the
     hero slams kept the loudest SFX; shared short convolution room; the non-hero part is gated shut 45 ms
     before each hero hit, like the music's breath; light glue compression; 4x-oversampled transient soft
     clip on the slam peaks), master (gentle EQ, HP150-keyed bus compressor, -14 LUFS integrated,
     look-ahead true-peak limiter at -1.5 dBTP, 0.3 s raised-cosine fade to digital zero at the end).
  4. Muxes: the video stream is copied untouched (-c:v copy), audio is AAC-LC 48 kHz stereo of exactly
     the video's duration at <= 256 kb/s (nominal rate calibrated once so the measured rate lands just
     under 256 k), -movflags +faststart.  The MP4's audio is decoded and measured; if the AAC overshoots
     -1.15 dBTP (8x) somewhere, the limiter ceiling is lowered around those spots and the master is redone
     (up to 6 passes, the first pass's make-up gain reused so a fix stays local; still over the limit
     after the last pass = a failed gate).
  5. Gates, always run (with or without --report): true peak, loudness, LRA, clipping / DC, durations,
     video md5, AAC format / rate, faststart, A/V sync, anchor offsets, frame-0 audibility, end fade.  If
     one fails the MP4 (and --wav) are NOT written and the exit code is 1.  The mix QC checks (SFX sync,
     hero slams, hit transients, section contour, masking) also always run; a failed QC check keeps the
     MP4 but exits 1.  --report <dir> writes cue_alignment.csv, anchor_alignment.csv, masking.csv,
     loudness.txt, master_spectrogram.png, master_waveform.png, ffprobe_output.json, video_md5.txt and
     mix_report.json.  Determinism key: the sha256 of the master's float32 samples (the --wav file is
     written without a timestamped PEAK chunk, so its md5 is stable too).
  Exit codes: 0 = everything passed, 1 = a gate or QC check failed, 2 = invalid cue sheet / anchors.

SYNC RULES (cue t = the time in the cue sheet; "target" = where the perceptual hit lands)
  impact, reject, pop, tick, check, downlifter : onset at t (sfx.anchor = 0)
  slash                                       : the cut (sfx.anchor = 0.065 s) at t
  scrub, shimmer                              : array starts at t, length = dur
  riser, reverse_swell                        : t is the START, the array ENDS on the hit at t + dur
                                                (the notes say "Hit = 2.0" etc.; with no dur and no
                                                "Hit =" it ends on the next hit anchor within 4 s, else on
                                                the next downbeat).  On a hero hit the array is gated
                                                45 ms early (the breath); QC checks end + 45 ms = hit.
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
  A cue needs at least one sample inside the video (start < end of video) and its sync point in
  0 <= target < duration - 0.3 s (before the final fade to silence), otherwise it is skipped with a note.
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
from scipy import signal                      # noqa: E402
from scipy.io import wavfile                  # noqa: E402
from scipy.ndimage import uniform_filter1d    # noqa: E402

import sfx                                    # noqa: E402

SR = 48000
BAR_S = 2.0                                   # 120 BPM, 4/4: every anchor is a bar line of this grid
ANCHOR_KEYS = ("drop1", "breakdown", "drop2", "end_hit")
HIT_ANCHORS = ("drop1", "drop2", "end_hit")          # music hits that carry a big SFX slam
DEFAULT_ANCHORS = dict(drop1=2.0, breakdown=14.0, drop2=18.0, end_hit=26.0)
SFX_TYPES = ("slash", "shimmer", "tick", "scrub", "riser", "reverse_swell", "impact", "whoosh",
             "pop", "reject", "check", "downlifter")
T_ALIASES = ("t", "time", "start")            # cue time field and accepted aliases
ANCHOR_NAME_FIELDS = ("anchor", "name", "id", "label", "key", "section")   # explicit anchor fields (win)
ON_ANCHOR_S = 0.001                           # a cue this close to an anchor's cue time is authored ON it
NUM_FIELDS = dict(dur=(0.0, 30.0), pitch=(-24.0, 24.0), intensity=(0.2, 2.0))   # optional numeric cue fields


class CueSheetError(Exception):
    """Invalid cue sheet or anchors: nothing is rendered or written (exit code 2)."""


# ------------------------------------------------------------------------------------------------
# mix settings (data)
# ------------------------------------------------------------------------------------------------
MUSIC_BUS_LUFS = -20.0      # music stems summed and set to this integrated loudness before the SFX
# music-bus lift per section (dB), applied after the level set; each ramp sits inside the music's breath
# before the section's downbeat, where the bed is silent.  Drop 2 must stay >= +1 LU over groove A in the
# MASTER, full band and on the HP150 phone model: the stems give it +1.5 / +2.1 LU, the denser groove-A SFX
# and the master limiter (more gain reduction in drop 2) ate ~0.5-1 LU of that.
# The intro is pulled down after the hook (from HOOK_S, 100 ms raised-cosine ramp under the slash at 0.5 s) so
# that, with its SFX, it sits >= 2 LU under groove A on the phone model; frame 0 keeps its level.
MUSIC_SECTION_DB = dict(intro=-1.5, drop2=+1.0)
MUSIC_SECTION_RAMP_S = 0.04
HOOK_S = (0.45, 0.55)
SFX_BUS_TRIM_DB = 0.0       # global SFX trim on top of SUGGESTED_GAIN_DB (SFX are -3 dBTP sounds)
HERO_SLAM_DB = +2.5         # big slam on drop1 / drop2 / end_hit: gives back the ~2.5 LU full-band loss of
                            # over_music_hit(), so the three hero slams are the loudest SFX full band AND on HP150
END_HIT_SLAM_DB = +2.0      # ... and the end hit alone gets this on top (it was the weakest hero on a phone) ...
END_HIT_LO_DUCK_DB = 3.0    # ... with the music's low band ducked this deep under it (boom/kick make room)
OTHER_BIG_DB = 0.0          # a big slam that is not on an anchor (e.g. EDITOR at 2.5) stays at its suggested level
HIERARCHY_MARGIN_DB = 1.0   # every non-hero SFX is capped this far under the quietest hero slam (LUFS-M, full
                            # band and HP150), e.g. an off-anchor big slam that keeps its sub
# per-type trims on top of SUGGESTED_GAIN_DB (musical judgement, measured in the story mix): the UI pops sit
# ~3 dB under the bed's momentary level, the glassy shimmers do not out-shine the hook slash, the scrubs
# (ruler ticks + tape noise, -27 LUFS-M role) are lifted so they read over the groove (they were 10 dB under
# the music and +0..3 dB in their best octave)
TYPE_TRIM_DB = dict(pop=-2.0, check=-2.5, reject=-1.5, shimmer=-2.0, scrub=+5.0)
# per-section trims (dB) by the section of the cue's sync point (risers / reverse swells: their start).  '*' =
# every non-hero type.  Phone-model contour: the intro must sit >= 2 LU under groove A on HP150 (the slash,
# riser, shimmer and pops filled the build to groove-A level); in the breakdown the stamps sit +6..11 dB over
# the dark bed instead of refilling it to groove-A level.
SECTION_TRIM_DB = {"intro": {"*": -6.0, "slash": -5.0}, "breakdown": {"reject": -4.0, "check": -5.0}}
# the cues in the last second before the end hit (VIDEO / EDITOR slams, pops, the reverse swell) give it room
PRE_END_HIT = dict(window_s=1.0, trim_db=-2.5)
# variable-length sounds: SUGGESTED_GAIN_DB was measured on one reference length (riser(1.5), shimmer(1.2),
# scrub(1.0), whoosh(0.45,'lr'), downlifter(1.0), reverse_swell(0.8)); a longer / louder render is pulled down
# to its ROLE_LUFS_M target (never boosted: a short whoosh is meant to be smaller), at most LEN_COMP_MAX_DB
LEN_COMP_TYPES = ("riser", "reverse_swell", "shimmer", "scrub", "whoosh", "downlifter", "pop", "slash")
LEN_COMP_MAX_DB = 3.0
STACK_WINDOW_S = 0.06       # a sound whose sync point is this close to a hero slam ...
STACK_ATTEN_DB = -3.0       # ... is pulled down so the slam stays the hero (slash / whoosh / shimmer ...)
STACK_TYPES = ("slash", "whoosh", "shimmer", "pop", "tick", "check")
# a shimmer starting in the first 0.3 s after a hero hit (not stacked on it) plays into the music's crash tail:
# +3 dB (the end card's glint at 26.1 sat 8 dB under the crash and its onset read 33 ms late)
CRASH_TAIL = dict(window_s=0.3, types=("shimmer",), gain_db=+3.0)
LIGHT_DB = -3.0             # cue note starting with "(light)"
LIGHT_INTENSITY = 0.7
LOW_PRESENT_DB = -12.0      # music LP150 peak (re its 99.5th pct) above which a slam uses over_music_hit
POP_IN_KEY = True           # snap out-of-key pop offsets to the nearest F-minor note (ties go up)
RISER_NEXT_HIT_S = 4.0      # a riser / reverse_swell without dur ends on the next hit anchor this close, else
                            # on the next downbeat
# transitions the music's own fx stem plays: an SFX riser / reverse_swell that duplicates a music fx transition
# ending on one of these anchors is REPLACED by the music's version (kept at full level, listed in the report)
# instead of ducking the music under it.  Into drop 1 the music already builds (riser, snare roll, hat
# crescendo); the SFX riser(1.0) on top filled the intro to groove-A level on a phone and was masked anyway.
MUSIC_OWNS_TRANSITIONS = ("drop1",)
# unmasking: a non-hero cue whose own energy in its dominant octave is under target_db over the music is helped,
# first by an octave-band duck of the music over the cue's window (zero-phase band-pass subtracted, so the music
# is untouched elsewhere; up to duck_max_db), then by raising the cue (up to boost_max_db, never past the hero
# hierarchy cap).  Up to `passes` re-measurements.  The QC gate is QC['mask_db'].
UNMASK = dict(target_db=3.5, duck_max_db=6.0, boost_max_db=6.0, attack=0.02, release=0.08, passes=3,
              duck_overshoot=1.5)    # an octave duck moves the octave's ratio by less than its depth (band skirts)
# the hero hits' breath: the music leaves a short gap before drop 1 / drop 2 / the end hit (tech-house 50 ms,
# cinematic one 16th); the non-hero part of the SFX bus (risers, reverse swells, whooshes, stacked slashes and
# every room tail) is gated shut for the same 45 ms, so the slam lands out of silence.  Raised-cosine fade-out
# ending pre_s before the hit, closed until the hit sample, back open over reopen_s (under the slam's click).
HIT_BREATH = dict(pre_s=0.045, fade_s=0.008, reopen_s=0.003)

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
# key: (hi-band depth dB, lo-band depth on a music hit, lo-band depth otherwise, attack s, hold s, release s);
# a hold of None = the whole array span (sustained cues)
DUCK_SPLIT_HZ = 150.0
DUCK = {
    "impact_big":    (4.5, 1.5, 5.0, 0.008, 0.06, 0.25),
    "impact_medium": (4.0, 1.0, 4.0, 0.008, 0.04, 0.20),
    "impact_small":  (3.0, 0.5, 3.0, 0.006, 0.03, 0.15),
    "reject":        (4.0, 4.0, 4.0, 0.006, 0.05, 0.20),
    "check":         (2.0, 0.0, 0.0, 0.008, 0.03, 0.15),
    "scrub":         (2.5, 0.0, 0.0, 0.020, None, 0.10),
    "shimmer":       (2.0, 0.0, 0.0, 0.030, None, 0.15),
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
MAX_MASTER_PASSES = 6        # with the limiter ceiling lowered only around the spots where the AAC overshoots;
                             # the next-to-last pass may drop it globally (AAC rate control is global), and if the
                             # last pass is still over the limit the true-peak gate fails (nothing is written)
AAC_FIX_PAD_S = 0.06         # (+- this around each overshooting cluster, 20 ms raised-cosine edges)
# the first AAC frames have no look-back (priming): a transient there overshot +2.9 dB (slam at t=0).  The
# ceiling is lowered this much over the first `samples` from the first pass, raised-cosine back over `ramp`.
AAC_START_GUARD = dict(samples=2048, ramp=1024, dip_db=2.0)
# the hero hits come out of a 45-125 ms breath (silence -> full-scale slam + kick in 1 ms), the transient AAC
# overshoots most (up to +1.9 dB measured, and it moves between passes).  The ceiling is lowered this much
# from pre_s before to post_s after each hit from the first pass (20 ms raised-cosine edges).
AAC_HIT_GUARD = dict(pre_s=0.005, post_s=0.040, dip_db=0.5)
LIMITER = dict(lookahead=0.005, release=0.08, oversample=4)
FADE_IN_S = 0.002            # click guard only: the hook is audible from frame 0
FADE_OUT_S = 0.3             # raised-cosine fade to digital zero at the video end
# AAC-LC 48 kHz stereo.  ffmpeg's native encoder overshoots its nominal rate by ~5-6 % in ABR (256k -> 270 k
# measured), so the nominal rate is calibrated once per run: encode at 256k, measure, re-aim at
# 256 k x 256 k / measured x AIM.  The spec gate wants the measured stream rate within [0.95, 1.0] x 256 kb/s.
AAC_BITRATE = 256000
AAC_BITRATE_AIM = 0.99
AAC_BASE = ["-c:a", "aac", "-profile:a", "aac_low", "-ar", "48000", "-ac", "2"]

# spec limits for the gates
SPEC = dict(lufs=(-14.5, -13.5), tp_max=-1.0, lra_max=8.0, frame_tol_s=1.0 / 30.0, dc_max=1e-4,
            bitrate=(0.95 * AAC_BITRATE, AAC_BITRATE))
# mix QC limits
QC = dict(drop2_lift_lu=1.0,          # master drop 2 - groove A, full band and HP150
          intro_under_lu=2.0,         # master intro <= groove A - this (HP150)
          dark_under_lu=2.0,          # master breakdown dark bar(s) <= groove A - this (HP150)
          hit_rise_db=10.0,           # each hero hit: RMS 0..30 ms after vs 40..5 ms before, master
          mask_db=3.0,                # each cue's own energy over the music in its dominant octave
          hero_flam_s=0.002)          # hero slam sync point vs the music's section time
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


# anchor keywords, matched as whole tokens (no letter / digit on either side) in lower-cased text.  The
# EARLIEST keyword in the text wins; at the same position the longer match wins ("end hit" over "end",
# "drop 2" over "drop").  "drop18" / "drop @ 18" are not drop1: the digit after "drop" must end the token.
_ANCHOR_RX = (
    ("end_hit", r"(?:end|final)[\s_\-]*(?:hit|slam)"),
    ("breakdown", r"break[\s_\-]*down|breakdwn"),
    ("drop1", r"drop[\s_\-#]*(?:0?1|one)|(?:1st|first)[\s_\-]*drop"),
    ("drop2", r"drop[\s_\-#]*(?:0?2|two)|(?:2nd|second)[\s_\-]*drop"),
    ("end", r"(?:video[\s_\-]*)?end(?:[\s_\-]*of[\s_\-]*(?:the[\s_\-]*)?video)?"),
    ("drop", r"drop"),
)
_ANCHOR_RE = tuple((k, re.compile(r"(?<![a-z0-9])(?:" + p + r")(?![a-z0-9])")) for k, p in _ANCHOR_RX)


def anchor_tokens(text):
    """[(position, -length, key)] of every anchor keyword in text, earliest first."""
    s = str(text).lower()
    return sorted((m.start(), -(m.end() - m.start()), k) for k, rx in _ANCHOR_RE for m in rx.finditer(s))


def anchor_key(cue):
    """(key, how) of an anchor cue: an explicit anchor/name/id/label/key/section field wins (two explicit
    fields that disagree -> (None, reason)); otherwise the earliest keyword of the note."""
    explicit = []
    for f in ANCHOR_NAME_FIELDS:
        v = cue.get(f)
        if isinstance(v, str) and v.strip():
            tok = anchor_tokens(v)
            if tok:
                explicit.append((f, tok[0][2]))
    keys = sorted({k for _, k in explicit})
    if len(keys) > 1:
        return None, f"explicit fields disagree {explicit}"
    if keys:
        return keys[0], f"field '{explicit[0][0]}'"
    note = cue.get("note")
    if isinstance(note, str):
        tok = anchor_tokens(note)
        if tok:
            return tok[0][2], "note"
    return None, "no anchor keyword in its fields or note"


def finite(v):
    """float(v) if v is a finite number (not bool / None / NaN / inf / text), else None."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def load_cues(path):
    """Parse and validate the cue sheet.  Returns dict(anchors={key: (t, cue index, how)}, items=[SFX cues],
    end_t, issues=[notes], errors=[dropped cues / bad values / conflicting anchors]).  Never guesses: a
    duplicate anchor with another time is an error, not 'use the first'."""
    with open(path) as f:
        data = json.load(f)
    cues = data.get("cues", data.get("events")) if isinstance(data, dict) else data
    if not isinstance(cues, list):
        raise CueSheetError(f"{path}: expected a list of cues (or {{'cues': [...]}})")
    anchors, plain, items, issues, errors, end_t = {}, [], [], [], [], None
    for i, c in enumerate(cues):
        if not isinstance(c, dict):
            errors.append(f"cue #{i} dropped: not an object: {c!r}"[:200])
            continue
        typ = c.get("type")
        if not isinstance(typ, str) or not typ.strip():
            errors.append(f"cue #{i} dropped: no 'type': {c!r}"[:200])
            continue
        typ = typ.strip().lower().replace("-", "_").replace(" ", "_")
        if typ == "reverseswell":
            typ = "reverse_swell"
        tkey = next((k for k in T_ALIASES if c.get(k) is not None), None)
        if tkey is None:
            errors.append(f"cue #{i} ({typ}) dropped: no time (needs one of {', '.join(T_ALIASES)})")
            continue
        t = finite(c[tkey])
        if t is None:
            errors.append(f"cue #{i} ({typ}) dropped: {tkey} = {c[tkey]!r} is not a finite number")
            continue
        if tkey != "t":
            issues.append(f"cue #{i} ({typ}): time given as '{tkey}' = {t:g}, read as t")
        c = dict(c, type=typ, t=t, index=i)
        if typ == "anchor":
            k, how = anchor_key(c)
            if k == "end":
                if end_t is not None and abs(end_t - t) > 1e-6:
                    errors.append(f"'end' anchor given twice ({end_t:g} and {t:g})")
                end_t = t if end_t is None else end_t
            elif k == "drop":
                plain.append((t, i))
            elif k in ANCHOR_KEYS:
                if k in anchors and abs(anchors[k][0] - t) > 1e-6:
                    errors.append(f"anchor {k} given twice: {anchors[k][0]:g} s (cue #{anchors[k][1]}) and {t:g} s "
                                  f"(cue #{i}, {how})")
                elif k not in anchors:
                    anchors[k] = (t, i, how)
            else:
                errors.append(f"anchor cue #{i} at {t:g} dropped: {how} ({c.get('note')!r})"[:240])
            continue
        if typ not in SFX_TYPES:
            errors.append(f"cue #{i} at {t:g} dropped: unknown type {typ!r}")
            continue
        for k, (lo, hi) in NUM_FIELDS.items():          # optional numbers: null = missing; anything else must be
            v = c.get(k)                                 # a finite number in range (checked before rendering)
            if v is not None and (finite(v) is None or not lo <= finite(v) <= hi):
                errors.append(f"cue #{i} {typ} at {t:g}: {k} = {v!r} is not a finite number in [{lo:g}, {hi:g}]")
        if typ == "impact" and c.get("size") is not None and norm_size(c["size"]) is None:
            errors.append(f"cue #{i} impact at {t:g}: size {c['size']!r} not recognised (big / medium / small)")
        if typ == "whoosh" and c.get("direction") is not None and norm_direction(c["direction"]) is None:
            errors.append(f"cue #{i} whoosh at {t:g}: direction {c['direction']!r} not recognised (lr / rl / up / down)")
        items.append(c)
    for t, i in sorted(plain):                       # unnamed "drop" anchors: drop1 if before the breakdown
        if any(abs(v[0] - t) <= 1e-6 for v in anchors.values()):
            continue                                 # same time as a named anchor: a duplicate, not a new one
        if "drop1" not in anchors and ("breakdown" not in anchors or t < anchors["breakdown"][0]):
            anchors["drop1"] = (t, i, "unnamed drop")
        elif "drop2" not in anchors:
            anchors["drop2"] = (t, i, "unnamed drop")
        else:
            errors.append(f"unnamed drop anchor cue #{i} at {t:g}: drop1 and drop2 are already set")
    items.sort(key=lambda c: (c["t"], c["index"]))
    return dict(anchors=anchors, items=items, end_t=end_t, issues=issues, errors=errors)


def resolve_anchors(cue_a, duration, fps):
    """Validate the four anchors and fit the music grid.  The bar grid (2 s) is anchored on drop1; its offset
    is the least-squares fit over all four anchors (each anchor's bar index re drop1, rounded), so an
    anchor one frame off spreads its error instead of dragging the whole grid.  Errors (CueSheetError):
    anchors not strictly increasing, any anchor more than one video frame off the fitted grid, a section
    shorter than one bar, an end hit leaving < 1 s of tail, drop1 < 0.  Returns dict(cue, music, resid_s,
    offset_s, bars)."""
    a = np.array([cue_a[k] for k in ANCHOR_KEYS], float)
    if not np.all(np.diff(a) > 0):
        raise CueSheetError("anchors out of order (need drop1 < breakdown < drop2 < end_hit): "
                            + ", ".join(f"{k} {cue_a[k]:g}" for k in ANCHOR_KEYS))
    frame = 1.0 / fps
    nb = np.floor((a - a[0]) / BAR_S + 0.5)
    off = float(np.mean(a - nb * BAR_S))
    if off < 0.0:
        if off < -frame:
            raise CueSheetError(f"drop1 {a[0]:g} s is before the start of the video")
        off = 0.0
    music = off + nb * BAR_S
    resid = a - music
    errs = []
    if np.any(np.abs(resid) > frame + 1e-9):
        # name the culprit: with one anchor far off, the least-squares fit spreads its error over all four, so
        # the anchors off the grid are reported against the median fit (the grid most anchors agree on)
        med = float(np.median(a - nb * BAR_S))
        r_med = a - (med + nb * BAR_S)
        bad = [i for i in range(4) if abs(r_med[i]) > frame + 1e-9] or [i for i in range(4) if abs(resid[i]) > frame + 1e-9]
        for i in bad:
            errs.append(f"anchor {ANCHOR_KEYS[i]} at {a[i]:g} s is {r_med[i] * 1000:+.1f} ms off the {BAR_S:g} s bar grid "
                        f"the other anchors sit on (grid offset {med:.4f} s); the limit is one frame ({frame * 1000:.1f} ms)")
    bars = np.diff(nb)
    for name, b in zip(("groove A", "breakdown", "drop 2"), bars):
        if b < 1:
            errs.append(f"{name} is {int(b)} bars long on the grid (needs >= 1 bar)")
    hit_t = float(music[-1]) if not errs else float(med + nb[-1] * BAR_S)     # (the grid named in the errors)
    if hit_t > duration - 1.0 + 1e-9:
        errs.append(f"end_hit on the grid ({hit_t:g} s) leaves {duration - hit_t:.3f} s before the video end (needs >= 1 s)")
    if errs:
        raise CueSheetError("\n  ".join(errs))
    mus = {k: round(float(m), 9) for k, m in zip(ANCHOR_KEYS, music)}
    return dict(cue=dict(cue_a), music=mus, resid_s={k: float(r) for k, r in zip(ANCHOR_KEYS, resid)},
                offset_s=round(off, 9), bars=[int(b) for b in bars], frame_s=frame)


def sections_of(mus_a, duration):
    """[(name, t0, t1)] of the story sections on the music's grid."""
    d1, bd, d2, hit = (mus_a[k] for k in ANCHOR_KEYS)
    return [("intro", 0.0, d1), ("grooveA", d1, bd), ("breakdown", bd, d2), ("drop2", d2, hit), ("end", hit, duration)]


def section_at(t, secs):
    for name, a, b in secs:
        if a - 1e-9 <= t < b - 1e-9:
            return name
    return secs[-1][0] if t >= secs[-1][1] else secs[0][0]


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
    hero = False
    on_hit = False
    breath = 0.0
    headroom_db = 0.0         # how far the cue may be raised before it reaches the hero hierarchy cap
    unmask_duck_db = 0.0      # octave-band duck of the music under this cue (unmasking)
    unmask_boost_db = 0.0     # gain added to this cue (unmasking)
    unmask_band = None        # octave index (OCT_CENTRES) of that duck


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
    """Whoosh direction, or None if unrecognised."""
    d = str(d).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    return {"lr": "lr", "l2r": "lr", "lefttoright": "lr", "right": "lr", "rl": "rl", "r2l": "rl",
            "righttoleft": "rl", "left": "rl", "up": "up", "u": "up", "down": "down", "d": "down"}.get(d)


def norm_size(s):
    """Impact size, or None if unrecognised."""
    s = str(s).strip().lower()
    return {"big": "big", "large": "big", "l": "big", "huge": "big", "medium": "medium", "med": "medium",
            "m": "medium", "small": "small", "s": "small", "light": "small"}.get(s)


def opt_num(c, key, default, report, lo=None, hi=None):
    """Optional numeric cue field: missing / null -> default; invalid or outside [lo, hi] -> reported
    (an error in strict mode) and the default is used."""
    v = c.get(key)
    if v is None:
        return default
    f = finite(v)
    if f is None or (lo is not None and f < lo) or (hi is not None and f > hi):
        report(f"cue #{c['index']} {c['type']} at {c['t']:g}: {key} = {v!r} is not a valid number"
               + (f" in [{lo:g}, {hi:g}]" if lo is not None and hi is not None else "") + f"; using {default!r}")
        return default
    return f


def breath_gate(n, hits):
    """Gain (per sample) of the non-hero SFX: raised-cosine fade-out ending HIT_BREATH['pre_s'] before each
    hit, closed until the hit sample, raised-cosine back open over reopen_s."""
    g = np.ones(n)
    pre, fade, reo = HIT_BREATH["pre_s"], HIT_BREATH["fade_s"], HIT_BREATH["reopen_s"]
    for h in hits:
        a, b, c, d = (int(round(v * SR)) for v in (h - pre - fade, h - pre, h, h + reo))
        for i0, i1, up in ((a, b, False), (c, d, True)):
            idx = np.arange(max(i0, 0), min(i1, n))
            if len(idx):
                q = (idx - i0) / max(i1 - i0, 1)
                w = np.sin(0.5 * np.pi * q) ** 2 if up else np.cos(0.5 * np.pi * q) ** 2
                g[idx] = np.minimum(g[idx], w)
        g[max(b, 0):min(c, n)] = 0.0
    return g


def plan_events(items, A, mlow, duration, whoosh_sync, issues, errors, meta):
    """Render and place every SFX cue.  A = resolve_anchors() (cue and music anchor times).  Hero / stack /
    section decisions use the MUSIC times; a cue authored on an anchor (within 1 ms of its cue time) moves
    with the anchor onto the music grid.  Returns (events, replaced): replaced = SFX transitions handed to
    the music's own fx stem (MUSIC_OWNS_TRANSITIONS)."""
    cue_a, mus_a = A["cue"], A["music"]
    hero_t = [mus_a[k] for k in HIT_ANCHORS]
    secs = sections_of(mus_a, duration)
    n = int(round(duration * SR))
    downbeats = sorted(float(b) for b in meta.get("downbeat_times", []))

    def report(msg):
        errors.append(msg)

    def to_grid(sync_t):
        for k in ANCHOR_KEYS:
            if abs(sync_t - cue_a[k]) <= ON_ANCHOR_S:
                return mus_a[k] - cue_a[k], k
        return 0.0, None

    def end_to_hit(end):              # a riser / reverse swell ends ON a hit: within a frame of its cue or music time
        for k in HIT_ANCHORS:
            if min(abs(end - cue_a[k]), abs(end - mus_a[k])) <= A.get("frame_s", 1 / 30) + 1e-9:
                return mus_a[k] - end, k
        return 0.0, None

    events, replaced = [], []
    for c in items:
        typ, t, idx = c["type"], c["t"], c["index"]
        note = c.get("note") if isinstance(c.get("note"), str) else ""
        e = Event()
        e.cue, e.typ, e.t, e.note = c, typ, t, note
        e.flags, e.params = [], {}
        gain_key = typ
        light = note.strip().lower().startswith("(light)")
        dur = opt_num(c, "dur", None, report, *NUM_FIELDS["dur"])
        shift, on_k = to_grid(t)
        if typ in ("riser", "reverse_swell"):
            m = re.search(r"hit\s*=\s*(\d+(?:\.\d+)?)", note, re.I)
            if dur is None:
                if m:
                    end = float(m.group(1))
                    shift, on_k = end_to_hit(end)
                    end += shift
                    how = f"note 'Hit = {m.group(1)}'"
                else:
                    t0 = t + sfx.MIN_DUR.get(typ, 0.2)
                    nxt = [h for h in hero_t if t0 - 1e-9 <= h <= t + RISER_NEXT_HIT_S]
                    if nxt:
                        end, how = nxt[0], "the next hit anchor"
                    else:
                        nd = [b for b in downbeats if b >= t0 - 1e-9]
                        end, how = (nd[0], "the next downbeat") if nd else (t + 1.0, "default 1 s (no downbeat)")
                    shift, on_k = 0.0, None
                d = end - t
                issues.append(f"cue #{idx} {typ} at {t:g} has no dur: ends on {how} at {end:.3f} s (dur {d:.3f})")
                e.flags.append(f"no dur: ends on {how} {end:.3f}")
                t_eff = t
            else:
                d = dur
                shift, on_k = end_to_hit(t + d)
                t_eff = t + shift
                if m and abs(t + d - float(m.group(1))) > 0.002:
                    issues.append(f"cue #{idx} {typ}: t + dur = {t + d:.3f} but the note says Hit = "
                                  f"{m.group(1)}; using t + dur")
        else:
            t_eff = t + shift
        if on_k and abs(shift) > 1e-9:
            e.flags.append(f"on {on_k}: moved {shift * 1000:+.1f} ms onto the music grid")
        e.t_eff = t_eff
        if typ == "impact":
            size = norm_size(c.get("size")) if c.get("size") is not None else None
            on_hero = any(abs(t_eff - h) <= 0.010 for h in hero_t)
            if size is None:
                size = "big" if on_hero else "medium"
                msg = (f"impact at {t:g} has {'no' if c.get('size') is None else 'an unrecognised'} size "
                       f"({c.get('size')!r}); using '{size}'" + (" (it sits on a hit anchor)" if on_hero else ""))
                (issues if c.get("size") is None else errors).append(f"cue #{idx} {msg}")
                e.flags.append(msg)
            e.params["size"] = size
            x = sfx.impact(size)
            gain_key = f"impact_{size}"
            e.sync_off, e.rule, e.detect = 0.0, "onset at t", "onset"
            rel = mlow.rel_db(t_eff)
            on_anchor = on_hero or abs(t_eff - mus_a["breakdown"]) <= 0.010
            e.music_low_db = rel
            e.on_hit = on_anchor or rel >= LOW_PRESENT_DB
            if e.on_hit:
                x = sfx.over_music_hit(x)
                e.flags.append(f"over_music_hit (music LP150 {rel:+.1f} dB re its kick level)")
            else:
                e.flags.append(f"plain impact with sub (music LP150 {rel:+.1f} dB: low-end gap)")
            e.hero = size == "big" and on_hero
            if e.hero:
                e.hero_key = min(HIT_ANCHORS, key=lambda k: abs(t_eff - mus_a[k]))
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
            p0 = opt_num(c, "pitch", 0.0, report, *NUM_FIELDS["pitch"])
            p, msg = snap_pop_pitch(p0)
            if msg:
                issues.append(f"cue #{idx} at {t:g}: {msg}")
                e.flags.append(msg)
            e.params["pitch"] = p
            x = sfx.pop(p)
            e.sync_off, e.rule, e.detect = 0.0, "onset at t", "onset"
        elif typ == "downlifter":
            d = dur if dur is not None else 1.0
            e.params["dur"] = d
            x = sfx.downlifter(d)
            e.sync_off, e.rule, e.detect = 0.0, "onset at t (falls over dur)", "onset"
        elif typ == "slash":
            d = dur if dur is not None else 0.35
            e.params["dur"] = d
            x = sfx.slash(d)
            e.sync_off, e.rule, e.detect = sfx.anchor("slash"), "cut (anchor 0.065 s) at t", "peak20"
        elif typ in ("scrub", "shimmer"):
            d = dur if dur is not None else (0.5 if typ == "scrub" else 1.2)
            e.params["dur"] = d
            x = sfx.scrub(d) if typ == "scrub" else sfx.shimmer(d)
            e.sync_off, e.rule, e.detect = 0.0, "starts at t, lasts dur", "start"
        elif typ in ("riser", "reverse_swell"):
            e.params["dur"] = round(d, 6)
            x = sfx.riser(d) if typ == "riser" else sfx.reverse_swell(d)
            e.sync_off = sfx.anchor(typ, dur=d)                       # = end of the array
            e.rule, e.detect = "t = start, ENDS on the hit at t + dur", "end"
            e.start_override = t_eff
        elif typ == "whoosh":
            d = dur if dur is not None else 0.45
            dr = c.get("direction")
            direction = norm_direction(dr) if dr is not None else "lr"
            if direction is None:
                report(f"cue #{idx} whoosh at {t:g}: direction {dr!r} not recognised; using 'lr'")
                direction = "lr"
            inten = opt_num(c, "intensity", LIGHT_INTENSITY if light else 1.0, report, *NUM_FIELDS["intensity"])
            e.params.update(dur=d, direction=direction, intensity=inten)
            x = sfx.whoosh(d, direction, inten)
            e.readme_anchor = sfx.anchor("whoosh", dur=d, direction=direction, intensity=inten)
            e.sync_off = e.readme_anchor
            e.detect = "peak3"
            if whoosh_sync == "window":
                e.rule = "spans the motion window [t, t+dur]; envelope peak (-3 dB centre) mid-motion"
                e.start_override = t             # the motion window is the picture's, never moved
            else:
                e.rule = "envelope peak at t"
        else:                                              # pragma: no cover (filtered in load_cues)
            continue
        e.x = np.asarray(x, np.float64)
        start_t = getattr(e, "start_override", t_eff - e.sync_off)
        e.start = int(round(start_t * SR))
        e.start_t = e.start / SR
        e.end_t = (e.start + len(e.x)) / SR
        if e.detect == "peak3" and hasattr(e, "start_override"):
            # window mode: the expected peak is this render's own -3 dB peak-region centre (robust; the
            # README anchor is the argmax at the trailing edge of a ~50 ms plateau, 0-19 ms later)
            e.target = e.start_t + detect(e.x, 0.0, "peak3")
        else:
            e.target = e.start_t + e.sync_off
        if e.detect == "end" and any(abs(e.target - h) <= 0.005 for h in hero_t):
            e.breath = HIT_BREATH["pre_s"]                 # gated with the bus before the hit (see breath_gate)
            e.rule += f"; gated {HIT_BREATH['pre_s'] * 1000:.0f} ms before the hit (breath)"
        # at least one sample inside the mix, and the sync point inside the video, before the final fade-out
        if not (e.start < n and e.start + len(e.x) > 0 and 0.0 <= e.target < duration):
            issues.append(f"cue #{idx} {typ} at {t:g}: sync point {e.target:.3f} s / array {e.start_t:.3f}-"
                          f"{e.end_t:.3f} s is not inside the video (0-{duration:.3f} s); skipped")
            continue
        if e.target >= duration - FADE_OUT_S:
            issues.append(f"cue #{idx} {typ} at {t:g}: sync point {e.target:.3f} s is inside the final {FADE_OUT_S:g} s "
                          "fade to silence; skipped")
            continue
        if e.start < 0:
            issues.append(f"cue #{idx} {typ} at {t:g}: starts {e.start_t:.3f} s, the part before 0 is cut")
        if e.end_t > duration:
            issues.append(f"cue #{idx} {typ} at {t:g}: runs to {e.end_t:.3f} s, cut at {duration:.3f}")
        if typ in ("riser", "reverse_swell"):
            owner = [k for k in MUSIC_OWNS_TRANSITIONS if abs(e.target - mus_a[k]) <= 0.005]
            dup = [tr for tr in (meta.get("fx_transitions") or []) if tr.get("kind") in FX_DUP_KINDS[typ]
                   and min(e.end_t, float(tr["t1"])) - max(e.start_t, float(tr["t0"])) > 0.05]
            if owner and dup:
                mtxt = ", ".join(f"{tr['kind']} {tr['t0']}-{tr['t1']}" for tr in dup)
                msg = (f"cue #{idx} {typ} at {t:g} ({e.start_t:.2f}-{e.end_t:.2f}) is replaced by the music's own "
                       f"transition into {owner[0]} ({mtxt}), kept at full level (MUSIC_OWNS_TRANSITIONS)")
                issues.append(msg)
                replaced.append(dict(cue_index=idx, cue_t=t, type=typ, span=[round(e.start_t, 4), round(e.end_t, 4)],
                                     anchor=owner[0], music=[f"{tr['kind']} {tr['t0']}-{tr['t1']}" for tr in dup]))
                continue
        # gain staging
        e.sec = section_at(e.start_t if typ in ("riser", "reverse_swell") else e.target, secs)
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
                e.flags.append(f"hero slam ({e.hero_key}) {HERO_SLAM_DB:+.1f} dB")
                if e.hero_key == "end_hit" and END_HIT_SLAM_DB:
                    gdb += END_HIT_SLAM_DB
                    e.flags.append(f"end hit {END_HIT_SLAM_DB:+.1f} dB")
            else:
                gdb += OTHER_BIG_DB
                e.flags.append(f"big slam off the anchors {OTHER_BIG_DB:+.1f} dB")
        if not e.hero:
            st_ = SECTION_TRIM_DB.get(e.sec, {})
            v = st_.get(typ, st_.get("*", 0.0))
            if v:
                gdb += v
                e.flags.append(f"{e.sec} trim {v:+.1f} dB")
            t_ref = e.start_t if typ in ("riser", "reverse_swell") else e.target
            if PRE_END_HIT["trim_db"] and mus_a["end_hit"] - PRE_END_HIT["window_s"] <= t_ref < mus_a["end_hit"] - 1e-9:
                gdb += PRE_END_HIT["trim_db"]
                e.flags.append(f"last {PRE_END_HIT['window_s']:g} s before the end hit {PRE_END_HIT['trim_db']:+.1f} dB")
        if light:
            gdb += LIGHT_DB
            e.flags.append(f"(light) {LIGHT_DB:+.1f} dB")
        if typ in STACK_TYPES and any(abs(e.target - h) <= STACK_WINDOW_S for h in hero_t):
            gdb += STACK_ATTEN_DB
            e.flags.append(f"stacked on a hero slam {STACK_ATTEN_DB:+.1f} dB")
        elif typ in CRASH_TAIL["types"] and any(STACK_WINDOW_S < e.target - h <= CRASH_TAIL["window_s"] for h in hero_t):
            gdb += CRASH_TAIL["gain_db"]
            e.flags.append(f"in a hero hit's crash tail {CRASH_TAIL['gain_db']:+.1f} dB")
        e.gain_db = gdb
        e.send = SEND.get(typ, 0.0)
        events.append(e)
    # hierarchy: the hero slams stay the loudest SFX, full band and on the phone model
    heroes = [e for e in events if e.hero]
    if heroes:
        hp = butter("highpass", 150.0, 4)

        def lv(e):
            y = e.x * g_db(e.gain_db)
            return sfx.lufs_momentary_max(y), sfx.lufs_momentary_max(signal.sosfilt(hp, y, axis=0))
        hmin = np.min([lv(e) for e in heroes], axis=0)
        for e in events:
            if e.hero:
                continue
            lf, lh = lv(e)
            cap = min(hmin[0] - lf, hmin[1] - lh) - HIERARCHY_MARGIN_DB
            if cap < 0.0:
                e.gain_db += cap
                e.flags.append(f"capped {cap:+.1f} dB under the hero slams")
            e.headroom_db = max(cap, 0.0)
    return events, replaced


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
    if not len(send):
        return np.zeros_like(send)
    ir = room_ir()
    w = np.stack([signal.oaconvolve(send[:, ch], ir[:, ch])[: len(send)] for ch in range(2)], axis=1)
    w = signal.sosfilt(butter("highpass", ROOM["hp"], 2), w, axis=0)
    w = signal.sosfilt(butter("lowpass", ROOM["lp"], 2), w, axis=0)
    return w * g_db(ROOM["ret_db"])


def build_sfx_bus(events, n, gate):
    """Hero slams + (everything else, room tails included) x the breath gate, then glue."""
    parts = {True: (np.zeros((n, 2)), np.zeros((n, 2))), False: (np.zeros((n, 2)), np.zeros((n, 2)))}
    for e in events:
        y = e.x * g_db(e.gain_db)
        dry, snd = parts[bool(e.hero)]
        add_at(dry, y, e.start)
        if e.send > 0:
            add_at(snd, y * e.send, e.start)
    hero = parts[True][0] + room(parts[True][1])
    rest = parts[False][0] + room(parts[False][1])
    pre = hero + rest * gate[:, None]
    det = signal.sosfilt(butter("highpass", GLUE["sc_hp"], 2), pre, axis=0)
    pk = float(np.max(uniform_filter1d(np.mean(det ** 2, axis=1), int(0.01 * SR), mode="constant")))
    pk = max(pk, 1e-12)
    thr = 10 * math.log10(pk + 1e-20) - GLUE["range_db"]
    glue, gr_max = compressor_gain(pre, thr, GLUE["ratio"], GLUE["attack"], GLUE["release"], GLUE["sc_hp"],
                                   knee_db=GLUE["knee_db"])
    return pre * glue[:, None], dict(glue=glue, glue_thr_db=thr, glue_gr_max_db=gr_max, gate=gate)


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


def event_in_bus(e, glue, gate, n, tail_s=0.7):
    """This event's own contribution to the SFX bus (dry + its room send, x the breath gate unless it is a
    hero slam, x the bus glue / transient gain).  Returns (segment, t0); the segment may be empty."""
    y = e.x * g_db(e.gain_db)
    a = min(max(e.start, 0), n)
    b = min(e.start + len(y) + int(tail_s * SR), n)
    if b <= a:
        return np.zeros((0, 2)), a / SR
    seg = np.zeros((b - a, 2))
    add_at(seg, y, e.start - a)
    if e.send > 0:
        snd = np.zeros_like(seg)
        add_at(snd, y * e.send, e.start - a)
        seg += room(snd)
    g = glue[a:b] if e.hero else glue[a:b] * gate[a:b]
    return seg * g[:, None], a / SR


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


def section_lift(n, mus_a):
    """dB curve of MUSIC_SECTION_DB: each section's lift ramps in (raised cosine) over MUSIC_SECTION_RAMP_S
    ending on its downbeat and out over the same length ending on the next anchor (both inside the music's
    breath, where the bed is silent)."""
    out = np.zeros(n)
    bounds = dict(intro=(None, "drop1"), grooveA=("drop1", "breakdown"), breakdown=("breakdown", "drop2"),
                  drop2=("drop2", "end_hit"))
    r = int(round(MUSIC_SECTION_RAMP_S * SR))
    for name, lift in MUSIC_SECTION_DB.items():
        if not lift or name not in bounds:
            continue
        a, b = (int(round(mus_a[k] * SR)) if k else None for k in bounds[name])
        ra = r
        if a is None:                                  # the intro: from the hook on (frame 0 keeps its level)
            h0, h1 = (min(v, 0.25 * mus_a["drop1"]) for v in HOOK_S)
            a, ra = int(round(h1 * SR)), max(int(round((h1 - h0) * SR)), 1)
        if b - r <= a:                                 # (no room: e.g. an intro shorter than the hook)
            continue
        w = np.zeros(n)
        w[max(a, 0):max(min(b - r, n), 0)] = 1.0
        for i0, ramp in ((a - ra, np.sin(0.5 * np.pi * np.arange(ra) / ra) ** 2),
                         (b - r, np.cos(0.5 * np.pi * np.arange(r) / r) ** 2)):
            j = np.arange(max(i0, 0), min(i0 + len(ramp), n))
            w[j] = ramp[j - i0]
        out += lift * w
    return out


def octave_duck(mus, events):
    """Unmasking ducks: for each octave band some cue needs, subtract (1 - g) x the zero-phase band-passed
    music, g = the duck curve of every cue on that band (raised-cosine ramps, deeper key wins).  g = 1 leaves
    the music bit-exact."""
    n = len(mus)
    keys = {}
    for e in events:
        if e.unmask_duck_db > 0 and e.unmask_band is not None:
            a_t, b_t = mask_window(e)
            keys.setdefault(e.unmask_band, []).append((a_t, e.unmask_duck_db, UNMASK["attack"], b_t - a_t,
                                                       UNMASK["release"]))
    out = mus
    depth = np.zeros(n)
    for band, kk in sorted(keys.items()):
        g, d = duck_curve(n, kk)
        c = OCT_CENTRES[band]
        sos = signal.butter(2, [c / math.sqrt(2), min(c * math.sqrt(2), 0.45 * SR)], "bandpass", fs=SR, output="sos")
        i = np.nonzero(d > 0)[0]
        a, b = max(i[0] - int(0.5 * SR), 0), min(i[-1] + int(0.5 * SR), n)       # filter only around the keys
        bp = signal.sosfiltfilt(sos, mus[a:b], axis=0)
        out = out.copy() if out is mus else out
        out[a:b] -= (1.0 - g[a:b])[:, None] * bp
        depth = np.maximum(depth, d)
    return out, depth


def build_music_bus(st, n, events, A, meta):
    mus_a = A["music"]
    stems = {k: fit_len(st[k], n) for k in ("drums", "bass", "music", "fx")}
    # 1. one source per transition: duck the music fx stem under duplicating SFX
    spans, ducked = [], []
    trans = meta.get("fx_transitions", []) or []
    hits = [mus_a[k] for k in ANCHOR_KEYS]
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
    lift = section_lift(n, mus_a)
    mus = mus * (g_db(mg) * 10 ** (lift / 20.0))[:, None]
    # 2. sidechain duck (two bands, zero-phase complementary split at 150 Hz)
    hi_keys, lo_keys = [], []
    for e in events:
        k = f"impact_{e.params['size']}" if e.typ == "impact" else e.typ
        if k not in DUCK:
            continue
        hi, lo_hit, lo_plain, att, hold, rel = DUCK[k]
        t_key = e.target
        if hold is None:                                    # sustained cue: duck over its whole span
            t_key, hold = max(e.start_t, 0.0), max(min(e.end_t, n / SR) - max(e.start_t, 0.0), 0.0)
        lo = lo_hit if e.on_hit else lo_plain
        if e.hero and getattr(e, "hero_key", None) == "end_hit":
            lo = max(lo, END_HIT_LO_DUCK_DB)
        hi_keys.append((t_key, hi, att, hold, rel))
        lo_keys.append((t_key, lo, att, hold, rel))
    g_hi, d_hi = duck_curve(n, hi_keys)
    g_lo, d_lo = duck_curve(n, lo_keys)
    low = signal.sosfiltfilt(butter("lowpass", DUCK_SPLIT_HZ, 2), mus, axis=0)
    bus = low * g_lo[:, None] + (mus - low) * g_hi[:, None]
    # 3. unmasking ducks (octave bands, see UNMASK)
    bus, d_oct = octave_duck(bus, events)
    return bus, dict(raw_lufs=raw_lufs, gain_db=mg, fx_duck=ducked, duck_hi_db=d_hi, duck_lo_db=d_lo,
                     section_lift_db=lift, duck_oct_db=d_oct)


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


def start_guard(n, hits=()):
    """Ceiling dip (dB) over the first AAC frames (AAC_START_GUARD) and around the hero hits (AAC_HIT_GUARD)."""
    g = np.zeros(n)
    a, r, d = AAC_START_GUARD["samples"], AAC_START_GUARD["ramp"], AAC_START_GUARD["dip_db"]
    g[:min(a, n)] = d
    j = np.arange(a, min(a + r, n))
    g[j] = d * np.cos(0.5 * np.pi * (j - a) / r) ** 2
    ramp = int(0.02 * SR)
    for h in hits:
        a, b = int(round((h - AAC_HIT_GUARD["pre_s"]) * SR)), int(round((h + AAC_HIT_GUARD["post_s"]) * SR))
        w = np.zeros(n)
        w[max(a, 0):min(b, n)] = 1.0
        for i0, q in ((a - ramp, np.sin(0.5 * np.pi * np.arange(ramp) / ramp) ** 2),
                      (b, np.cos(0.5 * np.pi * np.arange(ramp) / ramp) ** 2)):
            jj = np.arange(max(i0, 0), min(i0 + ramp, n))
            w[jj] = q[jj - i0]
        g = np.maximum(g, AAC_HIT_GUARD["dip_db"] * w)
    return g


def master(mix, ceiling=CEILING_DBTP, gain=None):
    """EQ -> HP150-keyed bus compressor -> make-up to TARGET_LUFS (iterated through the limiter, unless `gain`
    is given: the AAC passes reuse the first pass's make-up so a local ceiling dip changes only the samples
    around it, and the AAC encode elsewhere stays the same) -> true-peak limiter -> edge fades -> DC."""
    sos = np.stack([biquad(k, f, gdb, q) for k, f, gdb, q in EQ])
    y = signal.sosfilt(sos, mix, axis=0)
    thr = lufs(signal.sosfilt(butter("highpass", BUS_COMP["sc_hp"], 2), y, axis=0)) + BUS_COMP["thr_rel_db"]
    cg, comp_gr = compressor_gain(y, thr, BUS_COMP["ratio"], BUS_COMP["attack"], BUS_COMP["release"],
                                  BUS_COMP["sc_hp"])
    y = y * cg[:, None]
    fades = edge_fades(len(y), FADE_IN_S, FADE_OUT_S)
    fixed = gain is not None
    gain = gain if fixed else TARGET_LUFS - lufs(y)
    for it in range(6):
        z, lg = tp_limiter(y * g_db(gain), ceiling, LIMITER["lookahead"], LIMITER["release"],
                           LIMITER["oversample"])
        z = z * fades[:, None]
        if fixed:
            break
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
                makeup_gain_db=round(gain, 2), makeup_gain_exact_db=float(gain), limiter_gr_max_db=round(float(lim_gr.max()), 2),
                limiter_time_over_1db_s=round(float(np.sum(lim_gr > 1.0)) / SR, 3),
                limiter_time_over_3db_s=round(float(np.sum(lim_gr > 3.0)) / SR, 3), iterations=it + 1,
                lufs_py=round(lufs(z), 3), true_peak_8x_dbtp=round(db(true_peak(z)), 3), comp_gain=cg, lim_gain=lg)
    return z.astype(np.float32), info


# ------------------------------------------------------------------------------------------------
# mux / files
# ------------------------------------------------------------------------------------------------
def write_wav(path, y):
    """32-bit float WAV without a timestamped PEAK chunk (scipy.io.wavfile), so equal samples give equal
    files; the samples' sha256 is the determinism key."""
    wavfile.write(path, SR, np.ascontiguousarray(np.asarray(y, np.float32)))


def mux(video, wav, out, bitrate):
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video, "-i", wav,
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *AAC_BASE, "-b:a", f"{int(bitrate) // 1000}k",
           "-map_metadata", "0", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    return cmd


def audio_bitrate(path):
    j = ffprobe_json(path)
    a = [s for s in j["streams"] if s.get("codec_type") == "audio"]
    return int(a[0].get("bit_rate", 0) or 0) if a else 0


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
# detection
# ------------------------------------------------------------------------------------------------
def detect(seg, t0, mode):
    """Sync point of a segment (see the SYNC RULES); None if the segment is empty or silent."""
    if seg is None or not len(seg) or not np.any(seg):
        return None
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
    """Strongest level rise in the summed SFX bus within +-win of target: (time, rise dB) or (None, None)."""
    a = max(int((target - win - 0.02) * SR), 0)
    b = min(int((target + win + 0.005) * SR), len(bus))
    if b - a <= int(0.02 * SR):
        return None, None
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


def section_table(x, mus_a, n):
    """Master loudness per section (BS.1770 gated; full band and the HP150 phone model), plus the dark /
    build split of the breakdown and the like-for-like rows (groove A from bar 5, drop 2 without its last
    (turnaround) bar)."""
    d1, bd, d2, hit = (mus_a[k] for k in ANCHOR_KEYS)
    dur = n / SR
    spans = [("intro", 0.0, d1), ("grooveA", d1, bd), ("breakdown", bd, d2), ("drop2", d2, hit), ("end", hit, dur)]
    if d2 - bd >= 2 * BAR_S - 1e-6:                 # last bar of the breakdown is the build, the rest is dark
        spans[3:3] = [(" bd.dark", bd, d2 - BAR_S), (" bd.build", d2 - BAR_S, d2)]
    if bd - d1 >= 5 * BAR_S - 1e-6:
        spans.append((" grooveA.full", d1 + 4 * BAR_S, bd))
    if hit - d2 >= 2 * BAR_S - 1e-6:
        spans.append((" drop2.noturn", d2, hit - BAR_S))
    hp = signal.sosfilt(butter("highpass", 150.0, 4), x, axis=0)
    rows = []
    for name, a, b in spans:
        i, j = int(round(a * SR)), int(round(b * SR))
        if j - i < int(0.4 * SR):
            continue
        rows.append(dict(section=name, t0=a, t1=b, lufs=round(lufs(x[i:j]), 2), lufs_hp150=round(lufs(hp[i:j]), 2)))
    return rows


OCT_CENTRES = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)


def octave_energy(x):
    """Energy per octave band (OCT_CENTRES, edges at centre/sqrt2 .. centre*sqrt2) of a mono-summed,
    Hann-windowed segment."""
    m = np.mean(np.asarray(x, float), axis=1)
    if len(m) < 64:
        return np.zeros(len(OCT_CENTRES))
    S = np.abs(np.fft.rfft(m * np.hanning(len(m)))) ** 2
    f = np.fft.rfftfreq(len(m), 1.0 / SR)
    return np.array([S[(f >= c / math.sqrt(2)) & (f < c * math.sqrt(2))].sum() for c in OCT_CENTRES])


def mask_rows(events, music_bus, sinfo, gate, n):
    """Per non-hero cue: its own contribution to the SFX bus vs the music bus over mask_window(), per octave.
    Equal to the ratio in the master: the master's EQ is common to both and its compressor / limiter gain is
    one gain on their sum."""
    rows = []
    for e in events:
        if e.hero:
            continue
        seg_, t0 = event_in_bus(e, sinfo["glue"], gate, n)
        a_t, b_t = mask_window(e)
        a, b = max(int(round(a_t * SR)), 0), min(int(round(b_t * SR)), n)
        if b - a < int(0.02 * SR):
            continue
        sx = np.zeros((b - a, 2))
        add_at(sx, seg_, int(round(t0 * SR)) - a)
        es, em = octave_energy(sx), octave_energy(music_bus[a:b])
        k = int(np.argmax(es))
        ratio = 10 * np.log10((es + 1e-20) / (em + 1e-20))
        kb = int(np.argmax(np.where(es > es[k] * 0.01, ratio, -np.inf)))     # best among bands within 20 dB
        rows.append(dict(event=e, band=k, cue_t=f"{e.t:.3f}", type=e.typ, window=f"{a_t:.3f}-{b_t:.3f}",
                         dominant_octave_hz=int(OCT_CENTRES[k]), sfx_over_music_dominant_db=round(float(ratio[k]), 1),
                         best_octave_hz=int(OCT_CENTRES[kb]), sfx_over_music_best_db=round(float(ratio[kb]), 1),
                         broadband_db=round(10 * np.log10((np.sum(sx ** 2) + 1e-20) /
                                                          (np.sum(music_bus[a:b] ** 2) + 1e-20)), 1),
                         ok=bool(ratio[k] >= QC["mask_db"]), gain_db=round(e.gain_db, 1),
                         unmask_duck_db=round(e.unmask_duck_db, 1), unmask_boost_db=round(e.unmask_boost_db, 1)))
    return rows


def bus_levels(e, sinfo, gate, n):
    """(max momentary LUFS full band, on HP150) of an event's own contribution to the processed SFX bus."""
    seg_, _ = event_in_bus(e, sinfo["glue"], gate, n)
    if not len(seg_):
        return -120.0, -120.0
    return (sfx.lufs_momentary_max(seg_),
            sfx.lufs_momentary_max(signal.sosfilt(butter("highpass", 150.0, 4), seg_, axis=0)))


def unmask(events, music_bus, sinfo, gate, n):
    """One unmasking step (see UNMASK); returns the number of cues changed.  A boost never takes a cue past
    HIERARCHY_MARGIN_DB under the quietest hero slam, measured like the QC check (in the processed bus)."""
    changed = 0
    heroes = [bus_levels(e, sinfo, gate, n) for e in events if e.hero]
    hmin = np.min(heroes, axis=0) if heroes else None
    for r in mask_rows(events, music_bus, sinfo, gate, n):
        e, short = r["event"], UNMASK["target_db"] - r["sfx_over_music_dominant_db"]
        if short <= 0.05:
            continue
        if e.unmask_band is None:
            e.unmask_band = r["band"]
        want = UNMASK["duck_overshoot"] * short
        duck = min(e.unmask_duck_db + want, UNMASK["duck_max_db"]) if r["band"] == e.unmask_band else e.unmask_duck_db
        rest = short - (duck - e.unmask_duck_db) / UNMASK["duck_overshoot"]
        room_db = e.headroom_db
        if hmin is not None and rest > 0:
            lf, lh = bus_levels(e, sinfo, gate, n)
            room_db = min(room_db, e.unmask_boost_db + max(min(hmin[0] - lf, hmin[1] - lh) - HIERARCHY_MARGIN_DB, 0.0))
        boost = min(e.unmask_boost_db + max(rest, 0.0), UNMASK["boost_max_db"], room_db)
        if duck - e.unmask_duck_db < 0.05 and boost - e.unmask_boost_db < 0.05:
            continue
        e.gain_db += boost - e.unmask_boost_db
        e.unmask_duck_db, e.unmask_boost_db = duck, boost
        changed += 1
    return changed


def mask_window(e):
    """(t0, t1) where a cue's masking is judged: transients 0..120 ms after the sync point, the cut of a
    slash, the whole span of a sustained cue, the last 300 ms of a riser / reverse swell before its breath."""
    if e.typ in ("impact", "reject", "check", "pop", "tick"):
        return e.target, e.target + min(0.12, len(e.x) / SR)
    if e.typ == "slash":
        return e.target - 0.02, e.target + 0.10
    if e.typ in ("riser", "reverse_swell"):
        end = e.target - e.breath
        return max(end - 0.3, e.start_t), end
    if e.typ == "downlifter":                     # its bright onset and the start of the fall (the rest sinks
        return e.start_t, e.start_t + min(0.25, 0.5 * len(e.x) / SR)     # under the next downbeat by design)
    return e.start_t, e.end_t                       # scrub, shimmer, whoosh
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


# ------------------------------------------------------------------------------------------------
# run
# ------------------------------------------------------------------------------------------------
def prepare(args, vid, strict, issues):
    """Load + validate the cue sheet and anchors (CueSheetError on anything fatal)."""
    D = vid["duration"]
    cs = load_cues(args.cues)
    issues += cs["issues"]
    errors = list(cs["errors"])
    if cs["end_t"] is not None and abs(cs["end_t"] - D) > 0.5 / vid["fps"]:
        errors.append(f"cue sheet 'end' anchor {cs['end_t']:g} s differs from the video duration {D:.3f} s")
    cue_a = {}
    for k in ANCHOR_KEYS:
        if k in cs["anchors"]:
            cue_a[k] = cs["anchors"][k][0]
        elif strict:
            errors.append(f"anchor {k} missing from the cue sheet")
        else:
            cue_a[k] = DEFAULT_ANCHORS[k]
            issues.append(f"anchor {k} missing from the cue sheet; using the default {DEFAULT_ANCHORS[k]:g} (--lenient)")
    if errors:
        if strict:
            raise CueSheetError("\n  ".join(errors) + "\n(--lenient turns dropped cues / missing anchors into notes)")
        issues += [f"(--lenient) {m}" for m in errors]
    A = resolve_anchors(cue_a, D, vid["fps"])
    A["how"] = {k: cs["anchors"][k][2] if k in cs["anchors"] else "default" for k in ANCHOR_KEYS}
    for k in ANCHOR_KEYS:
        if abs(A["resid_s"][k]) > 1e-6:
            issues.append(f"anchor {k}: cue {A['cue'][k]:g} s, music grid {A['music'][k]:g} s "
                          f"({-A['resid_s'][k] * 1000:+.1f} ms, least-squares grid fit, within one frame)")
    return cs, A


def mix_buses(st, n, events, A, meta, gate):
    """Music bus + SFX bus (with the transient stage), then the unmasking passes (UNMASK)."""
    for it in range(UNMASK["passes"] + 1):
        music_bus, minfo = build_music_bus(st, n, events, A, meta)
        sfx_bus, sinfo = build_sfx_bus(events, n, gate)
        sfx_bus, sinfo["peak_gain"], sinfo["peak"] = sfx_peak_stage(sfx_bus, music_bus)
        sinfo["glue"] = sinfo["glue"] * sinfo["peak_gain"]          # per-event view = glue x transient stage
        if it == UNMASK["passes"] or not unmask(events, music_bus, sinfo, gate, n):
            break
    for e in events:
        if e.unmask_duck_db > 0 or e.unmask_boost_db > 0:
            e.flags.append(f"unmasked: music {OCT_CENTRES[e.unmask_band]:g} Hz octave -{e.unmask_duck_db:.1f} dB, "
                           f"cue {e.unmask_boost_db:+.1f} dB")
    return music_bus, sfx_bus, minfo, sinfo


def refuse(args, msg, code=2, extra=None):
    log(("CUE SHEET ERROR" if code == 2 else "ERROR") + " - nothing written:\n  " + msg)
    if args.report:
        os.makedirs(args.report, exist_ok=True)
        with open(os.path.join(args.report, "mix_report.json"), "w") as f:
            json.dump(dict(all_checks_pass=False, delivered=False, exit_code=code, error=msg, cues=args.cues,
                           video=args.video, music=args.music, **(extra or {})), f, indent=1)
    return code


def run(args):
    t_start = time.time()
    issues = []
    strict = not args.lenient
    vid = probe_video(args.video)
    D = vid["duration"]
    n = int(round(D * SR))
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            cs, A = prepare(args, vid, strict, issues)
        except CueSheetError as exc:
            return refuse(args, str(exc))
        log(f"video {D:.3f} s ({vid['nb_frames']} frames @ {vid['fps']:g} fps) -> {n} samples; anchors (cue) "
            f"{A['cue']}; music grid {A['music']}; {len(cs['items'])} SFX cues; music {args.music}")
        try:
            st = render_music(args.music, D, A["music"], args.seed, args.cache_dir)
        except ValueError as exc:                    # music_*.StoryGridError (one policy in both modules)
            return refuse(args, f"music_{args.music}.render_story refused the anchors: {exc}")
        meta = st["meta"]
        mt = meta.get("section_times", {})
        if any(abs(float(mt.get(k, np.nan)) - A["music"][k]) > 1e-6 for k in ANCHOR_KEYS):
            return refuse(args, f"the music module moved the anchors: asked {A['music']}, got {mt}", code=1)
        mlow = MusicLow(st, n)
        errors = []
        events, replaced = plan_events(cs["items"], A, mlow, D, args.whoosh_sync, issues, errors, meta)
        if errors:
            if strict:
                return refuse(args, "\n  ".join(errors) + "\n(--lenient uses the defaults instead)")
            issues += [f"(--lenient) {m}" for m in errors]
        hits = [A["music"][k] for k in HIT_ANCHORS]
        gate = breath_gate(n, hits)
        music_bus, sfx_bus, minfo, sinfo = mix_buses(st, n, events, A, meta, gate)
        mix = music_bus + sfx_bus
        caught = [f"{w.category.__name__}: {w.message}" for w in wlist]
    issues += caught
    for msg in issues:
        log("note:", msg)

    tmpdir = tempfile.mkdtemp(prefix="mix-")
    try:
        wav, mp4 = os.path.join(tmpdir, "mix.wav"), os.path.join(tmpdir, "out.mp4")
        # master -> WAV -> mux, then measure the decoded AAC; if its 8x true peak is above the spec minus a
        # margin (AAC overshoots the PCM by ~0.1-1 dB), lower the limiter ceiling there and redo
        limit = SPEC["tp_max"] - AAC_TP_MARGIN
        dips, passes, nominal = start_guard(n, hits), [], None
        for attempt in range(MAX_MASTER_PASSES):
            ceiling = CEILING_DBTP - dips
            y, mst = master(mix, ceiling, gain=passes[0]["makeup_exact"] if passes else None)
            if not np.isfinite(y).all():
                return refuse(args, "non-finite samples in the master", code=1)
            write_wav(wav, y)
            if nominal is None:                         # calibrate the AAC nominal rate once (see AAC_BITRATE)
                mux(args.video, wav, mp4, AAC_BITRATE)
                measured = audio_bitrate(mp4)
                nominal = AAC_BITRATE if not measured else \
                    min(AAC_BITRATE, int(AAC_BITRATE * AAC_BITRATE / measured * AAC_BITRATE_AIM / 1000) * 1000)
                bit_cal = dict(nominal_first=AAC_BITRATE, measured_first=measured, nominal=nominal)
            cmd = mux(args.video, wav, mp4, nominal)
            dec = decode_audio(mp4, n)[:n]
            tp_aac = db(true_peak(dec))
            p = dict(ceiling_dbtp=CEILING_DBTP, ceiling_min_dbtp=round(float(ceiling.min()), 3),
                     master_tp_8x=mst["true_peak_8x_dbtp"], aac_tp_8x=round(tp_aac, 3), lufs=mst["lufs_py"],
                     makeup_exact=mst["makeup_gain_exact_db"])
            passes.append(p)
            if tp_aac <= limit:
                break
            if attempt == MAX_MASTER_PASSES - 1:
                log(f"decoded AAC true peak {tp_aac:.2f} dBTP is still > {limit:.2f} after {MAX_MASTER_PASSES} passes")
                break
            dip, spots = local_ceiling_dips(dec, limit)
            if attempt >= MAX_MASTER_PASSES - 2:              # last chance: global (AAC rate control is global)
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
        mst["aac_bitrate_calibration"] = bit_cal
        mst["aac_tp_limit"] = round(limit, 3)
        ctx = dict(args=args, vid=vid, D=D, n=n, A=A, events=events, replaced=replaced, st=st, meta=meta, music_bus=music_bus,
                   sfx_bus=sfx_bus, sinfo=sinfo, minfo=minfo, y=y, mst=mst, wav=wav, mp4=mp4, cmd=cmd,
                   issues=issues, tmpdir=tmpdir, gate=gate)
        gates, qc, data = evaluate(ctx)
        gates_ok = all(v["ok"] for v in gates.values())
        qc_ok = all(v["ok"] for v in qc.values())
        if gates_ok:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
            shutil.copyfile(mp4, args.out + ".partial")
            os.replace(args.out + ".partial", args.out)
            if args.wav:
                os.makedirs(os.path.dirname(os.path.abspath(args.wav)) or ".", exist_ok=True)
                shutil.copyfile(wav, args.wav)
            log(f"wrote {args.out} (master {mst['lufs_py']:.2f} LUFS, {mst['true_peak_8x_dbtp']:.2f} dBTP, decoded AAC "
                f"{passes[-1]['aac_tp_8x']:.2f} dBTP, AAC nominal {nominal // 1000}k) in {time.time() - t_start:.1f} s")
        else:
            log(f"DELIVERY GATE FAILED ({', '.join(k for k, v in gates.items() if not v['ok'])}): {args.out} was NOT "
                "written" + (" (an older file at that path is left untouched)" if os.path.exists(args.out) else ""))
        for k, v in list(gates.items()) + list(qc.items()):
            log(f"{'PASS' if v['ok'] else 'FAIL'} {k}")
        if args.report:
            write_report(ctx, gates, qc, data, delivered=gates_ok)
        ok = gates_ok and qc_ok
        log(f"{'all gates and checks pass' if ok else 'SOME GATES / CHECKS FAIL'}; master sha256 {data['sha256']}")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def evaluate(ctx):
    """Delivery gates and mix QC checks, all measured on the master WAV / the muxed MP4 / the buses.
    Returns (gates, qc, data)."""
    args, vid, D, n, A = ctx["args"], ctx["vid"], ctx["D"], ctx["n"], ctx["A"]
    events, st, meta, y, mst = ctx["events"], ctx["st"], ctx["meta"], ctx["y"], ctx["mst"]
    music_bus, sfx_bus, sinfo = ctx["music_bus"], ctx["sfx_bus"], ctx["sinfo"]
    mus_a, cue_a = A["music"], A["cue"]
    tol = SPEC["frame_tol_s"]
    frame = 1.0 / vid["fps"]
    gm = g_db(mst["makeup_gain_db"])
    gates, qc, data = {}, {}, {}
    yd = y.astype(np.float64)

    # ---- loudness / peaks (gates) ----
    hp_wav = os.path.join(ctx["tmpdir"], "master_hp150.wav")
    write_wav(hp_wav, signal.sosfilt(butter("highpass", 150.0, 4), yd, axis=0))
    eb_m, eb_hp, eb_mp4 = ebur128(ctx["wav"]), ebur128(hp_wav), ebur128(ctx["mp4"])
    data.update(eb_m=eb_m, eb_hp=eb_hp, eb_mp4=eb_mp4)
    first_frame = yd[: int(round(SR / vid["fps"]))]
    ff_rms = db(np.sqrt(np.mean(first_frame ** 2)))
    tm, lm = loudness_curve(yd, 0.4, 0.01)
    m_first = float(lm[int(0.2 / 0.01)])                           # momentary over 0-400 ms
    last10 = db(np.sqrt(np.mean(yd[-int(0.01 * SR):] ** 2)))
    fo = yd[-int(FADE_OUT_S * SR):]
    fo_db = [round(db(np.sqrt(np.mean(c ** 2))), 1) for c in np.array_split(fo, 3)]
    dc = np.abs(np.mean(yd, axis=0)).max()
    clip_n = int(np.sum(np.abs(yd) >= 0.999))
    tp8 = db(true_peak(yd))
    dec = decode_audio(ctx["mp4"], n)
    tp8_aac = db(true_peak(dec[:n]))
    data.update(ff_rms=ff_rms, m_first=m_first, last10=last10, fo_db=fo_db, dc=dc, clip_n=clip_n, tp8=tp8,
                tp8_aac=tp8_aac)
    gates["integrated_loudness"] = dict(
        ok=eb_m["I"] is not None and SPEC["lufs"][0] <= eb_m["I"] <= SPEC["lufs"][1]
        and eb_mp4["I"] is not None and SPEC["lufs"][0] <= eb_mp4["I"] <= SPEC["lufs"][1],
        master_wav=eb_m["I"], mp4=eb_mp4["I"], target=TARGET_LUFS)
    gates["true_peak"] = dict(ok=eb_m["TP"] is not None and eb_m["TP"] <= SPEC["tp_max"] and tp8 <= SPEC["tp_max"]
                              and eb_mp4["TP"] is not None and eb_mp4["TP"] <= SPEC["tp_max"]
                              and tp8_aac <= SPEC["tp_max"] and mst["passes"][-1]["aac_tp_8x"] <= mst["aac_tp_limit"],
                              master_wav_ebur128=eb_m["TP"], master_wav_8x=round(tp8, 2), mp4_ebur128=eb_mp4["TP"],
                              mp4_decoded_8x=round(tp8_aac, 2), aac_loop_limit=mst["aac_tp_limit"],
                              master_passes=len(mst["passes"]))
    gates["lra"] = dict(ok=eb_m["LRA"] is not None and eb_m["LRA"] <= SPEC["lra_max"], master_wav=eb_m["LRA"],
                        mp4=eb_mp4["LRA"])
    gates["no_clipping_no_dc"] = dict(ok=clip_n == 0 and dc < SPEC["dc_max"] and bool(np.isfinite(yd).all()),
                                      samples_ge_0dBFS=clip_n, sample_peak_dbfs=round(db(np.abs(yd).max()), 2),
                                      dc=float(dc))
    gates["audible_from_frame0"] = dict(ok=ff_rms > -45.0 and m_first > -40.0,
                                        frame0_rms_dbfs=round(ff_rms, 1), momentary_0_400ms_lufs=round(m_first, 1),
                                        first_sample=float(yd[0, 0]))
    gates["fade_to_silence"] = dict(ok=float(np.abs(yd[-1]).max()) == 0.0 and last10 < -60.0
                                    and fo_db[0] > fo_db[1] > fo_db[2],
                                    last_sample=float(np.abs(yd[-1]).max()), last_10ms_rms_dbfs=round(last10, 1),
                                    fade_thirds_rms_dbfs=fo_db)

    # ---- video / container (gates) ----
    md5_in, md5_out = video_md5(args.video), video_md5(ctx["mp4"])
    pin, pout = ffprobe_json(args.video), ffprobe_json(ctx["mp4"])
    data.update(md5_in=md5_in, md5_out=md5_out, ffprobe_out=pout)
    vin = [s for s in pin["streams"] if s["codec_type"] == "video"][0]
    vout = [s for s in pout["streams"] if s["codec_type"] == "video"][0]
    aout = [s for s in pout["streams"] if s["codec_type"] == "audio"]
    keys = ("codec_name", "profile", "width", "height", "pix_fmt", "r_frame_rate", "avg_frame_rate", "time_base",
            "duration_ts", "nb_frames", "bit_rate")
    vdiff = {k: (vin.get(k), vout.get(k)) for k in keys if vin.get(k) != vout.get(k)}
    data["vdiff"] = vdiff
    gates["video_stream_bit_identical"] = dict(ok=md5_in == md5_out and not vdiff, md5_in=md5_in, md5_out=md5_out)
    a0 = aout[0] if aout else {}
    fdur = float(pout["format"]["duration"])
    adur = float(a0.get("duration", "nan"))
    vdur = float(vout.get("duration", "nan"))
    gates["durations_equal"] = dict(
        ok=abs(fdur - D) < 1e-3 and abs(adur - D) < 1e-3 and abs(vdur - D) < 1e-3 and len(y) == n,
        video_in=D, video_out=vdur, audio_out=adur, container=fdur, wav_samples=len(y), expected_samples=n)
    br = int(a0.get("bit_rate", 0) or 0)
    gates["audio_format"] = dict(
        ok=a0.get("codec_name") == "aac" and a0.get("profile") == "LC" and int(a0.get("sample_rate", 0)) == SR
        and int(a0.get("channels", 0)) == 2 and SPEC["bitrate"][0] <= br <= SPEC["bitrate"][1]
        and "faststart" in " ".join(ctx["cmd"]),
        codec=a0.get("codec_name"), profile=a0.get("profile"), sample_rate=a0.get("sample_rate"),
        channels=a0.get("channels"), bit_rate=br, bit_rate_window=[int(v) for v in SPEC["bitrate"]],
        aac_nominal=mst["aac_bitrate_calibration"]["nominal"])
    with open(ctx["mp4"], "rb") as f:
        head = f.read(1 << 16)
    moov, mdat = head.find(b"moov"), head.find(b"mdat")
    gates["faststart_moov_first"] = dict(ok=moov != -1 and (mdat == -1 or moov < mdat), moov_offset=moov,
                                         mdat_offset=mdat)
    seg = dec[:n]
    lags = np.arange(-2048, 2049)
    ref = yd[int(2 * SR):int(6 * SR), 0]
    cc = [float(np.dot(ref, dec[int(2 * SR) + l:int(6 * SR) + l, 0])) for l in lags]
    lag = int(lags[int(np.argmax(cc))])
    resid = db(np.sqrt(np.mean((seg[:len(yd)] - yd[:len(seg)]) ** 2)) / (np.sqrt(np.mean(yd ** 2)) + 1e-20))
    gates["mp4_audio_in_sync"] = dict(ok=lag == 0 and len(dec) >= n, lag_samples=lag, decoded_samples=len(dec),
                                      aac_residual_db=round(resid, 1))

    # ---- anchors (gate): music downbeat within one frame of the cue, kick onset on the music downbeat ----
    drums = fit_len(st["drums"], n)
    db_set = [float(b) for b in meta.get("downbeat_times", [])]
    sect = meta.get("section_times", {})
    heroes = {getattr(e, "hero_key", None): e for e in events if e.hero}
    arows = []
    for k in ANCHOR_KEYS:
        t, mt = cue_a[k], float(sect.get(k, float("nan")))
        on = kick_onset(drums, mt if np.isfinite(mt) else t)
        h = heroes.get(k)
        arows.append(dict(anchor=k, how=A["how"][k], cue_t=f"{t:.4f}", music_section_t=f"{mt:.4f}",
                          music_vs_cue_ms=f"{(mt - t) * 1000:+.2f}",
                          is_music_downbeat=any(abs(b - mt) < 1e-6 for b in db_set),
                          drums_onset_t=f"{on:.4f}" if on is not None else "",
                          drums_onset_vs_music_ms=f"{(on - mt) * 1000:+.2f}" if on is not None else "",
                          hero_slam_t=f"{h.target:.4f}" if h is not None else "",
                          hero_slam_vs_music_ms=f"{(h.target - mt) * 1000:+.2f}" if h is not None else ""))
    data["arows"] = arows
    gates["anchors_on_music_grid"] = dict(
        ok=all(r["is_music_downbeat"] and abs(float(r["music_vs_cue_ms"])) <= frame * 1000 + 1e-6
               and r["drums_onset_vs_music_ms"] != "" and abs(float(r["drums_onset_vs_music_ms"])) <= tol * 1000
               for r in arows),
        grid_offset_s=A["offset_s"],
        detail={r["anchor"]: f"cue {r['cue_t']} music {r['music_section_t']} ({r['music_vs_cue_ms']} ms) downbeat="
                             f"{r['is_music_downbeat']} kick {r['drums_onset_vs_music_ms']} ms" for r in arows})

    # ---- cue alignment (QC) ----
    rows, iso = [], {}
    for e in events:
        seg_, t0 = event_in_bus(e, sinfo["glue"], ctx["gate"], n)
        iso[id(e)] = (seg_, t0)
        det = detect(seg_, t0, e.detect)
        expect = e.target - e.breath
        err = (det - expect) * 1000 if det is not None else float("nan")
        bt, rise = (bus_onset(sfx_bus, e.target) if e.detect == "onset" else (None, None))
        rows.append(dict(
            cue_index=e.cue["index"], cue_t=f"{e.t:.3f}", type=e.typ,
            params=" ".join(f"{k}={v}" for k, v in e.params.items()),
            sync_rule=e.rule, array_start_t=f"{e.start_t:.4f}", array_end_t=f"{e.end_t:.4f}",
            target_t=f"{e.target:.4f}", breath_ms=f"{e.breath * 1000:.0f}", expected_detect_t=f"{expect:.4f}",
            detector=e.detect,
            readme_anchor_t=f"{e.start_t + e.readme_anchor:.4f}" if hasattr(e, "readme_anchor") else "",
            detected_t=f"{det:.4f}" if det is not None else "none (no samples in the mix)",
            error_ms=f"{err:+.2f}",
            within_1_frame="PASS" if det is not None and abs(err) <= tol * 1000 else "FAIL",
            error_vs_cue_t_ms=f"{(det - e.t) * 1000:+.2f}" if det is not None else "",
            bus_onset_t=f"{bt:.4f}" if bt is not None else "",
            bus_onset_error_ms=f"{(bt - e.target) * 1000:+.2f}" if bt is not None else "",
            bus_onset_rise_db=f"{rise:.1f}" if rise is not None else "",
            gain_db=f"{e.gain_db:+.1f}", send=f"{e.send:.2f}", processing="; ".join(e.flags)))
    data["rows"] = rows
    errs = [abs(float(r["error_ms"])) for r in rows if r["error_ms"] != "+nan"]
    qc["sfx_within_1_frame"] = dict(ok=all(r["within_1_frame"] == "PASS" for r in rows), n=len(rows),
                                    max_abs_error_ms=round(max(errs), 2) if errs else None)

    # ---- SFX hierarchy (QC): a hero slam on every hit anchor, ON the music hit, loudest full band + HP150 ----
    hp_sos = butter("highpass", 150.0, 4)
    lv = []
    for e in events:
        seg_, _ = iso[id(e)]
        segm = seg_ * gm
        lv.append((e, sfx.lufs_momentary_max(segm), sfx.lufs_momentary_max(signal.sosfilt(hp_sos, segm, axis=0))))
    missing = [k for k in HIT_ANCHORS if k not in heroes]
    flams = {k: round((heroes[k].target - mus_a[k]) * 1000, 2) for k in HIT_ANCHORS if k in heroes}
    qc["hero_slam_on_every_hit"] = dict(ok=not missing and all(abs(v) <= QC["hero_flam_s"] * 1000 for v in flams.values()),
                                        missing=missing, slam_vs_music_ms=flams)
    hl = [r for r in lv if r[0].hero]
    ot = [r for r in lv if not r[0].hero]
    if hl:
        hmin = (min(r[1] for r in hl), min(r[2] for r in hl))
        omax = max(ot, key=lambda r: r[1]) if ot else None
        omax_hp = max(ot, key=lambda r: r[2]) if ot else None
        qc["hero_slams_loudest_sfx"] = dict(
            ok=not missing and (omax is None or (hmin[0] > omax[1] and hmin[1] > omax_hp[2])),
            hero_min_lufs_m=round(hmin[0], 1), hero_min_lufs_m_hp150=round(hmin[1], 1),
            loudest_other=f"{omax[0].typ}@{omax[0].t} {omax[1]:.1f} LUFS-M" if omax else "none",
            loudest_other_hp150=f"{omax_hp[0].typ}@{omax_hp[0].t} {omax_hp[2]:.1f} LUFS-M" if omax_hp else "none")
    else:
        qc["hero_slams_loudest_sfx"] = dict(ok=False, reason="no hero slam on any hit anchor")
    # SFX over music at the keyed moments (for loudness.txt)
    bal = []
    mb = music_bus * gm
    for e, l_full, l_hp in lv:
        a, b = max(int((e.target - 0.05) * SR), 0), min(int((e.target + 0.35) * SR), n)
        if b - a < int(0.4 * SR):
            a = max(b - int(0.4 * SR), 0)
        lmus = sfx.lufs_momentary_max(mb[a:b]) if b - a >= int(0.4 * SR) else float("nan")
        bal.append((e, l_full, l_hp, lmus))
    data["bal"] = bal

    # ---- hero hit transients (QC): master RMS 0..30 ms after vs 40..5 ms before each hit ----
    yh = signal.sosfilt(hp_sos, yd, axis=0)
    tmh, lmh = loudness_curve(yh, 0.4, 0.01)
    rises = {}
    for k in HIT_ANCHORS:
        h = int(round(mus_a[k] * SR))
        post = db(np.sqrt(np.mean(yd[h:h + int(0.030 * SR)] ** 2)))
        pre = db(np.sqrt(np.mean(yd[max(h - int(0.040 * SR), 0):h - int(0.005 * SR)] ** 2)))
        sel = (tmh >= mus_a[k] - 0.005) & (tmh <= mus_a[k] + 0.25)
        bed = (tmh >= mus_a[k] - 0.75) & (tmh <= mus_a[k] - 0.25)
        rises[k] = dict(rise_db=round(post - pre, 1), pre_dbfs=round(pre, 1), post_dbfs=round(post, 1),
                        hp150_momentary_max=round(float(lmh[sel].max()), 1),
                        hp150_over_bed_lu=round(float(lmh[sel].max() - np.median(lmh[bed])), 1) if bed.any() else None,
                        limiter_gr_db=round(float(-20 * np.log10(max(mst["lim_gain"][h], 1e-9))), 2))
    data["rises"] = rises
    qc["hero_hit_transients"] = dict(ok=all(r["rise_db"] >= QC["hit_rise_db"] for r in rises.values()),
                                     min_rise_db=QC["hit_rise_db"],
                                     detail={k: f"{r['rise_db']:+.1f} dB ({r['pre_dbfs']} -> {r['post_dbfs']} dBFS), "
                                                f"HP150 M max {r['hp150_momentary_max']}" for k, r in rises.items()})

    # ---- section contour (QC) ----
    sec = section_table(yd, mus_a, n)
    data["sec"] = sec
    S = {r["section"].strip(): r for r in sec}
    c = {}
    if "drop2" in S and "grooveA" in S:
        c["drop2_minus_grooveA"] = (round(S["drop2"]["lufs"] - S["grooveA"]["lufs"], 2),
                                    round(S["drop2"]["lufs_hp150"] - S["grooveA"]["lufs_hp150"], 2))
    if "drop2.noturn" in S and "grooveA.full" in S:
        c["drop2noturn_minus_grooveAfull"] = (round(S["drop2.noturn"]["lufs"] - S["grooveA.full"]["lufs"], 2),
                                              round(S["drop2.noturn"]["lufs_hp150"] - S["grooveA.full"]["lufs_hp150"], 2))
    for nm in ("intro", "bd.dark"):
        if nm in S and "grooveA" in S:
            c[f"{nm}_minus_grooveA"] = (round(S[nm]["lufs"] - S["grooveA"]["lufs"], 2),
                                        round(S[nm]["lufs_hp150"] - S["grooveA"]["lufs_hp150"], 2))
    data["contrast"] = c
    d2c = c.get("drop2_minus_grooveA", (float("nan"),) * 2)
    qc["drop2_over_grooveA"] = dict(ok=min(d2c) >= QC["drop2_lift_lu"], min_lu=QC["drop2_lift_lu"],
                                    full_hp150_lu=list(d2c),
                                    like_for_like_full_hp150_lu=list(c.get("drop2noturn_minus_grooveAfull", ())))
    ic = c.get("intro_minus_grooveA", (float("nan"),) * 2)
    dk = c.get("bd.dark_minus_grooveA")
    qc["phone_contour"] = dict(ok=ic[1] <= -QC["intro_under_lu"] and (dk is None or dk[1] <= -QC["dark_under_lu"]),
                               intro_minus_grooveA_hp150=ic[1], dark_minus_grooveA_hp150=dk[1] if dk else None,
                               limits=[-QC["intro_under_lu"], -QC["dark_under_lu"]])

    # ---- masking (QC): each non-hero cue's own energy over the music in its dominant octave ----
    mrows = [{k: v for k, v in r.items() if k not in ("event", "band")}
             for r in mask_rows(events, music_bus, sinfo, ctx["gate"], n)]
    data["mrows"] = mrows
    bad = [f"{r['type']}@{r['cue_t']} {r['sfx_over_music_dominant_db']:+.1f} dB @ {r['dominant_octave_hz']} Hz"
           for r in mrows if not r["ok"]]
    worst = min(mrows, key=lambda r: r["sfx_over_music_dominant_db"]) if mrows else None
    qc["cues_not_masked"] = dict(ok=not bad, min_db=QC["mask_db"], n=len(mrows),
                                 worst=(f"{worst['type']}@{worst['cue_t']} {worst['sfx_over_music_dominant_db']:+.1f} dB "
                                        f"@ {worst['dominant_octave_hz']} Hz") if worst else None,
                                 failing=bad)
    data["sha256"] = hashlib.sha256(np.ascontiguousarray(y, np.float32).tobytes()).hexdigest()
    return gates, qc, data


def write_report(ctx, gates, qc, data, delivered):
    args, A, events, meta, mst = ctx["args"], ctx["A"], ctx["events"], ctx["meta"], ctx["mst"]
    minfo, sinfo, D, n = ctx["minfo"], ctx["sinfo"], ctx["D"], ctx["n"]
    rep = args.report
    os.makedirs(rep, exist_ok=True)
    yd = ctx["y"].astype(np.float64)

    def csv_out(name, rows):
        with open(os.path.join(rep, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["none"])
            w.writeheader()
            w.writerows(rows)
    csv_out("cue_alignment.csv", data["rows"])
    csv_out("anchor_alignment.csv", data["arows"])
    csv_out("masking.csv", data["mrows"])
    with open(os.path.join(rep, "ffprobe_output.json"), "w") as f:
        json.dump(data["ffprobe_out"], f, indent=1)
    with open(os.path.join(rep, "video_md5.txt"), "w") as f:
        f.write(f"input  {args.video}\n  video stream md5 (ffmpeg -map 0:v -c copy -f md5) {data['md5_in']}\n")
        f.write(f"output {args.out}{'' if delivered else ' (NOT written: a delivery gate failed)'}\n"
                f"  video stream md5 (ffmpeg -map 0:v -c copy -f md5) {data['md5_out']}\n")
        f.write(f"identical: {data['md5_in'] == data['md5_out']}\nffprobe video-stream fields that differ: "
                f"{data['vdiff'] or 'none'}\n")
    eb_m, eb_hp, eb_mp4 = data["eb_m"], data["eb_hp"], data["eb_mp4"]
    ast = astats(ctx["wav"])
    with open(os.path.join(rep, "loudness.txt"), "w") as f:
        f.write(f"mix.py report - {os.path.basename(args.out)}  (music: {args.music}, seed {args.seed})\n")
        f.write(f"cues {args.cues}\nvideo {args.video}\n\n")
        f.write(f"== ffmpeg ebur128, master WAV ==\n{eb_m['summary']}\n\n")
        f.write(f"== ffmpeg ebur128, master through the phone model (4th-order 150 Hz high-pass) ==\n{eb_hp['summary']}\n\n")
        f.write(f"== ffmpeg ebur128, output MP4 (decoded AAC) ==\n{eb_mp4['summary']}\n\n")
        f.write(f"== ffmpeg astats, master WAV ==\n{ast}\n\n")
        f.write("== python checks ==\n")
        f.write(f"8x true peak {data['tp8']:.2f} dBTP, integrated (python BS.1770) {mst['lufs_py']:.2f} LUFS, |DC| "
                f"{data['dc']:.2e}, samples >= -0.01 dBFS: {data['clip_n']}\n")
        f.write(f"frame 0 RMS {data['ff_rms']:.1f} dBFS, momentary 0-400 ms {data['m_first']:.1f} LUFS, last 10 ms "
                f"{data['last10']:.1f} dBFS, fade-out thirds {data['fo_db']} dBFS, last sample {float(np.abs(yd[-1]).max())}\n\n")
        f.write("== anchors ==\n")
        f.write(f"grid offset (least squares over the four anchors) {A['offset_s']:.4f} s; bars {A['bars']}\n")
        for r in data["arows"]:
            f.write(f"{r['anchor']:9s} ({r['how']}): cue {r['cue_t']} music {r['music_section_t']} ({r['music_vs_cue_ms']} ms), "
                    f"downbeat {r['is_music_downbeat']}, kick onset {r['drums_onset_vs_music_ms']} ms, hero slam "
                    f"{r['hero_slam_vs_music_ms'] or '-'} ms\n")
        f.write("\n== master chain ==\n")
        f.write(f"music bus: stems {minfo['raw_lufs']:.2f} LUFS -> {MUSIC_BUS_LUFS} LUFS ({minfo['gain_db']:+.2f} dB); "
                f"section lifts {MUSIC_SECTION_DB} (ramps {MUSIC_SECTION_RAMP_S * 1000:.0f} ms inside the breath); "
                f"sidechain duck max hi {minfo['duck_hi_db'].max():.1f} dB / lo {minfo['duck_lo_db'].max():.1f} dB\n")
        for d in minfo["fx_duck"]:
            f.write(f"  fx stem {sfx.MUSIC_FX_DUCK_DB:+.0f} dB under {d['sfx']} span {d['span']} (music {', '.join(d['music_fx'])})\n")
        hits_txt = ", ".join(f"{A['music'][k]:g}" for k in HIT_ANCHORS)
        f.write(f"SFX bus: non-hero part gated shut {HIT_BREATH['pre_s'] * 1000:.0f} ms before each hit "
                f"({hits_txt} s); glue threshold {sinfo['glue_thr_db']:.1f} dB, "
                f"max GR {sinfo['glue_gr_max_db']:.2f} dB, room rt {ROOM['rt']} s return {ROOM['ret_db']} dB; transient "
                f"stage ceiling {sinfo['peak']['ceiling_dbfs']} dBFS soft clip (make-up estimate "
                f"{sinfo['peak']['makeup_estimate_db']:+.2f} dB), bus true-peak reduction {sinfo['peak']['peak_reduction_db']} dB, "
                f"{sinfo['peak']['samples_shaped']} samples above the knee\n")
        f.write(f"master: EQ {EQ}; bus comp threshold {mst['bus_comp_thr_db']} dB max GR {mst['bus_comp_gr_max_db']} dB; "
                f"make-up {mst['makeup_gain_db']:+.2f} dB; limiter ceiling {mst['ceiling_dbtp']} dBTP (first "
                f"{AAC_START_GUARD['samples']} samples -{AAC_START_GUARD['dip_db']:g} dB), max GR "
                f"{mst['limiter_gr_max_db']} dB, time > 1 dB GR {mst['limiter_time_over_1db_s']} s, > 3 dB "
                f"{mst['limiter_time_over_3db_s']} s\n")
        f.write("master passes (limiter ceiling, lowest local ceiling -> 8x true peak of the WAV / of the decoded AAC): "
                + "; ".join(f"{p['ceiling_dbtp']}, {p['ceiling_min_dbtp']} -> {p['master_tp_8x']} / {p['aac_tp_8x']}"
                            + (f" (AAC overshoot at {', '.join(str(sp['t']) for sp in p['aac_overshoot_spots'])} s)"
                               if p.get('aac_overshoot_spots') else "") for p in mst["passes"]) + "\n")
        bc = mst["aac_bitrate_calibration"]
        f.write(f"AAC rate: nominal 256k measured {bc['measured_first']} b/s -> nominal {bc['nominal'] // 1000}k, "
                f"measured {gates['audio_format']['bit_rate']} b/s\n\n")
        f.write("== sections (master, BS.1770 gated over each window) ==\n")
        f.write(f"{'section':14s} {'window':>13s} {'full LUFS':>10s} {'HP150 LUFS':>11s}\n")
        for r in data["sec"]:
            f.write(f"{r['section']:14s} {r['t0']:6.2f}-{r['t1']:5.2f} {r['lufs']:10.2f} {r['lufs_hp150']:11.2f}\n")
        f.write("contrasts (full / HP150 LU): " + "; ".join(f"{k} {v[0]:+.2f} / {v[1]:+.2f}"
                                                        for k, v in data["contrast"].items()) + "\n\n")
        f.write("== hero hits (master): RMS 0..30 ms after vs 40..5 ms before; HP150 momentary max and over the bed ==\n")
        for k, r in data["rises"].items():
            f.write(f"{k:8s} rise {r['rise_db']:+.1f} dB ({r['pre_dbfs']} -> {r['post_dbfs']} dBFS); HP150 M max "
                    f"{r['hp150_momentary_max']} LUFS, {r['hp150_over_bed_lu']:+.1f} LU over the bed; limiter GR "
                    f"{r['limiter_gr_db']} dB\n")
        f.write("\n== SFX levels at master gain (max momentary of each SFX's own contribution; music bus = max momentary "
                "of the ducked music in the 0.4 s from 50 ms before the sync point) ==\n")
        f.write(f"{'t':>6s} {'sfx':14s} {'LUFS-M':>7s} {'HP150':>7s} {'music':>7s} {'sfx-mus':>8s}  processing\n")
        for e, l1, l2, lmu in data["bal"]:
            name = e.typ + (f"({e.params['size']})" if e.typ == "impact" else "")
            f.write(f"{e.t:6.2f} {name:14s} {l1:7.1f} {l2:7.1f} {lmu:7.1f} {l1 - lmu:+8.1f}  {'; '.join(e.flags)}\n")
        f.write("\nloudest SFX, full band: " + ", ".join(f"{e.typ}@{e.t:g} {l1:.1f}" for e, l1, l2, _ in
                                                       sorted(data["bal"], key=lambda r: -r[1])[:6]) + "\n")
        f.write("loudest SFX, HP150:     " + ", ".join(f"{e.typ}@{e.t:g} {l2:.1f}" for e, l1, l2, _ in
                                                       sorted(data["bal"], key=lambda r: -r[2])[:6]) + "\n")
        f.write("\n== masking: each cue's own energy over the music in its dominant octave (masking.csv) ==\n")
        for r in data["mrows"]:
            f.write(f"{float(r['cue_t']):6.2f} {r['type']:14s} dominant {r['dominant_octave_hz']:>5d} Hz "
                    f"{r['sfx_over_music_dominant_db']:+6.1f} dB, best {r['best_octave_hz']:>5d} Hz "
                    f"{r['sfx_over_music_best_db']:+6.1f} dB, broadband {r['broadband_db']:+6.1f} dB {'' if r['ok'] else ' <- MASKED'}\n")
        f.write("\n== delivery gates (a failure blocks the MP4) ==\n")
        for k, v in gates.items():
            f.write(f"{'PASS' if v['ok'] else 'FAIL'}  {k}: " + ", ".join(f"{a}={b}" for a, b in v.items() if a != "ok") + "\n")
        f.write("\n== mix QC checks ==\n")
        for k, v in qc.items():
            f.write(f"{'PASS' if v['ok'] else 'FAIL'}  {k}: " + ", ".join(f"{a}={b}" for a, b in v.items() if a != "ok") + "\n")
        f.write(f"\nmaster sha256 (float32 samples; the determinism key) {data['sha256']}\n")
        if ctx["issues"]:
            f.write("\n== notes ==\n" + "\n".join(ctx["issues"]) + "\n")

    anchors_plot = A["music"]
    spec_png, wave_png = draw_reports(rep, ctx["y"], ctx["sfx_bus"], ctx["music_bus"], events, anchors_plot, meta,
                                      mst["makeup_gain_db"])
    ok = all(v["ok"] for v in gates.values()) and all(v["ok"] for v in qc.values())
    out = dict(
        all_checks_pass=ok, delivered=delivered, gates=gates, qc=qc, out=args.out,
        wav=args.wav if delivered else None, music=args.music, seed=args.seed, strict=not args.lenient,
        whoosh_sync=args.whoosh_sync, cues=args.cues, video=args.video, duration=D, samples=n,
        anchors=dict(cue=A["cue"], music=A["music"], grid_offset_s=A["offset_s"], how=A["how"]),
        music_meta=dict(section_times=meta.get("section_times"), section_boundaries=meta.get("section_boundaries"),
                        fx_transitions=meta.get("fx_transitions")),
        sections=data["sec"], contrasts_lu=data["contrast"], hero_hits=data["rises"],
        master=dict((k, v) for k, v in mst.items() if k not in ("comp_gain", "lim_gain")),
        music_bus=dict(raw_lufs=round(minfo["raw_lufs"], 2), gain_db=round(minfo["gain_db"], 2), fx_duck=minfo["fx_duck"],
                       section_lift_db=MUSIC_SECTION_DB),
        sfx_bus=dict(glue_thr_db=round(sinfo["glue_thr_db"], 2), glue_gr_max_db=round(sinfo["glue_gr_max_db"], 2),
                     transient_stage=sinfo["peak"], hit_breath=HIT_BREATH),
        replaced_by_music=ctx.get("replaced", []), ffmpeg_mux=ctx["cmd"], notes=ctx["issues"], master_sha256=data["sha256"],
        files=dict(cue_alignment="cue_alignment.csv", anchor_alignment="anchor_alignment.csv", masking="masking.csv",
                   loudness="loudness.txt", spectrogram=os.path.basename(spec_png), waveform=os.path.basename(wave_png),
                   ffprobe="ffprobe_output.json", video_md5="video_md5.txt"))
    with open(os.path.join(rep, "mix_report.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log(f"report -> {rep}  ({'all gates and checks pass' if ok else 'SOME GATES / CHECKS FAIL'})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cues", required=True, help="cue sheet JSON (list of {t, type, ...})")
    ap.add_argument("--video", required=True, help="silent video; its stream is copied untouched")
    ap.add_argument("--music", required=True, choices=("techhouse", "cinematic"))
    ap.add_argument("--out", required=True, help="output MP4 (written only if every delivery gate passes)")
    ap.add_argument("--wav", help="also keep the 48 kHz float32 master WAV here (same condition)")
    ap.add_argument("--report", help="write the QC report into this directory")
    ap.add_argument("--seed", type=int, default=1, help="music seed (default 1)")
    ap.add_argument("--cache-dir", help="cache rendered music stems here (keyed by module source + args)")
    ap.add_argument("--whoosh-sync", choices=("window", "peak"), default="window",
                    help="window (default): a whoosh cue's (t, dur) is the motion window, the array spans it; "
                         "peak: the whoosh envelope peak lands on t")
    ap.add_argument("--lenient", action="store_true",
                    help="do not stop on dropped cues / invalid optional values / missing anchors (use the defaults "
                         "2/14/18/26 and note it); anchor order, grid and section-length errors still stop the run")
    args = ap.parse_args(argv)
    for p in (args.cues, args.video):
        if not os.path.exists(p):
            ap.error(f"not found: {p}")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
