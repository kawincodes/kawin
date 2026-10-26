#!/usr/bin/env python3
"""
music_cinematic.py -- "Cinematic hybrid pulse": an original, fully synthesized trailer-style score for the
KRS Tech Solutions "WE'RE HIRING - VIDEO EDITOR" Instagram Story.

No samples, loops or downloaded audio. Every sound is generated in this file from polyBLEP saws, sines and
seeded noise, shaped with analytic envelopes, time-varying biquads, waveshaping and a synthetic-IR convolution
reverb. Output is deterministic for a given `seed`.

Contract shared with the other audio modules
    key   F minor (F G Ab Bb C Db Eb), tonal centre F
    tempo 120 BPM, straight 16ths
    48 kHz, stereo, float32 arrays shaped (N, 2)

API
    render(duration, drop_time, end_hit_time, seed=1) -> {
        'drums', 'bass', 'music', 'fx' : float32 (N, 2), N = round(duration * 48000), unmastered,
                                         each with true peak <= -6 dBTP (common gain, so the balance holds),
        'meta' : {'bpm', 'key', 'beat_times', 'drop_time', 'end_hit_time_snapped', ...extras}}
    beat_grid(duration, drop_time, end_hit_time) -> grid dict (beats, bars, snapped hit)
    arrange(grid) -> score (the arrangement as data: per-bar note lists + timed events)
    check_score(score, grid) -> list of musical-rule violations (render() raises if any)
    render_demo() -> demo/music_cinematic_demo.wav + spectrogram/waveform PNGs + stem report

Run:  python3 -I music_cinematic.py      (renders the demo)
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import zlib
from collections import defaultdict

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import minimum_filter1d

# ==============================================================================================
# contract / music theory
# ==============================================================================================
SR = 48000
BPM = 120.0
BEAT = 60.0 / BPM            # 0.5 s
S16 = BEAT / 4.0             # 0.125 s
KEY = "F minor"
SCALE = ("F", "G", "Ab", "Bb", "C", "Db", "Eb")
PC = {"C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "Gb": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}
SCALE_PCS = frozenset(PC[n] for n in SCALE)
NOTE_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

PRE = 2.0                    # pre-roll (s) rendered before t=0 so the pad/reverbs are already "running" at t=0
POST = 0.5                   # render margin after `duration` (cropped)
GAP = S16                    # the "breath": sustained parts drop out for one 16th before the drop and the end hit
FADE_IN = 0.005              # click guard at t=0
TAIL_FADE = 0.25             # last 0.25 s fades to digital silence, ending exactly at `duration`
STEM_PEAK_DBTP = -6.5        # all stems share one gain so the hottest true peak sits here (<= -6 dBTP)


def m2f(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0)


def note_name(m: int) -> str:
    m = int(m)
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


# ----------------------------------------------------------------------------------------------
# harmony (all voicings are MIDI numbers; every one is checked against the chord + scale in check_score)
# ----------------------------------------------------------------------------------------------
CHORDS = {
    # name     root   chord tones            bass (sub) note   pad voicing (supersaw strings)   ostinato tones (16ths)    braam (brass) voicing
    "Fm":    dict(root="F",  tones=("F", "Ab", "C"),      bass=29, pad=(53, 56, 60, 65),       ost=(72, 77, 80, 84),  braam=(41, 48, 53)),
    "Db":    dict(root="Db", tones=("Db", "F", "Ab"),     bass=37, pad=(53, 56, 61, 65),       ost=(68, 73, 77, 80),  braam=(37, 44, 49)),
    "Bbm":   dict(root="Bb", tones=("Bb", "Db", "F"),     bass=34, pad=(53, 58, 61, 65),       ost=(70, 73, 77, 82),  braam=(46, 53, 58)),
    "Csus4": dict(root="C",  tones=("C", "F", "G"),       bass=36, pad=(53, 60, 65, 67),       ost=(67, 72, 77, 79),  braam=(36, 43, 48)),
    "Eb":    dict(root="Eb", tones=("Eb", "G", "Bb"),     bass=39, pad=(55, 58, 63, 67),       ost=(70, 75, 79, 82),  braam=(39, 46, 51)),
    "Fm9":   dict(root="F",  tones=("F", "Ab", "C", "G"), bass=29, pad=(53, 60, 68, 72, 79),   ost=(72, 77, 79, 80),  braam=(41, 48, 53)),
}
PROGRESSION = ("Fm", "Db", "Bbm", "Csus4")   # i - VI - iv - Vsus4, one chord per bar from the drop
INTRO_LAST = "Csus4"                          # dominant (sus) pedal in the bar before the drop -> resolves on the drop
INTRO_CYCLE = ("Db", "Fm")                    # earlier intro bars, walking backwards from the drop (long intros only)
TURNAROUND = ("Db", "Eb")                     # last bar before the end hit: VI -> VII -> (i)  "aeolian cadence"
FINAL_CHORD = "Fm9"                           # end hit + tail: Fm(add9)
DRUM_PITCH = {"kick": 32, "low": 41, "mid": 48, "hi": 53, "tick": 113, "tock": 108, "boom": 29, "clapbody": 55}

# ----------------------------------------------------------------------------------------------
# rhythm patterns (16th-step index inside a 4/4 bar). Steps >= 16 (5/4, 6/4 bars) reuse the bar's 2nd half.
# ----------------------------------------------------------------------------------------------
KICK_A = ((0, 1.00), (8, 0.90), (10, 0.72))
KICK_B = ((0, 1.00), (6, 0.70), (8, 0.90), (11, 0.62))
CLAP = ((4, 1.00), (12, 1.00))
HAT_VEL = (0.42, 0.26, 0.80, 0.30)           # per 16th inside a beat: offbeat 8th accented
OPEN_HAT_STEP = 14                            # open hat on the "and" of 4 in odd bars
TOM_GROOVE = ((3, "mid", 0.50), (6, "low", 0.62), (10, "low", 0.55), (13, "mid", 0.46), (14, "low", 0.52))
TOM_FILL_SMALL = ((14, "mid", 0.62), (15, "hi", 0.74))
TOM_FILL_BIG = ((12, "low", 0.62), (13, "low", 0.68), (14, "mid", 0.80), (15, "hi", 0.92))
TOM_FILL_LAST_BEAT = (("low", 0.72), ("mid", 0.78), ("mid", 0.88), ("hi", 1.00))     # 16ths into a hit
CLAP_ROLL_LAST_BEAT = (0.38, 0.50, 0.64, 0.80)
BASS_8THS = ((0, 1.00, 0), (2, 0.62, 0), (4, 0.84, 0), (6, 0.62, 0),
             (8, 0.95, 0), (10, 0.62, 0), (12, 0.84, 0), (14, 0.74, 12))   # (step, vel, octave shift)
OST_PATTERN = (0, 1, 2, 1, 3, 1, 2, 1, 0, 1, 2, 1, 3, 2, 1, 2)              # index into CHORDS[c]['ost']
OST_VEL = (1.00, 0.55, 0.80, 0.55)


def _pstep(s: int) -> int:
    return s if s < 16 else 8 + (s - 16) % 8


# ==============================================================================================
# beat grid
# ==============================================================================================
def beat_grid(duration: float, drop_time: float, end_hit_time: float) -> dict:
    """Beat k sits at drop_time + k*BEAT (k=0 is the drop downbeat). Bars before the drop are counted
    backwards from it (the first one may be partial). Between the drop and the snapped end hit the bars are
    4/4; if the hit is not a multiple of 4 beats after the drop, the last bar becomes 5/4 or 6/4 (or a 3/4 bar
    is appended) so the end hit is ALSO a downbeat. After the hit, 4/4 bars again."""
    D = float(duration)
    drop = float(min(max(drop_time, 0.0), max(D - 3.0, 0.0)))
    k_first = -int(math.floor(drop / BEAT + 1e-9))
    k_last = int(math.ceil((D - drop) / BEAT - 1e-9)) - 1
    hk = int(math.floor((float(end_hit_time) - drop) / BEAT + 0.5))           # nearest beat
    hk_max = int(math.floor((D - 1.0 - drop) / BEAT + 1e-9))                   # keep >= 1 s of tail
    hk = max(1, min(hk, hk_max))
    hit = drop + hk * BEAT
    n4, r = divmod(hk, 4)
    lens = [4] * n4
    if r:
        if n4 and r <= 2:
            lens[-1] += r
        else:
            lens.append(r)
    bars, b = [], 0
    for n in lens:
        bars.append((drop + b * BEAT, n))
        b += n
    beat_times = [round(drop + k * BEAT, 9) for k in range(k_first, k_last + 1)]
    intro_bars = [round(drop - 4 * BEAT * j, 9) for j in range(1, 64) if drop - 4 * BEAT * j >= -1e-9]
    tail_bars = [round(hit + 4 * BEAT * j, 9) for j in range(0, 64) if hit + 4 * BEAT * j < D - 1e-9]
    bar_times = sorted(intro_bars + [round(t, 9) for t, _ in bars] + tail_bars)
    return dict(duration=D, drop_time=drop, end_hit_time_snapped=hit, hit_beats=hk, bars=bars,
                beat_times=beat_times, bar_times=bar_times,
                meter=[dict(t=round(t, 9), beats=n) for t, n in bars])


# ==============================================================================================
# arrangement (data -> timed events)
# ==============================================================================================
def arrange(g: dict) -> dict:
    """Returns the score: chord segments, per-bar note lists and per-instrument event lists.
    Event tuples:  kick (t, vel, big)          tom (t, kind, vel, big)    clap (t, vel)
                   hat (t, vel, open)          tick (t, midi, vel)        crash (t, vel, dur)
                   bass (t, dur, midi, vel, long)  ost (t, dur, midi, vel)
                   braam (t, dur, chord, vel, kind)  boom (t, vel, decay)  riser (t0, t1, vel)  revcym (t_end, dur, vel)
    """
    drop, hit, D = g["drop_time"], g["end_hit_time_snapped"], g["duration"]
    E = defaultdict(list)
    chords = []
    bars_out = []

    # ------------------------------------------------------------------ INTRO / BUILD (0 .. drop)
    if drop > 1e-9:
        t1, k = drop, 0
        while t1 > 1e-9:
            name = INTRO_LAST if k == 0 else INTRO_CYCLE[(k - 1) % len(INTRO_CYCLE)]
            chords.append([t1 - 4 * BEAT, t1, name, "intro"])
            t1 -= 4 * BEAT
            k += 1
        chords.sort(key=lambda c: c[0])
        chords[0][0] = -PRE

        def ch_at(t):
            for c in chords:
                if c[0] - 1e-9 <= t < c[1] - 1e-9:
                    return c[2]
            return chords[-1][2]

        k8 = -int(math.floor(drop / (2 * S16) + 1e-9))
        for j in range(k8, 0):                                    # 8th notes from the first one >= 0 s
            t = drop + j * 2 * S16
            u = max(t, 0.0) / drop
            on_beat = j % 2 == 0
            E["tick"].append((t, DRUM_PITCH["tick"] if on_beat else DRUM_PITCH["tock"], (0.78 if on_beat else 0.6) + 0.22 * u))
            if t < drop - GAP - 1e-9:
                E["bass"].append((t, 0.19, CHORDS[ch_at(t)]["bass"], (0.62 if on_beat else 0.5) + 0.38 * u, False))
        for kb in range(-int(math.floor(drop / BEAT + 1e-9)), -1):  # heartbeat toms on the intro beats
            t = drop + kb * BEAT
            E["tom"].append((t, "low", 0.42 + 0.25 * (t / drop), False))
        for j, (kind, v) in enumerate(TOM_FILL_LAST_BEAT):         # 16th fill on the last beat into the drop
            t = drop - BEAT + j * S16
            if t >= -1e-9:
                E["tom"].append((t, kind, 0.85 * v, False))
        for j in range(-8, 0):                                     # 16th hats over the last 2 beats (crescendo)
            t = drop + j * S16
            if t >= -1e-9 and t < drop - GAP - 1e-9:
                E["hat"].append((t, 0.25 + 0.45 * (j + 8) / 8, False))
        E["riser"].append((max(0.0, drop - 3.0), drop, 0.9))
        E["revcym"].append((drop, min(0.9, drop), 0.9))
        bars_out.append(dict(section="intro", t=0.0, beats=round(drop / BEAT, 3),
                             chords=[(max(0.0, c[0]), c[2]) for c in chords]))

    # ------------------------------------------------------------------ DROP .. END HIT
    nb = len(g["bars"])
    for bi, (bt, nbeats) in enumerate(g["bars"]):
        last = bi == nb - 1
        nsteps = 4 * nbeats
        energy = 0.86 if bi < 2 else (0.93 if bi < 4 else 1.0)
        if last:
            h = nbeats // 2
            segs = ([(bt, bt + h * BEAT, TURNAROUND[0])] if h else []) + [(bt + h * BEAT, bt + nbeats * BEAT, TURNAROUND[1])]
        else:
            segs = [(bt, bt + nbeats * BEAT, PROGRESSION[bi % len(PROGRESSION)])]
        for a, b, c in segs:
            chords.append([a, b, c, "turn" if last else "main"])

        def chord_of(t):
            for a, b, c in segs:
                if a - 1e-9 <= t < b - 1e-9:
                    return c
            return segs[-1][2]

        fill0 = nsteps - 4 if last else nsteps          # last bar: final beat is a fill
        kick = KICK_A if bi % 2 == 0 else KICK_B
        bar_notes = dict(section="last" if last else "groove", t=bt, beats=nbeats,
                         chords=[(a, c) for a, _, c in segs], bass=[], ost=[], pad=[])
        tom_fill = ()
        if not last and bi % 2 == 1:
            tom_fill = TOM_FILL_BIG if bi % 4 == 3 else TOM_FILL_SMALL
        fill_from = min((s for s, _, _ in tom_fill), default=99)

        for s in range(nsteps):
            t = bt + s * S16
            p = _pstep(s)
            if s < fill0:
                for ks, v in kick:
                    if ks == p:
                        E["kick"].append((t, v, bi == 0 and s == 0))
                for cs, v in CLAP:
                    if cs == p:
                        E["clap"].append((t, v * energy))
            if s < nsteps - 1 or not last:
                E["hat"].append((t, HAT_VEL[s % 4] * (0.8 + 0.2 * energy) * (1.0 + (0.35 * (s - fill0) / 4 if s >= fill0 else 0.0)),
                                 p == OPEN_HAT_STEP and bi % 2 == 1 and s < fill0))
            if s % 2 == 0:
                on_beat = s % 4 == 0
                E["tick"].append((t, DRUM_PITCH["tick"] if on_beat else DRUM_PITCH["tock"], 0.85 if on_beat else 0.62))
            # toms
            if s < fill0 and bi >= 2 and p < fill_from:
                for ts, kind, v in TOM_GROOVE:
                    if ts == p:
                        E["tom"].append((t, kind, v * energy, False))
            for ts, kind, v in tom_fill:
                if ts == p:
                    E["tom"].append((t, kind, v, False))
            # bass ostinato (8ths, root of the chord)
            c = chord_of(t)
            if s < fill0:
                for bs, v, octv in BASS_8THS:
                    if bs == p:
                        m = CHORDS[c]["bass"] + octv
                        E["bass"].append((t, 0.2, m, v, False))
                        bar_notes["bass"].append(m)
            elif s < nsteps - 1:                     # last beat: 16th drive into the hit (gap on the last 16th)
                m = CHORDS[c]["bass"]
                E["bass"].append((t, 0.1, m, 0.8 + 0.06 * (s - fill0), False))
                bar_notes["bass"].append(m)
            # spiccato-style ostinato (16ths, chord tones)
            if not (last and s == nsteps - 1):
                m = CHORDS[c]["ost"][OST_PATTERN[p]]
                v = OST_VEL[s % 4] * (0.9 if bi < 2 else 1.0)
                E["ost"].append((t, 0.085, m, v))
                bar_notes["ost"].append(m)
        if last:                                      # turnaround fill into the end hit
            E["tom"].append((bt + (fill0 - 4) * S16, "low", 0.66, False))
            E["tom"].append((bt + (fill0 - 2) * S16, "mid", 0.70, False))
            for j, (kind, v) in enumerate(TOM_FILL_LAST_BEAT):
                E["tom"].append((bt + (fill0 + j) * S16, kind, v, False))
            for j, v in enumerate(CLAP_ROLL_LAST_BEAT):
                E["clap"].append((bt + (fill0 + j) * S16, v))
        for a, b, c in segs:
            bar_notes["pad"].append(list(CHORDS[c]["pad"]))
        # accents / ear candy
        if bi == 0:
            E["crash"].append((bt, 1.0, 2.6))
            E["boom"].append((bt, 1.0, 1.0))
            E["tom"].append((bt, "low", 1.0, True))
            E["braam"].append((bt, 1.9, segs[0][2], 1.0, "hit"))
        elif bi % 4 == 0 and not last:
            E["crash"].append((bt, 0.7, 2.0))
            E["boom"].append((bt, 0.55, 0.6))
            E["braam"].append((bt, 1.5, segs[0][2], 0.6, "hit"))
            E["revcym"].append((bt, 0.5, 0.6))
        if last:
            a, b, c = segs[-1]
            t0 = max(a, hit - GAP - 2.0)
            E["braam"].append((t0, hit - GAP - t0, c, 0.8, "rise"))
            E["riser"].append((max(bt, hit - 2.5), hit, 1.0))
            E["revcym"].append((hit, 1.2, 1.0))
        bars_out.append(bar_notes)

    # ------------------------------------------------------------------ END HIT + TAIL
    chords.append([hit, D + POST, FINAL_CHORD, "final"])
    E["kick"].append((hit, 1.0, True))
    E["tom"].append((hit, "low", 1.0, True))
    E["clap"].append((hit, 1.0))
    E["crash"].append((hit, 1.0, min(3.2, D - hit + POST)))
    E["boom"].append((hit, 1.0, 1.5))
    E["braam"].append((hit, D - hit, "Fm", 1.0, "tail"))
    E["bass"].append((hit, D - hit, CHORDS[FINAL_CHORD]["bass"], 1.0, True))
    for j in range(1, 7):                                       # the clock keeps ticking, fading away
        t = hit + j * 2 * S16
        if t < D - 0.3:
            E["tick"].append((t, DRUM_PITCH["tock"] if j % 2 else DRUM_PITCH["tick"], 0.55 * 0.72 ** j))
    bars_out.append(dict(section="tail", t=hit, beats=round((D - hit) / BEAT, 3),
                         chords=[(hit, FINAL_CHORD)], pad=[list(CHORDS[FINAL_CHORD]["pad"])]))
    for k in E:
        E[k].sort(key=lambda e: e[0])
    return dict(chords=[tuple(c) for c in sorted(chords, key=lambda c: c[0])], events=dict(E), bars=bars_out)


def chord_at(score: dict, t: float) -> str:
    for a, b, c, _ in score["chords"]:
        if a - 1e-9 <= t < b - 1e-9:
            return c
    return score["chords"][-1][2]


def check_score(score: dict, g: dict) -> list:
    """Musical correctness: every pitched note in F natural minor, harmony notes are chord tones,
    bass plays chord roots, chord tones are diatonic, and the grid puts bar lines on the drop and the hit."""
    bad = []
    for name, c in CHORDS.items():
        for n in c["tones"]:
            if PC[n] not in SCALE_PCS:
                bad.append(f"chord {name}: tone {n} not in {KEY}")
        tones = {PC[n] for n in c["tones"]}
        if c["bass"] % 12 != PC[c["root"]]:
            bad.append(f"chord {name}: bass {note_name(c['bass'])} is not the root")
        for part in ("pad", "ost", "braam"):
            for m in c[part]:
                if m % 12 not in tones:
                    bad.append(f"chord {name}: {part} note {note_name(m)} not a chord tone")
    E = score["events"]
    pitched = 0
    for t, dur, m, v, lng in E.get("bass", []):
        pitched += 1
        c = chord_at(score, t)
        if m % 12 != PC[CHORDS[c]["root"]]:
            bad.append(f"bass {note_name(m)} at {t:.3f}s is not the root of {c}")
    for t, dur, m, v in E.get("ost", []):
        pitched += 1
        c = chord_at(score, t)
        if m % 12 not in {PC[n] for n in CHORDS[c]["tones"]}:
            bad.append(f"ostinato {note_name(m)} at {t:.3f}s not in {c}")
    for t, dur, c, v, kind in E.get("braam", []):
        pitched += len(CHORDS[c]["braam"])
        if c != chord_at(score, t + 1e-6) and kind != "tail":
            bad.append(f"braam {c} at {t:.3f}s disagrees with the chord {chord_at(score, t + 1e-6)}")
    for a, b, c, _ in score["chords"]:
        pitched += len(CHORDS[c]["pad"])
    every = ([e[2] for e in E.get("bass", [])]
             + [e[2] for e in E.get("ost", [])] + [e[1] for e in E.get("tick", [])]
             + [m for e in E.get("braam", []) for m in CHORDS[e[2]]["braam"]]
             + [m for c in score["chords"] for m in CHORDS[c[2]]["pad"]]
             + [DRUM_PITCH[e[1]] for e in E.get("tom", [])] + list(DRUM_PITCH.values()))
    for m in every:
        if m % 12 not in SCALE_PCS:
            bad.append(f"note {note_name(m)} is outside {KEY}")
    drop, hit = g["drop_time"], g["end_hit_time_snapped"]
    if not any(abs(t - drop) < 1e-9 for t in g["bar_times"]) and drop > 0:
        bad.append("no bar line on the drop")
    if not any(abs(t - hit) < 1e-9 for t in g["bar_times"]):
        bad.append("no bar line on the end hit")
    if abs(((hit - drop) / BEAT) - round((hit - drop) / BEAT)) > 1e-9:
        bad.append("end hit is off the beat grid")
    bt = np.array(g["beat_times"])
    if len(bt) > 1 and np.max(np.abs(np.diff(bt) - BEAT)) > 1e-9:
        bad.append("beat grid is not uniform")
    score["n_pitched_checked"] = pitched + len(every)
    return bad


# ==============================================================================================
# DSP helpers
# ==============================================================================================
def _db(v):
    return 20.0 * math.log10(max(float(v), 1e-12))


def _g(db):
    return 10.0 ** (db / 20.0)


def _rng(seed, name, i=0):
    return np.random.default_rng([int(seed), zlib.crc32(name.encode()), int(i)])


def _idx(t):
    return int(round((t + PRE) * SR))


def _edge(n, a, r):
    """Envelope of length n with raised-cosine fade-in of a samples and fade-out of r samples."""
    e = np.ones(n)
    a, r = min(a, n), min(r, n)
    if a > 0:
        e[:a] = 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / a)
    if r > 0:
        e[n - r:] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(r) + 1) / r)
    return e


def _pan(x, p):
    th = (np.clip(p, -1, 1) + 1.0) * np.pi / 4.0
    return np.stack([x * np.cos(th), x * np.sin(th)], axis=1) * math.sqrt(2.0)


def _st(x):
    return x if x.ndim == 2 else np.stack([x, x], axis=1)


def _place(bus, x, t, gain=1.0):
    x = _st(x)
    i = _idx(t)
    a, b = max(i, 0), min(i + len(x), len(bus))
    if b > a:
        bus[a:b] += gain * x[a - i:b - i]


def _place_end(bus, x, t_end, gain=1.0):
    _place(bus, x, t_end - len(x) / SR, gain)


def _sos(kind, f, order=2):
    bt = {"lp": "lowpass", "hp": "highpass", "bp": "bandpass"}[kind]
    return signal.butter(order, f, btype=bt, fs=SR, output="sos")


def _filt(x, kind, f, order=2):
    return signal.sosfilt(_sos(kind, f, order), x, axis=0)


def _rbj(kind, fc, q):
    w0 = 2.0 * np.pi * fc / SR
    cw, sw = np.cos(w0), np.sin(w0)
    al = sw / (2.0 * q)
    if kind == "lp":
        b0 = b2 = (1.0 - cw) / 2.0
        b1 = 1.0 - cw
    elif kind == "hp":
        b0 = b2 = (1.0 + cw) / 2.0
        b1 = -(1.0 + cw)
    else:                                   # band-pass, 0 dB peak
        b0, b1, b2 = al, 0.0 * al, -al
    a0 = 1.0 + al
    return np.stack([b0 / a0, b1 / a0, b2 / a0, np.ones_like(a0), -2.0 * cw / a0, (1.0 - al) / a0], axis=-1)


def _tvf(x, kind, fc, q=0.707, stages=1, block=32):
    """Time-varying RBJ biquad cascade: coefficients updated every `block` samples, state carried over."""
    mono = x.ndim == 1
    X = x[:, None] if mono else x
    N, C = X.shape
    if np.ndim(fc) == 0:
        sos = np.tile(_rbj(kind, float(np.clip(fc, 15.0, 0.45 * SR)), q), (stages, 1))
        Y = signal.sosfilt(sos, X, axis=0)
        return Y[:, 0] if mono else Y
    fc = np.asarray(fc, dtype=np.float64)
    starts = np.arange(0, N, block)
    centers = np.minimum(starts + block // 2, N - 1)
    coefs = _rbj(kind, np.clip(fc[centers], 15.0, 0.45 * SR), q)
    Y = np.empty((N, C))
    zi = np.zeros((stages, 2, C))
    for j, s in enumerate(starts):
        sos = np.tile(coefs[j], (stages, 1))
        Y[s:s + block], zi = signal.sosfilt(sos, X[s:s + block], axis=0, zi=zi)
    return Y[:, 0] if mono else Y


def _saw(freq, phase0=0.0):
    """polyBLEP (band-limited) sawtooth for a per-sample frequency array."""
    dt = np.asarray(freq, dtype=np.float64) / SR
    ph = (phase0 + np.cumsum(dt) - dt[0]) % 1.0
    y = 2.0 * ph - 1.0
    m = ph < dt
    u = ph[m] / dt[m]
    y[m] -= u + u - u * u - 1.0
    m = ph > 1.0 - dt
    u = (ph[m] - 1.0) / dt[m]
    y[m] -= u * u + u + u + 1.0
    return y


def _sine(freq, phase0=0.0):
    return np.sin(2.0 * np.pi * (phase0 + np.cumsum(np.asarray(freq, dtype=np.float64)) / SR))


def _norm(x):
    return x / (np.max(np.abs(x)) + 1e-12)


def _pink(n, rng):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    X /= np.sqrt(np.maximum(f, 20.0))
    return _norm(np.fft.irfft(X, n))


def _mono_below(x, fc=140.0):
    lo = signal.sosfiltfilt(_sos("lp", fc, 4), x, axis=0)
    return x - lo + lo.mean(axis=1, keepdims=True)


_IR = {}


def _ir(name):
    """Synthetic stereo impulse response: decorrelated noise, three bands with their own decay, soft build-up."""
    if name in _IR:
        return _IR[name]
    rt_lo, rt_mid, rt_hi, length, pre, seed = {"hall": (2.4, 2.1, 1.1, 3.0, 0.020, 11),
                                               "room": (0.8, 0.7, 0.4, 1.1, 0.006, 12)}[name]
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    ir = np.zeros((n, 2))
    for c in range(2):
        w = rng.standard_normal(n)
        lo = _filt(w, "lp", 350.0, 2)
        hi = _filt(w, "hp", 3500.0, 2)
        mid = w - lo - hi
        ir[:, c] = (lo * 10 ** (-3 * t / rt_lo) + mid * 10 ** (-3 * t / rt_mid) + hi * 10 ** (-3 * t / rt_hi))
    ir *= (1.0 - np.exp(-t / 0.014))[:, None]
    ir *= _edge(n, 0, int(0.15 * n))[:, None]
    ir = np.concatenate([np.zeros((int(pre * SR), 2)), ir])
    ir /= np.sqrt(np.sum(ir ** 2, axis=0, keepdims=True))
    _IR[name] = ir
    return ir


def _reverb(x, name, hp=250.0, lp=9000.0):
    X = _st(x)
    ir = _ir(name)
    wet = np.stack([signal.fftconvolve(X[:, c], ir[:, c])[:len(X)] for c in range(2)], axis=1)
    return _filt(_filt(wet, "hp", hp, 2), "lp", lp, 2)


def _pingpong(x, delay, fb=0.38, taps=6, hp=350.0, lp=2800.0):
    X = _st(x).mean(axis=1)
    d = int(round(delay * SR))
    wet = np.zeros((len(X), 2))
    for k in range(1, taps + 1):
        if k * d >= len(X):
            break
        wet[k * d:, (k - 1) % 2] += fb ** (k - 1) * X[:len(X) - k * d]
    return _filt(_filt(wet, "hp", hp, 2), "lp", lp, 2)


def _auto(points, L, log=True):
    """Automation curve over the bus timeline from [(t_seconds, value), ...]."""
    pts = sorted(points)
    tt = np.array([_idx(t) for t, _ in pts], dtype=np.float64)
    vv = np.array([v for _, v in pts], dtype=np.float64)
    x = np.arange(L, dtype=np.float64)
    return np.exp(np.interp(x, tt, np.log(vv))) if log else np.interp(x, tt, vv)


def _curve(t0, t1, v0, v1, shape=1.5, n=24):
    """Breakpoints for a curved ramp v0 -> v1 between t0 and t1 (log-domain interpolation in _auto)."""
    u = np.linspace(0, 1, n)
    return [(t0 + (t1 - t0) * a, v0 * (v1 / v0) ** (a ** shape)) for a in u]


def _duck(times, L, depth, release=0.13, hold=0.012, pre=0.004):
    g = np.ones(L)
    a, h, r = int(pre * SR), int(hold * SR), int(release * 6 * SR)
    shape = np.concatenate([0.5 - 0.5 * np.cos(np.pi * np.arange(a) / a), np.ones(h),
                            np.exp(-np.arange(r) / SR / release)])
    seg = 1.0 - depth * shape
    for t, dep in times:
        i = _idx(t) - a
        lo, hi = max(i, 0), min(i + len(seg), L)
        if hi > lo:
            g[lo:hi] = np.minimum(g[lo:hi], 1.0 - dep * (1.0 - seg[lo - i:hi - i]) / depth)
    return g


def _gap_gate(times, L, gap=GAP, ramp=0.004):
    g = np.ones(L)
    r = int(ramp * SR)
    for T in times:
        a, b = _idx(T - gap), _idx(T)
        g[a - r:a] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(r) + 1) / r)
        g[a:b] = 0.0
        g[b:b + r] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / r)
    return g


def _true_peak(x):
    up = signal.resample_poly(_st(x), 4, 1, axis=0)
    return float(np.max(np.abs(up)))


_KW = ((np.array([1.53512485958697, -2.69169618940638, 1.19839281085285]),
        np.array([1.0, -1.69065929318241, 0.73248077421585])),
       (np.array([1.0, -2.0, 1.0]), np.array([1.0, -1.99004745483398, 0.99007225036621])))


def _block_loudness(x, win=0.4, hop=0.1):
    y = _st(x)
    for b, a in _KW:
        y = signal.lfilter(b, a, y, axis=0)
    w, h = int(win * SR), int(hop * SR)
    if len(y) < w:
        return np.array([-np.inf])
    c = np.cumsum(np.concatenate([np.zeros((1, 2)), y ** 2]), axis=0)
    starts = np.arange(0, len(y) - w + 1, h)
    ms = (c[starts + w] - c[starts]) / w
    return -0.691 + 10.0 * np.log10(np.sum(ms, axis=1) + 1e-20)


def lufs(x):
    """ITU-R BS.1770-4 integrated loudness (absolute -70 and relative -10 LU gates)."""
    lb = _block_loudness(x)
    lb = lb[lb > -70.0]
    if not len(lb):
        return -np.inf
    rel = 10 * np.log10(np.mean(10 ** (lb / 10))) - 10.0
    lb = lb[lb > rel]
    return float(10 * np.log10(np.mean(10 ** (lb / 10))))


def _norm_lufs(x, target):
    return x * _g(target - lufs(x))


# ==============================================================================================
# instruments
# ==============================================================================================
def _kick(vel, rng, big=False):
    dur = 1.15 if big else 0.55
    n = int(dur * SR)
    t = np.arange(n) / SR
    f1 = float(m2f(DRUM_PITCH["kick"]))                       # Ab1 51.9 Hz tail (F-minor chord tone)
    f = f1 + ((215.0 if big else 165.0) - f1) * np.exp(-t / (0.048 if big else 0.032))
    env = np.exp(-np.maximum(t - (0.03 if big else 0.012), 0) / (0.34 if big else 0.16))
    body = np.tanh(2.2 * _sine(f) * env) / np.tanh(2.2)       # saturation -> 87/131 Hz harmonics for phones
    click = _norm(signal.sosfilt(_sos("bp", (2800.0, 5200.0), 2), rng.standard_normal(n))) * np.exp(-t / 0.0016)
    knock = _norm(signal.sosfilt(_sos("bp", (180.0, 600.0), 2), rng.standard_normal(n))) * np.exp(-t / 0.01)
    x = body + 0.2 * click + (0.22 if big else 0.12) * knock
    return x * _edge(n, int(0.002 * SR), int(0.03 * SR)) * vel


def _tom(kind, vel, rng, big=False):
    f0 = float(m2f(DRUM_PITCH[kind]))
    decay = {"low": 0.34, "mid": 0.24, "hi": 0.18}[kind] * (2.6 if big else 1.0)
    n = int(min(decay * 5.0, 2.4) * SR)
    t = np.arange(n) / SR
    f = f0 * (1.0 + 0.55 * np.exp(-t / 0.028))
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = (np.sin(ph) * np.exp(-t / decay) + 0.30 * np.sin(1.593 * ph + 0.4) * np.exp(-t / (0.35 * decay))
            + 0.14 * np.sin(2.136 * ph + 1.1) * np.exp(-t / (0.2 * decay)))
    skin = _norm(signal.sosfilt(_sos("bp", (160.0, 1100.0), 2), rng.standard_normal(n))) * np.exp(-t / 0.02)
    stick = _norm(signal.sosfilt(_sos("hp", 2200.0, 2), rng.standard_normal(n))) * np.exp(-t / 0.0025)
    x = np.tanh(1.7 * (body + 0.32 * skin + 0.08 * stick)) / np.tanh(1.7)
    return x * _edge(n, int(0.002 * SR), int(0.02 * SR)) * vel


def _clap(vel, rng):
    n = int(0.55 * SR)
    t = np.arange(n) / SR
    env = np.zeros(n)
    for dt, gg in ((0.0, 0.55), (0.0085, 0.72), (0.0175, 0.86), (0.029, 1.0)):
        env += gg * (t >= dt) * np.exp(-np.maximum(t - dt, 0) / 0.0045)
    env += 0.42 * (t >= 0.029) * np.exp(-np.maximum(t - 0.029, 0) / 0.085)
    out = np.zeros((n, 2))
    for c in range(2):
        nz = rng.standard_normal(n)
        core = _norm(_tvf(nz, "bp", 1350.0, q=0.75))
        core = _filt(core, "hp", 450.0, 2)
        air = _norm(_filt(nz, "hp", 6500.0, 2))
        out[:, c] = (core + 0.22 * air) * env
    body = np.sin(2 * np.pi * float(m2f(DRUM_PITCH["clapbody"])) * t) * np.exp(-t / 0.045) * 0.35
    x = np.tanh(1.6 * (out + body[:, None])) / np.tanh(1.6)
    return x * _edge(n, int(0.002 * SR), int(0.03 * SR))[:, None] * vel


def _hat(vel, rng, open_=False):
    n = int((0.38 if open_ else 0.07) * SR)
    t = np.arange(n) / SR
    nz = rng.standard_normal(n)
    x = _norm(_filt(nz, "hp", 7200.0, 4)) + 0.35 * _norm(_tvf(nz, "bp", 10500.0, q=2.5))
    x = _filt(x, "lp", 15500.0, 2) * np.exp(-t / (0.11 if open_ else 0.013))
    return x * _edge(n, int(0.002 * SR), int(0.01 * SR)) * vel


def _tick(midi, vel, rng):
    """Clock tick: a tiny tuned modal click (fundamental + fifth) with a 1 ms noise snap."""
    n = int(0.05 * SR)
    t = np.arange(n) / SR
    f0 = float(m2f(midi))
    x = (np.sin(2 * np.pi * f0 * t) * np.exp(-t / 0.0065) + 0.45 * np.sin(2 * np.pi * 1.5 * f0 * t + 0.7) * np.exp(-t / 0.0035))
    snap = _norm(_filt(rng.standard_normal(n), "hp", 6000.0, 2)) * np.exp(-t / 0.001)
    x = x + 0.4 * snap
    return x * _edge(n, int(0.002 * SR), int(0.008 * SR)) * vel


def _crash(vel, rng, dur=2.6):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    fr = np.exp(rng.uniform(np.log(4500.0), np.log(12000.0), 36))
    for c in range(2):
        nz = _filt(rng.standard_normal(n), "hp", 4500.0, 4)
        nz = _norm(_filt(nz, "lp", 14000.0, 2)) * np.exp(-t / 0.75)
        sh = np.zeros(n)
        for f in fr:
            sh += np.sin(2 * np.pi * f * (1 + rng.uniform(-0.002, 0.002)) * t + rng.uniform(0, 6.3)) * np.exp(-t / rng.uniform(0.35, 1.3))
        out[:, c] = nz + 0.35 * _norm(sh) * 0.6
    att = 1.0 - np.exp(-t / 0.002)
    return out * (att * _edge(n, int(0.002 * SR), int(0.25 * n)))[:, None] * vel


def _boom(vel, rng, decay=1.0):
    n = int(min(decay * 4.0, 4.0) * SR)
    t = np.arange(n) / SR
    f1 = float(m2f(DRUM_PITCH["boom"]))
    f = f1 + (2.0 * f1 - f1) * np.exp(-t / 0.07)
    x = np.tanh(1.8 * _sine(f) * np.exp(-t / decay)) / np.tanh(1.8)
    thump = _norm(_filt(rng.standard_normal(n), "lp", 220.0, 2)) * np.exp(-t / 0.03)
    return _filt((x + 0.25 * thump) * _edge(n, int(0.003 * SR), int(0.1 * n)), "hp", 30.0, 2) * vel


def _riser(T, vel, rng):
    """Filtered pink noise rising to its end; tremolo locked to the grid: 8ths -> 16ths -> 32nds."""
    n = int(T * SR)
    t = np.arange(n) / SR
    u = t / T
    rem = T - t
    rate = np.where(rem > 1.0, 4.0, np.where(rem > 0.5, 8.0, 16.0))
    ph = np.cumsum(rate) / SR
    ph -= ph[-1]
    trem = 0.62 + 0.38 * np.cos(2 * np.pi * ph)
    fc = 320.0 * (7500.0 / 320.0) ** (u ** 1.25)
    out = np.zeros((n, 2))
    for c in range(2):
        x = _tvf(_pink(n, rng), "bp", fc, q=1.4, stages=1, block=64)
        out[:, c] = _norm(x)
    mid = out.mean(axis=1, keepdims=True)
    width = (0.3 + 0.7 * u)[:, None]
    out = mid + width * (out - mid)
    env = (0.05 + 0.95 * u ** 2.2) * trem
    return out * (env * _edge(n, int(0.02 * SR), int(0.003 * SR)))[:, None] * vel


def _revcym(dur, vel, rng):
    x = _crash(1.0, rng, dur=max(dur * 1.6, 1.2))[::-1]
    x = x[-int(dur * SR):]
    n = len(x)
    x = _reverb(x, "hall", hp=2000.0, lp=12000.0) * 0.6 + x
    u = np.arange(n) / n
    return x * (u ** 1.5 * _edge(n, int(0.01 * SR), int(0.003 * SR)))[:, None] * vel


SS_DETUNE = np.array([-1.0, -0.64, -0.31, 0.0, 0.29, 0.62, 1.0])
SS_GAIN = np.array([0.70, 0.80, 0.90, 1.00, 0.90, 0.80, 0.70])
SS_PAN = np.array([-0.85, 0.55, -0.35, 0.0, 0.35, -0.55, 0.85])


def _supersaw(midi, n, rng, spread_c=15.0):
    f0 = float(m2f(midi))
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for d, gg, p in zip(SS_DETUNE, SS_GAIN, SS_PAN):
        cents = d * spread_c + rng.uniform(1.5, 3.5) * np.sin(2 * np.pi * rng.uniform(0.12, 0.45) * t + rng.uniform(0, 6.3))
        out += _pan(_saw(f0 * 2.0 ** (cents / 1200.0), rng.random()) * gg, p)
    return out / SS_GAIN.sum()


def _pad_bus(chords, L, seed, D):
    """String-like supersaw ensemble. Voices that hold the same pitch across chords are tied (no re-attack)."""
    lines = defaultdict(list)                    # voice index -> [t0, t1, midi]
    for a, b, c, kind in chords:
        for v, m in enumerate(CHORDS[c]["pad"]):
            ln = lines[v]
            if ln and ln[-1][2] == m and abs(ln[-1][1] - a) < 1e-6:
                ln[-1][1] = b
            else:
                ln.append([a, b, m, kind])
    bus = np.zeros((L, 2))
    k = 0
    for v, ln in lines.items():
        for a, b, m, kind in ln:
            att = 0.07 if kind == "final" else (0.45 if a < -1.0 else 0.32)
            rel = 0.45
            n = int((b - a + rel) * SR)
            x = _supersaw(m, n, _rng(seed, "pad", k))
            x *= _edge(n, int(att * SR), int(rel * SR))[:, None]
            _place(bus, x, a)
            k += 1
    return bus


def _mono_synth(notes, L, osc, fenv_tau, amp_shape):
    """Monophonic synth voice: stepped pitch with 2 ms glide, per-note amp and filter envelopes."""
    freq = np.zeros(L)
    amp = np.zeros(L)
    fenv = np.zeros(L)
    notes = sorted(notes, key=lambda e: e[0])
    for j, (t, dur, m, v, lng) in enumerate(notes):
        i0 = _idx(t)
        if i0 >= L:
            continue
        freq[max(i0, 0):] = float(m2f(m))
        env, fe = amp_shape(dur, lng)
        i1 = min(i0 + len(env), L)
        if j + 1 < len(notes):
            i1 = min(i1, _idx(notes[j + 1][0]) + int(0.005 * SR))
        if i1 <= max(i0, 0):
            continue
        lo = max(i0, 0)
        e = env[lo - i0:i1 - i0].copy()
        if i1 - i0 < len(env):                   # cut by the next note: short release, no click
            r = min(int(0.005 * SR), len(e))
            e[len(e) - r:] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(r) + 1) / r)
        amp[lo:i1] = np.maximum(amp[lo:i1], e * v)
        fenv[lo:i1] = np.maximum(fenv[lo:i1], fe[lo - i0:i1 - i0] * v)
    first = next((float(m2f(e[2])) for e in notes), 100.0)
    freq[freq == 0] = first
    lf = signal.lfilter([1 - math.exp(-1 / (0.002 * SR))], [1, -math.exp(-1 / (0.002 * SR))], np.log(freq),
                        zi=[math.exp(-1 / (0.002 * SR)) * math.log(first)])[0]
    return np.exp(lf), amp, fenv


def _bass_bus(notes, L, cutoff):
    """8th-note ostinato bass: sine sub + two detuned saws at pitch + a saw an octave up (mid 'grit' that
    phone speakers can reproduce), 24 dB low-pass with a per-note envelope, gentle drive."""
    def shape(dur, lng):
        if lng:
            n = int((dur + 0.1) * SR)
            t = np.arange(n) / SR
            return np.exp(-t / 0.9) * _edge(n, int(0.006 * SR), int(0.05 * SR)), np.exp(-t / 0.35)
        n = int((dur + 0.03) * SR)
        t = np.arange(n) / SR
        e = (0.74 + 0.26 * np.exp(-t / 0.05)) * _edge(n, int(0.006 * SR), int(0.03 * SR))
        return e, np.exp(-t / 0.08)
    f, amp, fenv = _mono_synth(notes, L, "bass", 0.08, shape)
    sub = _sine(f)
    saw = 0.5 * (_saw(f * 2 ** (-7 / 1200.0), 0.1) + _saw(f * 2 ** (7 / 1200.0), 0.6)) + 0.45 * _saw(2.0 * f, 0.35)
    fc = np.clip(cutoff * (1.0 + 2.6 * fenv), 40.0, 9000.0)
    saw = _tvf(saw, "lp", fc, q=0.85, stages=2, block=32)
    x = (0.85 * sub + 0.8 * saw) * amp
    x = np.tanh(1.6 * x) / np.tanh(1.6)
    x = _filt(x, "hp", 30.0, 4)
    return x


def _ost_bus(notes, L, cutoff):
    def shape(dur, lng):
        n = int((dur + 0.03) * SR)
        t = np.arange(n) / SR
        e = (0.22 + 0.78 * np.exp(-t / 0.05)) * _edge(n, int(0.002 * SR), int(0.025 * SR))
        return e, np.exp(-t / 0.04)
    f, amp, fenv = _mono_synth([(t, d, m, v, False) for t, d, m, v in notes], L, "ost", 0.04, shape)
    x = 0.6 * _saw(f, 0.0) + 0.4 * (_saw(f * 2 ** (7 / 1200.0), 0.2) - _saw(f * 2 ** (7 / 1200.0), 0.5))
    fc = np.clip(cutoff * (1.0 + 2.4 * fenv), 200.0, 9000.0)
    x = _tvf(x, "lp", fc, q=0.9, stages=2, block=32) * amp
    x = _filt(x, "hp", 260.0, 2)
    return x


def _braam(chord, dur, vel, kind, rng):
    """Brass-like 'braam': 4 detuned saws per note, pitch scoop, low-pass envelope, drive."""
    rel = 0.3 if kind != "rise" else 0.0
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for m, w in zip(CHORDS[chord]["braam"], (1.0, 0.75, 0.6)):
        f0 = float(m2f(m))
        scoop = -40.0 * np.exp(-t / 0.07) if kind != "rise" else -18.0 * (1 - np.minimum(t / max(dur, 1e-3), 1))
        for d, p in ((-12.0, -0.7), (-4.0, 0.25), (5.0, -0.25), (13.0, 0.7)):
            vib = 3.0 * np.sin(2 * np.pi * rng.uniform(4.5, 5.5) * t + rng.uniform(0, 6.3)) * np.minimum(t / 0.6, 1.0)
            out += _pan(_saw(f0 * 2.0 ** ((d + scoop + vib) / 1200.0), rng.random()) * w, p)
    if kind == "rise":
        u = np.clip(t / max(dur, 1e-3), 0, 1)
        fc = 170.0 * (2400.0 / 170.0) ** (u ** 1.5)
        amp = 0.04 + 0.96 * u ** 1.8
    else:
        fc = 120.0 + 1900.0 * (1.0 - np.exp(-t / 0.05)) * np.exp(-t / (0.5 if kind == "hit" else 0.8)) + 300.0 * np.exp(-t / 2.5)
        amp = 0.4 + 0.6 * np.exp(-t / (0.35 if kind == "hit" else 0.6))
        if kind == "tail":
            amp = amp * np.exp(-t / 2.2)
    out = _tvf(out, "lp", fc, q=0.9, stages=2, block=64)
    out = np.tanh(2.0 * _norm(out)) / np.tanh(2.0)
    out *= (amp * _edge(n, int(0.012 * SR), int((rel if rel else 0.003) * SR)))[:, None]
    out = _filt(out, "hp", 70.0, 4)
    return out * vel


# ==============================================================================================
# mix
# ==============================================================================================
BUS_LUFS = {                 # loudness of each bus while it plays (before the common stem gain)
    "kick": -19.5, "tom": -22.0, "clap": -23.0, "hat": -31.0, "tick": -30.0, "crash": -27.0,
    "bass": -19.5,
    "pad": -21.5, "ost": -26.0, "braam": -22.5,
    "boom": -21.0, "riser": -25.0, "revcym": -27.0,
}
STEMS = {"drums": ("kick", "tom", "clap", "hat", "tick", "crash"), "bass": ("bass",),
         "music": ("pad", "ost", "braam"), "fx": ("boom", "riser", "revcym")}


def render(duration, drop_time, end_hit_time, seed=1, _buses=False):
    g = beat_grid(duration, drop_time, end_hit_time)
    score = arrange(g)
    bad = check_score(score, g)
    if bad:
        raise ValueError("musical check failed:\n  " + "\n  ".join(bad))
    D, drop, hit = g["duration"], g["drop_time"], g["end_hit_time_snapped"]
    N = int(round(D * SR))
    L = int(round((PRE + D + POST) * SR))
    E = score["events"]
    bars = g["bars"]
    bar4 = bars[4][0] if len(bars) > 5 else None              # second cycle (only when it is not the last bar)
    last_bt = bars[-1][0] if bars else drop
    B = {k: np.zeros((L, 2)) for k in BUS_LUFS}

    # ---------------------------------------------------------------- drums
    for i, (t, v, big) in enumerate(E.get("kick", [])):
        _place(B["kick"], _kick(v, _rng(seed, "kick", i), big), t)
    pan_t = {"low": -0.12, "mid": 0.22, "hi": -0.28}
    for i, (t, kind, v, big) in enumerate(E.get("tom", [])):
        _place(B["tom"], _pan(_tom(kind, v, _rng(seed, "tom", i), big), pan_t[kind]), t)
    for i, (t, v) in enumerate(E.get("clap", [])):
        _place(B["clap"], _clap(v, _rng(seed, "clap", i)), t)
    for i, (t, v, op) in enumerate(E.get("hat", [])):
        _place(B["hat"], _pan(_hat(v, _rng(seed, "hat", i), op), 0.3), t)
    for i, (t, m, v) in enumerate(E.get("tick", [])):
        _place(B["tick"], _pan(_tick(m, v, _rng(seed, "tick", i)), -0.35 if m == DRUM_PITCH["tick"] else 0.35), t)
    for i, (t, v, d) in enumerate(E.get("crash", [])):
        _place(B["crash"], _crash(v, _rng(seed, "crash", i), d), t)
    B["tom"] += 0.30 * _reverb(B["tom"], "room", hp=200.0, lp=6000.0)
    B["clap"] += 0.35 * _reverb(B["clap"], "room", hp=400.0, lp=8000.0) + 0.12 * _reverb(B["clap"], "hall", hp=500.0, lp=7000.0)
    B["tick"] += 0.18 * _pingpong(B["tick"], 3 * S16, fb=0.35, taps=4, hp=2500.0, lp=12000.0)
    B["kick"] = _mono_below(B["kick"], 150.0)

    # ---------------------------------------------------------------- bass (8th ostinato, sidechained)
    pts = [(-PRE, 170.0)] + (_curve(0.0, drop - GAP, 170.0, 850.0, 1.3) if drop > 0.05 else []) + [(drop, 560.0)]
    if bar4 is not None:
        pts += [(bar4 - 0.01, 600.0), (bar4, 700.0)]
    pts += [(last_bt, 700.0), (hit - GAP, 1100.0), (hit, 520.0), (hit + 1.5, 220.0), (D + POST, 160.0)]
    bass = _bass_bus(E.get("bass", []), L, _auto(pts, L))
    B["bass"] = _st(bass)

    # ---------------------------------------------------------------- music: pad, ostinato, braam
    pad = _pad_bus(score["chords"], L, seed, D)
    pts = [(-PRE, 850.0)] + (_curve(0.0, drop, 850.0, 3600.0, 1.3) if drop > 0.05 else []) + [(drop, 2900.0)]
    if bar4 is not None:
        pts += [(bar4 - 0.01, 3000.0), (bar4, 3900.0)]
    pts += [(last_bt, 3900.0), (hit - GAP, 5600.0), (hit, 5200.0), (hit + 0.4, 4800.0), (D + POST, 1300.0)]
    pad = _tvf(pad, "lp", _auto(pts, L), q=0.6, stages=2, block=64)
    pad = _filt(pad, "hp", 150.0, 2)
    lvl = [(-PRE, 0.5)] + (_curve(0.0, drop, 0.5, 1.0, 1.5) if drop > 0.05 else []) + [(drop, 0.82)]
    if bar4 is not None:
        lvl += [(bar4 - 0.01, 0.84), (bar4, 0.95)]
    lvl += [(last_bt, 0.95), (hit - GAP, 1.15), (hit, 1.0), (D + POST, 1.0)]
    pad = pad * _auto(lvl, L)[:, None]
    pad = pad + 0.32 * _reverb(pad, "hall", hp=300.0, lp=8000.0)
    B["pad"] = pad

    pts = [(-PRE, 900.0), (drop, 900.0)]
    if bar4 is not None:
        pts += [(bar4 - 0.01, 1350.0), (bar4, 1700.0)]
    pts += [(last_bt, 1800.0), (hit, 2600.0), (D + POST, 2600.0)]
    ost = _ost_bus(E.get("ost", []), L, _auto(pts, L))
    ost = _pan(ost, 0.0)
    ost = ost + 0.42 * _pingpong(ost, 3 * S16, fb=0.4, taps=6, hp=500.0, lp=3200.0) + 0.18 * _reverb(ost, "hall", hp=400.0, lp=7000.0)
    B["ost"] = ost

    for i, (t, d, c, v, kind) in enumerate(E.get("braam", [])):
        x = _braam(c, d, v, kind, _rng(seed, "braam", i))
        _place(B["braam"], x, t)
    B["braam"] = _mono_below(B["braam"] + 0.25 * _reverb(B["braam"], "hall", hp=250.0, lp=6000.0), 140.0)

    # ---------------------------------------------------------------- fx
    for i, (t, v, d) in enumerate(E.get("boom", [])):
        _place(B["boom"], _boom(v, _rng(seed, "boom", i), d), t)
    for i, (t0, t1, v) in enumerate(E.get("riser", [])):
        _place_end(B["riser"], _riser(t1 - t0, v, _rng(seed, "riser", i)), t1)
    for i, (t, d, v) in enumerate(E.get("revcym", [])):
        _place_end(B["revcym"], _revcym(d, v, _rng(seed, "revcym", i)), t)

    # ---------------------------------------------------------------- sidechain, breath gaps, levels
    kicks = [(t, v) for t, v, big in E.get("kick", [])]
    B["bass"] *= _duck([(t, 0.82 * min(v, 1.0)) for t, v in kicks], L, 0.82, release=0.12)[:, None]
    gpad = _duck([(t, 0.3 * v) for t, v in kicks], L, 0.3, release=0.22)[:, None]
    B["pad"] *= gpad
    B["ost"] *= _duck([(t, 0.25 * v) for t, v in kicks], L, 0.25, release=0.15)[:, None]
    gate = _gap_gate([drop, hit] if drop > GAP else [hit], L)[:, None]
    for k in ("bass", "pad", "ost", "hat", "tick", "braam"):
        B[k] *= gate
    for k in B:
        if np.max(np.abs(B[k])) > 0:
            B[k] = _norm_lufs(B[k], BUS_LUFS[k])

    # ---------------------------------------------------------------- stems: crop, tail, fades, DC, common gain
    s0 = int(round(PRE * SR))
    t = np.arange(N) / SR
    tail = np.ones(N)
    th = hit + 0.35
    tau = max(D - TAIL_FADE - th, 0.3) / math.log(_g(9.0))    # ~ -9 dB by the start of the final fade
    tail[t > th] = np.exp(-(t[t > th] - th) / tau)
    env = tail * _edge(N, int(FADE_IN * SR), int(TAIL_FADE * SR))
    stems = {}
    for name, parts in STEMS.items():
        x = sum(B[k] for k in parts)
        x = signal.sosfilt(_sos("hp", 12.0, 2), x, axis=0)[s0:s0 + N]
        stems[name] = x * env[:, None]
    tps = {k: _true_peak(v) for k, v in stems.items()}
    gain = _g(STEM_PEAK_DBTP) / max(tps.values())
    for k in stems:
        stems[k] = (stems[k] * gain).astype(np.float32)

    beat_times = g["beat_times"]
    meta = dict(
        bpm=BPM, key=KEY, beat_times=beat_times, drop_time=float(drop), end_hit_time_snapped=float(hit),
        # extras
        bar_times=g["bar_times"], meter=g["meter"], scale=list(SCALE),
        progression=list(PROGRESSION), intro_chord=INTRO_LAST, turnaround=list(TURNAROUND), final_chord=FINAL_CHORD,
        chord_timeline=[(round(max(0.0, a), 4), round(min(b, D), 4), c) for a, b, c, _ in score["chords"] if b > 0 and a < D],
        stem_gain_db=round(_db(gain), 2), stem_peak_dbtp=STEM_PEAK_DBTP, sample_rate=SR,
        musical_check=dict(violations=0, notes_checked=score["n_pitched_checked"]),
        events={k: len(v) for k, v in E.items()},
    )
    out = dict(stems)
    out["meta"] = meta
    if _buses:
        out["_score"] = score
        out["_grid"] = g
    return out


# ==============================================================================================
# demo + measurement
# ==============================================================================================
DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo")


def _ebur128(path):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-filter_complex",
                        "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    txt = p.stderr[p.stderr.rfind("Summary:"):]
    get = lambda pat: float(re.search(pat, txt, re.S).group(1))
    return dict(I_LUFS=get(r"I:\s+(-?[\d.]+) LUFS"), LRA_LU=get(r"LRA:\s+(-?[\d.]+) LU"),
                TP_dBTP=get(r"True peak:\s+Peak:\s+(-?[\d.inf]+) dBFS"))


OCTAVES = (31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)


def octave_db(x):
    f, P = signal.welch(_st(x).mean(axis=1), SR, nperseg=16384)
    out = {}
    for fc in OCTAVES:
        m = (f >= fc / math.sqrt(2)) & (f < fc * math.sqrt(2))
        out[str(fc)] = round(10 * math.log10(np.sum(P[m]) + 1e-20), 1)
    ref = max(out.values())
    return {k: round(v - ref, 1) for k, v in out.items()}


def stem_report(st):
    rep = {}
    for k in ("drums", "bass", "music", "fx"):
        x = st[k].astype(np.float64)
        rep[k] = dict(peak_dBFS=round(_db(np.max(np.abs(x))), 2), true_peak_dBTP=round(_db(_true_peak(x)), 2),
                      lufs_I=round(lufs(x), 2), max_momentary=round(float(np.max(_block_loudness(x))), 2),
                      dc=[float(f"{v:.2e}") for v in np.mean(x, axis=0)], finite=bool(np.all(np.isfinite(x))),
                      first_last_abs=[float(np.max(np.abs(x[0]))), float(np.max(np.abs(x[-1])))],
                      level_at_t0_dBFS=round(_db(np.sqrt(np.mean(x[:int(0.25 * SR)] ** 2))), 1),
                      octave_bands_dB=octave_db(x))
    return rep


def master_preview(mix, target_lufs=-14.0, ceiling_dbtp=-1.0):
    """Preview master only: gain to target loudness, then a look-ahead true-peak limiter."""
    x = mix * _g(target_lufs - lufs(mix))
    ceil = _g(ceiling_dbtp - 0.4)
    la = int(0.004 * SR)
    for _ in range(4):
        up = np.abs(signal.resample_poly(x, 4, 1, axis=0)).max(axis=1)
        pk = up[:len(x) * 4].reshape(-1, 4).max(axis=1)
        need = np.minimum(1.0, ceil / np.maximum(pk, 1e-12))
        g1 = minimum_filter1d(need, size=2 * la + 1, origin=0)
        # release: one-pole back towards 1 (attack handled by the look-ahead min)
        rel = math.exp(-1.0 / (0.06 * SR))
        g2 = signal.lfilter([1 - rel], [1, -rel], np.concatenate([[1.0], g1]))[1:]
        g2 = np.minimum(g1, g2)
        k = np.ones(la) / la
        g3 = np.convolve(np.pad(g2, (la, 0), constant_values=1.0), k, mode="valid")[:len(x)]
        g3 = np.minimum(g3, minimum_filter1d(need, size=4 * la + 1))
        x = x * g3[:, None]
        if _db(_true_peak(x)) <= ceiling_dbtp - 0.15:
            break
    return x


def render_demo(duration=17.5, drop_time=1.8, end_hit_time=14.5, seed=1, out_dir=DEMO_DIR):
    os.makedirs(out_dir, exist_ok=True)
    st = render(duration, drop_time, end_hit_time, seed=seed, _buses=True)
    meta = st["meta"]
    mix = sum(st[k].astype(np.float64) for k in ("drums", "bass", "music", "fx"))
    y = master_preview(mix)
    n = len(y)
    y = y * _edge(n, int(FADE_IN * SR), int(TAIL_FADE * SR))[:, None]
    wav = os.path.join(out_dir, "music_cinematic_demo.wav")
    sf.write(wav, y.astype(np.float32), SR, subtype="FLOAT")
    png = os.path.join(out_dir, "music_cinematic_demo_spectrogram.png")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", wav, "-lavfi",
                    "showspectrumpic=s=1600x600:legend=1:scale=log:fscale=log", png], check=True)
    wpng = os.path.join(out_dir, "music_cinematic_demo_waveform.png")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", wav, "-filter_complex",
                    "showwavespic=s=1600x300:split_channels=1:scale=sqrt", wpng], check=True)
    eb = _ebur128(wav)
    drop, hit = meta["drop_time"], meta["end_hit_time_snapped"]
    sec = [("intro 0-drop", 0.0, drop), ("drop bar", drop, drop + 2.0), ("groove", drop + 2.0, hit - 2.5),
           ("turnaround", hit - 2.5, hit), ("hit+tail", hit, duration)]
    lb = _block_loudness(y)
    sections = {}
    for name, a, b in sec:
        i0, i1 = int(a / 0.1), max(int((b - 0.4) / 0.1), int(a / 0.1) + 1)
        sections[name] = round(float(np.max(lb[i0:i1])), 1) if i1 > i0 else None
    # drop onset accuracy: first sample of the drums stem over 25 % of the local max after drop-5ms
    d = np.abs(st["drums"][:, 0].astype(np.float64)) + np.abs(st["drums"][:, 1].astype(np.float64))
    i_drop = int(round(drop * SR))
    seg = d[i_drop - int(0.005 * SR): i_drop + int(0.05 * SR)]
    on = int(np.argmax(seg > 0.25 * seg.max())) - int(0.005 * SR)
    rep = dict(params=dict(duration=duration, drop_time=drop_time, end_hit_time=end_hit_time, seed=seed),
               meta={k: v for k, v in meta.items() if k != "beat_times"},
               beat_times=meta["beat_times"],
               stems=stem_report(st),
               demo=dict(ebur128=eb, true_peak_internal_dBTP=round(_db(_true_peak(y)), 2),
                         dc=[float(f"{v:.2e}") for v in np.mean(y, axis=0)],
                         first_last=[float(np.max(np.abs(y[0]))), float(np.max(np.abs(y[-1])))],
                         max_momentary_by_section=sections, octave_bands_dB=octave_db(y),
                         drop_onset_error_ms=round(1000.0 * on / SR, 3)),
               bars=[{k: (v if k != "t" else round(v, 3)) for k, v in b.items()} for b in st["_score"]["bars"]])
    with open(os.path.join(out_dir, "music_cinematic_demo_report.json"), "w") as fh:
        json.dump(rep, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return rep


if __name__ == "__main__":
    r = render_demo()
    print(json.dumps({k: r[k] for k in ("stems", "demo")}, indent=1))
    print(json.dumps({k: r["meta"][k] for k in ("drop_time", "end_hit_time_snapped", "meter", "chord_timeline", "stem_gain_db", "musical_check")}, indent=1))
