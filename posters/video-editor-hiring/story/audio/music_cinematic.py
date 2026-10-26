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
        'meta' : {'bpm', 'key', 'beat_times', 'drop_time', 'end_hit_time_snapped', 'downbeat_times', ...extras}}
        (drop / hit rounded to 9 decimals like beat_times; clamping either one issues warnings.warn)
    beat_grid(duration, drop_time, end_hit_time) -> grid dict (beats, bars, snapped hit)
    arrange(grid) -> score (the arrangement as data: per-bar note lists + timed events)
    check_score(score, grid) -> list of musical-rule violations (render() raises if any)
    render_demo() -> demo/music_cinematic_demo.wav + spectrogram/waveform PNGs + stem report
    render_story(duration=30.0, drop1=2.0, breakdown=14.0, drop2=18.0, end_hit=26.0, seed=1) -> same stems + 'meta'
        (intro -> groove A -> breakdown -> drop 2 -> end hit + Fm(add9) tail; meta adds the section times,
        'sections' and 'fx_transitions'). story_grid() / arrange_story() / check_story_score() are its parts.
    render_story_demo() -> demo/music_cinematic_story30_demo.wav + spectrogram/waveform PNGs + section report

Run:  python3 -I music_cinematic.py           (renders the demo)
      python3 -I music_cinematic.py --story   (renders the 30 s story demo; exit code 1 if a check fails)
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import warnings
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
TOM_CHOKE = 0.04             # tom fills (and their room) are choked for the last 40 ms before the drop / end hit ...
TOM_CHOKE_FADE = 0.012       # ... with a 12 ms raised-cosine fade, so the hit's transient starts from near silence
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
    is appended) so the end hit is ALSO a downbeat. After the hit, 4/4 bars again.
    drop_time is clamped to 0 .. duration-3 s and the end hit to 1 beat after the drop .. duration-1 s; either
    clamp issues warnings.warn (the clamped values are what meta reports). Drop and hit are rounded to 9
    decimals, exactly like beat_times, so `hit in beat_times` is an exact membership test."""
    D = float(duration)
    drop_req, hit_req = float(drop_time), float(end_hit_time)
    if not all(math.isfinite(v) for v in (D, drop_req, hit_req)):
        raise ValueError(f"non-finite timing: duration={duration}, drop_time={drop_time}, end_hit_time={end_hit_time}")
    drop = round(float(min(max(drop_req, 0.0), max(D - 3.0, 0.0))), 9)
    if abs(drop - drop_req) > 1e-9:
        warnings.warn(f"music_cinematic: drop_time {drop_req:g} s clamped to {drop:g} s "
                      f"(must lie in 0 .. duration-3 = {max(D - 3.0, 0.0):g} s)", stacklevel=3)
    k_first = -int(math.floor(drop / BEAT + 1e-9))
    k_last = int(math.ceil((D - drop) / BEAT - 1e-9)) - 1
    hk_req = int(math.floor((hit_req - drop) / BEAT + 0.5))                    # nearest beat
    hk_max = int(math.floor((D - 1.0 - drop) / BEAT + 1e-9))                   # keep >= 1 s of tail
    hk = max(1, min(hk_req, hk_max))
    hit = round(drop + hk * BEAT, 9)
    if hk != hk_req:
        warnings.warn(f"music_cinematic: end_hit_time {hit_req:g} s clamped to {hit:g} s (nearest beat is out of "
                      f"range: >= 1 beat after the drop and >= 1 s before the end)", stacklevel=3)
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
def _arrange_intro(drop, E, chords, bars_out):
    """0 .. drop: the build (shared by arrange() and arrange_story()). Appends the intro chords, events and
    bar summary in place."""
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
            # flat-ish velocity: the un-ducked intro pulse must stay under the (side-chained) groove bass
            E["bass"].append((t, 0.19, CHORDS[ch_at(t)]["bass"], (0.5 if on_beat else 0.4) + 0.15 * u, False))
    for kb in range(-int(math.floor(drop / BEAT + 1e-9)), -1):  # heartbeat toms on the intro beats
        t = drop + kb * BEAT
        E["tom"].append((t, "low", 0.36 + 0.2 * (t / drop), False))
    for j, (kind, v) in enumerate(TOM_FILL_LAST_BEAT):         # 16th fill on the last beat into the drop
        t = drop - BEAT + j * S16
        if t >= -1e-9:
            E["tom"].append((t, kind, 0.72 * v, False))
    for j in range(-8, 0):                                     # 16th hats over the last 2 beats (crescendo)
        t = drop + j * S16
        if t >= -1e-9 and t < drop - GAP - 1e-9:
            E["hat"].append((t, 0.25 + 0.45 * (j + 8) / 8, False))
    E["riser"].append((max(0.0, drop - 3.0), drop, 0.8))
    E["revcym"].append((drop, min(0.9, drop), 0.8))
    bars_out.append(dict(section="intro", t=0.0, beats=round(drop / BEAT, 3),
                         chords=[(max(0.0, c[0]), c[2]) for c in chords]))


def _arrange_end(E, chords, bars_out, hit, D):
    """End hit + tail (shared by arrange() and arrange_story())."""
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
        _arrange_intro(drop, E, chords, bars_out)

    # ------------------------------------------------------------------ DROP .. END HIT
    nb = len(g["bars"])
    for bi, (bt, nbeats) in enumerate(g["bars"]):
        last = bi == nb - 1
        nsteps = 4 * nbeats
        energy = 0.86 if bi < 2 else (0.93 if bi < 4 else 1.0)
        if last:
            h = nbeats // 2
            first = PROGRESSION[0] if bi == 0 else TURNAROUND[0]          # the drop itself always lands on i
            segs = ([(bt, bt + h * BEAT, first)] if h else []) + [(bt + h * BEAT, bt + nbeats * BEAT, TURNAROUND[1])]
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
    _arrange_end(E, chords, bars_out, hit, D)
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
    x = body + 0.16 * click + (0.22 if big else 0.12) * knock
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
    return _filt(x + 0.25 * thump, "hp", 30.0, 2) * _edge(n, int(0.003 * SR), int(0.1 * n)) * vel


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


def _pad_bus(chords, L, seed, D, voicing=None, spread_c=15.0, tag="pad"):
    """String-like supersaw ensemble. Voices that hold the same pitch across chords are tied (no re-attack).
    voicing: {chord: midi tuple} instead of CHORDS[c]['pad'] (render_story's wide octave layer)."""
    lines = defaultdict(list)                    # voice index -> [t0, t1, midi]
    for a, b, c, kind in chords:
        for v, m in enumerate(CHORDS[c]["pad"] if voicing is None else voicing[c]):
            ln = lines[v]
            if ln and ln[-1][2] == m and abs(ln[-1][1] - a) < 1e-6:
                ln[-1][1] = b
            else:
                ln.append([a, b, m, kind])
    bus = np.zeros((L, 2))
    k = 0
    for v, ln in lines.items():
        for a, b, m, kind in ln:
            att = 0.07 if kind == "final" else (0.45 if a < -1.0 else 0.25)
            rel = 0.3
            n = int((b - a + rel) * SR)
            x = _supersaw(m, n, _rng(seed, tag, k), spread_c)
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


def _bass_bus(notes, L, cutoff, fenv_attack=0.0):
    """8th-note ostinato bass: sine sub + two detuned saws at pitch + a saw an octave up (mid 'grit' that
    phone speakers can reproduce), 24 dB low-pass with a per-note envelope, gentle drive.
    fenv_attack > 0 (render_story: 3 ms) ramps the filter envelope in instead of stepping it, which removes the
    small waveform kink an un-ducked 16th note otherwise gets when the cutoff jumps."""
    def shape(dur, lng):
        if lng:
            n = int((dur + 0.1) * SR)
            t = np.arange(n) / SR
            fe = np.exp(-t / 0.35)
            if fenv_attack > 0:
                fe = fe * (1.0 - np.exp(-t / fenv_attack))
            return np.exp(-t / 0.9) * _edge(n, int(0.006 * SR), int(0.05 * SR)), fe
        n = int((dur + 0.03) * SR)
        t = np.arange(n) / SR
        e = (0.74 + 0.26 * np.exp(-t / 0.05)) * _edge(n, int(0.006 * SR), int(0.03 * SR))
        fe = np.exp(-t / 0.08)
        if fenv_attack > 0:
            fe = fe * (1.0 - np.exp(-t / fenv_attack))
        return e, fe
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
    rel = 0.3 if kind != "rise" else 0.0          # kinds: hit, tail, rise, dark (render_story's breakdown)
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
    elif kind == "dark":                          # breakdown: closed low-pass, slow bloom, long decay
        fc = 110.0 + 520.0 * (1.0 - np.exp(-t / 0.12)) * np.exp(-t / 1.4) + 160.0 * np.exp(-t / 3.0)
        amp = (0.35 + 0.65 * np.exp(-t / 0.9)) * np.exp(-t / 2.6)
    else:
        fc = 120.0 + 1900.0 * (1.0 - np.exp(-t / 0.05)) * np.exp(-t / (0.5 if kind == "hit" else 0.8)) + 300.0 * np.exp(-t / 2.5)
        amp = 0.4 + 0.6 * np.exp(-t / (0.35 if kind == "hit" else 0.6))
        if kind == "tail":
            amp = amp * np.exp(-t / 2.2)
    out = _tvf(out, "lp", fc, q=0.9, stages=2, block=64)
    out = np.tanh(2.0 * _norm(out)) / np.tanh(2.0)
    out = _filt(out, "hp", 70.0, 4)
    out *= (amp * _edge(n, int(0.012 * SR), int((rel if rel else 0.003) * SR)))[:, None]
    return out * vel


# ==============================================================================================
# mix
# ==============================================================================================
BUS_LUFS = {                 # loudness of each bus while it plays (before the common stem gain)
    "kick": -19.5, "tom": -22.0, "clap": -23.0, "hat": -31.0, "tick": -32.5, "crash": -27.0,
    "bass": -19.5,
    "pad": -21.5, "ost": -23.0, "braam": -22.5,          # ost +3 dB (groove only: it starts at the drop)
    "boom": -21.0, "riser": -27.0, "revcym": -27.0,      # riser -2 dB: the build must sit >= 2 LU under the groove
}
STEMS = {"drums": ("kick", "tom", "clap", "hat", "tick", "crash"), "bass": ("bass",),
         "music": ("pad", "ost", "braam"), "fx": ("boom", "riser", "revcym")}


def _drum_buses(E, L, seed, hit_times):
    """kick / tom / clap / hat / tick / crash buses from the score events (shared by render() and
    render_story()). Toms that land ON a time in hit_times go to an ungated bus; every other tom is choked
    TOM_CHOKE before each of those hits."""
    B = {k: np.zeros((L, 2)) for k in ("kick", "tom", "clap", "hat", "tick", "crash")}
    for i, (t, v, big) in enumerate(E.get("kick", [])):
        _place(B["kick"], _kick(v, _rng(seed, "kick", i), big), t)
    # toms: the hits that land ON the drop / end hit go to their own bus; every other tom (fills, heartbeat,
    # groove) is choked before those hits: each event is cut with a raised-cosine fade ending TOM_CHOKE before
    # the hit (so its ring cannot come back after it), and the processed fill bus (room included) is gated shut
    # over the same window. Without this the last fill tom rings into the drop and blurs its transient.
    pan_t = {"low": -0.12, "mid": 0.22, "hi": -0.28}
    tom_fill, tom_hits = np.zeros((L, 2)), np.zeros((L, 2))
    nf = int(TOM_CHOKE_FADE * SR)
    for i, (t, kind, v, big) in enumerate(E.get("tom", [])):
        x = _pan(_tom(kind, v, _rng(seed, "tom", i), big), pan_t[kind])
        if any(abs(t - T) < 1e-9 for T in hit_times):
            _place(tom_hits, x, t)
            continue
        for T in hit_times:
            keep = _idx(T - TOM_CHOKE) - _idx(t)
            if t < T and len(x) > keep:                     # still ringing at the choke point: cut + fade
                x = x[:max(keep, 0)] * _edge(max(keep, 0), 0, nf)[:, None]
        if len(x) >= int(0.002 * SR):
            _place(tom_fill, x, t)

    def _tom_proc(x):
        return _mono_below(x + 0.30 * _reverb(x, "room", hp=200.0, lp=6000.0), 120.0)
    B["tom"] = (_tom_proc(tom_fill) * _gap_gate(hit_times, L, gap=TOM_CHOKE, ramp=TOM_CHOKE_FADE)[:, None]
                + _tom_proc(tom_hits))
    for i, (t, v) in enumerate(E.get("clap", [])):
        _place(B["clap"], _clap(v, _rng(seed, "clap", i)), t)
    for i, (t, v, op) in enumerate(E.get("hat", [])):
        _place(B["hat"], _pan(_hat(v, _rng(seed, "hat", i), op), 0.3), t)
    for i, (t, m, v) in enumerate(E.get("tick", [])):
        _place(B["tick"], _pan(_tick(m, v, _rng(seed, "tick", i)), -0.35 if m == DRUM_PITCH["tick"] else 0.35), t)
    for i, (t, v, d) in enumerate(E.get("crash", [])):
        _place(B["crash"], _crash(v, _rng(seed, "crash", i), d), t)
    B["clap"] += 0.35 * _reverb(B["clap"], "room", hp=400.0, lp=8000.0) + 0.12 * _reverb(B["clap"], "hall", hp=500.0, lp=7000.0)
    B["tick"] += 0.18 * _pingpong(B["tick"], 3 * S16, fb=0.35, taps=4, hp=2500.0, lp=12000.0)
    B["kick"] = _mono_below(B["kick"], 150.0)
    return B


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
    hit_times = [hit] + ([drop] if drop > 1e-9 else [])
    B.update(_drum_buses(E, L, seed, hit_times))

    # ---------------------------------------------------------------- bass (8th ostinato, sidechained)
    # starts at 550 Hz (the octave-up saw + filter envelope put a 250 Hz - 1.5 kHz pulse in from t=0 for phone
    # speakers), opens through the build and does NOT pull back at the drop (800 Hz, 900 Hz from bar 5)
    pts = [(-PRE, 550.0)] + (_curve(0.0, drop - GAP, 550.0, 850.0, 1.3) if drop > 0.05 else []) + [(drop, 800.0)]
    if bar4 is not None:
        pts += [(bar4 - 0.01, 820.0), (bar4, 900.0)]
    pts += [(last_bt, 900.0), (hit - GAP, 1100.0), (hit, 520.0), (hit + 1.5, 220.0), (D + POST, 160.0)]
    bass = _bass_bus(E.get("bass", []), L, _auto(pts, L))
    B["bass"] = _st(bass)

    # ---------------------------------------------------------------- music: pad, ostinato, braam
    pad = _pad_bus(score["chords"], L, seed, D)
    # pad: mid-range from t=0 (LP 1.8 kHz, level 0.85), opening through the build; at the drop it keeps opening
    # (3.7 kHz, level 0.97) instead of pulling back. The build peaks lower (0.88) than the groove, so the drop
    # is carried by the whole arrangement, not by the single hit.
    pts = [(-PRE, 1800.0)] + (_curve(0.0, drop, 1800.0, 3600.0, 1.3) if drop > 0.05 else []) + [(drop, 3700.0)]
    if bar4 is not None:
        pts += [(bar4 - 0.01, 3800.0), (bar4, 4200.0)]
    pts += [(last_bt, 4200.0), (hit - GAP, 5600.0), (hit, 5200.0), (hit + 0.4, 4800.0), (D + POST, 1300.0)]
    pad = _tvf(pad, "lp", _auto(pts, L), q=0.6, stages=2, block=64)
    pad = _filt(pad, "hp", 150.0, 2)
    lvl = [(-PRE, 0.85)] + (_curve(0.0, drop, 0.85, 0.88, 1.5) if drop > 0.05 else []) + [(drop, 0.97)]
    if bar4 is not None:
        lvl += [(bar4 - 0.01, 0.97), (bar4, 1.0)]
    lvl += [(last_bt, 1.0), (hit - GAP, 1.15), (hit, 1.0), (D + POST, 1.0)]
    pad = pad * _auto(lvl, L)[:, None]
    pad = pad + 0.32 * _reverb(pad, "hall", hp=300.0, lp=8000.0)
    B["pad"] = _mono_below(pad, 140.0)

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
        bpm=BPM, key=KEY, beat_times=beat_times, drop_time=round(float(drop), 9),       # clamped value if clamped
        end_hit_time_snapped=round(float(hit), 9),                                      # exact member of beat_times
        # extras. downbeat_times is the key shared with music_techhouse (here: an alias of bar_times, which
        # also holds the 5/4 / 6/4 bar lines of the last bar, so drop and end hit are both members)
        downbeat_times=list(g["bar_times"]), bar_times=g["bar_times"], meter=g["meter"], scale=list(SCALE),
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
# 30 s Instagram Story arrangement -- render_story()
#   0 .. drop1          intro / build (the same build as render())
#   drop1 .. breakdown  groove A: a lighter 4-bar phrase, then the full phrase; a turnaround every 4 bars
#   breakdown .. drop2  breakdown: no kick, dark low-passed pad, pulsing sub drone, the clock ticking; the last
#                       bar is the build (snare roll 8ths -> 16ths -> 32nds, riser, ostinato opening up)
#   drop2 .. end_hit    drop 2: groove A's full groove + a counter-melody lead + a wide octave pad layer
#   end_hit .. end      the end hit, then a sustained Fm(add9) chord + tail fading to digital silence
# ==============================================================================================
BAR = 4 * BEAT                                 # 2.0 s; every section boundary is a bar line
STORY_ANCHORS = ("drop1", "breakdown", "drop2", "end_hit")
STORY_TAIL_FADE = 0.3                          # the last 0.3 s fades to digital silence (delivery spec)


def _P(**kw):
    """One bar of a groove program (defaults = groove A's lightest bar)."""
    d = dict(kick="A", kick_vel=1.0, toms=False, energy=0.86, ost=0.90, ostpat="A", fill=None, open_hat=False,
             accent=None, lead=False, padhi=False, kind="groove")
    d.update(kw)
    return d


# Per-bar programs. A section's bars walk through its phrases (4 bars each; the last phrase repeats), then the
# section's LAST bar takes the section's `exit` fill.
#   kick 'A'/'B' (KICK_A / KICK_B), kick_vel | toms: tom groove | energy: clap / tom scale (hats capped at 1)
#   ost: ostinato velocity | ostpat: OST_PATTERNS key
#   fill: None | 'small' / 'big' (tom fills) | 'turn' (4-bar turnaround: last beat drops the kick for a 16th tom
#         fill, a clap roll and a 16th bass drive) | 'exit' (groove A -> breakdown: a unison stop on beat 4) |
#         'build' (breakdown's last bar) | 'end' (Db -> Eb turnaround into the end hit)
#   accent: event bundle on the downbeat ('drop', 'lift', 'breakdown', 'drop2')
#   lead / padhi: drop 2's extra layers (counter-melody lead, wide octave pad)
STORY_PLAN = {
    "grooveA": dict(exit="exit", phrases=(
        (_P(kick="A", energy=0.86, ost=0.90, accent="drop"),
         _P(kick="B", energy=0.86, ost=0.85, fill="small", open_hat=True),
         _P(kick="A", energy=0.93, ost=0.92, toms=True),
         _P(kick="B", energy=0.93, ost=0.95, toms=True, fill="turn", open_hat=True)),
        (_P(kick="A", energy=1.00, ost=1.00, ostpat="B", toms=True, accent="lift"),
         _P(kick="B", energy=1.00, ost=1.00, ostpat="B", toms=True, fill="small", open_hat=True),
         _P(kick="A", energy=1.00, ost=1.00, ostpat="B", toms=True, open_hat=True),
         _P(kick="B", energy=1.00, ost=1.00, ostpat="B", toms=True, fill="turn", open_hat=True)))),
    "breakdown": dict(exit="build", phrases=(
        (_P(kind="dark", accent="breakdown"), _P(kind="dark"), _P(kind="dark"), _P(kind="dark")),)),
    "drop2": dict(exit="end", phrases=(
        (_P(kick="A", kick_vel=1.1, energy=1.12, ost=1.05, ostpat="B", toms=True, open_hat=True, lead=True, padhi=True, accent="drop2"),
         _P(kick="B", kick_vel=1.1, energy=1.12, ost=1.05, ostpat="B", toms=True, open_hat=True, lead=True, padhi=True, fill="small"),
         _P(kick="A", kick_vel=1.1, energy=1.12, ost=1.05, ostpat="B", toms=True, open_hat=True, lead=True, padhi=True),
         _P(kick="B", kick_vel=1.1, energy=1.12, ost=1.05, ostpat="B", toms=True, open_hat=True, lead=True, padhi=True, fill="turn")),
        (_P(kick="A", kick_vel=1.1, energy=1.12, ost=1.05, ostpat="B", toms=True, open_hat=True, lead=True, padhi=True, accent="lift"),
         _P(kick="B", kick_vel=1.1, energy=1.12, ost=1.05, ostpat="B", toms=True, open_hat=True, lead=True, padhi=True, fill="small"),
         _P(kick="A", kick_vel=1.1, energy=1.12, ost=1.05, ostpat="B", toms=True, open_hat=True, lead=True, padhi=True),
         _P(kick="B", kick_vel=1.1, energy=1.12, ost=1.05, ostpat="B", toms=True, open_hat=True, lead=True, padhi=True, fill="turn")))),
}
OST_PATTERNS = {"A": OST_PATTERN,                                      # render()'s pattern (index into 'ost')
                "B": (0, 1, 2, 3, 2, 1, 2, 3, 0, 1, 2, 3, 2, 3, 1, 2)}  # full phrases: rising arpeggio variant
# harmony per section (groove A and drop 2 walk PROGRESSION from their first bar; drop 2 ends on TURNAROUND)
BREAKDOWN_DARK = ("Fm", "Db")                # dark bars cycle i, VI
BREAKDOWN_BUILD = ("Db", "Csus4")            # build bar: VI (beats 1-2) -> Vsus4 (beats 3-4) -> i at drop 2
DRONE_PULSE = {"dark": 1, "build": 2}        # sub-drone pump per beat: quarters (heartbeat), then 8ths ...
DRONE_DIP = 0.6                              # ... dipping 0.6 (-8 dB) on the pulse, swelling between them
SNARE_ROLL = (2, 2, 4, 8)                    # build bar: hits per beat (8ths, 8ths, 16ths, 32nds) ...
SNARE_STEPS = (55, 56, 58, 60)               # ... body pitch per beat G3 Ab3 Bb3 C4 (F-minor degrees)
SNARE_VEL = (0.22, 0.85)                     # crescendo over the bar (exponential)
# Drop-2 counter-melody, (16th step, length in 16ths, midi) per chord bar. Notes on a beat are chord tones, the
# others are passing tones of F minor (check_story_score enforces both). The synth doubles it an octave below.
LEAD_MOTIF = {
    "Fm":    ((0, 6, 72), (6, 2, 77), (8, 6, 80), (14, 2, 79)),        # C5 . F5 Ab5 . G5
    "Db":    ((0, 6, 77), (6, 2, 75), (8, 8, 73)),                     # F5 . Eb5 Db5
    "Bbm":   ((0, 6, 73), (6, 2, 77), (8, 6, 82), (14, 2, 80)),        # Db5 . F5 Bb5 . Ab5
    "Csus4": ((0, 6, 79), (6, 2, 77), (8, 8, 72)),                     # G5 . F5 C5
    "end":   ((0, 4, 80), (4, 4, 77), (8, 4, 79), (12, 3, 82)),        # Db: Ab5 F5 | Eb: G5 Bb5 -> C6 on the hit
}
LEAD_FINAL = (84, 1.8)                       # C6 (5th of Fm9) rings from the end hit into the tail
PADHI_VOICING = {c: tuple(m + 12 for m in v["pad"][-2:]) for c, v in CHORDS.items()}   # top 2 pad voices, 8va
STORY_BUS_LUFS = dict(BUS_LUFS, snare=-27.0, drone=-27.0, lead=-23.5, padhi=-28.5)
STORY_STEMS = {"drums": ("kick", "tom", "clap", "hat", "tick", "crash", "snare"), "bass": ("bass", "drone"),
               "music": ("pad", "padhi", "ost", "braam", "lead"), "fx": ("boom", "riser", "revcym")}
# Automation as data: (anchor, offset s, value) breakpoints, or ("~", anchor0, off0, anchor1, off1, v0, v1, shape)
# for a curved ramp. Anchors: pre (-PRE), zero, d1, full (groove A's 2nd phrase; skipped if A has <= 4 bars),
# exit (last beat of groove A), bd, build (breakdown's last bar), d2, turn (drop 2's last bar), hit, end.
STORY_AUTO = {
    "pad_lp": (("pre", 0, 1800.0), ("~", "zero", 0, "d1", 0, 1800.0, 3600.0, 1.3), ("d1", 0, 3700.0),
               ("full", -0.01, 3800.0), ("full", 0, 4200.0), ("exit", 0, 4200.0), ("bd", -GAP, 1100.0),
               ("bd", 0, 700.0), ("build", 0, 760.0), ("~", "build", 0, "d2", -GAP, 760.0, 3800.0, 1.6),
               ("d2", 0, 4600.0), ("turn", 0, 4800.0), ("hit", -GAP, 5600.0), ("hit", 0, 5200.0),
               ("hit", 0.4, 4800.0), ("end", 0, 1300.0)),
    "pad_lvl": (("pre", 0, 0.85), ("~", "zero", 0, "d1", 0, 0.85, 0.88, 1.5), ("d1", 0, 0.92), ("full", -0.01, 0.95),
                ("full", 0, 1.0), ("bd", -GAP, 1.0), ("bd", 0, 0.48), ("build", 0, 0.5),
                ("~", "build", 0, "d2", -GAP, 0.5, 1.0, 1.4), ("d2", 0, 1.06), ("turn", 0, 1.06),
                ("hit", -GAP, 1.15), ("hit", 0, 1.0), ("end", 0, 1.0)),
    "bass_lp": (("pre", 0, 550.0), ("~", "zero", 0, "d1", -GAP, 550.0, 850.0, 1.3), ("d1", 0, 800.0),
                ("full", -0.01, 820.0), ("full", 0, 900.0), ("bd", 0, 900.0), ("d2", 0, 950.0), ("turn", 0, 950.0),
                ("hit", -GAP, 1100.0), ("hit", 0, 520.0), ("hit", 1.5, 220.0), ("end", 0, 160.0)),
    "bass_lvl": (("pre", 0, 1.0), ("d2", -GAP, 1.0), ("d2", 0, 1.2), ("turn", 0, 1.2), ("hit", -GAP, 1.2),
                 ("hit", 0, 1.0), ("end", 0, 1.0)),                      # drop 2: bass +1.6 dB
    "ost_lp": (("pre", 0, 900.0), ("d1", 0, 900.0), ("full", -0.01, 1350.0), ("full", 0, 1700.0), ("bd", 0, 1700.0),
               ("build", 0, 420.0), ("~", "build", 0, "d2", -GAP, 420.0, 2000.0, 1.5), ("d2", 0, 2000.0),
               ("turn", 0, 2100.0), ("hit", 0, 2600.0), ("end", 0, 2600.0)),
    "lead_lp": (("pre", 0, 1500.0), ("d2", 0, 1500.0), ("turn", 0, 1800.0), ("hit", 0, 2000.0),
                ("hit", 1.5, 900.0), ("end", 0, 700.0)),
}


def story_grid(duration=30.0, drop1=2.0, breakdown=14.0, drop2=18.0, end_hit=26.0) -> dict:
    """Bar grid for render_story(). Beat k is at drop1 + k*BEAT. breakdown, drop2 and end_hit snap to the nearest
    bar line after drop1 (warnings.warn if one moves), keeping >= 1 bar per section and >= 1 s of tail."""
    D = float(duration)
    req = dict(drop1=float(drop1), breakdown=float(breakdown), drop2=float(drop2), end_hit=float(end_hit))
    if not all(math.isfinite(v) for v in (D, *req.values())):
        raise ValueError(f"non-finite timing: duration={duration}, anchors={req}")
    if D < 3 * BAR + 1.0:
        raise ValueError(f"duration {D:g} s is too short for a story (needs >= {3 * BAR + 1.0:g} s)")
    d1 = round(float(min(max(req["drop1"], 0.0), D - 3 * BAR - 1.0)), 9)
    k_max = int(math.floor((D - 1.0 - d1) / BAR + 1e-9))            # last bar line leaving >= 1 s of tail
    kb = max(1, int(math.floor((req["breakdown"] - d1) / BAR + 0.5)))
    k2 = max(kb + 1, int(math.floor((req["drop2"] - d1) / BAR + 0.5)))
    kh = max(k2 + 1, int(math.floor((req["end_hit"] - d1) / BAR + 0.5)))
    kh = min(kh, k_max)
    k2 = min(k2, kh - 1)
    kb = min(kb, k2 - 1)
    got = dict(drop1=d1, breakdown=round(d1 + kb * BAR, 9), drop2=round(d1 + k2 * BAR, 9), end_hit=round(d1 + kh * BAR, 9))
    for k in STORY_ANCHORS:
        if abs(got[k] - req[k]) > 1e-6:
            warnings.warn(f"music_cinematic.render_story: {k} {req[k]:g} s moved to {got[k]:g} s (bar lines every "
                          f"{BAR:g} s from drop1; >= 1 bar per section, >= 1 s tail)", stacklevel=3)
    bars = []
    for k in range(kh):
        sec, j0, j1 = (("grooveA", 0, kb) if k < kb else ("breakdown", kb, k2) if k < k2 else ("drop2", k2, kh))
        bars.append(dict(t=round(d1 + k * BAR, 9), beats=4, section=sec, j=k - j0, n=j1 - j0))
    k_first = -int(math.floor(d1 / BEAT + 1e-9))
    k_last = int(math.ceil((D - d1) / BEAT - 1e-9)) - 1
    beat_times = [round(d1 + k * BEAT, 9) for k in range(k_first, k_last + 1)]
    intro_bars = [round(d1 - BAR * j, 9) for j in range(1, 64) if d1 - BAR * j >= -1e-9]
    hit = got["end_hit"]
    tail_bars = [round(hit + BAR * j, 9) for j in range(0, 64) if hit + BAR * j < D - 1e-9]
    bar_times = sorted(set(intro_bars + [b["t"] for b in bars] + tail_bars))
    sections = [dict(name="intro", t0=0.0, t1=d1), dict(name="grooveA", t0=d1, t1=got["breakdown"]),
                dict(name="breakdown", t0=got["breakdown"], t1=got["drop2"]),
                dict(name="drop2", t0=got["drop2"], t1=hit), dict(name="end", t0=hit, t1=D)]
    return dict(duration=D, drop_time=d1, end_hit_time_snapped=hit, anchors=got, requested=req, bars=bars,
                sections=sections, beat_times=beat_times, bar_times=bar_times,
                meter=[dict(t=b["t"], beats=4, section=b["section"]) for b in bars])


def _story_bar_spec(b):
    plan = STORY_PLAN[b["section"]]
    ph = plan["phrases"][min(b["j"] // 4, len(plan["phrases"]) - 1)]
    sp = dict(ph[b["j"] % len(ph)])
    if b["j"] == b["n"] - 1:
        sp["fill"] = plan["exit"]
        if b["section"] == "breakdown":
            sp["kind"] = "build"
    return sp


def _story_segs(b, sp):
    bt, sec, j = b["t"], b["section"], b["j"]
    h = bt + 2 * BEAT
    if sec == "breakdown":
        if sp["kind"] == "build":
            return [(bt, h, BREAKDOWN_BUILD[0]), (h, bt + BAR, BREAKDOWN_BUILD[1])]
        return [(bt, bt + BAR, BREAKDOWN_DARK[j % len(BREAKDOWN_DARK)])]
    if sp["fill"] == "end":
        return [(bt, h, PROGRESSION[0] if j == 0 else TURNAROUND[0]), (h, bt + BAR, TURNAROUND[1])]
    return [(bt, bt + BAR, PROGRESSION[j % len(PROGRESSION)])]


def _story_accent(E, bt, sp, chord):
    a = sp["accent"]
    if a in ("drop", "drop2"):
        E["crash"].append((bt, 1.0, 2.6))
        E["boom"].append((bt, 1.0, 1.0))
        E["tom"].append((bt, "low", 1.0, True))
        E["braam"].append((bt, 1.9, chord, 1.0, "hit"))
    elif a == "lift":
        E["crash"].append((bt, 0.7, 2.0))
        E["boom"].append((bt, 0.55, 0.6))
        E["braam"].append((bt, 1.5, chord, 0.6, "hit"))
        E["revcym"].append((bt, 0.5, 0.6))
    elif a == "breakdown":                       # a dark thud: taiko + short sub boom + closed braam, no crash
        E["tom"].append((bt, "low", 0.72, True))
        E["boom"].append((bt, 0.5, 0.45))
        E["braam"].append((bt, BAR - GAP, chord, 0.5, "dark"))


def _story_groove_bar(E, b, segs, sp, hit):
    bt = b["t"]
    fill = sp["fill"]
    stop = 12 if fill in ("turn", "end", "exit") else 16       # first 16th of the fill beat
    en = sp["energy"]
    kick = KICK_A if sp["kick"] == "A" else KICK_B
    tom_fill = TOM_FILL_SMALL if fill == "small" else (TOM_FILL_BIG if fill == "big" else ())
    fill_from = min((s for s, _, _ in tom_fill), default=99)

    def chord_of(t):
        for a, bb, c in segs:
            if a - 1e-9 <= t < bb - 1e-9:
                return c
        return segs[-1][2]
    for s in range(16):
        t = bt + s * S16
        if s < stop:
            for ks, v in kick:
                if ks == s:
                    E["kick"].append((t, v * sp["kick_vel"], s == 0 and sp["accent"] in ("drop", "drop2")))
            for cs, v in CLAP:
                if cs == s:
                    E["clap"].append((t, v * en))
        if (s < 12 if fill == "exit" else (s < 15 if fill == "end" else True)):
            ramp = 1.0 + (0.35 * (s - stop) / 4 if s >= stop else 0.0)
            E["hat"].append((t, HAT_VEL[s % 4] * (0.8 + 0.2 * min(en, 1.0)) * ramp,
                             s == OPEN_HAT_STEP and sp["open_hat"] and s < stop))
        if s % 2 == 0:
            E["tick"].append((t, DRUM_PITCH["tick"] if s % 4 == 0 else DRUM_PITCH["tock"], 0.85 if s % 4 == 0 else 0.62))
        if s < stop and sp["toms"] and s < fill_from:
            for ts, kind, v in TOM_GROOVE:
                if ts == s:
                    E["tom"].append((t, kind, v * en, False))
        for ts, kind, v in tom_fill:
            if ts == s:
                E["tom"].append((t, kind, v, False))
        c = chord_of(t)
        if s < stop:
            for bs, v, octv in BASS_8THS:
                if bs == s:
                    E["bass"].append((t, 0.2, CHORDS[c]["bass"] + octv, v, False))
        elif fill == "exit":
            if s == stop:
                E["bass"].append((t, 0.4, CHORDS[c]["bass"], 0.9, False))
        elif s < 15 or fill == "turn":
            E["bass"].append((t, 0.1, CHORDS[c]["bass"], 0.8 + 0.06 * (s - stop), False))
        if not ((fill == "exit" and s > stop) or (fill == "end" and s == 15)):
            m = CHORDS[c]["ost"][OST_PATTERNS[sp["ostpat"]][s]]
            E["ost"].append((t, 0.25 if (fill == "exit" and s == stop) else 0.085, m, OST_VEL[s % 4] * sp["ost"]))
    if fill in ("turn", "end"):                  # last-beat fill: 16th toms + clap roll (the bass drives in 16ths)
        sc = 1.0 if fill == "end" else 0.85
        if fill == "end":
            E["tom"].append((bt + (stop - 4) * S16, "low", 0.66, False))
            E["tom"].append((bt + (stop - 2) * S16, "mid", 0.70, False))
        for j, (kind, v) in enumerate(TOM_FILL_LAST_BEAT):
            E["tom"].append((bt + (stop + j) * S16, kind, v * sc, False))
        for j, v in enumerate(CLAP_ROLL_LAST_BEAT):
            E["clap"].append((bt + (stop + j) * S16, v * sc))
    if fill == "exit":                           # unison stop on beat 4, then the breakdown's breath
        t = bt + stop * S16
        E["kick"].append((t, 0.95, False))
        E["clap"].append((t, 0.9))
        E["tom"].append((t, "low", 0.8, False))
    if fill == "end":                            # turnaround into the end hit (as in render())
        a, bb, c = segs[-1]
        t0 = max(a, hit - GAP - 2.0)
        E["braam"].append((t0, hit - GAP - t0, c, 0.8, "rise"))
        E["riser"].append((max(bt, hit - 2.5), hit, 1.0))
        E["revcym"].append((hit, 1.2, 1.0))
    _story_accent(E, bt, sp, segs[0][2])
    if sp["lead"]:
        motif = LEAD_MOTIF["end"] if fill == "end" else LEAD_MOTIF[segs[0][2]]
        for st, ln, m in motif:
            E["lead"].append((bt + st * S16, ln * S16 - 0.03, m, 1.0 if st % 4 == 0 else 0.86))
    if sp["padhi"]:
        for a, bb, c in segs:
            E["padhi"].append((a, bb, c))


def _story_breakdown_bar(E, b, segs, sp, nxt):
    bt = b["t"]
    build = sp["kind"] == "build"
    for a, bb, c in segs:
        E["drone"].append((a, bb - a, CHORDS[c]["bass"], 1.0))
    per = DRONE_PULSE[sp["kind"]]
    for k in range(4 * per):
        t = bt + k * BEAT / per
        if t < nxt - GAP - 1e-9:
            E["drone_pulse"].append((t, 1.0 if k % per == 0 else 0.7))
    for s in range(16):                          # the clock: tick-tock 8ths (+ ghost 16ths in the build)
        t = bt + s * S16
        if s % 2 == 0:
            E["tick"].append((t, DRUM_PITCH["tick"] if s % 4 == 0 else DRUM_PITCH["tock"], 0.9 if s % 4 == 0 else 0.66))
        elif build and t < nxt - GAP - 1e-9:
            E["tick"].append((t, DRUM_PITCH["tock"], 0.22 + 0.2 * s / 16))
    if build:
        for k in range(1 if sp["accent"] else 0, 3):   # heartbeat toms, then the 16th fill into drop 2 (as the intro)
            E["tom"].append((bt + k * BEAT, "low", 0.42 + 0.06 * k, False))
        for j, (kind, v) in enumerate(TOM_FILL_LAST_BEAT):
            E["tom"].append((nxt - BEAT + j * S16, kind, 0.72 * v, False))
        hits = [(bt + k * BEAT + i * BEAT / n, k) for k, n in enumerate(SNARE_ROLL) for i in range(n)]
        hits = [(t, k) for t, k in hits if t < nxt - GAP - 1e-9]
        for i, (t, k) in enumerate(hits):
            u = i / max(len(hits) - 1, 1)
            E["snare"].append((t, SNARE_VEL[0] * (SNARE_VEL[1] / SNARE_VEL[0]) ** (u ** 1.3), SNARE_STEPS[k]))
        for s in range(8, 16):                   # 16th hats over the last 2 beats (crescendo)
            t = bt + s * S16
            if t < nxt - GAP - 1e-9:
                E["hat"].append((t, 0.25 + 0.45 * (s - 8) / 8, False))
        for s in range(16):                      # the ostinato comes back filtered, swelling into drop 2
            t = bt + s * S16
            if t < nxt - GAP - 1e-9:
                c = segs[0][2] if t < segs[0][1] - 1e-9 else segs[-1][2]
                E["ost"].append((t, 0.085, CHORDS[c]["ost"][OST_PATTERN[s]], OST_VEL[s % 4] * (0.45 + 0.55 * s / 15)))
        E["riser"].append((bt, nxt, 1.0))
        E["revcym"].append((nxt, 1.0, 0.85))
    elif sp["accent"] is None:
        E["tom"].append((bt, "low", 0.5, False))          # later dark bars: a soft heartbeat on the downbeat
    _story_accent(E, bt, sp, segs[0][2])


def arrange_story(g: dict) -> dict:
    """The story score: chord segments, per-bar summaries and per-instrument events (render()'s tuples plus
    lead (t, dur, midi, vel), padhi (t0, t1, chord), drone (t, dur, midi, vel), drone_pulse (t, vel),
    snare (t, vel, midi))."""
    E = defaultdict(list)
    chords, bars_out = [], []
    A = g["anchors"]
    drop1, hit, D = A["drop1"], A["end_hit"], g["duration"]
    if drop1 > 1e-9:
        _arrange_intro(drop1, E, chords, bars_out)
    for i, b in enumerate(g["bars"]):
        sp = _story_bar_spec(b)
        segs = _story_segs(b, sp)
        nxt = g["bars"][i + 1]["t"] if i + 1 < len(g["bars"]) else hit
        kind = {"grooveA": "main", "breakdown": "breakdown", "drop2": "main"}[b["section"]]
        for a, bb, c in segs:
            chords.append([a, bb, c, "turn" if sp["fill"] == "end" else kind])
        if b["section"] == "breakdown":
            _story_breakdown_bar(E, b, segs, sp, nxt)
        else:
            _story_groove_bar(E, b, segs, sp, hit)
        bars_out.append(dict(section=b["section"], t=b["t"], beats=4, chords=[(a, c) for a, _, c in segs],
                             fill=sp["fill"], accent=sp["accent"], kind=sp["kind"], lead=sp["lead"]))
    _arrange_end(E, chords, bars_out, hit, D)
    if E.get("lead"):
        E["lead"].append((hit, LEAD_FINAL[1], LEAD_FINAL[0], 1.0))
    for k in E:
        E[k].sort(key=lambda e: e[0])
    return dict(chords=[tuple(c) for c in sorted(chords, key=lambda c: c[0])], events=dict(E), bars=bars_out)


def check_story_score(score: dict, g: dict) -> list:
    """check_score() plus the story layers: lead notes in F minor and chord tones on every beat, drone = chord
    root, wide-pad voicings = chord tones, tuned snare in key, every anchor on a bar line."""
    bad = check_score(score, g)
    E = score["events"]
    d1 = g["anchors"]["drop1"]
    for t, d, m, v in E.get("lead", []):
        c = chord_at(score, t)
        if m % 12 not in SCALE_PCS:
            bad.append(f"lead {note_name(m)} at {t:.3f}s is outside {KEY}")
        on_beat = abs((t - d1) / BEAT - round((t - d1) / BEAT)) < 1e-6
        if on_beat and m % 12 not in {PC[n] for n in CHORDS[c]["tones"]}:
            bad.append(f"lead {note_name(m)} on the beat at {t:.3f}s is not a tone of {c}")
    for t, d, m, v in E.get("drone", []):
        c = chord_at(score, t)
        if m % 12 != PC[CHORDS[c]["root"]]:
            bad.append(f"drone {note_name(m)} at {t:.3f}s is not the root of {c}")
    for a, b, c in E.get("padhi", []):
        if c != chord_at(score, a + 1e-6):
            bad.append(f"wide pad {c} at {a:.3f}s disagrees with {chord_at(score, a + 1e-6)}")
        for m in PADHI_VOICING[c]:
            if m % 12 not in {PC[n] for n in CHORDS[c]["tones"]}:
                bad.append(f"wide pad note {note_name(m)} not a tone of {c}")
    for t, v, m in E.get("snare", []):
        if m % 12 not in SCALE_PCS:
            bad.append(f"snare body {note_name(m)} at {t:.3f}s is outside {KEY}")
    for k, t in g["anchors"].items():
        if not any(abs(t - x) < 1e-9 for x in g["bar_times"]) and t > 0:
            bad.append(f"{k} {t} is not a bar line")
    score["n_pitched_checked"] += len(E.get("lead", [])) + len(E.get("drone", [])) + len(E.get("snare", []))
    return bad


def _snare(vel, rng, midi=60):
    """Tuned snare for the build roll: a pitched body (fundamental + octave, fast pitch drop) and a band-passed
    noise rattle; a little decorrelated width on the rattle only."""
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    f = float(m2f(midi)) * (1.0 + 0.35 * np.exp(-t / 0.012))
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.045) + 0.25 * np.sin(2 * ph + 0.5) * np.exp(-t / 0.02)
    att = 1.0 - np.exp(-t / 0.0008)
    out = np.zeros((n, 2))
    nzs = [_norm(_filt(_filt(rng.standard_normal(n), "hp", 1400.0, 2), "lp", 6500.0, 2)) for _ in range(2)]
    mid = 0.5 * (nzs[0] + nzs[1])
    for c in range(2):
        rattle = mid + 0.3 * (nzs[c] - mid)
        out[:, c] = 0.6 * body + 0.75 * rattle * att * np.exp(-t / 0.07)
    x = np.tanh(1.4 * out) / np.tanh(1.4)
    return x * _edge(n, int(0.002 * SR), int(0.04 * SR))[:, None] * vel


def _drone_bus(notes, pulses, L):
    """Breakdown bass: one continuous sub drone on the chord roots (sine + low-passed detuned saws + a soft
    octave), gliding 40 ms between roots. It pulses with a sidechain shape: -8 dB AT each pulse time (12 ms
    raised-cosine dip that ends on the beat, 0.16 s recovery), so the swell falls between the beats and anything
    landing on a beat (the story's reject / check stamps) lands in a dip of the sub."""
    out = np.zeros(L)
    if not notes:
        return _st(out)
    notes = sorted(notes)
    i0, i1 = max(_idx(notes[0][0]), 0), min(_idx(notes[-1][0] + notes[-1][1]), L)
    f = np.full(L, float(m2f(notes[0][2])))
    for t, d, m, v in notes:
        f[max(_idx(t), 0):] = float(m2f(m))
    a = math.exp(-1.0 / (0.04 * SR))
    f = np.exp(signal.lfilter([1 - a], [1, -a], np.log(f), zi=[a * math.log(f[0])])[0])
    saw = 0.5 * (_saw(f * 2 ** (-6 / 1200.0), 0.2) + _saw(f * 2 ** (6 / 1200.0), 0.7))
    saw = _tvf(saw, "lp", 380.0, q=0.7, stages=2)
    x = 0.9 * _sine(f) + 0.55 * saw + 0.25 * _sine(2.0 * f, 0.25)
    x = np.tanh(1.5 * x) / np.tanh(1.5)
    amp = np.zeros(L)
    amp[i0:i1] = _edge(i1 - i0, int(0.08 * SR), int(0.03 * SR))
    pump = _duck([(t, DRONE_DIP * v) for t, v in pulses], L, DRONE_DIP, release=0.16, hold=0.03, pre=0.012) if pulses else 1.0
    x = x * amp * pump
    return _st(_filt(x, "hp", 30.0, 4))


def _lead_bus(notes, L, cutoff):
    """Drop-2 counter-melody: a brassy saw lead doubled an octave below (+ a soft square sub-octave for body),
    35 ms portamento, gentle vibrato, filter envelope. Detuned pair on the side only (mono-safe)."""
    def shape(dur, lng):
        n = int((dur + 0.14) * SR)
        t = np.arange(n) / SR
        e = (0.8 + 0.2 * np.exp(-t / 0.15)) * np.where(t < dur, 1.0, np.exp(-np.maximum(t - dur, 0) / 0.05))
        return e * _edge(n, int(0.014 * SR), int(0.03 * SR)), np.exp(-t / 0.3)
    f, amp, fenv = _mono_synth([(t, d, m, v, False) for t, d, m, v in notes], L, "lead", 0.3, shape)
    a = math.exp(-1.0 / (0.035 * SR))
    f = np.exp(signal.lfilter([1 - a], [1, -a], np.log(f), zi=[a * math.log(f[0])])[0])
    tt = np.arange(L) / SR
    f = f * 2.0 ** (6.0 * np.sin(2 * np.pi * 5.2 * tt) / 1200.0)
    lo = 0.5 * f
    centre = 0.5 * _saw(f, 0.0) + 0.5 * _saw(lo, 0.3) + 0.22 * (_saw(lo, 0.0) - _saw(lo, 0.5))
    fc = np.clip(cutoff * (1.0 + 1.6 * fenv), 300.0, 7000.0)
    C = _tvf(centre, "lp", fc, q=0.8, stages=2)
    Dm = _tvf(_saw(f * 2 ** (-8 / 1200.0), 0.6) + _saw(lo * 2 ** (8 / 1200.0), 0.1), "lp", fc, q=0.8, stages=2)
    Dp = _tvf(_saw(f * 2 ** (8 / 1200.0), 0.15) + _saw(lo * 2 ** (-8 / 1200.0), 0.85), "lp", fc, q=0.8, stages=2)
    x = np.stack([C + 0.22 * Dm, C + 0.22 * Dp], axis=1) * amp[:, None]
    x = np.tanh(1.3 * x) / np.tanh(1.3)
    return _filt(x, "hp", 220.0, 2)


def _story_points(spec, T):
    pts = []
    for e in spec:
        if e[0] == "~":
            _, a0, o0, a1, o1, v0, v1, shp = e
            if T.get(a0) is not None and T.get(a1) is not None and T[a1] + o1 - T[a0] - o0 > 0.05:
                pts += _curve(T[a0] + o0, T[a1] + o1, v0, v1, shp)
        elif T.get(e[0]) is not None:
            pts.append((T[e[0]] + e[1], e[2]))
    pts.sort(key=lambda p: p[0])
    out = []
    for t, v in pts:                              # strictly increasing times for np.interp
        if out and t <= out[-1][0] + 1e-6:
            t = out[-1][0] + 2e-5
        out.append((t, v))
    return out


def render_story(duration=30.0, drop1=2.0, breakdown=14.0, drop2=18.0, end_hit=26.0, seed=1, _buses=False):
    """The 30 s story arrangement (see the section comment above). Returns render()'s stems dict
    {'drums','bass','music','fx'} (float32 (N, 2), unmastered, common gain, true peak <= -6 dBTP) + 'meta'
    (render()'s keys + section times, 'sections', 'fx_transitions')."""
    g = story_grid(duration, drop1, breakdown, drop2, end_hit)
    score = arrange_story(g)
    bad = check_story_score(score, g)
    if bad:
        raise ValueError("musical check failed:\n  " + "\n  ".join(bad))
    A = g["anchors"]
    D, d1, bd, d2, hit = g["duration"], A["drop1"], A["breakdown"], A["drop2"], A["end_hit"]
    N = int(round(D * SR))
    L = int(round((PRE + D + POST) * SR))
    E = score["events"]
    na = sum(1 for b in g["bars"] if b["section"] == "grooveA")
    T = dict(pre=-PRE, zero=0.0, d1=d1 if d1 > 0.05 else None, full=(d1 + 4 * BAR) if na > 4 else None,
             exit=bd - BEAT, bd=bd, build=d2 - BAR, d2=d2, turn=hit - BAR, hit=hit, end=D + POST)
    if T["d1"] is None:
        T["zero"] = None
    auto = {k: _auto(_story_points(v, T), L) for k, v in STORY_AUTO.items()}
    B = {k: np.zeros((L, 2)) for k in STORY_BUS_LUFS}

    # ---------------------------------------------------------------- drums
    hit_times = [hit, d2, bd] + ([d1] if d1 > 1e-9 else [])
    B.update(_drum_buses(E, L, seed, hit_times))
    stop = _gap_gate([bd], L, gap=TOM_CHOKE, ramp=TOM_CHOKE_FADE)[:, None]   # groove A's stop: kick + clap (and
    B["kick"] *= stop                                                          # its room) choked before the
    B["clap"] *= stop                                                          # breakdown's downbeat
    for i, (t, v, m) in enumerate(E.get("snare", [])):
        _place(B["snare"], _snare(v, _rng(seed, "snare", i), m), t)
    B["snare"] = B["snare"] + 0.3 * _reverb(B["snare"], "room", hp=300.0, lp=7000.0)

    # ---------------------------------------------------------------- bass: 8th ostinato + breakdown drone
    B["bass"] = _st(_bass_bus(E.get("bass", []), L, auto["bass_lp"], fenv_attack=0.003)) * auto["bass_lvl"][:, None]
    B["drone"] = _drone_bus(E.get("drone", []), E.get("drone_pulse", []), L)

    # ---------------------------------------------------------------- music
    pad = _pad_bus(score["chords"], L, seed, D)
    pad = _tvf(pad, "lp", auto["pad_lp"], q=0.6, stages=2, block=64)
    pad = _filt(pad, "hp", 150.0, 2) * auto["pad_lvl"][:, None]
    pad = pad + 0.32 * _reverb(pad, "hall", hp=300.0, lp=8000.0)
    B["pad"] = _mono_below(pad, 140.0)
    ph = [(a, b, c, "main") for a, b, c in E.get("padhi", [])]
    if ph:
        x = _pad_bus(ph, L, seed, D, voicing=PADHI_VOICING, spread_c=24.0, tag="padhi")
        x = _filt(_filt(x, "hp", 400.0, 2), "lp", 5200.0, 2)
        B["padhi"] = x + 0.35 * _reverb(x, "hall", hp=500.0, lp=8000.0)
    ost = _pan(_ost_bus(E.get("ost", []), L, auto["ost_lp"]), 0.0)
    B["ost"] = ost + 0.42 * _pingpong(ost, 3 * S16, fb=0.4, taps=6, hp=500.0, lp=3200.0) + 0.18 * _reverb(ost, "hall", hp=400.0, lp=7000.0)
    for i, (t, d, c, v, kind) in enumerate(E.get("braam", [])):
        _place(B["braam"], _braam(c, d, v, kind, _rng(seed, "braam", i)), t)
    B["braam"] = _mono_below(B["braam"] + 0.25 * _reverb(B["braam"], "hall", hp=250.0, lp=6000.0), 140.0)
    if E.get("lead"):
        x = _lead_bus(E["lead"], L, auto["lead_lp"])
        B["lead"] = x + 0.3 * _pingpong(x, 3 * S16, fb=0.35, taps=5, hp=500.0, lp=4000.0) + 0.22 * _reverb(x, "hall", hp=350.0, lp=7000.0)

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
    B["padhi"] *= gpad
    B["ost"] *= _duck([(t, 0.25 * v) for t, v in kicks], L, 0.25, release=0.15)[:, None]
    B["lead"] *= _duck([(t, 0.15 * v) for t, v in kicks], L, 0.15, release=0.15)[:, None]
    gate = _gap_gate([t for t in (d1, bd, d2, hit) if t > GAP], L)[:, None]
    for k in ("bass", "drone", "pad", "padhi", "ost", "hat", "tick", "braam", "lead", "snare"):
        B[k] *= gate
    for k in B:
        if np.max(np.abs(B[k])) > 0:
            B[k] = _norm_lufs(B[k], STORY_BUS_LUFS[k])

    # ---------------------------------------------------------------- stems: crop, tail, fades, DC, common gain
    s0 = int(round(PRE * SR))
    t = np.arange(N) / SR
    tail = np.ones(N)
    th = hit + 0.35
    tau = max(D - STORY_TAIL_FADE - th, 0.3) / math.log(_g(9.0))   # ~ -9 dB by the start of the final fade
    tail[t > th] = np.exp(-(t[t > th] - th) / tau)
    env = tail * _edge(N, int(FADE_IN * SR), int(STORY_TAIL_FADE * SR))
    stems = {}
    for name, parts in STORY_STEMS.items():
        x = sum(B[k] for k in parts)
        x = signal.sosfilt(_sos("hp", 12.0, 2), x, axis=0)[s0:s0 + N]
        stems[name] = x * env[:, None]
    tps = {k: _true_peak(v) for k, v in stems.items()}
    gain = _g(STEM_PEAK_DBTP) / max(tps.values())
    for k in stems:
        stems[k] = (stems[k] * gain).astype(np.float32)

    fx_tr = ([dict(kind="riser", t0=round(max(a, 0.0), 4), t1=round(b, 4)) for a, b, v in E.get("riser", [])]
             + [dict(kind="reverse_cymbal", t0=round(max(te - d, 0.0), 4), t1=round(te, 4)) for te, d, v in E.get("revcym", [])]
             + [dict(kind="sub_boom", t0=round(tb, 4), t1=round(min(tb + 4 * dc, D), 4)) for tb, v, dc in E.get("boom", [])])
    meta = dict(
        bpm=BPM, key=KEY, beat_times=g["beat_times"], downbeat_times=list(g["bar_times"]), bar_times=g["bar_times"],
        drop_time=round(float(d1), 9), end_hit_time_snapped=round(float(hit), 9),
        drop1=d1, breakdown=bd, drop2=d2, end_hit=hit, duration=D,
        section_times=dict(A, duration=D), requested_anchors=g["requested"],
        sections=g["sections"], meter=g["meter"], scale=list(SCALE), progression=list(PROGRESSION),
        breakdown_chords=dict(dark=list(BREAKDOWN_DARK), build=list(BREAKDOWN_BUILD)), turnaround=list(TURNAROUND),
        final_chord=FINAL_CHORD,
        chord_timeline=[(round(max(0.0, a), 4), round(min(b, D), 4), c) for a, b, c, _ in score["chords"] if b > 0 and a < D],
        fx_transitions=sorted(fx_tr, key=lambda e: e["t0"]),
        stem_gain_db=round(_db(gain), 2), stem_peak_dbtp=STEM_PEAK_DBTP, sample_rate=SR, tail_fade_s=STORY_TAIL_FADE,
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


PHONE_HP = (150.0, 250.0, 400.0)      # crude phone-speaker models: 4th-order high-pass at these frequencies


def section_lufs(x, a, b, hp=None):
    """Gated BS.1770 loudness of x between a and b seconds (400 ms blocks, 100 ms hop, -70 / -10 LU gates).
    The optional high-pass and the K-weighting run over the whole signal first, so the window holds no
    filter start-up. Returns None for a window shorter than one block."""
    if b - a < 0.4 or a < 0:
        return None
    y = _st(x).astype(np.float64)
    if hp:
        y = signal.sosfilt(_sos("hp", hp, 4), y, axis=0)
    for bb, aa in _KW:
        y = signal.lfilter(bb, aa, y, axis=0)
    y = y[int(round(a * SR)):int(round(b * SR))]
    w, h = int(0.4 * SR), int(0.1 * SR)
    c = np.concatenate([[0.0], np.cumsum(np.sum(y ** 2, axis=1))])
    i = np.arange(0, len(y) - w + 1, h)
    lb = -0.691 + 10.0 * np.log10((c[i + w] - c[i]) / w + 1e-20)
    lb = lb[lb > -70.0]
    if not len(lb):
        return -np.inf
    lb = lb[lb > 10 * np.log10(np.mean(10 ** (lb / 10))) - 10.0]
    return round(float(10 * np.log10(np.mean(10 ** (lb / 10)))), 2)


def hit_onset(x, t, lp=None, frac=0.3, pre=0.02, post=0.06):
    """Onset error (ms) of a hit at t: first sample of |L|+|R| (optionally low-passed) above frac x the local max
    in [t-pre, t+post]; plus the 'breath': max level in [t-30 ms, t-1 ms] in dB re that max. A breath above
    about -10.5 dB (30 %) makes the detector fire early, which is what a ringing tom used to do."""
    a = _st(x).astype(np.float64)
    if lp:
        a = signal.sosfilt(_sos("lp", lp, 4), a, axis=0)
    e = np.abs(a).sum(axis=1)
    i0, i1 = max(0, int(round((t - pre) * SR))), min(len(e), int(round((t + post) * SR)))
    seg = e[i0:i1]
    k = int(np.argmax(seg > frac * seg.max()))
    br = e[max(0, int(round((t - 0.03) * SR))):int(round((t - 0.001) * SR))]
    return round(1000.0 * ((i0 + k) / SR - t), 3), round(_db(br.max() / seg.max()) if len(br) else -240.0, 1)


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
    # hit transients + breath (drums stem, broadband and LP 200 Hz) at the drop and the end hit
    onsets = {f"{name}_{lbl}": dict(zip(("onset_ms", "breath_dB"), hit_onset(st["drums"], T, lp=lp)))
              for name, T in (("drop", drop), ("end_hit", hit)) for lbl, lp in (("broadband", None), ("lp200", 200.0))}
    # build -> drop contrast, full band and through the phone high-pass models (unmastered stem sum)
    wins = dict(first_1s=(0.0, 1.0), build_end=(drop - 1.0, drop), drop_bar=(drop, drop + 2.0), groove=(drop + 2.0, hit - 2.5))
    sec_lufs = {w: dict(full=section_lufs(mix, a, b), **{f"hp{int(f)}": section_lufs(mix, a, b, hp=f) for f in PHONE_HP})
                for w, (a, b) in wins.items()}

    def _diff(p, q):
        return {k: (round(sec_lufs[p][k] - sec_lufs[q][k], 2) if sec_lufs[p][k] is not None and sec_lufs[q][k] is not None
                    else None) for k in sec_lufs[p]}
    contrast = dict(drop_bar_minus_build_end=_diff("drop_bar", "build_end"),
                    groove_minus_build_end=_diff("groove", "build_end"), groove_minus_first_1s=_diff("groove", "first_1s"))
    sr = stem_report(st)
    checks = dict(
        stems_true_peak_le_minus6=all(v["true_peak_dBTP"] <= -6.0 for v in sr.values()),
        stems_dc_lt_1e4=all(max(abs(d) for d in v["dc"]) < 1e-4 for v in sr.values()),
        stems_finite=all(v["finite"] for v in sr.values()),
        stems_start_end_silent=all(max(v["first_last_abs"]) < 1e-6 for v in sr.values()),
        stems_shape=all(st[k].shape == (int(round(duration * SR)), 2) and st[k].dtype == np.float32 for k in STEMS),
        demo_true_peak_le_minus1=eb["TP_dBTP"] <= -1.0,
        demo_dc_lt_1e4=float(np.max(np.abs(np.mean(y, axis=0)))) < 1e-4,
        drop_is_bar_line=any(abs(t - drop) < 1e-9 for t in meta["bar_times"]),
        end_hit_is_bar_line=any(abs(t - hit) < 1e-9 for t in meta["bar_times"]),
        meta_exact_membership=(hit in meta["beat_times"] and drop in meta["beat_times"]
                               and hit in meta["downbeat_times"] and drop in meta["downbeat_times"]),
        drop_onset_within_1ms=abs(on / SR) <= 0.001,
        hit_onsets_within_4ms=all(-1.0 <= v["onset_ms"] <= 4.0 for v in onsets.values()),
        breath_before_hits_le_minus15dB=all(v["breath_dB"] <= -15.0 for v in onsets.values()),
        notes_in_key=meta["musical_check"]["violations"] == 0,
        audible_from_t0=max(v["level_at_t0_dBFS"] for v in sr.values()) > -40.0)
    if contrast["groove_minus_build_end"]["full"] is not None:
        checks["groove_ge_build_end_plus_2LU"] = contrast["groove_minus_build_end"]["full"] >= 2.0
    rep = dict(checks=checks, all_checks_pass=all(checks.values()),
               params=dict(duration=duration, drop_time=drop_time, end_hit_time=end_hit_time, seed=seed),
               meta={k: v for k, v in meta.items() if k != "beat_times"},
               beat_times=meta["beat_times"],
               stems=sr,
               demo=dict(ebur128=eb, true_peak_internal_dBTP=round(_db(_true_peak(y)), 2),
                         dc=[float(f"{v:.2e}") for v in np.mean(y, axis=0)],
                         first_last=[float(np.max(np.abs(y[0]))), float(np.max(np.abs(y[-1])))],
                         max_momentary_by_section=sections, octave_bands_dB=octave_db(y),
                         drop_onset_error_ms=round(1000.0 * on / SR, 3)),
               transients=onsets,
               section_loudness_LUFS=dict(note="unmastered stem sum; hpNNN = 4th-order high-pass at NNN Hz (phone "
                                               "speaker model); build_end = last 1 s before the drop; groove = "
                                               "drop+2 s .. end hit-2.5 s", **sec_lufs),
               contrast_LU=contrast,
               bars=[{k: (v if k != "t" else round(v, 3)) for k, v in b.items()} for b in st["_score"]["bars"]])
    with open(os.path.join(out_dir, "music_cinematic_demo_report.json"), "w") as fh:
        json.dump(rep, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return rep


def _centroid(x, a, b):
    """Spectral centroid over 150 Hz - 16 kHz (the phone band, so the sub does not decide it) and the share (%)
    of the energy above 1 kHz."""
    s = _st(x)[int(round(a * SR)):int(round(b * SR))].astype(np.float64).mean(axis=1)
    f, P = signal.welch(s, SR, nperseg=8192)
    m = (f >= 150.0) & (f <= 16000.0)
    return float(np.sum(f[m] * P[m]) / (np.sum(P[m]) + 1e-30)), float(100.0 * P[f >= 1000.0].sum() / (P.sum() + 1e-30))


def _chroma_out_of_key(x, a, b, lo, hi):
    """Share (%) of the a..b energy (lo..hi Hz) on pitch classes outside F minor, and the top pitch classes."""
    s = _st(x)[int(round(a * SR)):int(round(b * SR))].astype(np.float64).mean(axis=1)
    nfft = 1 << int(math.ceil(math.log2(len(s))) + 1)
    S = np.abs(np.fft.rfft(s * np.hanning(len(s)), nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / SR)
    m = (f >= lo) & (f <= hi)
    pc = np.round(69 + 12 * np.log2(f[m] / 440.0)).astype(int) % 12
    c = np.bincount(pc, weights=S[m], minlength=12)
    c = c / (c.sum() + 1e-30)
    return round(100.0 * sum(c[i] for i in range(12) if i not in SCALE_PCS), 2), [NOTE_NAMES[i] for i in np.argsort(c)[::-1][:3]]


def _click_events(x, a, b, hf=12000.0):
    """Clicks in a..b s: sample-level discontinuities (|2nd difference| > 12x its local RMS and > 1e-3) and
    broadband HF bursts (> hf Hz energy in 2 ms windows > 20 dB over the 200 ms median and > -80 dBFS)."""
    from scipy.ndimage import median_filter, uniform_filter1d
    y = _st(x).astype(np.float64)
    i0, i1 = max(int(round(a * SR)), 0), min(int(round(b * SR)), len(y))
    s = y.sum(axis=1)
    d2 = np.abs(np.diff(s, 2))
    loc = np.sqrt(np.maximum(uniform_filter1d(d2 ** 2, size=481), 0.0)) + 1e-12
    idx = np.nonzero((d2 > 12 * loc) & (d2 > 1e-3))[0]
    disc = sorted({round(float(i) / SR, 3) for i in idx if i0 <= i < i1})
    hp = signal.sosfilt(_sos("hp", hf, 4), y, axis=0)
    p = np.sum(hp ** 2, axis=1)
    w = int(0.002 * SR)
    e = p[:len(p) // w * w].reshape(-1, w).mean(axis=1) + 1e-24
    ex = 10 * np.log10(e / (median_filter(e, size=101, mode="nearest") + 1e-24))
    hf = [round(i * w / SR, 3) for i in np.nonzero((ex > 20.0) & (10 * np.log10(e) > -80.0))[0] if i0 <= i * w < i1]
    return dict(discontinuities=disc[:10], n_discontinuities=len(disc), hf_bursts=hf[:10], n_hf_bursts=len(hf))


def render_story_demo(duration=30.0, drop1=2.0, breakdown=14.0, drop2=18.0, end_hit=26.0, seed=1, out_dir=DEMO_DIR):
    """Renders render_story(), sums the stems through master_preview() (-14 LUFS, <= -1 dBTP) and writes
    demo/music_cinematic_story30_demo.wav, _spectrogram.png, _waveform.png and _report.json (section loudness
    full-band and through the phone high-pass models, contrasts, anchor transients, clicks, tail, checks)."""
    os.makedirs(out_dir, exist_ok=True)
    st = render_story(duration, drop1, breakdown, drop2, end_hit, seed=seed, _buses=True)
    meta = st["meta"]
    D, d1, bd, d2, hit = meta["duration"], meta["drop1"], meta["breakdown"], meta["drop2"], meta["end_hit"]
    mix = sum(st[k].astype(np.float64) for k in ("drums", "bass", "music", "fx"))
    y = master_preview(mix)
    y = y * _edge(len(y), int(FADE_IN * SR), int(STORY_TAIL_FADE * SR))[:, None]
    base = os.path.join(out_dir, "music_cinematic_story30_demo")
    wav = base + ".wav"
    sf.write(wav, y.astype(np.float32), SR, subtype="FLOAT")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", wav, "-lavfi",
                    "showspectrumpic=s=1800x640:legend=1:scale=log:fscale=log", base + "_spectrogram.png"], check=True)
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", wav, "-filter_complex",
                    "showwavespic=s=1800x300:split_channels=1:scale=sqrt", base + "_waveform.png"], check=True)
    eb = _ebur128(wav)
    na = sum(1 for b in meta["meter"] if b["section"] == "grooveA")
    wins = [("intro", 0.0, d1), ("grooveA", d1, bd)]
    if na > 4:
        wins += [("grooveA_light", d1, d1 + 4 * BAR), ("grooveA_full", d1 + 4 * BAR, bd)]
    if d2 - bd > BAR:
        wins += [("breakdown_dark", bd, d2 - BAR)]
    wins += [("breakdown_build", d2 - BAR, d2), ("breakdown", bd, d2), ("drop2", d2, hit)]
    if hit - d2 > BAR:
        wins += [("drop2_body", d2, hit - BAR)]
    wins += [("end_hit_tail", hit, D)]
    bands = (None,) + PHONE_HP
    sec = {}
    for name, a, b in wins:
        r = {("full" if f is None else f"hp{int(f)}"): section_lufs(y, a, b, hp=f) for f in bands}
        r["unmastered_full"] = section_lufs(mix, a, b)
        r["unmastered_hp150"] = section_lufs(mix, a, b, hp=150.0)
        c, hf = _centroid(y, a, b)
        r.update(t0=round(a, 3), t1=round(b, 3), centroid_Hz=round(c), energy_above_1k_pct=round(hf, 2),
                 max_momentary=round(float(np.max(_block_loudness(y[int(a * SR):int(b * SR)]))), 2) if b - a >= 0.4 else None)
        sec[name] = r

    def diff(p, q, k):
        return round(sec[p][k] - sec[q][k], 2)
    contrast = {}
    for p, q in (("grooveA", "intro"), ("drop2", "grooveA"), ("drop2_body", "grooveA"), ("drop2", "grooveA_full"),
                 ("drop2_body", "grooveA_full"), ("breakdown", "grooveA"), ("breakdown_dark", "grooveA"),
                 ("breakdown_build", "breakdown_dark"), ("grooveA_full", "grooveA_light")):
        if p in sec and q in sec:
            contrast[f"{p}_minus_{q}"] = {k: diff(p, q, k) for k in ("full", "hp150", "hp250", "hp400", "unmastered_full", "unmastered_hp150")}
    anchors = dict(drop1=d1, breakdown=bd, drop2=d2, end_hit=hit)
    onsets = {f"{n}_{lbl}": dict(zip(("onset_ms", "breath_dB"), hit_onset(st["drums"], T, lp=lp)))
              for n, T in anchors.items() for lbl, lp in (("broadband", None), ("lp200", 200.0))}
    # clicks: the discontinuity scan runs over every stem end to end; the HF-burst scan runs +-0.3 s around each
    # section boundary on music / fx (> 12 kHz). Drums (by design full of HF transients) and the bass (> 3 kHz
    # fires on every note attack, also in render()) only get the discontinuity scan.
    clicks = {"whole_length": {k: _click_events(st[k], 0.0, D) for k in STEMS}}
    for n, T in list(anchors.items()) + [("end", D - 0.25)]:
        clicks[n] = {k: _click_events(st[k], T - 0.3, T + 0.3) for k in ("music", "fx")}
    n_clicks = (sum(v["n_discontinuities"] for v in clicks["whole_length"].values())
                + sum(v["n_hf_bursts"] for n, d in clicks.items() if n != "whole_length" for v in d.values()))
    key_audio = []
    for a, b, c in meta["chord_timeline"]:
        if b - a >= 0.9:
            ob, tb = _chroma_out_of_key(st["bass"], a + 0.05, b - 0.05, 40.0, 300.0)
            om, tm = _chroma_out_of_key(st["music"], a + 0.05, b - 0.05, 150.0, 2000.0)
            key_audio.append(dict(t0=a, t1=b, chord=c, bass_out_of_key_pct=ob, bass_top=tb[0],
                                  music_out_of_key_pct=om, music_top=tm))
    last = y[-int(0.01 * SR):]
    tail = dict(last_sample_abs=float(np.max(np.abs(y[-1]))), last_10ms_rms_dBFS=round(_db(np.sqrt(np.mean(last ** 2))), 1),
                last_0p3s_max_momentary=round(float(np.max(_block_loudness(y[-int(0.4 * SR):]))), 1),
                stems_last_sample_abs=max(float(np.max(np.abs(st[k][-1]))) for k in STEMS))
    srp = stem_report(st)
    dk = "breakdown_dark" if "breakdown_dark" in sec else "breakdown"
    checks = dict(
        stems_true_peak_le_minus6=all(v["true_peak_dBTP"] <= -6.0 for v in srp.values()),
        stems_dc_lt_1e4=all(max(abs(d) for d in v["dc"]) < 1e-4 for v in srp.values()),
        stems_finite=all(v["finite"] for v in srp.values()),
        stems_start_end_silent=all(max(v["first_last_abs"]) < 1e-6 for v in srp.values()),
        stems_shape=all(st[k].shape == (int(round(D * SR)), 2) and st[k].dtype == np.float32 for k in STEMS),
        demo_I_minus14_pm05=abs(eb["I_LUFS"] + 14.0) <= 0.5,
        demo_true_peak_le_minus1=eb["TP_dBTP"] <= -1.0,
        demo_LRA_le_8=eb["LRA_LU"] <= 8.0,
        anchors_are_downbeats=all(T in meta["downbeat_times"] and T in meta["beat_times"] for T in anchors.values()),
        anchors_as_requested=all(abs(meta["section_times"][k] - meta["requested_anchors"][k]) < 1e-9 for k in anchors),
        anchor_onsets_within_4ms=all(-1.0 <= v["onset_ms"] <= 4.0 for v in onsets.values()),
        breath_before_drops_and_hit_le_minus15dB=all(onsets[f"{n}_{l}"]["breath_dB"] <= -15.0
                                                     for n in ("drop1", "drop2", "end_hit") for l in ("broadband", "lp200")),
        notes_in_key=meta["musical_check"]["violations"] == 0,
        # audio cross-check of the note check. The "out of key" share is never 0: saw harmonics land on D, E, A
        # (render()'s own audit: bass 6-12 %, music 2-6 %; an Eb major chord's G and Bb put their 3rd/5th
        # harmonics on D). So: bass's strongest pitch class is the chord root, music's top 3 are in key and its
        # strongest is a chord tone, and the shares stay under 13 % (bass) / 10 % (music).
        audio_key_check=all(r["bass_top"] == CHORDS[r["chord"]]["root"] and r["bass_out_of_key_pct"] < 13
                            and all(PC[n] in SCALE_PCS for n in r["music_top"]) and r["music_out_of_key_pct"] < 10
                            and r["music_top"][0] in CHORDS[r["chord"]]["tones"] for r in key_audio),
        intro_lt_grooveA_lt_drop2=all(sec["intro"][k] < sec["grooveA"][k] < sec["drop2"][k] for k in ("full", "hp150")),
        drop2_ge_grooveA_plus_1LU=all(contrast[c][k] >= 1.0 for c in ("drop2_minus_grooveA", "drop2_minus_grooveA_full")
                                      if c in contrast for k in ("full", "hp150")),
        breakdown_quieter_ge_4LU=all(contrast["breakdown_minus_grooveA"][k] <= -4.0 for k in ("full", "hp150")),
        breakdown_darker=(sec[dk]["centroid_Hz"] < 0.75 * sec["grooveA"]["centroid_Hz"]
                          and sec[dk]["energy_above_1k_pct"] < 0.5 * sec["grooveA"]["energy_above_1k_pct"]),
        no_clicks=n_clicks == 0,
        tail_fades_to_zero=tail["last_sample_abs"] < 1e-6 and tail["stems_last_sample_abs"] < 1e-6 and tail["last_10ms_rms_dBFS"] < -50.0,
        audible_from_t0=max(v["level_at_t0_dBFS"] for v in srp.values()) > -40.0)
    rep = dict(checks=checks, all_checks_pass=all(checks.values()),
               params=dict(duration=duration, drop1=drop1, breakdown=breakdown, drop2=drop2, end_hit=end_hit, seed=seed),
               demo=dict(ebur128=eb, true_peak_internal_dBTP=round(_db(_true_peak(y)), 2),
                         dc=[float(f"{v:.2e}") for v in np.mean(y, axis=0)], octave_bands_dB=octave_db(y)),
               section_loudness_LUFS=dict(note="demo master (master_preview, -14 LUFS / -1 dBTP) unless 'unmastered_*'; "
                                               "hpNNN = 4th-order high-pass at NNN Hz (phone-speaker model); gated BS.1770 "
                                               "over the window; centroid / >1 kHz share of the master", **sec),
               contrast_LU=contrast, transients=onsets, clicks=clicks, tail=tail, key_audio_check=key_audio,
               stems=srp, meta={k: v for k, v in meta.items() if k != "beat_times"}, beat_times=meta["beat_times"],
               bars=[{k: (v if k != "t" else round(v, 3)) for k, v in b.items()} for b in st["_score"]["bars"]])
    with open(base + "_report.json", "w") as fh:
        json.dump(rep, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return rep


if __name__ == "__main__":
    import sys
    if "--story" in sys.argv:
        r = render_story_demo()
        print(json.dumps({k: r[k] for k in ("checks", "all_checks_pass", "demo", "contrast_LU")}, indent=1))
        sys.exit(0 if r["all_checks_pass"] else 1)
    r = render_demo()
    print(json.dumps({k: r[k] for k in ("checks", "all_checks_pass", "demo")}, indent=1))
    print(json.dumps({k: r["meta"][k] for k in ("drop_time", "end_hit_time_snapped", "meter", "chord_timeline", "stem_gain_db", "musical_check")}, indent=1))
