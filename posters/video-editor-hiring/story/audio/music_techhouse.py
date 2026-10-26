#!/usr/bin/env python3
"""
music_techhouse.py -- "Driving tech pulse": an original, fully synthesized tech-house bed for the
KRS Tech Solutions "WE'RE HIRING - VIDEO EDITOR" Instagram Story.

No samples, loops or downloaded audio: every sound is generated in this file (band-limited additive
oscillators, polyBLEP saws, noise, envelopes, filters) and processed with pedalboard/scipy.

Contract shared with the other audio modules
    key   F minor (F G Ab Bb C Db Eb), tonal centre F
    tempo 120 BPM, 4/4, 16th-note swing ~10 ms on the off-16ths
    48 kHz, stereo, float32 arrays shaped (N, 2)

API
    render(duration, drop_time, end_hit_time, seed=1) -> {
        'drums', 'bass', 'music', 'fx'  : float32 (N, 2), N = round(duration * 48000), unmastered,
                                          scaled together so the hottest true peak is -7 dBTP,
        'meta' : {'bpm', 'key', 'beat_times', 'drop_time', 'end_hit_time_snapped', ...extras}}
    check_arrangement(...)  -> list of musical-rule violations (render() raises if any)
    render_demo()           -> writes demo/music_techhouse_demo.wav + spectrogram PNG + stem report
    render_story(duration=30.0, drop1=2.0, breakdown=14.0, drop2=18.0, end_hit=26.0, seed=1)
                            -> same stems + meta; intro / groove A / breakdown / drop 2 / end hit + held tail
                               (meta adds drop1, breakdown, drop2, end_hit, sections, section_boundaries,
                               fx_transitions); arrangement data in STORY_SECTIONS / STORY_BREAKDOWN / STORY_LEAD
    render_story_demo()     -> demo/music_techhouse_story30_demo.wav + _spectrogram.png + _report.json

Run:  python3 -I music_techhouse.py               (renders the demo + report)
      python3 -I music_techhouse.py --selftest    (renders the edge cases in SELF_TEST_CASES and asserts)
      python3 -I music_techhouse.py --story       (renders the 30 s Story demo + section report)
      python3 -I music_techhouse.py --story-selftest   (STORY_SELF_TEST_CASES)
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import warnings

import numpy as np
import soundfile as sf
from scipy import signal as sps
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from pedalboard import Chorus, Compressor, LadderFilter

# ----------------------------------------------------------------------------------------------
# constants / contract
# ----------------------------------------------------------------------------------------------
SR = 48000
BPM = 120.0
BEAT = 60.0 / BPM          # 0.5 s
S16 = BEAT / 4.0           # 0.125 s
BAR = 4.0 * BEAT           # 2.0 s
KEY = "F minor"
SWING = 0.010              # off-16ths ("e" and "a") land 10 ms late  (~54 % swing)
GAP = 0.05                 # the "breath" of silence right before the drop and the end hit
PRE = 2.5                  # pre-roll (s) rendered before t=0 so the intro is already sounding at t=0
TAIL_FADE = 0.25           # final fade to digital silence, ends exactly at `duration`
STEM_PEAK_DBTP = -7.0      # stems are scaled (all together) so the hottest true peak sits here

PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
      "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
SCALE = ("F", "G", "Ab", "Bb", "C", "Db", "Eb")
SCALE_PCS = frozenset(PC[n] for n in SCALE)


def nm(name: str) -> int:
    """'Ab4' -> MIDI 68 (C4 = 60)."""
    m = re.fullmatch(r"([A-G][b#]?)(-?\d)", name)
    if not m:
        raise ValueError(name)
    return PC[m.group(1)] + 12 * (int(m.group(2)) + 1)


def hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


# ----------------------------------------------------------------------------------------------
# ARRANGEMENT AS DATA
# ----------------------------------------------------------------------------------------------
# i - VI - III - VII, one bar each, cycling from the drop downbeat (the intro bar is the VII, so the
# drop resolves Eb -> Fm).  The two beats before the end hit are a VII turnaround, the hit is Fm9.
PROGRESSION = ("Fm", "Db", "Ab", "Eb")
TURNAROUND_CHORD = "Eb"
FINAL_CHORD = "Fm9"

CHORDS = {
    #        chord tones                    bass (root, fifth)  pad voicing (voice-led)           arp ladder (ascending chord tones)
    "Fm":  dict(root="F", tones=("F", "Ab", "C"), bass=("F1", "C2"), pad=("F3", "C4", "F4", "Ab4"),
                arp=("F4", "Ab4", "C5", "F5", "Ab5", "C6")),
    "Db":  dict(root="Db", tones=("Db", "F", "Ab"), bass=("Db2", "Ab1"), pad=("F3", "Db4", "F4", "Ab4"),
                arp=("F4", "Ab4", "Db5", "F5", "Ab5", "Db6")),
    "Ab":  dict(root="Ab", tones=("Ab", "C", "Eb"), bass=("Ab1", "Eb2"), pad=("Eb3", "C4", "Eb4", "Ab4"),
                arp=("Eb4", "Ab4", "C5", "Eb5", "Ab5", "C6")),
    "Eb":  dict(root="Eb", tones=("Eb", "G", "Bb"), bass=("Eb2", "Bb1"), pad=("Eb3", "Bb3", "Eb4", "G4"),
                arp=("Eb4", "G4", "Bb4", "Eb5", "G5", "Bb5")),
    "Fm9": dict(root="F", tones=("F", "Ab", "C", "Eb", "G"), bass=("F1", "C2"),
                pad=("F3", "Ab3", "C4", "Eb4", "G4"), arp=("F4", "Ab4", "C5", "Eb5", "G5", "C6")),
    # render_story() end chord: open Fm(add9), the same colour the cinematic bed ends on
    "Fmadd9": dict(root="F", tones=("F", "Ab", "C", "G"), bass=("F1", "C2"),
                   pad=("F3", "Ab3", "C4", "G4"), arp=("F4", "Ab4", "C5", "G5", "C6", "F6")),
}

# Arp: one bar = 16 sixteenths, values index the chord's arp ladder.  Accents follow a 3-3-2 / 3-3-2
# grouping so the straight 16ths feel syncopated against the four-on-the-floor.
ARP_A = (0, 1, 2, 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3)          # rolling, climbing
ARP_B = (5, 3, 4, 2, 3, 1, 2, 0, 1, 2, 3, 1, 2, 3, 4, 5)          # answer: falls then climbs back
ARP_ACCENTS = frozenset((0, 3, 6, 8, 11, 14))
ARP_TURN = (0, 1, 2, 3, 2, 3, 4, 5)                               # 2-beat turnaround run (16ths)

# Bass: per beat, (sixteenth position, role, velocity).  Position 0 is the kick -> left empty.
BASS_BEAT = ((1, "root", 0.70), (2, "root", 1.00), (3, "oct", 0.80))
BASS_LAST_BEAT = ((1, "root", 0.70), (2, "root", 1.00), (3, "fifth", 0.85))   # 4th beat of a bar
BASS_TURN = (                                                    # walk-up over the VII turnaround
    ((1, "Eb2", "root", 0.75), (2, "Eb2", "root", 1.00), (3, "Bb1", "fifth", 0.85)),
    ((0, "Bb1", "walk", 0.80), (1, "C2", "walk", 0.85), (2, "Db2", "walk", 0.90), (3, "Eb2", "walk", 1.0)),
)

# Hats: per 16th position in a beat: closed-hat velocity, 'open' = offbeat open hat.
HAT_GROOVE = (0.60, 0.36, "open", 0.46)

# Snare body pitch: F-minor degrees above G3 (G3 Ab3 Bb3 C4 Db4), as semitones; the build steps up through
# them one per beat (Db4 on the last 8th), fills climb Ab3-Bb3-C4.
SNARE_STEPS = (0, 1, 3, 5, 6)


def snare_ratio(step: int) -> float:
    return 2.0 ** (SNARE_STEPS[max(0, min(step, len(SNARE_STEPS) - 1))] / 12.0)


# ----------------------------------------------------------------------------------------------
# beat grid
# ----------------------------------------------------------------------------------------------
def beat_grid(duration: float, drop_time: float, end_hit_time: float) -> dict:
    """Beat k is at drop_time + k*BEAT; k = 0 is the drop downbeat, k % 4 == 0 are bar lines.

    drop_time is clamped to 0 .. duration-3 s (warnings.warn, the clamped value is reported in meta), so
    there is always room for >= 3 beats of groove before the end hit and >= 1 s of tail after it.
    k0 is the intro start: the last bar line at or before 0 s (always inside the PRE pre-roll), so the
    intro pad/arp are already sounding at t=0 whatever the drop time."""
    duration, drop, end_hit_time = float(duration), float(drop_time), float(end_hit_time)
    if not all(math.isfinite(v) for v in (duration, drop, end_hit_time)):
        raise ValueError(f"non-finite timing: duration={duration}, drop_time={drop}, end_hit_time={end_hit_time}")
    if duration < 3.0:
        raise ValueError(f"duration must be >= 3 s (got {duration})")
    lim = duration - 3.0
    if not 0.0 <= drop <= lim:
        clamped = min(max(drop, 0.0), lim)
        warnings.warn(f"music_techhouse: drop_time {drop:g} s clamped to {clamped:g} s "
                      f"(must lie in 0 .. duration-3 = {lim:g} s)", stacklevel=3)
        drop = clamped
    k_first = -int(math.floor(drop / BEAT + 1e-9))                         # first beat >= 0 s
    k_last = int(math.ceil((duration - drop) / BEAT - 1e-9)) - 1          # last beat < duration
    hk = int(math.floor((end_hit_time - drop) / BEAT + 0.5))               # nearest beat (ties up)
    hk_max = int(math.floor((duration - 1.0 - drop) / BEAT + 1e-9))        # leave >= 1 s of tail (>= 4)
    hk = min(max(3, hk), hk_max)
    k0 = 4 * int(math.floor(-drop / BAR - 1e-9))                           # intro bar line <= 0 s
    beats = [drop + k * BEAT for k in range(k_first, k_last + 1)]
    return dict(k_first=k_first, k_last=k_last, k0=k0, hk=hk, beats=beats,
                t_hit=drop + hk * BEAT, drop=drop, duration=duration)


def chord_of_beat(k: int, hk: int) -> str:
    if k >= hk:
        return FINAL_CHORD
    if k >= hk - 2:
        return TURNAROUND_CHORD
    return PROGRESSION[(k // 4) % 4]


def chord_segments(g: dict) -> list:
    """[(t0, t1, chord)] -- merges beats with the same chord; starts at the intro bar line (<= 0 s)."""
    drop, hk, k0 = g["drop"], g["hk"], g["k0"]
    segs = []
    for k in range(k0, hk + 1):
        ch = chord_of_beat(k, hk)
        t0 = drop + k * BEAT
        if segs and segs[-1][2] == ch:
            continue
        if segs:
            segs[-1][1] = t0
        segs.append([t0, None, ch])
    segs[-1][1] = g["duration"]
    return [tuple(s) for s in segs]


# ----------------------------------------------------------------------------------------------
# arrangement -> event list
# ----------------------------------------------------------------------------------------------
def _swing(pos16: int) -> float:
    return SWING if pos16 % 2 == 1 else 0.0


def arrange(g: dict, rng) -> list:
    """Build every note / hit as data.  t = performed time (with swing), tn = nominal grid time."""
    drop, hk, t_hit, k0 = g["drop"], g["hk"], g["t_hit"], g["k0"]
    ev = []

    def jit(v, amt=0.05):
        return float(np.clip(v * (1.0 + rng.uniform(-amt, amt)), 0.05, 1.2))

    # ---------------- arp (+ octave lift layer in the second half) ----------------
    for k in range(k0, hk):
        ch = chord_of_beat(k, hk)
        ladder = [nm(n) for n in CHORDS[ch]["arp"]]
        bar = k // 4
        for s in range(4):
            tn = drop + k * BEAT + s * S16
            if tn < -S16:
                continue
            if k >= hk - 2:                                   # turnaround run
                idx = ARP_TURN[(k - (hk - 2)) * 4 + s]
                step = 99
                vel = 0.70 + 0.30 * ((k - (hk - 2)) * 4 + s) / 7.0
            else:
                step = (k % 4) * 4 + s
                pat = ARP_A if (bar < 0 or bar % 2 == 0) else ARP_B
                idx = pat[step]
                vel = 1.0 if step in ARP_ACCENTS else 0.62
            if tn >= t_hit - GAP:
                continue
            ev.append(dict(inst="arp", t=tn + _swing(s), tn=tn, dur=0.30, midi=ladder[idx], vel=jit(vel)))
            if bar >= 4 and k < hk - 2:                        # lift: quiet octave double
                ev.append(dict(inst="arp", t=tn + _swing(s), tn=tn, dur=0.22, midi=ladder[idx] + 12,
                               vel=jit(vel) * 0.38, layer="lift"))

    # ---------------- chord stabs: drop and end hit ----------------
    ev.append(dict(inst="stab", t=drop, tn=drop, dur=0.9, chord="Fm",
                   midis=[nm(n) for n in CHORDS["Fm"]["arp"][:4]], vel=1.0))
    ev.append(dict(inst="stab", t=t_hit, tn=t_hit, dur=2.6, chord=FINAL_CHORD,
                   midis=[nm(n) for n in CHORDS[FINAL_CHORD]["arp"][:5]], vel=1.0, final=True))

    # ---------------- pad (one event per chord segment) ----------------
    for (t0, t1, ch) in chord_segments(g):
        ev.append(dict(inst="pad", t=t0, tn=t0, dur=t1 - t0, chord=ch,
                       midis=[nm(n) for n in CHORDS[ch]["pad"]], final=(ch == FINAL_CHORD)))

    # ---------------- bass ----------------
    for k in range(0, hk - 2):
        ch = chord_of_beat(k, hk)
        root, fifth = (nm(n) for n in CHORDS[ch]["bass"])
        pat = BASS_LAST_BEAT if k % 4 == 3 else BASS_BEAT
        for (pos, role, vel) in pat:
            tn = drop + k * BEAT + pos * S16
            midi = {"root": root, "oct": root + 12, "fifth": fifth}[role]
            ev.append(dict(inst="bass", t=tn + _swing(pos), tn=tn, dur=0.105, midi=midi, role=role,
                           vel=jit(vel, 0.03)))
    for j, beat in enumerate(BASS_TURN):
        k = hk - 2 + j
        for (pos, name, role, vel) in beat:
            tn = drop + k * BEAT + pos * S16
            ev.append(dict(inst="bass", t=tn + _swing(pos), tn=tn, dur=0.105, midi=nm(name), role=role,
                           vel=vel))
    ev.append(dict(inst="bass", t=t_hit, tn=t_hit, dur=g["duration"] - t_hit, midi=nm("F1"), role="root",
                   vel=1.0, final=True))

    # ---------------- drums ----------------
    for k in range(0, hk - 1):                                   # kick: drop .. beat before turnaround end
        ev.append(dict(inst="kick", t=drop + k * BEAT, tn=drop + k * BEAT, vel=jit(1.0, 0.02)))
    ev.append(dict(inst="kick", t=t_hit, tn=t_hit, vel=1.0, big=True))
    for k in range(0, hk - 1):
        if k % 4 in (1, 3):
            ev.append(dict(inst="clap", t=drop + k * BEAT, tn=drop + k * BEAT, vel=jit(1.0, 0.04)))
    # hats -- intro: closed 16ths crescendo; groove: closed 16ths + offbeat open hats
    for k in range(g["k_first"] - 1, hk - 1):
        for s in range(4):
            tn = drop + k * BEAT + s * S16
            if tn < 0.0:
                continue
            if k < 0:
                prog = (tn / max(drop, 1e-3)) if drop > 0 else 1.0
                if tn >= drop - GAP:
                    continue
                v = (0.18 + 0.45 * prog) * (1.0 if s % 2 == 0 else 0.7)
                ev.append(dict(inst="hat", t=tn + _swing(s), tn=tn, vel=jit(v)))
            else:
                h = HAT_GROOVE[s]
                if h == "open":
                    ev.append(dict(inst="ohat", t=tn, tn=tn, vel=jit(0.85 if k // 4 >= 4 else 0.75)))
                else:
                    ev.append(dict(inst="hat", t=tn + _swing(s), tn=tn, vel=jit(h)))
    # snare build into the drop: 8ths -> 16ths -> 32nds, crescendo, high-pass opening
    rolls = []
    for b in np.arange(-4.0, 0.0, 0.0625):
        if b < -2.0 and (b * 2) % 1:          # 8ths before beat -2
            continue
        if -2.0 <= b < -1.0 and (b * 4) % 1:  # 16ths in beat -2
            continue
        tn = drop + b * BEAT
        if tn < 0.0 or tn >= drop - GAP:
            continue
        if b >= -1.0 and (b * 8) % 1:         # 32nds in the last beat
            continue
        rolls.append((tn, (b + 4.0) / 4.0))
    for tn, p in rolls:
        step = 4 if p >= 0.875 else int(p * 4.0 + 1e-9)                # one degree per beat, Db4 last 8th
        ev.append(dict(inst="snare", t=tn, tn=tn, vel=(0.30 + 0.70 * p ** 1.3) * (1.0 - 0.2 * p),   # top x0.8
                       pitch=snare_ratio(step), hp=250 + 900 * p))
    # small fill at the end of every 4th bar (not inside the turnaround)
    for k in range(15, hk - 2, 16):
        for s, v in ((1, 0.45), (2, 0.6), (3, 0.8)):
            tn = drop + k * BEAT + s * S16
            ev.append(dict(inst="snare", t=tn + _swing(s), tn=tn, vel=v, pitch=snare_ratio(s), hp=300))
        ev.append(dict(inst="crash", t=drop + (k + 1) * BEAT, tn=drop + (k + 1) * BEAT, vel=0.55))
    # turnaround fill: beat hk-2 = 16ths on the 2nd half, beat hk-1 = 32nds crescendo (no kick)
    for s in (2, 3):
        tn = drop + (hk - 2) * BEAT + s * S16
        ev.append(dict(inst="snare", t=tn + _swing(s), tn=tn, vel=0.45 + 0.1 * s, pitch=snare_ratio(1), hp=300))
    for i in range(8):
        tn = drop + (hk - 1) * BEAT + i * S16 / 2
        if tn >= t_hit - GAP:
            continue
        p = i / 7.0
        ev.append(dict(inst="snare", t=tn, tn=tn, vel=0.55 + 0.45 * p, pitch=snare_ratio(1 + min(3, int(p * 4.0))),
                       hp=400 + 800 * p))
    ev.append(dict(inst="crash", t=drop, tn=drop, vel=1.0))
    ev.append(dict(inst="crash", t=t_hit, tn=t_hit, vel=1.0))

    # ---------------- fx ----------------
    r0 = max(0.0, drop - 2.0 * BAR)
    if drop - r0 > 0.25:
        ev.append(dict(inst="riser", t=r0, tn=r0, dur=drop - r0 - 0.02,
                       midi_start=nm("F3"), midi_end=nm("F5")))
    ev.append(dict(inst="boom", t=drop, tn=drop, dur=1.6, vel=0.8))
    ev.append(dict(inst="downlifter", t=drop, tn=drop, dur=BAR))
    ev.append(dict(inst="revcym", t=t_hit - BEAT * 2, tn=t_hit - BEAT * 2, dur=BEAT * 2))
    ev.append(dict(inst="boom", t=t_hit, tn=t_hit, dur=g["duration"] - t_hit, vel=1.0, final=True))
    ev.append(dict(inst="impact", t=t_hit, tn=t_hit, dur=g["duration"] - t_hit))
    ev.sort(key=lambda e: e["t"])
    return ev


# ----------------------------------------------------------------------------------------------
# musical-correctness check
# ----------------------------------------------------------------------------------------------
def check_arrangement(events: list, g: dict, segs: list | None = None) -> list:
    """Musical rules (key, chord tones, bass roles, grid).  `segs` = chord timeline; default chord_segments(g)
    (render_story() passes its own)."""
    segs = chord_segments(g) if segs is None else segs
    issues = []

    def chord_at(t):
        for (t0, t1, ch) in segs:
            if t0 - 1e-6 <= t < t1 - 1e-6:
                return ch
        return segs[-1][2] if t >= segs[-1][0] else segs[0][2]

    def pcs(ch):
        return {PC[n] for n in CHORDS[ch]["tones"]}

    for ch, d in CHORDS.items():                                     # chord table itself
        for n in d["tones"]:
            if PC[n] not in SCALE_PCS:
                issues.append(f"chord {ch}: tone {n} out of key")
        for n in d["pad"] + d["arp"]:
            if nm(n) % 12 not in pcs(ch):
                issues.append(f"chord {ch}: voicing note {n} not a chord tone")
        if nm(d["bass"][0]) % 12 != PC[d["root"]]:
            issues.append(f"chord {ch}: bass root mismatch")
        if (nm(d["bass"][1]) - nm(d["bass"][0])) % 12 != 7:
            issues.append(f"chord {ch}: bass fifth mismatch")
    for e in events:
        midis = e.get("midis") or ([e["midi"]] if "midi" in e else [])
        if e["inst"] == "riser":
            midis = [e["midi_start"], e["midi_end"]]
        ch = e.get("chord") or chord_at(e["tn"])
        for m in midis:
            if m % 12 not in SCALE_PCS:
                issues.append(f"{e['inst']} @ {e['t']:.3f}s: MIDI {m} out of key")
            if e["inst"] in ("arp", "pad", "stab", "chord") and m % 12 not in pcs(ch):
                issues.append(f"{e['inst']} @ {e['t']:.3f}s: MIDI {m} not in chord {ch}")
        if e["inst"] == "snare":                                     # tuned snare body (G3 x pitch)
            sm = nm("G3") + 12.0 * math.log2(e["pitch"])
            if abs(sm - round(sm)) > 1e-6 or round(sm) % 12 not in SCALE_PCS:
                issues.append(f"snare @ {e['t']:.3f}s: body pitch MIDI {sm:.2f} not an F-minor note")
        if e["inst"] == "bass":
            root = PC[CHORDS[ch]["root"]]
            want = {"root": root, "oct": root, "fifth": (root + 7) % 12}.get(e["role"])
            if want is not None and e["midi"] % 12 != want:
                issues.append(f"bass @ {e['t']:.3f}s: {e['role']} {e['midi']} wrong for {ch}")
    for (t0, t1, ch) in segs:                                        # bass enters each chord on its root
        bn = [e for e in events if e["inst"] == "bass" and t0 - 1e-6 <= e["tn"] < t1 - 1e-6]
        if bn:
            first = min(bn, key=lambda e: e["tn"])
            if first["midi"] % 12 != PC[CHORDS[ch]["root"]]:
                issues.append(f"bass: first note of {ch} @ {t0:.3f}s is not the root")
    # grid: the drop is a beat, a downbeat, and has the drop kick on its exact sample
    drop = g["drop"]
    if not any(abs(b - drop) < 1e-9 for b in g["beats"]):
        issues.append("drop_time not on the beat grid")
    kicks = [e for e in events if e["inst"] == "kick"]
    if not any(round(e["t"] * SR) == round(drop * SR) for e in kicks):
        issues.append("no kick on the drop sample")
    if not any(abs(b - g["t_hit"]) < 1e-9 for b in g["beats"]):
        issues.append("end hit not on the beat grid")
    if any(e["inst"] == "kick" and e["t"] < drop - 1e-9 for e in events):
        issues.append("kick in the intro")
    return issues


# ----------------------------------------------------------------------------------------------
# DSP helpers
# ----------------------------------------------------------------------------------------------
def _fades(n: int, a: float, r: float) -> np.ndarray:
    """Raised-cosine in/out envelope; first and last sample are exactly 0. Min 2 ms each."""
    env = np.ones(n)
    na = min(n // 2, max(int(round(a * SR)), int(0.002 * SR)))
    nr = min(n - na, max(int(round(r * SR)), int(0.002 * SR)))
    if na > 0:
        env[:na] = 0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)
    if nr > 0:
        env[n - nr:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(1, nr + 1) / nr)
    return env


def _sos(kind, f, order=2):
    return sps.butter(order, f, btype=kind, fs=SR, output="sos")


def _filt(x, kind, f, order=2):
    return sps.sosfilt(_sos(kind, f, order), x, axis=0)


def _peq(x, f0, gain_db, q=1.0):
    """RBJ peaking EQ."""
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / SR
    al = np.sin(w) / (2 * q)
    b = np.array([1 + al * A, -2 * np.cos(w), 1 - al * A])
    a = np.array([1 + al / A, -2 * np.cos(w), 1 - al / A])
    return sps.lfilter(b / a[0], a / a[0], x, axis=0)


def _pan(sig: np.ndarray, pan: float) -> np.ndarray:
    th = (pan + 1.0) * np.pi / 4.0
    return np.stack([sig * np.cos(th), sig * np.sin(th)], axis=1) * np.sqrt(2.0)


def _polyblep_saw(f: float, n: int, sr: float, phase0: float = 0.0) -> np.ndarray:
    dt = f / sr
    ph = (phase0 + dt * np.arange(n)) % 1.0
    y = 2.0 * ph - 1.0
    m = ph < dt
    t = ph[m] / dt
    y[m] -= t + t - t * t - 1.0
    m = ph > 1.0 - dt
    t = (ph[m] - 1.0) / dt
    y[m] -= t * t + t + t + 1.0
    return y


def _polyblep_saw_glide(freqs: np.ndarray, sr: float, phase0: float = 0.0) -> np.ndarray:
    dt = freqs / sr
    ph = (phase0 + np.concatenate([[0.0], np.cumsum(dt[:-1])])) % 1.0
    y = 2.0 * ph - 1.0
    m = ph < dt
    t = ph[m] / dt[m]
    y[m] -= t + t - t * t - 1.0
    m = ph > 1.0 - dt
    t = (ph[m] - 1.0) / dt[m]
    y[m] -= t * t + t + t + 1.0
    return y


def _additive(f0, n, amps, fc, q=0.8, poles=4, fmax=19000.0, block=2048):
    """Band-limited oscillator with a (time-varying) resonant low-pass applied per harmonic.
    amps[k-1] = amplitude of harmonic k; fc = cutoff (scalar or per-sample array). Alias-free.
    Harmonics beyond 14x the block's highest cutoff (< -90 dB for 24 dB/oct) are skipped per block.
    sin(k*th) comes from the recurrence sin((k+1)th) = 2cos(th) sin(k th) - sin((k-1)th): one sin/cos per
    sample instead of one per harmonic (~5x faster, identical to direct evaluation within -240 dB)."""
    K = int(min(len(amps), fmax // f0))
    fc = np.broadcast_to(np.asarray(fc, dtype=np.float64), (n,))
    out = np.zeros(n)
    iq2 = 1.0 / (q * q)
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        kmax = int(min(K, (fc[i0:i1].max() * 14.0) // f0))
        if kmax < 1:
            continue
        th = np.mod(2.0 * np.pi * f0 * np.arange(i0, i1) / SR, 2.0 * np.pi)
        c2 = 2.0 * np.cos(th)
        x = f0 / fc[i0:i1]                                   # harmonic-1 frequency / cutoff
        acc = np.zeros(i1 - i0)
        s_prev, s_cur = np.zeros(i1 - i0), np.sin(th)
        for k in range(1, kmax + 1):
            a = float(amps[k - 1])
            if a != 0.0:
                r2 = (k * x) ** 2
                H = 1.0 / ((1.0 - r2) ** 2 + r2 * iq2)       # |2-pole|^2 = 4-pole magnitude
                acc += a * (H if poles == 4 else np.sqrt(H)) * s_cur
            s_prev, s_cur = s_cur, c2 * s_cur - s_prev
        out[i0:i1] = acc
    return out


_K = np.arange(1, 1200, dtype=np.float64)
SAW_AMPS = 1.0 / _K
SQR_AMPS = np.where(_K % 2 == 1, 1.0 / _K, 0.0)


def _ladder(x: np.ndarray, cutoff, mode="LPF24", res=0.1, drive=1.0, block=1024) -> np.ndarray:
    """Pedalboard Moog-ladder with per-block cutoff automation (state carried across blocks)."""
    st = x.ndim == 2
    xx = np.ascontiguousarray((x.T if st else x[None, :]).astype(np.float32))
    n = xx.shape[1]
    cut = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), (n,))
    lf = LadderFilter(mode=getattr(LadderFilter.Mode, mode), cutoff_hz=float(cut[0]),
                      resonance=res, drive=drive)
    out = np.zeros_like(xx)
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        lf.cutoff_hz = float(np.clip(cut[(i0 + i1) // 2], 20.0, 20000.0))
        y = lf.process(xx[:, i0:i1], SR, buffer_size=block, reset=(i0 == 0))
        out[:, i0:i0 + y.shape[1]] = y
    out = out.astype(np.float64)
    return out.T if st else out[0]


def _pb(plugin, x: np.ndarray) -> np.ndarray:
    """Run a pedalboard plugin over a stereo (n, 2) float64 buffer."""
    y = plugin.process(np.ascontiguousarray(x.T.astype(np.float32)), SR, buffer_size=8192, reset=True)
    out = np.zeros_like(x)
    out[: y.shape[1]] = y.T
    return out


def _delay(x: np.ndarray, d: float) -> np.ndarray:
    n = int(round(d * SR))
    y = np.zeros_like(x)
    if n < len(x):
        y[n:] = x[: len(x) - n]
    return y


PP_NEAR, PP_FAR = 0.7 / math.hypot(0.7, 0.3), 0.3 / math.hypot(0.7, 0.3)   # 70/30 taps, constant power


def _pingpong(x: np.ndarray, delay_s: float, fb: float, taps: int = 7) -> np.ndarray:
    """Ping-pong echoes of the mono sum, each repeat a little darker; first repeat left-ish.  Taps are
    placed 70/30 (not hard L/R) so the echoes survive a mono fold-down."""
    src = x.mean(axis=1)
    src = _filt(src, "highpass", 280.0)
    wet = np.zeros_like(x)
    cur = src
    g = 1.0
    lp = _sos("lowpass", 5200.0, 1)
    for i in range(1, taps + 1):
        g *= fb
        cur = sps.sosfilt(lp, cur)
        tap = g * _delay(cur[:, None], delay_s * i)[:, 0]
        wet[:, (i - 1) % 2] += PP_NEAR * tap
        wet[:, i % 2] += PP_FAR * tap
    return wet


def _side_shelf(x: np.ndarray, fc: float = 1000.0, g: float = 0.7) -> np.ndarray:
    """Scale the side (L-R) signal by ~g above fc (complementary split, mid untouched): mono safety."""
    m = 0.5 * (x[:, 0] + x[:, 1])
    sd = 0.5 * (x[:, 0] - x[:, 1])
    sd = sd - (1.0 - g) * _filt(sd, "highpass", fc, 2)
    return np.stack([m + sd, m - sd], axis=1)


def _peak_limit(x: np.ndarray, ceiling: float, win_ms: float = 5.0, true_peak: bool = False,
                smooth: bool = False) -> np.ndarray:
    """Look-ahead brick-wall limiter (no make-up gain).  gain = box-average of a min-filtered gain
    curve whose window covers the averaging window, so |out| <= ceiling at every (4x) sample.
    smooth=True (render_story): two box passes (triangular kernel) over a min filter twice as wide, so the
    gain is C1-continuous -- no slope corners, which on a loud sub put 2-5 kHz "kinks" on the waveform."""
    a = np.max(np.abs(x), axis=1)
    if true_peak:
        up = np.abs(sps.resample_poly(x, 4, 1, axis=0)).max(axis=1)
        a = np.maximum(a, up[: 4 * len(x)].reshape(len(x), 4).max(axis=1))
    g = np.minimum(1.0, ceiling / np.maximum(a, 1e-12))
    L = max(3, int(win_ms * 1e-3 * SR)) | 1
    if smooth:
        g = uniform_filter1d(uniform_filter1d(minimum_filter1d(g, size=4 * L + 1), size=L), size=L)
    else:
        g = uniform_filter1d(minimum_filter1d(g, size=2 * L + 1), size=L)
    return x * g[:, None]


def _smooth_comp(x: np.ndarray, thr_db: float, ratio: float, attack: float = 0.004, release: float = 0.08,
                 win: float = 0.025) -> np.ndarray:
    """Feed-forward RMS compressor with a C1-smooth gain (render_story bass bus).  25 ms RMS detector (longer
    than an F1 cycle, so it does not follow the waveform), static curve, one-pole attack/release on a 1 ms
    grid, then two 2 ms box passes at audio rate.  pedalboard's Compressor switches its peak ballistics at
    every waveform peak, which on a sustained sub put slope corners (2-5 kHz "kinks", -32 dB) on the peaks."""
    hop = 48
    ms = np.maximum(uniform_filter1d(np.mean(x ** 2, axis=1), int(win * SR))[::hop], 0.0)   # running sums can dip < 0
    lv = 10.0 * np.log10(ms + 1e-20)
    tgt = np.minimum(0.0, (thr_db - lv) * (1.0 - 1.0 / ratio))
    a_att, a_rel = math.exp(-hop / (attack * SR)), math.exp(-hop / (release * SR))
    g = np.empty_like(tgt)
    cur = 0.0
    for i, v in enumerate(tgt.tolist()):
        a = a_att if v < cur else a_rel
        cur = a * cur + (1.0 - a) * v
        g[i] = cur
    gs = np.interp(np.arange(len(x)) / hop, np.arange(len(g)), g)
    gs = uniform_filter1d(uniform_filter1d(gs, 97), 97)
    return x * _db2(gs)[:, None]


_IR_CACHE = {}


def _synth_ir(rt60: float, seed: int, predelay: float = 0.010, build: float = 0.012, hi_damp: float = 0.5,
              lo_mult: float = 1.1, er: int = 6) -> np.ndarray:
    """Synthetic stereo reverb impulse response: decorrelated noise in three bands, each decaying at its
    own rate (highs die faster = damping), a smooth build-up, a few early reflections and a pre-delay.
    Used instead of Freeverb, whose comb/allpass network rings at ~3.8 and ~5.8 kHz (measured +17 dB
    narrow peaks on a clap tail vs +7 dB for plain decaying noise)."""
    key = (rt60, seed, predelay, build, hi_damp, lo_mult, er)
    if key in _IR_CACHE:
        return _IR_CACHE[key]
    rng = np.random.default_rng(seed)
    n = int(rt60 * 1.25 * SR)
    t = np.arange(n) / SR
    ir = np.zeros((n, 2))
    for c in range(2):
        lo = _filt(rng.standard_normal(n), "lowpass", 450.0, 2)
        mid = _filt(rng.standard_normal(n), "bandpass", [450.0, 4000.0], 2)
        hi = _filt(rng.standard_normal(n), "highpass", 4000.0, 2)
        ir[:, c] = (lo * np.exp(-6.91 * t / (rt60 * lo_mult)) + mid * np.exp(-6.91 * t / rt60)
                    + hi * np.exp(-6.91 * t / (rt60 * hi_damp)))
        for j in range(er):                                            # sparse early reflections
            d = int(rng.uniform(0.004, 0.035) * SR)
            ir[d, c] += rng.choice((-1.0, 1.0)) * rng.uniform(0.6, 1.2) * np.exp(-j * 0.25) * 8.0
    ir *= (1.0 - np.exp(-t / build))[:, None]
    ir *= _fades(n, 0.001, 0.05)[:, None]
    ir = np.concatenate([np.zeros((int(predelay * SR), 2)), ir])
    ir /= np.sqrt(np.sum(ir ** 2, axis=0, keepdims=True))             # unit energy per channel
    _IR_CACHE[key] = ir
    return ir


def _reverb(x: np.ndarray, rt60: float, seed: int, hp: float = 200.0, lp: float = 9000.0, **kw) -> np.ndarray:
    """Wet-only convolution reverb (mono in -> decorrelated stereo out), band-limited return."""
    ir = _synth_ir(rt60, seed, **kw)
    m = x.mean(axis=1)
    wet = np.stack([sps.fftconvolve(m, ir[:, c])[: len(m)] for c in range(2)], axis=1)
    return _filt(_filt(wet, "highpass", hp, 2), "lowpass", lp, 2)


def _true_peak(x: np.ndarray) -> float:
    """4x-oversampled peak, linear."""
    y = sps.resample_poly(x, 4, 1, axis=0)
    return float(max(np.max(np.abs(y)), np.max(np.abs(x))))


def _db(v: float) -> float:
    return 20.0 * math.log10(max(v, 1e-12))


def lufs(x: np.ndarray) -> float:
    """ITU-R BS.1770-4 integrated loudness (48 kHz), gated."""
    b1, a1 = [1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585]
    b2, a2 = [1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621]
    y = sps.lfilter(b2, a2, sps.lfilter(b1, a1, x, axis=0), axis=0)
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    if len(y) < blk:
        return -120.0
    ms = np.array([np.sum(np.mean(y[i:i + blk] ** 2, axis=0)) for i in range(0, len(y) - blk + 1, hop)])
    l = -0.691 + 10 * np.log10(np.maximum(ms, 1e-20))
    ms = ms[l > -70]
    if len(ms) == 0:
        return -120.0
    rel = -0.691 + 10 * np.log10(np.mean(ms)) - 10
    ms2 = ms[(-0.691 + 10 * np.log10(ms)) > rel]
    return float(-0.691 + 10 * np.log10(np.mean(ms2)))


# ----------------------------------------------------------------------------------------------
# instruments
# ----------------------------------------------------------------------------------------------
KICK_F_END = hz(nm("F1"))                                         # kick body tuned to the tonic, 43.7 Hz
KNOCK_F = hz(nm("F3"))                                            # 120-300 Hz "knock" layer, settles on F3
KICK_KNOCK, KICK_CLICK = 0.9, 0.75                                # layer levels re the body (phone speakers)


def _kick_freq(t, big=False):
    return KICK_F_END + ((210.0 if big else 175.0) - KICK_F_END) * np.exp(-t / 0.028) + 380.0 * np.exp(-t / 0.0035)


def _kick_phase(dt, big=False):
    """Phase (rad) of the kick body `dt` s after its onset -- used to start subs/booms in phase."""
    n = max(1, int(round(dt * SR)))
    return float(2 * np.pi * np.sum(_kick_freq(np.arange(n) / SR, big)) / SR)


def _kick(vel, rng, big=False):
    dur = 1.0 if big else 0.42
    n = int(dur * SR)
    t = np.arange(n) / SR
    fk = _kick_freq(t, big)
    hold, tau = (0.05, 0.28) if big else (0.035, 0.17)
    amp = np.where(t < hold, 1.0, np.exp(-(t - hold) / tau))
    # knock: the beater "thock" a phone speaker can reproduce (the F1 body is inaudible there).  It rides
    # the body's own pitch sweep shifted up to settle on F3, so it starts in phase with the body and adds
    # to it instead of cancelling; band-limited to 110-420 Hz and saturated together with the body, so it
    # lifts the 120-300 Hz band ~3.5 dB without raising the kick's peak.
    knock = np.sin(2 * np.pi * np.cumsum(KNOCK_F + (fk - KICK_F_END)) / SR) * (1.0 - np.exp(-t / 0.0007)) \
        * np.exp(-t / (0.08 if big else 0.06))
    knock = _filt(_filt(knock, "highpass", 110.0, 2), "lowpass", 420.0, 2)
    knock /= np.max(np.abs(knock)) + 1e-9
    body = np.sin(2 * np.pi * np.cumsum(fk) / SR) * amp
    body = np.tanh(1.7 * (body + KICK_KNOCK * knock)) / np.tanh(1.7)
    click = _filt(rng.standard_normal(n), "bandpass", [3000.0, 5200.0], 2) * np.exp(-t / 0.0035)
    click /= np.max(np.abs(click[: int(0.01 * SR)])) + 1e-9
    y = body + KICK_CLICK * click
    return vel * y * _fades(n, 0.002, 0.03 if not big else 0.25)


def _clap(vel, rng):
    n = int(0.42 * SR)
    t = np.arange(n) / SR
    nz = rng.standard_normal(n)
    body = _filt(nz, "bandpass", [800.0, 2300.0], 2)
    air = _filt(rng.standard_normal(n), "highpass", 6500.0, 2) * 0.22
    env = np.zeros(n)
    for off, g in ((0.0, 0.65), (0.0095, 0.8), (0.0185, 0.9), (0.028, 1.0)):
        lt = t - off
        on = np.clip(lt / 0.0006, 0, 1)
        env += g * on * np.where(lt >= 0, np.exp(-np.maximum(lt, 0) / 0.0045), 0.0)
    env += 0.55 * np.where(t >= 0.028, np.exp(-np.maximum(t - 0.028, 0) / 0.065), 0.0)
    y = _peq((body + air) * env, 3000.0, -3.0, 1.0)
    y /= np.max(np.abs(y)) + 1e-9
    return vel * y * _fades(n, 0.002, 0.05)


_METAL = (205.3, 304.4, 369.6, 522.7, 540.0, 800.0)


def _bl_square(f, n, phase, fmax=20000.0):
    """Band-limited square (odd harmonics 4/(pi k) up to fmax) -- additive, via the recurrence
    sin((k+2)x) = 2 cos(2x) sin(kx) - sin((k-2)x), so no harmonic can fold back (alias-free)."""
    th = 2 * np.pi * f * np.arange(n) / SR + phase
    s_cur = np.sin(th)
    c2 = 2.0 * np.cos(2.0 * th)
    s_prev = -s_cur
    out = s_cur.copy()
    for k in range(3, int(fmax // f) + 1, 2):
        s_prev, s_cur = s_cur, c2 * s_cur - s_prev
        out += s_cur / k
    return out * (4.0 / np.pi)


def _metal(n, rng, scale=1.0):
    """808-style metallic source: six band-limited squares at inharmonic ratios (alias-free)."""
    x = np.zeros(n)
    for f in _METAL:
        x += _bl_square(f * scale, n, rng.uniform(0, 2 * np.pi))
    return x


def _hat(vel, rng, open_=False):
    dur = 0.36 if open_ else 0.08
    n = int(dur * SR)
    t = np.arange(n) / SR
    m = _filt(_metal(n, rng, 1.35), "highpass", 7200.0, 4)
    nz = _filt(rng.standard_normal(n), "highpass", 8500.0, 2)
    tau = 0.11 if open_ else 0.016
    y = (0.75 * m + 0.6 * nz) * np.exp(-t / tau)
    y = _filt(y, "lowpass", 16000.0, 2)
    y /= np.max(np.abs(y)) + 1e-9
    return vel * y * _fades(n, 0.002, 0.08 if open_ else 0.02)


def _snare(vel, rng, pitch=1.0, hp=250.0):
    n = int(0.22 * SR)
    t = np.arange(n) / SR
    f0 = hz(nm("G3")) * pitch
    f = f0 * (1.0 + 0.5 * np.exp(-t / 0.008))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.05)
    nz = _filt(rng.standard_normal(n), "bandpass", [1100.0, 7500.0], 2) * np.exp(-t / 0.075)
    nz /= np.max(np.abs(nz)) + 1e-9
    y = 0.6 * body + 0.75 * nz
    y = _peq(y, 3000.0, -3.0, 0.9)
    y = _filt(y, "highpass", hp, 2)
    y /= np.max(np.abs(y)) + 1e-9
    return vel * y * _fades(n, 0.002, 0.04)


def _crash(vel, rng, dur=2.4):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for c in range(2):
        m = _metal(n, rng, 2.1) + 0.7 * _metal(n, rng, 3.3)
        nz = rng.standard_normal(n)
        y = 0.6 * _filt(m, "bandpass", [4200.0, 14000.0], 2) + 0.55 * _filt(nz, "highpass", 5000.0, 2)
        y = _peq(y, 3200.0, -4.0, 0.8)
        env = np.exp(-t / 0.55) * (0.7 + 0.3 * np.exp(-t / 0.05))
        out[:, c] = y * env
    out /= np.max(np.abs(out)) + 1e-9
    return vel * out * _fades(n, 0.002, 0.4)[:, None]


PLUCK_SIDE = 0.5            # detuned +/- pair level re the centred saw (-6 dB): width without mono loss


def _pluck(midi, dur, vel, base_fc, env_amt=3.2, env_tau=0.045, amp_tau=0.11, detune_c=7.0, cache=None):
    """Filtered saw+square pluck.  L = C + s*D(-detune), R = C + s*D(+detune): the centred oscillator C is
    identical in both channels, the detuned pair only adds width (s = -6 dB) so the mono sum stays solid.
    The timbre is cached on the velocity-scaled cutoff quantised to 1/12 octave (inaudible for a plucked
    note) so most arp notes hit the cache; the velocity amplitude is applied exactly, outside the cache."""
    bfc = base_fc * (0.65 + 0.35 * vel)
    qfc = int(round(12.0 * math.log2(bfc / 1000.0)))
    key = (midi, round(dur, 3), qfc, round(env_amt, 2), round(env_tau, 3), round(amp_tau, 3), round(detune_c, 2))
    y = cache.get(key) if cache is not None else None
    if y is None:
        n = int(dur * SR)
        t = np.arange(n) / SR
        f0 = hz(midi)
        fc = np.minimum(1000.0 * 2.0 ** (qfc / 12.0) * (1.0 + env_amt * np.exp(-t / env_tau)), 17000.0)
        c = _additive(f0, n, SAW_AMPS + 0.45 * SQR_AMPS, fc, q=0.9)
        dl = _additive(f0 * 2 ** (-detune_c / 1200), n, SAW_AMPS, fc, q=0.9)
        dr = _additive(f0 * 2 ** (+detune_c / 1200), n, SAW_AMPS, fc, q=0.9)
        amp = np.exp(-t / amp_tau) * _fades(n, 0.002, min(0.06, dur * 0.3)) / math.sqrt(1.0 + PLUCK_SIDE ** 2)
        y = np.stack([c + PLUCK_SIDE * dl, c + PLUCK_SIDE * dr], axis=1) * amp[:, None]
        if cache is not None:
            cache[key] = y
    return y * vel ** 1.2


def _bass_note(midi, dur, vel, cache, final=False, phase0=0.0, pulse=False):
    """Returns (sub, mid, vel).  The filtered-saw mid layer is cached on a velocity quantised to 0.05 (it only
    sets the filter-envelope depth); the sine sub is cheap and computed per note with its exact phase.
    pulse=True (render_story breakdown): soft 12 ms attack, ~0.15 s decay and a dark mid layer."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f0 = hz(midi)
    if final:                                     # the long tonic note under the end card
        amp = np.exp(-t / 1.1) * _fades(n, 0.003, 0.2)
    elif pulse:                                   # breakdown sub pulse (heartbeat-like, no click)
        amp = _fades(n, 0.012, 0.03) * np.exp(-t / 0.15)
    else:
        amp = _fades(n, 0.003, 0.014) * (0.75 + 0.25 * np.exp(-t / 0.05))
    vq = round(round(vel / 0.05) * 0.05, 2)
    key = (midi, round(dur, 3), vq, final, pulse)
    mid = cache.get(key)
    if mid is None:
        if pulse:
            fc = 150.0 + 450.0 * np.exp(-t / 0.05) * vq
        else:
            fc = 160.0 + 900.0 * np.exp(-t / 0.08) if final else 420.0 + 2200.0 * np.exp(-t / 0.04) * vq
        mid = _additive(f0, n, SAW_AMPS, fc, q=1.1) * amp
        mid = sps.resample_poly(np.tanh(2.0 * sps.resample_poly(mid, 4, 1)) / 2.0, 1, 4)[:n]   # 4x-OS drive
        cache[key] = mid
    sub = np.sin(2 * np.pi * f0 * t + phase0) * amp
    return sub, mid, vel


def _supersaw_chord(midis, n, rng, voices=5, spread_c=15.0, os=2):
    n2 = n * os
    L = np.zeros(n2)
    R = np.zeros(n2)
    det = np.linspace(-spread_c, spread_c, voices)
    pans = rng.permutation(np.linspace(-0.85, 0.85, voices))
    for m in midis:
        for v in range(voices):
            fv = hz(m) * 2 ** ((det[v] + rng.normal(0, 1.2)) / 1200.0)
            s = _polyblep_saw(fv, n2, SR * os, rng.random())
            th = (pans[v] + 1) * np.pi / 4
            L += np.cos(th) * s
            R += np.sin(th) * s
    y = np.stack([L, R], axis=1) * (np.sqrt(2) / (len(midis) * np.sqrt(voices)))
    return sps.resample_poly(y, 1, os, axis=0)


def _swept_noise(n, f0, f1, rng, bw_oct=0.6, stereo=True, curve=1.0):
    """Noise through a smoothly sweeping band (STFT-domain gaussian band, f0 -> f1)."""
    nper, hop = 2048, 256
    chans = []
    for _ in range(2 if stereo else 1):
        x = rng.standard_normal(n + nper)
        f, tt, Z = sps.stft(x, SR, nperseg=nper, noverlap=nper - hop)
        p = np.clip(tt / (n / SR), 0, 1) ** curve
        fc = f0 * (f1 / f0) ** p
        G = np.exp(-0.5 * (np.log2(np.maximum(f[:, None], 1.0) / fc[None, :]) / bw_oct) ** 2)
        _, y = sps.istft(Z * G, SR, nperseg=nper, noverlap=nper - hop)
        y = y[:n]
        chans.append(y / (np.sqrt(np.mean(y ** 2)) + 1e-12))
    return np.stack(chans, axis=1) if stereo else chans[0]


# ----------------------------------------------------------------------------------------------
# story-only instruments (render_story; never instantiated by render())
# ----------------------------------------------------------------------------------------------
LEAD_REL = 0.09             # lead release after the written length (s)
LEAD_SUB = 0.40             # octave-down body layer re the lead (-8 dB)
STORY_CHORD_FC = 1700.0     # offbeat chord-stab pluck base cutoff (Hz)


def _clock(midi, rng):
    """Clock / playhead tick for the story breakdown: a damped sine on `midi` plus its 12th (3rd harmonic,
    also in key: F6 -> C8, C6 -> G7) and a 1 ms band-passed noise snap.  45 ms, peak 1, mono."""
    n = int(0.045 * SR)
    t = np.arange(n) / SR
    f = hz(midi)
    body = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.010) + 0.35 * np.sin(2 * np.pi * 3 * f * t) * np.exp(-t / 0.005)
    snap = _filt(rng.standard_normal(n), "bandpass", [2500.0, 7000.0], 2) * np.exp(-t / 0.0010)
    snap /= np.max(np.abs(snap)) + 1e-9
    y = body + 0.35 * snap
    y /= np.max(np.abs(y)) + 1e-9
    return y * _fades(n, 0.002, 0.012)


def _lead(midi, dur, cache=None):
    """Drop-2 counter-melody voice: a key-tracked low-passed saw (about two audible harmonics, a soft filter
    'bite' on the attack) with a +/-8 cent detuned pair on the side only (mono-safe, like the pluck) and an
    octave-down body layer.  12 ms attack, gentle decay to 80 %, LEAD_REL release.  Peak 1."""
    key = (midi, round(dur, 3))
    if cache is not None and key in cache:
        return cache[key]
    nd = int(dur * SR)
    n = nd + int(LEAD_REL * SR)
    t = np.arange(n) / SR
    f0 = hz(midi)
    fc = f0 * (2.2 + 1.6 * np.exp(-t / 0.08))
    c = _additive(f0, n, SAW_AMPS, fc, q=0.8)
    dl = _additive(f0 * 2 ** (-8 / 1200), n, SAW_AMPS, fc, q=0.8)
    dr = _additive(f0 * 2 ** (+8 / 1200), n, SAW_AMPS, fc, q=0.8)
    lo = _additive(f0 / 2, n, SAW_AMPS, f0 * 1.6, q=0.7)
    env = 0.8 + 0.2 * np.exp(-t / 0.12)
    na = int(0.012 * SR)
    env[:na] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)
    nr = n - nd
    env[nd:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(1, nr + 1) / nr)
    y = np.stack([c + 0.5 * dl + LEAD_SUB * lo, c + 0.5 * dr + LEAD_SUB * lo], axis=1) * env[:, None]
    y /= np.max(np.abs(y)) + 1e-9
    if cache is not None:
        cache[key] = y
    return y


def _lead_bus(events, n_tot, kicks, seed):
    bus = _Bus(n_tot)
    cache = {}
    for e in events:
        if e["inst"] == "lead":
            bus.add(e["t"], _lead(e["midi"], e["dur"], cache) * e["vel"], _db2(STORY_LV["lead"]))
    L = _filt(bus.x, "highpass", 240.0, 2)
    L = _peq(L, 2800.0, -3.0, 1.0)
    L = L + _pingpong(L, 0.375, 0.35) * _db2(STORY_LV["lead_delay"])          # dotted-8th echoes
    L = L + _db2(STORY_LV["lead_verb"]) * _reverb(L, 2.2, seed + 15, hp=250.0, lp=7000.0, predelay=0.025)
    L *= _pump(kicks, n_tot, 0.25, rel=0.18)[:, None]
    return L


def _chord_bus(events, n_tot, kicks, seed):
    bus = _Bus(n_tot)
    cache = {}
    for e in events:
        if e["inst"] == "chord":
            for j, m in enumerate(e["midis"]):
                y = _pluck(m, e["dur"], e["vel"], STORY_CHORD_FC, env_amt=2.0, env_tau=0.03, amp_tau=0.09, cache=cache)
                bus.add(e["t"] + 0.003 * j, y, _db2(STORY_LV["chord"]) / np.sqrt(len(e["midis"])))
    C = _filt(bus.x, "highpass", 170.0, 2)
    C = _peq(C, 2800.0, -2.5, 1.2)
    C = C + _db2(STORY_LV["chord_verb"]) * _reverb(C, 1.1, seed + 16, hp=300.0, lp=7500.0, predelay=0.012)
    C *= _pump(kicks, n_tot, 0.35, rel=0.14)[:, None]                         # offbeat bounce
    return C


# ----------------------------------------------------------------------------------------------
# automation curves (video time, seconds)
# ----------------------------------------------------------------------------------------------
def _pad_cutoff(t, g):
    drop, t_hit, dur = g["drop"], g["t_hit"], g["duration"]
    t0 = drop + g["k0"] * BEAT                                    # intro start (<= 0 s)
    p = np.clip((t - t0) / max(drop - t0, 1e-3), 0, 1)
    intro = 520.0 * (2300.0 / 520.0) ** (p ** 1.6)
    lfo = 2000.0 * 2 ** (0.35 * np.sin(2 * np.pi * (t - drop) / (4 * BAR) - np.pi / 2))
    main = lfo * np.where(t >= drop + 4 * BAR, 1.15, 1.0)
    tr = np.clip((t - (t_hit - BAR / 2)) / (BAR / 2), 0, 1)
    main = main * (1 + 0.6 * tr)
    q = np.clip((t - t_hit) / max(dur - t_hit, 1e-3), 0, 1)
    tail = 3000.0 * (600.0 / 3000.0) ** (q ** 0.8)
    return np.where(t < drop, intro, np.where(t < t_hit, main, tail))


def _arp_cutoff(t, g):
    drop, t_hit = g["drop"], g["t_hit"]
    t0 = drop + g["k0"] * BEAT                                    # intro start (<= 0 s)
    p = np.clip((t - t0) / max(drop - t0, 1e-3), 0, 1)
    intro = 650.0 * (2400.0 / 650.0) ** (p ** 1.4)
    phrase = ((t - drop) % (2 * BAR)) / (2 * BAR)                 # 2-bar filter ramps
    main = 1500.0 * 2 ** (0.75 * phrase) * np.where(t >= drop + 4 * BAR, 1.2, 1.0)
    tr = np.clip((t - (t_hit - BAR / 2)) / (BAR / 2), 0, 1)
    main = main * (1 + 0.8 * tr)
    return np.where(t < drop, intro, np.where(t < t_hit, main, 2600.0))


def _pump(times, n_tot, depth, rel=0.18, hold=0.018):
    """Sidechain-style gain curve: ducks `depth` at each time, cosine recovery over `rel`."""
    env = np.zeros(n_tot)
    L = int((hold + rel) * SR)
    tt = np.arange(L) / SR
    shape = np.where(tt < hold, 1.0, 0.5 + 0.5 * np.cos(np.pi * np.clip((tt - hold) / rel, 0, 1)))
    pre = int(0.003 * SR)
    ramp = np.linspace(0, 1, pre, endpoint=False)
    for t in times:
        i = int(round((t + PRE) * SR))
        a, b = i - pre, min(n_tot, i + L)
        if b <= 0 or a >= n_tot:
            continue
        seg = np.concatenate([ramp, shape])[: b - a]
        lo = max(a, 0)
        env[lo:b] = np.maximum(env[lo:b], seg[lo - a:])
    return 1.0 - depth * env


def _step_gain(tl, t_step, g_before, g_after, ramp=0.02):
    """Gain g_before until t_step, g_after from there (raised-cosine ramp of `ramp` s centred on t_step)."""
    q = np.clip((tl - t_step) / ramp + 0.5, 0.0, 1.0)
    return g_before + (g_after - g_before) * (0.5 - 0.5 * np.cos(np.pi * q))


def _gap_gate(n_tot, times):
    """Silence the last GAP seconds before each given time (4 ms ramps)."""
    g = np.ones(n_tot)
    r = int(0.004 * SR)
    for t in times:
        i1 = int(round((t + PRE) * SR))
        i0 = i1 - int(GAP * SR)
        g[i0 - r:i0] = np.minimum(g[i0 - r:i0], 0.5 + 0.5 * np.cos(np.pi * np.arange(r) / r))
        g[i0:i1] = 0.0
        g[i1:i1 + r // 2] = np.minimum(g[i1:i1 + r // 2], 0.5 - 0.5 * np.cos(np.pi * np.arange(r // 2) / (r // 2)))
    return g


# ----------------------------------------------------------------------------------------------
# render
# ----------------------------------------------------------------------------------------------
class _Bus:
    def __init__(self, n_tot):
        self.x = np.zeros((n_tot, 2))

    def add(self, t, sig, gain=1.0, pan=0.0):
        sig = np.asarray(sig, dtype=np.float64)
        n = sig.shape[0]
        env = _fades(n, 0.002, 0.002)                   # every sound starts/ends with >= 2 ms fades
        sig = (sig * env) if sig.ndim == 1 else sig * env[:, None]
        if sig.ndim == 1:
            sig = _pan(sig, pan)
        i0 = int(round((t + PRE) * SR))
        a, b = max(i0, 0), min(i0 + n, len(self.x))
        if b > a:
            self.x[a:b] += gain * sig[a - i0:b - i0]


def _db2(g):
    return 10 ** (g / 20.0)


# stem-internal levels (dB)
LV = dict(kick=-5.5, clap=-10.0, clap_plate=-19.5, hat=-11.5, ohat=-13.0, snare=-15.5, crash=-18.5,
          sub=-6.0, midbass=-0.5, pad=-9.4, arp=-11.5, arp_intro=-3.0, lift=-12.0, stab=-8.0, stab_drop=-6.0, delay=-18.0,
          riser=-25.0, riser_tone=-29.0, boom=-12.0, downlifter=-27.0, revcym=-19.0, impact=-21.0)
MUSIC_SIDE_HI = 0.7                                # music-bus side gain above 1 kHz
PUMP_PAD = 0.35                                    # pad sidechain depth on the kicks (drop on)
STEM_HP = 30.0                                     # stems high-passed here (4th order): 22-30 Hz is headroom only


def render(duration, drop_time, end_hit_time, seed=1, _parts=None):
    """See module docstring. `_parts` (optional dict) receives cropped sub-buses for analysis."""
    rng = np.random.default_rng(seed)
    N = int(round(duration * SR))
    g = beat_grid(duration, drop_time, end_hit_time)
    events = arrange(g, np.random.default_rng(seed + 1000))
    issues = check_arrangement(events, g)
    if issues:
        raise AssertionError("musical check failed:\n  " + "\n  ".join(issues))

    drop, t_hit = g["drop"], g["t_hit"]
    auto = dict(gaps=[drop, t_hit],                                     # the 50 ms breaths
                ghost=[drop + k * BEAT for k in range(g["k0"], 0)],     # intro pad pulse, no kick
                pad_cutoff=lambda tl: _pad_cutoff(tl, g),
                arp_cutoff=lambda t: _arp_cutoff(t, g),
                arp_gain=lambda tl: _step_gain(tl, drop - GAP / 2, _db2(LV["arp_intro"]), 1.0),  # build arp under the drop
                tail_fade=TAIL_FADE)
    stems, scale = _render_events(events, N, seed, rng, auto, _parts)

    beats = [round(float(b), 9) for b in g["beats"]]                    # drop_time stays exact
    segs = chord_segments(g)
    stems["meta"] = dict(
        bpm=BPM, key=KEY, beat_times=beats, drop_time=float(drop),             # clamped value if clamped
        end_hit_time_snapped=round(float(t_hit), 9),
        # extras
        downbeat_times=[round(float(drop + k * BEAT), 9) for k in range(g["k_first"], g["k_last"] + 1) if k % 4 == 0],
        scale=list(SCALE), progression=list(PROGRESSION), turnaround_chord=TURNAROUND_CHORD,
        final_chord=FINAL_CHORD,
        chord_timeline=[(max(0.0, float(a)), float(b), c) for (a, b, c) in segs if b > 0],
        stem_scale_db=float(_db(scale)), sample_rate=SR, swing_s=SWING,
        n_events=len(events),
        musical_check=dict(violations=0, pitched_events=sum(1 for e in events if 'midi' in e or 'midis' in e),
                           pitched_notes=sum(len(e.get('midis', [])) or ('midi' in e) for e in events),
                           tuned_snare_hits=sum(1 for e in events if e["inst"] == "snare")),
    )
    return stems


def _render_events(events, N, seed, rng, auto, _parts=None):
    """Render an event list (from arrange() or arrange_story()) into the four cropped, cleaned, jointly
    scaled stems.  `auto` holds the arrangement-specific automation (video time):
        gaps        times of the 50 ms breaths (all buses gated shut before each)
        ghost       kick-less pad pulse times (intro / builds)
        pad_cutoff  f(tl) -> pad ladder cutoff per buffer sample
        arp_cutoff  f(t)  -> arp pluck base cutoff at note time
        arp_gain    f(tl) -> arp bus gain per buffer sample
        tail_fade   final raised-cosine fade (s), ends exactly at `duration`
        pad_gain    optional f(tl) -> extra pad gain (render_story only)
        smooth_dynamics optional bool -> C1-smooth bass compressor + bass/arp limiters (render_story only)
        final_pad_amp optional f(t since hit, q = t / (tail - tail_fade)) -> final pad envelope (render_story)
    Story-only instruments (clock, lead, chord, pulse bass) are rendered only when such events exist, so
    render() output is bit-identical to the pre-refactor module.  Returns (stems, common scale)."""
    n_tot = int(PRE * SR) + N + int(0.5 * SR)
    tl = np.arange(n_tot) / SR - PRE                   # video time of every buffer sample
    kicks = [e["t"] for e in events if e["inst"] == "kick"]
    ghost, gaps = auto["ghost"], auto["gaps"]

    # ------------------------------------------------ DRUMS
    drums = _Bus(n_tot)
    plate_send = _Bus(n_tot)
    hat_bank = [_hat(1.0, rng) for _ in range(6)]
    ohat_bank = [_hat(1.0, rng, open_=True) for _ in range(3)]
    clock_bank = {}
    hi = 0
    for e in events:
        i = e["inst"]
        if i == "kick":
            drums.add(e["t"], _kick(e["vel"], rng, big=e.get("big", False)), _db2(LV["kick"] + (1.0 if e.get("big") else 0)))
        elif i == "clap":
            c = _clap(e["vel"], rng)
            drums.add(e["t"], c, _db2(LV["clap"]), pan=0.0)
            plate_send.add(e["t"], c, 1.0)
        elif i == "hat":
            hi += 1
            drums.add(e["t"], hat_bank[hi % 6] * e["vel"], _db2(LV["hat"]), pan=0.22)
        elif i == "ohat":
            drums.add(e["t"], ohat_bank[hi % 3] * e["vel"], _db2(LV["ohat"]), pan=-0.18)
        elif i == "snare":
            s = _snare(e["vel"], rng, e["pitch"], e["hp"])
            drums.add(e["t"], s, _db2(LV["snare"]), pan=0.05)
            plate_send.add(e["t"], s, 0.6)
        elif i == "crash":
            drums.add(e["t"], _crash(e["vel"], rng), _db2(LV["crash"]))
        elif i == "clock":                                               # render_story breakdown tick-tock
            if e["midi"] not in clock_bank:
                clock_bank[e["midi"]] = _clock(e["midi"], rng)
            drums.add(e["t"], clock_bank[e["midi"]] * e["vel"], _db2(STORY_LV["clock"]), pan=e.get("pan", 0.0))
    plate = _reverb(plate_send.x, 0.8, seed + 11, hp=450.0, lp=7500.0, predelay=0.012, hi_damp=0.6)  # short plate
    D = drums.x + _db2(LV["clap_plate"]) * plate * 3.2
    D = D * _gap_gate(n_tot, gaps)[:, None]

    # ------------------------------------------------ BASS
    sub_bus, mid_bus = _Bus(n_tot), _Bus(n_tot)
    bcache = {}
    for e in events:
        if e["inst"] != "bass":
            continue
        prev = [kt for kt in kicks if kt <= e["t"] + 1e-9]                 # start the sub in phase with
        dt = e["t"] - prev[-1] if prev else 1.0                            # the ringing kick body
        ph0 = (_kick_phase(dt, e.get("final", False)) - 2 * np.pi * hz(e["midi"]) * dt) % (2 * np.pi) \
            if dt < BEAT else 0.0
        sub, mid, vel = _bass_note(e["midi"], e["dur"], e["vel"], bcache, final=e.get("final", False), phase0=ph0,
                                   pulse=e.get("shape") == "pulse")
        sub_bus.add(e["t"], sub, vel * _db2(LV["sub"]))
        mid_bus.add(e["t"], mid, vel * _db2(LV["midbass"]))
    sub_x = _filt(sub_bus.x, "lowpass", 140.0, 2)
    mid_x = _filt(_filt(mid_bus.x, "highpass", 110.0, 2), "lowpass", 5000.0, 2)
    mid_x = _peq(mid_x, 900.0, 2.0, 0.8)                                # phone-speaker audibility
    B = sub_x + mid_x
    b_rms = _db(np.sqrt(np.mean(B[int(PRE * SR):] ** 2)) + 1e-12)
    if auto.get("smooth_dynamics", False):
        B = _smooth_comp(B, b_rms + 4.0, 3.0, attack=0.004, release=0.08)
    else:
        B = _pb(Compressor(threshold_db=b_rms + 4.0, ratio=3.0, attack_ms=4.0, release_ms=80.0), B)
    B = _peak_limit(B, _db2(b_rms + 8.0), win_ms=6.0, smooth=auto.get("smooth_dynamics", False))
    B *= _pump(kicks, n_tot, 0.85, rel=0.16)[:, None]
    B *= _gap_gate(n_tot, gaps)[:, None]

    # ------------------------------------------------ MUSIC: pad
    pad_bus = _Bus(n_tot)
    for e in events:
        if e["inst"] != "pad":
            continue
        rel = 0.25
        if e["final"]:
            n = int((e["dur"] + 0.05) * SR)
            y = _supersaw_chord(e["midis"], n, rng)
            tt = np.arange(n) / SR
            q = np.clip(tt / max(e["dur"] - auto["tail_fade"], 0.1), 0, 1)
            if "final_pad_amp" in auto:                                       # render_story: held chord
                amp = auto["final_pad_amp"](tt, q)
            else:
                amp = (1.0 + 0.35 * np.exp(-tt / 0.25)) * (1.0 - 0.6 * q ** 1.5)    # swell on hit, ring out
            y *= (amp * _fades(n, 0.004, 0.05))[:, None]
        else:
            n = int((e["dur"] + rel) * SR)
            y = _supersaw_chord(e["midis"], n, rng)
            y *= _fades(n, 0.03 if e["t"] > 0 else 0.002, rel)[:, None]
        pad_bus.add(e["t"], y, _db2(LV["pad"]))
    P = _ladder(pad_bus.x, auto["pad_cutoff"](tl), res=0.12)
    P = _filt(P, "highpass", 150.0, 4)
    P = _peq(P, 320.0, -2.0, 1.0)
    P = _pb(Chorus(rate_hz=0.33, depth=0.22, centre_delay_ms=7.0, feedback=0.0, mix=0.30), P)
    P = P + 0.40 * _reverb(P, 2.6, seed + 12, hp=150.0, lp=8000.0, predelay=0.02)
    pump_pad = np.minimum(_pump(kicks, n_tot, PUMP_PAD, rel=0.2), _pump(ghost, n_tot, 0.4, rel=0.24))
    P *= pump_pad[:, None]
    if "pad_gain" in auto:
        P *= auto["pad_gain"](tl)[:, None]

    # ------------------------------------------------ MUSIC: arp + stabs
    arp_bus = _Bus(n_tot)
    pcache = {}
    for e in events:
        if e["inst"] == "arp":
            fc = float(auto["arp_cutoff"](np.array([e["t"]]))[0])
            y = _pluck(e["midi"], e["dur"], e["vel"], fc, cache=pcache)
            lvl = LV["lift"] if e.get("layer") == "lift" else LV["arp"]
            arp_bus.add(e["t"], y, _db2(lvl))
        elif e["inst"] == "stab":
            fin = e.get("final", False)
            for j, m in enumerate(e["midis"]):
                y = _pluck(m, e["dur"], 0.95, 3200.0 if fin else 2800.0, env_amt=2.5, env_tau=0.08,
                           amp_tau=0.9 if fin else 0.28, cache=pcache)
                arp_bus.add(e["t"] + 0.004 * j, y, _db2(LV["stab"] if fin else LV["stab_drop"]) / np.sqrt(len(e["midis"])))
    A = _filt(arp_bus.x, "highpass", 220.0, 2)
    a_rms = _db(np.sqrt(np.mean(A[int(PRE * SR):] ** 2)) + 1e-12)
    A = _pb(Compressor(threshold_db=a_rms + 5.0, ratio=4.0, attack_ms=1.5, release_ms=70.0), A)
    A = _peak_limit(A, _db2(a_rms + 9.0), win_ms=5.0, smooth=auto.get("smooth_dynamics", False))
    A = _peq(A, 2800.0, -2.5, 1.2)
    A *= auto["arp_gain"](tl)[:, None]
    A_d = _pingpong(A, 0.375, 0.42) * _db2(LV["delay"] - LV["arp"]) * 4.0
    A = A + A_d
    A = A + 0.135 * _reverb(A, 1.4, seed + 13, hp=250.0, lp=8000.0, predelay=0.015)
    A *= _pump(kicks, n_tot, 0.30, rel=0.16)[:, None]
    X = P + A
    if any(e["inst"] == "lead" for e in events):                      # render_story drop 2 layers
        X = X + _lead_bus(events, n_tot, kicks, seed)
    if any(e["inst"] == "chord" for e in events):
        X = X + _chord_bus(events, n_tot, kicks, seed)
    M = _side_shelf(X, 1000.0, MUSIC_SIDE_HI)                         # mono-safe width above 1 kHz
    if _parts is not None:
        _parts.update(pad=P[int(PRE * SR):int(PRE * SR) + N].copy(), arp=A[int(PRE * SR):int(PRE * SR) + N].copy(),
                      sub=sub_x[int(PRE * SR):int(PRE * SR) + N].copy(), midbass=mid_x[int(PRE * SR):int(PRE * SR) + N].copy())
    M *= _gap_gate(n_tot, gaps)[:, None]

    # ------------------------------------------------ FX
    fx = _Bus(n_tot)
    send = _Bus(n_tot)                                   # end-hit impact -> big reverb
    for e in events:
        i = e["inst"]
        if i == "riser":
            n = int(e["dur"] * SR)
            tt = np.arange(n) / SR
            p = tt / e["dur"]
            nz = _swept_noise(n, 350.0, 9000.0, rng, bw_oct=0.7, curve=1.3)
            env = (0.15 + 0.85 * p ** 2.2) * _fades(n, 0.01, 0.012)
            fx.add(e["t"], nz * env[:, None], _db2(LV["riser"]))
            fr = hz(e["midi_start"]) * (hz(e["midi_end"]) / hz(e["midi_start"])) ** (p ** 1.5)
            tone = np.zeros((n, 2))
            for c, dc in enumerate((-9.0, 9.0)):
                os2 = 2
                fr2 = np.repeat(fr * 2 ** (dc / 1200), os2)
                s = _polyblep_saw_glide(fr2, SR * os2, rng.random()) + \
                    _polyblep_saw_glide(fr2 * 2.0, SR * os2, rng.random()) * 0.5
                tone[:, c] = sps.resample_poly(s, 1, os2)[:n]
            tone = _ladder(tone, 500.0 * (7000.0 / 500.0) ** p, res=0.25)
            fx.add(e["t"], tone * env[:, None], _db2(LV["riser_tone"]))
        elif i == "downlifter":
            n = int(e["dur"] * SR)
            tt = np.arange(n) / SR
            nz = _swept_noise(n, 7000.0, 260.0, rng, bw_oct=0.8, curve=0.6)
            fx.add(e["t"], nz * (np.exp(-tt / 0.5) * _fades(n, 0.004, 0.3))[:, None], _db2(LV["downlifter"]))
        elif i == "revcym":
            n = int(e["dur"] * SR)
            c = _crash(1.0, rng, dur=e["dur"] + 0.6)[::-1][-n:]
            c = c * ((np.arange(n) / n) ** 1.5 * _fades(n, 0.01, 0.003))[:, None]
            fx.add(e["t"] - 0.003, c / (np.max(np.abs(c)) + 1e-9), _db2(LV["revcym"]))
        elif i == "boom":
            n = int(e["dur"] * SR)
            tt = np.arange(n) / SR
            tau = 0.9 if e.get("final") else 0.45                          # same pitch path as the kick
            y = np.sin(2 * np.pi * np.cumsum(_kick_freq(tt, e.get("final", False))) / SR) * np.exp(-tt / tau)
            th = _filt(rng.standard_normal(n), "lowpass", 380.0, 2) * np.exp(-tt / 0.035) * 0.6
            y = np.tanh(1.4 * (y + th)) / np.tanh(1.4)
            y = y * _fades(n, 0.002, min(0.4, e["dur"] * 0.3))
            fx.add(e["t"], y, _db2(LV["boom"]) * e["vel"])
            if e.get("final"):
                send.add(e["t"], _filt(y, "highpass", 200.0, 2), _db2(LV["boom"]))
        elif i == "impact":
            n = int(min(e["dur"], 2.5) * SR)
            tt = np.arange(n) / SR
            nz = _filt(rng.standard_normal((n, 2)), "bandpass", [300.0, 5000.0], 2)
            y = nz * (np.exp(-tt / 0.09))[:, None]
            fx.add(e["t"], y, _db2(LV["impact"]))
            send.add(e["t"], y, _db2(LV["impact"]))
    F = fx.x + 1.14 * _reverb(send.x, 3.0, seed + 14, hp=200.0, lp=9000.0, predelay=0.02)
    F = _filt(F, "highpass", 28.0, 2)

    # ------------------------------------------------ crop, clean, fades, level
    stems = {}
    s0 = int(PRE * SR)
    n_fade = int(auto["tail_fade"] * SR)
    out_env = np.ones(N)
    out_env[:int(0.005 * SR)] = 0.5 - 0.5 * np.cos(np.pi * np.arange(int(0.005 * SR)) / int(0.005 * SR))
    out_env[N - n_fade:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(1, n_fade + 1) / n_fade)
    for name, X in (("drums", D), ("bass", B), ("music", M), ("fx", F)):
        X = _filt(X, "highpass", STEM_HP, 4)                            # DC / rumble / sub headroom
        X = X[s0:s0 + N] * out_env[:, None]
        X = X - (np.mean(X, axis=0) * out_env[:, None])                 # residual DC, keeps the 0 ends
        stems[name] = X
    tps = {k: _true_peak(v) for k, v in stems.items()}
    scale = _db2(STEM_PEAK_DBTP) / max(tps.values())
    for k in stems:
        stems[k] = np.nan_to_num(stems[k] * scale).astype(np.float32)
    return stems, scale


# ----------------------------------------------------------------------------------------------
# demo + measurement
# ----------------------------------------------------------------------------------------------
DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo")


def _ebur128(path: str) -> dict:
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-filter_complex",
                        "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    txt = p.stderr[p.stderr.rfind("Summary:"):]
    get = lambda pat: float(re.search(pat, txt, re.S).group(1))
    return dict(I_LUFS=get(r"I:\s+(-?[\d.]+) LUFS"), LRA_LU=get(r"LRA:\s+(-?[\d.]+) LU"),
                TP_dBTP=get(r"True peak:\s+Peak:\s+(-?[\d.inf]+) dBFS"))


def band_energy(x: np.ndarray) -> dict:
    f, P = sps.welch(x.mean(axis=1), SR, nperseg=8192)
    bands = [(20, 60), (60, 120), (120, 250), (250, 1000), (1000, 2000), (2000, 4000), (4000, 6000),
             (6000, 12000), (12000, 20000)]
    tot = np.sum(P[(f >= 20) & (f < 20000)]) + 1e-20
    return {f"{a}-{b}": round(float(100 * np.sum(P[(f >= a) & (f < b)]) / tot), 2) for a, b in bands}


def stem_report(stems: dict) -> dict:
    rep = {}
    for k in ("drums", "bass", "music", "fx"):
        x = stems[k].astype(np.float64)
        nz = np.nonzero(np.max(np.abs(x), axis=1) > 10 ** (-60 / 20))[0]
        rep[k] = dict(
            peak_dBFS=round(_db(np.max(np.abs(x))), 2), true_peak_dBTP=round(_db(_true_peak(x)), 2),
            rms_dBFS=round(_db(np.sqrt(np.mean(x ** 2))), 2), lufs=round(lufs(x), 2),
            dc=[float(f"{v:.2e}") for v in np.mean(x, axis=0)],
            finite=bool(np.all(np.isfinite(x))),
            first_audible_s=round(float(nz[0] / SR), 4) if len(nz) else None,
            first_last_sample=[float(np.max(np.abs(x[0]))), float(np.max(np.abs(x[-1])))],
            bands_pct=band_energy(x))
    return rep


STEMS = ("drums", "bass", "music", "fx")


def translation_report(mix: np.ndarray, drop: float) -> dict:
    """Small-speaker check: loudness of the drop bar vs the last second of the build, full band and with
    the lows removed (4th-order high-pass = phone-speaker model).  Positive LU = the drop is louder."""
    out = {}
    for name, f in (("full", None), ("HP150", 150.0), ("HP400", 400.0)):
        y = mix if f is None else _filt(mix, "highpass", f, 4)
        b = lufs(y[int(round(max(drop - 1.0, 0.0) * SR)):int(round(drop * SR))])
        d = lufs(y[int(round(drop * SR)):int(round((drop + BAR) * SR))])
        out[name] = dict(build_last_1s=round(b, 2), drop_bar=round(d, 2), drop_minus_build_LU=round(d - b, 2))
    return out


def stereo_report(x: np.ndarray, win: float = 0.5) -> dict:
    """L/R correlation overall, per band and the worst `win`-second window (windows with signal only)."""
    x = np.asarray(x, np.float64)

    def corr(a):
        return float(np.sum(a[:, 0] * a[:, 1]) / (math.sqrt(np.sum(a[:, 0] ** 2) * np.sum(a[:, 1] ** 2)) + 1e-30))
    e_lr = 0.5 * np.sum(x ** 2)
    e_m = np.sum((0.5 * (x[:, 0] + x[:, 1])) ** 2)
    w = int(win * SR)
    wins = [(i / SR, corr(x[i:i + w])) for i in range(0, len(x) - w + 1, w) if np.sum(x[i:i + w] ** 2) > 1e-7 * w]
    worst = min(wins, key=lambda z: z[1]) if wins else (None, 1.0)
    return dict(corr=round(corr(x), 3), mono_loss_dB=round(10 * math.log10((e_m + 1e-30) / (e_lr + 1e-30)), 2),
                corr_1k_3k=round(corr(_filt(x, "bandpass", [1000.0, 3000.0], 4)), 3),
                corr_3k_8k=round(corr(_filt(x, "bandpass", [3000.0, 8000.0], 4)), 3),
                worst_window=dict(t=round(worst[0], 2) if worst[0] is not None else None, corr=round(worst[1], 3), win_s=win))


def _glue_comp(x: np.ndarray, thr_rel_db=3.0, ratio=2.0, attack=0.015, release=0.15, sc_hp=150.0) -> np.ndarray:
    """Bus 'glue' compressor whose detector is high-passed at `sc_hp` (so the kick/sub do not pump the
    whole bus) with the threshold set relative to the detector's loudness.  10 ms RMS detector, gain
    computed on a 1 ms grid with attack/release smoothing, then interpolated back to audio rate."""
    sc = _filt(x, "highpass", sc_hp, 2)
    hop = 48
    lv = 10.0 * np.log10(uniform_filter1d(np.mean(sc ** 2, axis=1), int(0.010 * SR))[::hop] + 1e-20)
    tgt = np.minimum(0.0, (lufs(sc) + thr_rel_db - lv) * (1.0 - 1.0 / ratio))
    a_att, a_rel = math.exp(-hop / (attack * SR)), math.exp(-hop / (release * SR))
    g = np.empty_like(tgt)
    cur = 0.0
    for i, v in enumerate(tgt.tolist()):
        a = a_att if v < cur else a_rel
        cur = a * cur + (1.0 - a) * v
        g[i] = cur
    gs = np.interp(np.arange(len(x)) / hop, np.arange(len(g)), g)
    return x * _db2(gs)[:, None]


def master_preview(mix: np.ndarray, target_lufs=-14.0, ceiling_dbtp=-1.0) -> np.ndarray:
    """Quick preview master (NOT the final mixer): side-chain-HP150 glue comp, gain to target, true-peak
    limiter.  (A full-band detector with a fixed -20 dBFS threshold let the kick pump the bus and flattened
    the drop on phone speakers: HP150 drop-vs-build +2.4 LU in the stems became -0.5 LU after mastering.)"""
    x = _glue_comp(mix)
    x = x * _db2(target_lufs - lufs(x))
    y = _peak_limit(x, _db2(ceiling_dbtp - 0.3), win_ms=8.0, true_peak=True)
    tp = _true_peak(y)
    if _db(tp) > ceiling_dbtp - 0.1:
        y = y * _db2(ceiling_dbtp - 0.15) / tp
    return y


def render_demo(duration=17.5, drop_time=1.8, end_hit_time=14.5, seed=1, out_dir=DEMO_DIR):
    import time
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    st = render(duration, drop_time, end_hit_time, seed=seed)
    render_s = time.time() - t0
    mix = sum(st[k].astype(np.float64) for k in STEMS)
    y = master_preview(mix)
    y = (y - np.mean(y, axis=0) * _fades(len(y), 0.005, TAIL_FADE)[:, None])
    wav = os.path.join(out_dir, "music_techhouse_demo.wav")
    sf.write(wav, y.astype(np.float32), SR, subtype="FLOAT")
    png = os.path.join(out_dir, "music_techhouse_demo_spectrogram.png")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", wav, "-lavfi",
                    "showspectrumpic=s=1600x600:legend=1", png], check=True)
    meta = st["meta"]
    sr_ = stem_report(st)
    eb = _ebur128(wav)
    tr = translation_report(mix, meta["drop_time"])
    tr_demo = translation_report(y, meta["drop_time"])
    mono = stereo_report(st["music"])
    first100 = _db(np.sqrt(np.mean(mix[:int(0.1 * SR)] ** 2)))
    checks = dict(
        stems_true_peak_le_minus6=all(v["true_peak_dBTP"] <= -6.0 for v in sr_.values()),
        stems_dc_lt_1e4=all(max(abs(d) for d in v["dc"]) < 1e-4 for v in sr_.values()),
        stems_finite=all(v["finite"] for v in sr_.values()),
        stems_start_end_zero=all(max(v["first_last_sample"]) < 1e-6 for v in sr_.values()),
        demo_true_peak_le_minus1=eb["TP_dBTP"] <= -1.0,
        notes_in_key=meta["musical_check"]["violations"] == 0,
        audible_from_t0=first100 > -35.0,
        drop_beats_build_on_HP150_by_2LU=tr["HP150"]["drop_minus_build_LU"] >= 2.0,
        demo_drop_beats_build_on_HP150_by_2LU=tr_demo["HP150"]["drop_minus_build_LU"] >= 2.0,
        music_mono_corr_ge_045=mono["corr"] >= 0.45,
        music_no_window_corr_lt_02=mono["worst_window"]["corr"] >= 0.2)
    rep = dict(checks=checks, all_checks_pass=all(checks.values()),
               params=dict(duration=duration, drop_time=drop_time, end_hit_time=end_hit_time, seed=seed),
               render_s=round(render_s, 1),
               meta={k: v for k, v in meta.items() if k != "beat_times"},
               n_beats=len(meta["beat_times"]),
               stems=sr_,
               unmastered_sum=dict(lufs=round(lufs(mix), 2), true_peak_dBTP=round(_db(_true_peak(mix)), 2),
                                   first100ms_rms_dBFS=round(first100, 1)),
               translation_drop_vs_build=dict(stems_sum=tr, demo_master=tr_demo),
               music_stereo=mono,
               demo=dict(ebur128=eb, python_lufs=round(lufs(y), 2),
                         true_peak_dBTP=round(_db(_true_peak(y)), 2),
                         dc=[float(f"{v:.2e}") for v in np.mean(y, axis=0)]))
    with open(os.path.join(out_dir, "music_techhouse_demo_report.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    return rep


SELF_TEST_CASES = (
    (17.5, 1.8, 14.5),       # the demo
    (17.5, 2.37, 14.5),      # intro bar starts at -1.63 s (used to be silent until 0.37 s)
    (16.0, 0.4, 13.0),       # very short intro (used to be hats + riser only)
    (17.5, 2.2, 14.5),       # used to be silent until 0.20 s
    (17.5, 4.3, 15.0),       # two intro bars (Ab, Eb); used to be silent until 0.30 s
    (17.5, 0.0, 14.5),       # drop on the very first sample
    (17.5, 16.0, 17.0),      # drop too late: clamped to duration-3 with a warning (used to raise)
)


def self_test(cases=SELF_TEST_CASES, seed=1, verbose=True) -> list:
    """Render each (duration, drop_time, end_hit_time) case and assert the contract: shapes/dtype, finite,
    stem true peak <= -6 dBTP, |DC| < 1e-4, zero first/last samples, drop and snapped hit on the beat grid,
    drop on a downbeat, >= 1 s tail, the first 100 ms of the stem sum above -35 dBFS RMS (audible from
    t=0), and a warning exactly when drop_time had to be clamped.  Raises AssertionError on any failure."""
    results, failed = [], []
    for (dur, drop, hit) in cases:
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            st = render(dur, drop, hit, seed=seed)
        m = st["meta"]
        N = int(round(dur * SR))
        xs = {k: st[k].astype(np.float64) for k in STEMS}
        mix = sum(xs.values())
        first100 = _db(np.sqrt(np.mean(mix[:int(0.1 * SR)] ** 2)))
        want_drop = min(max(drop, 0.0), dur - 3.0)
        on_grid = lambda t: any(abs(b - t) < 1e-9 for b in m["beat_times"])
        chk = dict(
            shape_dtype=all(st[k].shape == (N, 2) and st[k].dtype == np.float32 for k in STEMS),
            finite=all(np.all(np.isfinite(v)) for v in xs.values()),
            true_peak_le_minus6=all(_db(_true_peak(v)) <= -6.0 for v in xs.values()),
            dc_lt_1e4=all(np.max(np.abs(np.mean(v, axis=0))) < 1e-4 for v in xs.values()),
            ends_zero=all(np.max(np.abs(v[[0, -1]])) < 1e-6 for v in xs.values()),
            drop_reported=abs(m["drop_time"] - want_drop) < 1e-9,
            drop_on_grid=on_grid(m["drop_time"]),
            drop_is_downbeat=any(abs(b - m["drop_time"]) < 1e-9 for b in m["downbeat_times"]),
            hit_on_grid=on_grid(m["end_hit_time_snapped"]),
            hit_leaves_1s_tail=m["end_hit_time_snapped"] <= dur - 1.0 + 1e-9,
            first100ms_gt_minus35dBFS=first100 > -35.0,
            clamp_warning_iff_clamped=(len(wl) > 0) == (abs(want_drop - drop) > 1e-12))
        r = dict(case=(dur, drop, hit), drop=m["drop_time"], hit=m["end_hit_time_snapped"],
                 first100ms_dBFS=round(first100, 1), ok=all(chk.values()),
                 failed=[k for k, v in chk.items() if not v])
        results.append(r)
        if verbose:
            print(json.dumps(r), flush=True)
        if not r["ok"]:
            failed.append(r)
    if failed:
        raise AssertionError("self-test failed: " + json.dumps(failed))
    return results


# ==============================================================================================
# render_story(): the 30 s Instagram Story arrangement (four fixed music anchors)
# ==============================================================================================
# Sections (all boundaries are bar lines of the grid anchored on drop1):
#   0 .. drop1            intro / build  -- the render() intro, unchanged
#   drop1 .. breakdown    groove A       -- lighter first 4 bars, then "full" (lift, brighter); a fill every
#                                           4 bars and a falling exit fill into the breakdown
#   breakdown .. drop2    breakdown      -- dark bar(s): one kick+crash+boom on the downbeat, then no kick,
#                                           low-passed pad, 8th-note sub pulse, clock tick-tock; the last bar is
#                                           a build (VI-VII, riser, snare roll, arp re-enters filtered)
#   drop2 .. end_hit      drop 2         -- groove A "full" + backbeat snare, offbeat chord stabs, a lead
#                                           counter-melody, brighter pad/arp; Eb turnaround into the hit
#   end_hit .. duration   end            -- big hit on the downbeat, then a sustained Fm(add9) + F1 + tails,
#                                           raised-cosine fade to digital zero at `duration`
STORY_DEFAULTS = dict(duration=30.0, drop1=2.0, breakdown=14.0, drop2=18.0, end_hit=26.0)
STORY_FINAL_CHORD = "Fmadd9"
STORY_TAIL_FADE = 0.30
BREAKDOWN_DARK = ("Fm", "Db")                 # dark-bar chords, one bar each, cycling (one dark bar: Fm)
BREAKDOWN_BUILD = ("Db", "Db", "Eb", "Eb")    # build bar, per beat: VI -> VII -> i on drop 2

STORY_SECTIONS = dict(
    # bars count from the section's own downbeat; (a, b) pairs = value before / from `full_from_bar`
    groove_a=dict(arp_patterns=(ARP_A, ARP_B), full_from_bar=4, ohat_vel=(0.75, 0.85), hat_gain=1.0,
                  kick_vel=1.0, bass_vel=1.0, pad_db=0.0,
                  pad_bright=(1.0, 1.15), arp_bright=(1.0, 1.2), fill_every=4, exit_fill=True,
                  backbeat_snare=False, chord_stabs=False, lead=False),
    drop2=dict(arp_patterns=(ARP_A, ARP_B), full_from_bar=0, ohat_vel=(0.9, 0.9), hat_gain=1.2,
               kick_vel=1.08, bass_vel=1.08, pad_db=1.5,
               pad_bright=(1.3, 1.3), arp_bright=(1.3, 1.3), fill_every=4, exit_fill=False,
               backbeat_snare=True, chord_stabs=True, lead=True),
)
STORY_BREAKDOWN = dict(
    entry=dict(kick=0.9, crash=0.5, boom=0.55, boom_dur=1.2,      # the breakdown downbeat accent; no music
               downlifter=False),                                 # downlifter: the cue sheet's SFX one owns it
    dark_pad_fc=480.0, dark_glide=0.35, dark_wobble_oct=0.15,      # pad LP: glide down, slow 1-bar breathing
    pad_dark_db=2.0,                                              # pad lift while dark (it is low-passed)
    pulse=((0, 0.5, 0.2), (2, 1.0, 0.24)),                        # dark bars: (16th pos, vel, dur) per beat --
                                                                  # off-8th accent, so stamps on the beat land in a dip
    clock=((0, "F6", 0.9, 0.3), (2, "C6", 0.65, -0.3)),           # tick on the beat, tock on the off-8th
    build_pad_fc=(480.0, 2300.0), build_arp_fc=(650.0, 2400.0),
)
# exit fill of groove A: falling Db4 C4 Ab3 G3 16ths (decrescendo) on its last beat
EXIT_FILL = ((0, 4, 0.72), (1, 3, 0.62), (2, 1, 0.52), (3, 0, 0.44))       # (16th pos, snare step, vel)
# drop-2 counter-melody per bar chord: (16th position in the bar, length in 16ths, note).  3-3-2 rhythm in
# 8ths (dotted quarter, dotted quarter, 8th, 8th); beats 1 and 3 always land on chord tones.
STORY_LEAD = {
    "Fm": ((0, 6, "C6"), (6, 6, "Ab5"), (12, 2, "C6"), (14, 2, "Eb6")),
    "Db": ((0, 6, "Db6"), (6, 6, "Ab5"), (12, 2, "F5"), (14, 2, "Ab5")),
    "Ab": ((0, 6, "C6"), (6, 6, "Eb6"), (12, 2, "C6"), (14, 2, "Bb5")),
    "Eb": ((0, 6, "Bb5"), (6, 6, "G5"), (12, 2, "Ab5"), (14, 2, "Bb5")),
}
# story-only levels (dB; *_delay / *_verb are wet returns re dry)
STORY_LV = dict(clock=-19.0, lead=-10.0, lead_delay=-9.0, lead_verb=-13.0, chord=-10.5, chord_verb=-15.0,
                backbeat_snare=0.5, pad_end_db=6.0)          # pad_end_db: held Fm(add9) under the end card
STORY_TAIL_PAD_FC = (3200.0, 1400.0, 1.3)     # end: pad LP closes start -> end Hz (curve exponent), slower than render()
STORY_END_HOLD = (0.45, 2.5)                  # end pad envelope 1 - a*q^b: holds (-1 dB at mid-tail), -5 dB at the fade


def story_grid(duration=30.0, drop1=2.0, breakdown=14.0, drop2=18.0, end_hit=26.0) -> dict:
    """beat_grid() of the story plus its section anchors.  breakdown / drop2 / end_hit are snapped to the
    nearest bar line of the grid anchored on drop1 (warnings.warn when one moves); every section must be at
    least one bar and the hit must leave >= 1 s of tail (ValueError otherwise)."""
    req = dict(duration=duration, drop1=drop1, breakdown=breakdown, drop2=drop2, end_hit=end_hit)
    req = {k: float(v) for k, v in req.items()}
    if not all(math.isfinite(v) for v in req.values()):
        raise ValueError(f"render_story: non-finite timing {req}")
    d1 = req["drop1"]
    if d1 < 0.0:
        raise ValueError(f"render_story: drop1 must be >= 0 (got {d1})")
    bars = {}
    for name in ("breakdown", "drop2", "end_hit"):
        nb = int(round((req[name] - d1) / BAR))
        t = d1 + nb * BAR
        if abs(t - req[name]) > 1e-6:
            warnings.warn(f"music_techhouse.render_story: {name} {req[name]:g} s snapped to the bar line "
                          f"{t:g} s (bars are {BAR:g} s from drop1 = {d1:g} s)", stacklevel=3)
        bars[name] = nb
    nA, nB, n2 = bars["breakdown"], bars["drop2"] - bars["breakdown"], bars["end_hit"] - bars["drop2"]
    if min(nA, nB, n2) < 1:
        raise ValueError(f"render_story: groove A / breakdown / drop 2 must each be >= 1 bar "
                         f"(got {nA} / {nB} / {n2} bars of {BAR:g} s)")
    t_hit = d1 + bars["end_hit"] * BAR
    if t_hit > req["duration"] - 1.0 + 1e-9:
        raise ValueError(f"render_story: end_hit {t_hit:g} s must leave >= 1 s before duration {req['duration']:g} s")
    g = beat_grid(req["duration"], d1, t_hit)
    kb, kd = 4 * nA, 4 * (nA + nB)
    assert g["hk"] == 4 * bars["end_hit"] and abs(g["drop"] - d1) < 1e-12
    g.update(kb=kb, kd=kd, kbuild=kd - 4, t_bd=d1 + kb * BEAT, t_d2=d1 + kd * BEAT, t_build=d1 + (kd - 4) * BEAT,
             bars=dict(groove_a=nA, breakdown=nB, drop2=n2), requested=req)
    return g


def story_chord_of_beat(k: int, g: dict) -> str:
    kb, kd, hk = g["kb"], g["kd"], g["hk"]
    if k >= hk:
        return STORY_FINAL_CHORD
    if k < kb:                                                   # intro (Ab, Eb into drop 1) and groove A
        return PROGRESSION[(k // 4) % 4]
    if k < kd - 4:                                               # breakdown, dark bars
        return BREAKDOWN_DARK[((k - kb) // 4) % len(BREAKDOWN_DARK)]
    if k < kd:                                                   # breakdown, build bar
        return BREAKDOWN_BUILD[k - (kd - 4)]
    if k >= hk - 2:
        return TURNAROUND_CHORD
    return PROGRESSION[((k - kd) // 4) % 4]


def story_segments(g: dict) -> list:
    """[(t0, t1, chord)] of the story, from the intro bar line (<= 0 s) to duration."""
    drop, hk, k0 = g["drop"], g["hk"], g["k0"]
    segs = []
    for k in range(k0, hk + 1):
        ch = story_chord_of_beat(k, g)
        if segs and segs[-1][2] == ch:
            continue
        if segs:
            segs[-1][1] = drop + k * BEAT
        segs.append([drop + k * BEAT, None, ch])
    segs[-1][1] = g["duration"]
    return [tuple(s) for s in segs]


def _story_section(k: int, g: dict):
    """(section name, beat index within the section) of beat k."""
    if k < 0:
        return "intro", k - g["k0"]
    if k < g["kb"]:
        return "groove_a", k
    if k < g["kd"]:
        return "breakdown", k - g["kb"]
    if k < g["hk"]:
        return "drop2", k - g["kd"]
    return "end", k - g["hk"]


def _snare_roll(t_target: float, t_min: float) -> list:
    """The render() build roll into t_target: 8ths -> 16ths (beat -2) -> 32nds (last beat), (tn, progress)."""
    rolls = []
    for b in np.arange(-4.0, 0.0, 0.0625):
        if b < -2.0 and (b * 2) % 1:
            continue
        if -2.0 <= b < -1.0 and (b * 4) % 1:
            continue
        tn = t_target + b * BEAT
        if tn < t_min or tn >= t_target - GAP:
            continue
        if b >= -1.0 and (b * 8) % 1:
            continue
        rolls.append((tn, (b + 4.0) / 4.0))
    return rolls


def arrange_story(g: dict, rng) -> list:
    """Every note / hit of the story as data (same event format as arrange())."""
    drop, hk, k0, kb, kd, kbuild = g["drop"], g["hk"], g["k0"], g["kb"], g["kd"], g["kbuild"]
    t_bd, t_d2, t_build, t_hit = g["t_bd"], g["t_d2"], g["t_build"], g["t_hit"]
    BD = STORY_BREAKDOWN
    ev = []

    def T(k, s=0):
        return drop + k * BEAT + s * S16

    def jit(v, amt=0.05):
        return float(np.clip(v * (1.0 + rng.uniform(-amt, amt)), 0.05, 1.2))

    def chord(k):
        return story_chord_of_beat(k, g)

    # ---------------- arp: intro, groove A, breakdown build bar, drop 2 (+ lift layer in "full" bars)
    for k in range(k0, hk):
        name, kk = _story_section(k, g)
        if name == "breakdown" and k < kbuild:
            continue                                                 # dark bars: no arp
        ladder = [nm(n) for n in CHORDS[chord(k)]["arp"]]
        for s in range(4):
            tn = T(k, s)
            if tn < -S16 or tn >= t_hit - GAP:
                continue
            lift = False
            if k >= hk - 2:                                          # turnaround run into the hit
                idx = ARP_TURN[(k - (hk - 2)) * 4 + s]
                vel = 0.70 + 0.30 * ((k - (hk - 2)) * 4 + s) / 7.0
            else:
                step = (k % 4) * 4 + s
                if name in STORY_SECTIONS:
                    cfg = STORY_SECTIONS[name]
                    pat = cfg["arp_patterns"][(kk // 4) % 2]
                    lift = kk // 4 >= cfg["full_from_bar"]
                else:                                                # intro and build bar: the intro arp
                    pat = ARP_A
                idx = pat[step]
                vel = 1.0 if step in ARP_ACCENTS else 0.62
            ev.append(dict(inst="arp", t=tn + _swing(s), tn=tn, dur=0.30, midi=ladder[idx], vel=jit(vel)))
            if lift:
                ev.append(dict(inst="arp", t=tn + _swing(s), tn=tn, dur=0.22, midi=ladder[idx] + 12,
                               vel=jit(vel) * 0.38, layer="lift"))

    # ---------------- chord stabs on the drops, the final Fm(add9) stab
    for t in (drop, t_d2):
        ev.append(dict(inst="stab", t=t, tn=t, dur=0.9, chord="Fm", midis=[nm(n) for n in CHORDS["Fm"]["arp"][:4]],
                       vel=1.0))
    ev.append(dict(inst="stab", t=t_hit, tn=t_hit, dur=2.6, chord=STORY_FINAL_CHORD,
                   midis=[nm(n) for n in CHORDS[STORY_FINAL_CHORD]["arp"][:5]], vel=1.0, final=True))

    # ---------------- pad (one event per chord segment)
    for (t0, t1, ch) in story_segments(g):
        ev.append(dict(inst="pad", t=t0, tn=t0, dur=t1 - t0, chord=ch, midis=[nm(n) for n in CHORDS[ch]["pad"]],
                       final=(ch == STORY_FINAL_CHORD)))

    # ---------------- drop-2 layers: offbeat chord stabs + lead counter-melody
    for name, k_a, k_b in (("groove_a", 0, kb), ("drop2", kd, hk)):
        cfg = STORY_SECTIONS[name]
        if cfg["chord_stabs"]:
            for k in range(k_a, min(k_b, hk - 1)):
                tn = T(k, 2)
                ch = chord(k)
                ev.append(dict(inst="chord", t=tn, tn=tn, dur=0.20, chord=ch, midis=[nm(n) for n in CHORDS[ch]["pad"]],
                               vel=jit(1.0 if k % 4 == 3 else 0.85)))
        if cfg["lead"]:
            for k in range(k_a, k_b, 4):
                for (pos, n16, note) in STORY_LEAD[chord(k)]:
                    tn = T(k, pos)
                    if tn >= t_hit - GAP:
                        continue
                    dur = min(n16 * S16 * 0.92, t_hit - GAP - tn)
                    ev.append(dict(inst="lead", t=tn + _swing(pos), tn=tn, dur=dur, midi=nm(note), pos=pos,
                                   vel=jit(1.0 if pos == 0 else 0.85, 0.03)))

    # ---------------- bass: groove patterns, breakdown sub pulse, turnaround walk, final F1
    for k in range(0, hk - 2):
        name, kk = _story_section(k, g)
        root, fifth = (nm(n) for n in CHORDS[chord(k)]["bass"])
        if name == "breakdown":
            if k < kbuild:
                pat = BD["pulse"]
            else:                                                    # build: 8ths, then 16ths, crescendo
                j = k - kbuild
                p = j / 4.0
                pat = (((0, 0.62 + 0.3 * p, 0.24), (2, 0.5 + 0.3 * p, 0.24)) if j < 2 else
                       tuple((s, 0.55 + 0.4 * (p + s / 16.0), 0.115) for s in range(4)))
            for (pos, vel, dur) in pat:
                tn = T(k, pos)
                if tn >= t_d2 - GAP - 0.02:
                    continue
                ev.append(dict(inst="bass", t=tn, tn=tn, dur=min(dur, t_d2 - GAP - tn), midi=root, role="root",
                               vel=jit(vel, 0.03), shape="pulse"))
            continue
        for (pos, role, vel) in (BASS_LAST_BEAT if k % 4 == 3 else BASS_BEAT):
            tn = T(k, pos)
            midi = {"root": root, "oct": root + 12, "fifth": fifth}[role]
            ev.append(dict(inst="bass", t=tn + _swing(pos), tn=tn, dur=0.105, midi=midi, role=role,
                           vel=jit(vel, 0.03) * STORY_SECTIONS[name]["bass_vel"]))
    for j, beat in enumerate(BASS_TURN):
        k = hk - 2 + j
        for (pos, name_, role, vel) in beat:
            tn = T(k, pos)
            ev.append(dict(inst="bass", t=tn + _swing(pos), tn=tn, dur=0.105, midi=nm(name_), role=role, vel=vel))
    ev.append(dict(inst="bass", t=t_hit, tn=t_hit, dur=g["duration"] - t_hit, midi=nm("F1"), role="root",
                   vel=1.0, final=True))

    # ---------------- drums
    groove_beats = list(range(0, kb)) + list(range(kd, hk - 1))      # no kick on the beat before the hit
    for k in groove_beats:
        ev.append(dict(inst="kick", t=T(k), tn=T(k), vel=jit(1.0, 0.02) * STORY_SECTIONS[_story_section(k, g)[0]]["kick_vel"]))
    ev.append(dict(inst="kick", t=t_bd, tn=t_bd, vel=BD["entry"]["kick"]))          # breakdown downbeat accent
    ev.append(dict(inst="kick", t=t_hit, tn=t_hit, vel=1.0, big=True))
    for k in groove_beats:
        if k % 4 in (1, 3):
            ev.append(dict(inst="clap", t=T(k), tn=T(k), vel=jit(1.0, 0.04)))
            if STORY_SECTIONS[_story_section(k, g)[0]]["backbeat_snare"]:
                ev.append(dict(inst="snare", t=T(k), tn=T(k), vel=jit(STORY_LV["backbeat_snare"], 0.04),
                               pitch=snare_ratio(3), hp=350.0))
    # hats: intro / build crescendo (closed 16ths), groove patterns, none in the dark bars or the last beat
    for k in range(g["k_first"] - 1, hk - 1):
        name, kk = _story_section(k, g)
        for s in range(4):
            tn = T(k, s)
            if tn < 0.0:
                continue
            if name == "intro" or (name == "breakdown" and k >= kbuild):
                t_a, t_b = (0.0, drop) if name == "intro" else (t_build, t_d2)
                if tn >= t_b - GAP:
                    continue
                prog = (tn - t_a) / max(t_b - t_a, 1e-3)
                ev.append(dict(inst="hat", t=tn + _swing(s), tn=tn,
                               vel=jit((0.18 + 0.45 * prog) * (1.0 if s % 2 == 0 else 0.7))))
            elif name in STORY_SECTIONS:
                cfg = STORY_SECTIONS[name]
                full = kk // 4 >= cfg["full_from_bar"]
                h = HAT_GROOVE[s]
                if h == "open":
                    ev.append(dict(inst="ohat", t=tn, tn=tn, vel=jit(cfg["ohat_vel"][1 if full else 0])))
                else:
                    ev.append(dict(inst="hat", t=tn + _swing(s), tn=tn, vel=jit(h * cfg["hat_gain"])))
    # snare build rolls into drop 1 (the intro) and into drop 2 (the build bar)
    for t_target, t_min in ((drop, 0.0), (t_d2, t_build)):
        for tn, p in _snare_roll(t_target, t_min):
            step = 4 if p >= 0.875 else int(p * 4.0 + 1e-9)
            ev.append(dict(inst="snare", t=tn, tn=tn, vel=(0.30 + 0.70 * p ** 1.3) * (1.0 - 0.2 * p),
                           pitch=snare_ratio(step), hp=250 + 900 * p))
    # fills every `fill_every` bars inside groove A / drop 2 (never on a section's last beat)
    for name, k_a, k_b in (("groove_a", 0, kb), ("drop2", kd, hk)):
        fe = 4 * STORY_SECTIONS[name]["fill_every"]
        for k in range(k_a + fe - 1, min(k_b - 1, hk - 2), fe):
            for s, v in ((1, 0.45), (2, 0.6), (3, 0.8)):
                ev.append(dict(inst="snare", t=T(k, s) + _swing(s), tn=T(k, s), vel=v, pitch=snare_ratio(s), hp=300))
            ev.append(dict(inst="crash", t=T(k + 1), tn=T(k + 1), vel=0.55))
    if STORY_SECTIONS["groove_a"]["exit_fill"]:                      # falling fill into the breakdown
        for (s, step, v) in EXIT_FILL:
            ev.append(dict(inst="snare", t=T(kb - 1, s) + _swing(s), tn=T(kb - 1, s), vel=v, pitch=snare_ratio(step),
                           hp=300))
    # turnaround fill into the hit (render() logic): 16ths on the 2nd half of beat hk-2, 32nds on hk-1
    for s in (2, 3):
        tn = T(hk - 2, s)
        ev.append(dict(inst="snare", t=tn + _swing(s), tn=tn, vel=0.45 + 0.1 * s, pitch=snare_ratio(1), hp=300))
    for i in range(8):
        tn = drop + (hk - 1) * BEAT + i * S16 / 2
        if tn >= t_hit - GAP:
            continue
        p = i / 7.0
        ev.append(dict(inst="snare", t=tn, tn=tn, vel=0.55 + 0.45 * p, pitch=snare_ratio(1 + min(3, int(p * 4.0))),
                       hp=400 + 800 * p))
    for t, v in ((drop, 1.0), (t_bd, BD["entry"]["crash"]), (t_d2, 1.0), (t_hit, 1.0)):
        ev.append(dict(inst="crash", t=t, tn=t, vel=v))
    # breakdown clock: tick (F6) on the beat, tock (C6) on the off-8th, dark bars only
    for k in range(kb, kbuild):
        for (pos, note, vel, pan) in BD["clock"]:
            tn = T(k, pos)
            ev.append(dict(inst="clock", t=tn, tn=tn, midi=nm(note), vel=jit(vel, 0.04), pan=pan))

    # ---------------- fx (musical transitions)
    r0 = max(0.0, drop - 2.0 * BAR)
    if drop - r0 > 0.25:
        ev.append(dict(inst="riser", t=r0, tn=r0, dur=drop - r0 - 0.02, midi_start=nm("F3"), midi_end=nm("F5")))
    ev.append(dict(inst="riser", t=t_build, tn=t_build, dur=t_d2 - t_build - 0.02, midi_start=nm("F3"),
                   midi_end=nm("F5")))
    ev.append(dict(inst="revcym", t=t_d2 - BEAT * 2, tn=t_d2 - BEAT * 2, dur=BEAT * 2))
    ev.append(dict(inst="revcym", t=t_hit - BEAT * 2, tn=t_hit - BEAT * 2, dur=BEAT * 2))
    for t, v, d, dl in ((drop, 0.8, 1.6, True), (t_bd, BD["entry"]["boom"], BD["entry"]["boom_dur"], BD["entry"]["downlifter"]),
                        (t_d2, 0.9, 1.6, True)):
        ev.append(dict(inst="boom", t=t, tn=t, dur=d, vel=v))
        if dl:
            ev.append(dict(inst="downlifter", t=t, tn=t, dur=BAR))
    ev.append(dict(inst="boom", t=t_hit, tn=t_hit, dur=g["duration"] - t_hit, vel=1.0, final=True))
    ev.append(dict(inst="impact", t=t_hit, tn=t_hit, dur=g["duration"] - t_hit))
    ev.sort(key=lambda e: e["t"])
    return ev


def check_story(events: list, g: dict, segs: list) -> list:
    """Story-specific rules on top of check_arrangement(): anchors are bar lines, kicks land on the exact
    samples of drop 1 / breakdown / drop 2 / end hit, no kick inside the breakdown after its downbeat, lead
    notes on beats 1 and 3 are chord tones, the end chord is Fm(add9)."""
    issues = []
    drop = g["drop"]

    def chord_at(t):
        for (t0, t1, ch) in segs:
            if t0 - 1e-6 <= t < t1 - 1e-6:
                return ch
        return segs[-1][2]
    kick_samples = {round(e["t"] * SR) for e in events if e["inst"] == "kick"}
    for name, t in (("drop1", drop), ("breakdown", g["t_bd"]), ("drop2", g["t_d2"]), ("end_hit", g["t_hit"])):
        if abs(((t - drop) / BAR) - round((t - drop) / BAR)) > 1e-9:
            issues.append(f"{name} {t} is not a bar line")
        if round(t * SR) not in kick_samples:
            issues.append(f"no kick on the {name} sample")
    for e in events:
        if e["inst"] == "kick" and g["t_bd"] + 1e-9 < e["t"] < g["t_d2"] - 1e-9:
            issues.append(f"kick inside the breakdown @ {e['t']:.3f}s")
        if e["inst"] == "lead" and e["pos"] % 8 == 0:
            ch = chord_at(e["tn"])
            if e["midi"] % 12 not in {PC[n] for n in CHORDS[ch]["tones"]}:
                issues.append(f"lead @ {e['t']:.3f}s: strong-beat MIDI {e['midi']} not in chord {ch}")
    if segs[-1][2] != STORY_FINAL_CHORD:
        issues.append("story does not end on the final chord")
    return issues


def _story_auto(g: dict) -> dict:
    """Automation for _render_events(): pad / arp cutoffs, arp gain, ghost pulse, breaths, pad gain."""
    drop, t_bd, t_build, t_d2, t_hit, dur = g["drop"], g["t_bd"], g["t_build"], g["t_d2"], g["t_hit"], g["duration"]
    A_, D2, BD = STORY_SECTIONS["groove_a"], STORY_SECTIONS["drop2"], STORY_BREAKDOWN
    t0 = drop + g["k0"] * BEAT                                       # intro bar line (<= 0 s)

    def lfo(t):
        return 2000.0 * 2 ** (0.35 * np.sin(2 * np.pi * (t - drop) / (4 * BAR) - np.pi / 2))

    def bright(t, start, cfg, key):
        return np.where(t >= start + cfg["full_from_bar"] * BAR, cfg[key][1], cfg[key][0])

    def ramp(t, a, b, e=1.0):
        return np.clip((t - a) / max(b - a, 1e-3), 0.0, 1.0) ** e

    def turn(t):                                                     # last half bar before the hit
        return np.clip((t - (t_hit - BAR / 2)) / (BAR / 2), 0, 1)

    def pad_cutoff(t):
        intro = 520.0 * (2300.0 / 520.0) ** ramp(t, t0, drop, 1.6)
        ga = lfo(t) * bright(t, drop, A_, "pad_bright")
        fc_bd = float(lfo(np.array([t_bd]))[0] * bright(np.array([t_bd]), drop, A_, "pad_bright")[0])
        dark_fc = BD["dark_pad_fc"] * 2 ** (BD["dark_wobble_oct"] * np.sin(2 * np.pi * (t - t_bd) / BAR))
        dark = dark_fc * (fc_bd / dark_fc) ** (1.0 - ramp(t, t_bd, t_bd + BD["dark_glide"]))
        lo, hi = BD["build_pad_fc"]
        build = lo * (hi / lo) ** ramp(t, t_build, t_d2, 1.6)
        d2 = lfo(t) * bright(t, t_d2, D2, "pad_bright") * (1 + 0.6 * turn(t))
        f_a, f_b, ex = STORY_TAIL_PAD_FC
        tail = f_a * (f_b / f_a) ** ramp(t, t_hit, dur, ex)
        return np.select([t < drop, t < t_bd, t < t_build, t < t_d2, t < t_hit], [intro, ga, dark, build, d2], tail)

    def arp_cutoff(t):
        intro = 650.0 * (2400.0 / 650.0) ** ramp(t, t0, drop, 1.4)

        def groove(start, cfg):
            phrase = ((t - start) % (2 * BAR)) / (2 * BAR)            # 2-bar filter ramps
            return 1500.0 * 2 ** (0.75 * phrase) * bright(t, start, cfg, "arp_bright")
        lo, hi = BD["build_arp_fc"]
        build = lo * (hi / lo) ** ramp(t, t_build, t_d2, 1.4)
        d2 = groove(t_d2, D2) * (1 + 0.8 * turn(t))
        return np.select([t < drop, t < t_bd, t < t_d2, t < t_hit], [intro, groove(drop, A_), build, d2], 2600.0)

    a = _db2(LV["arp_intro"])

    def arp_gain(tl):                                                # build arps sit 3 dB under the drops
        w = (1.0 - _step_gain(tl, drop - GAP / 2, 0.0, 1.0)) + \
            (_step_gain(tl, t_build, 0.0, 1.0) - _step_gain(tl, t_d2 - GAP / 2, 0.0, 1.0))
        return 1.0 + (a - 1.0) * w

    def pad_gain(tl):            # dB: breakdown (low-passed) +, drop 2 +, held end chord + (swells in after the hit)
        bd = _step_gain(tl, t_bd + 0.1, 0.0, 1.0, ramp=0.2) - _step_gain(tl, t_d2 - GAP / 2, 0.0, 1.0)
        d2 = _step_gain(tl, t_d2 - GAP / 2, 0.0, 1.0) - _step_gain(tl, t_hit - GAP / 2, 0.0, 1.0)
        end = _step_gain(tl, t_hit + 0.35, 0.0, 1.0, ramp=0.6)
        return _db2(BD["pad_dark_db"] * bd + D2["pad_db"] * d2 + STORY_LV["pad_end_db"] * end)

    ghost = [drop + k * BEAT for k in range(g["k0"], 0)] + [drop + k * BEAT for k in range(g["kbuild"], g["kd"])]
    return dict(gaps=[drop, t_d2, t_hit], ghost=ghost, pad_cutoff=pad_cutoff, arp_cutoff=arp_cutoff,
                arp_gain=arp_gain, pad_gain=pad_gain, tail_fade=STORY_TAIL_FADE, smooth_dynamics=True,
                final_pad_amp=lambda tt, q: (1.0 + 0.35 * np.exp(-tt / 0.25)) * (1.0 - STORY_END_HOLD[0] * q ** STORY_END_HOLD[1]))


def render_story(duration=30.0, drop1=2.0, breakdown=14.0, drop2=18.0, end_hit=26.0, seed=1, _parts=None):
    """The 30 s Story arrangement: intro -> groove A -> breakdown -> drop 2 -> end hit + held Fm(add9) tail.
    Same return format as render(): {'drums','bass','music','fx'} float32 (N, 2) (unmastered, scaled together so
    the hottest true peak is -7 dBTP) + 'meta' (beat_times, downbeat_times, sections, section_boundaries,
    drop_time (= drop1), end_hit_time_snapped (= end_hit), fx_transitions, chord_timeline, ...).
    The anchors must be bar lines of the 2 s grid anchored on drop1 (off-grid values snap with a warning)."""
    rng = np.random.default_rng(seed)
    g = story_grid(duration, drop1, breakdown, drop2, end_hit)
    N = int(round(g["duration"] * SR))
    events = arrange_story(g, np.random.default_rng(seed + 1000))
    segs = story_segments(g)
    issues = check_arrangement(events, g, segs) + check_story(events, g, segs)
    if issues:
        raise AssertionError("musical check failed:\n  " + "\n  ".join(issues))
    stems, scale = _render_events(events, N, seed, rng, _story_auto(g), _parts)

    r9 = lambda v: round(float(v), 9)
    d1, bd, d2, hit, dur = (r9(g[k]) for k in ("drop", "t_bd", "t_d2", "t_hit", "duration"))
    tb = r9(g["t_build"])
    stems["meta"] = dict(
        bpm=BPM, key=KEY, beat_times=[r9(b) for b in g["beats"]],
        downbeat_times=[r9(g["drop"] + k * BEAT) for k in range(g["k_first"], g["k_last"] + 1) if k % 4 == 0],
        drop_time=d1, end_hit_time_snapped=hit,                      # render()-compatible keys
        drop1=d1, breakdown=bd, drop2=d2, end_hit=hit, duration=dur,
        section_times=dict(drop1=d1, breakdown=bd, drop2=d2, end_hit=hit, duration=dur),   # = music_cinematic
        section_boundaries=[0.0, d1, bd, d2, hit, dur],
        sections=[dict(name="intro", t0=0.0, t1=d1),
                  dict(name="grooveA", t0=d1, t1=bd, bars=g["bars"]["groove_a"],
                       full_from=r9(g["drop"] + STORY_SECTIONS["groove_a"]["full_from_bar"] * BAR)),
                  dict(name="breakdown", t0=bd, t1=d2, bars=g["bars"]["breakdown"], dark=[bd, tb], build=[tb, d2]),
                  dict(name="drop2", t0=d2, t1=hit, bars=g["bars"]["drop2"], turnaround=[r9(g["t_hit"] - 2 * BEAT), hit]),
                  dict(name="end", t0=hit, t1=dur)],
        requested_anchors=g["requested"],
        # fx-stem transition spans for sfx.fx_duck_gain(); kinds as in music_cinematic (+ downlifter)
        fx_transitions=sorted(
            [dict(kind={"revcym": "reverse_cymbal"}.get(e["inst"], e["inst"]), t0=r9(e["t"]), t1=r9(e["t"] + e["dur"]))
             for e in events if e["inst"] in ("riser", "revcym", "downlifter")]
            + [dict(kind="sub_boom", t0=r9(e["t"]), t1=r9(e["t"] + min(4 * (0.9 if e.get("final") else 0.45), e["dur"])))
               for e in events if e["inst"] == "boom"], key=lambda d: d["t0"]),
        scale=list(SCALE), progression=list(PROGRESSION), breakdown_chords=dict(dark=list(BREAKDOWN_DARK),
                                                                                build=list(BREAKDOWN_BUILD)),
        turnaround_chord=TURNAROUND_CHORD, final_chord=STORY_FINAL_CHORD,
        chord_timeline=[(max(0.0, float(a)), float(b), c) for (a, b, c) in segs if b > 0],
        stem_scale_db=float(_db(scale)), sample_rate=SR, swing_s=SWING, n_events=len(events),
        musical_check=dict(violations=0, pitched_events=sum(1 for e in events if 'midi' in e or 'midis' in e),
                           pitched_notes=sum(len(e.get('midis', [])) or ('midi' in e) for e in events),
                           lead_notes=sum(1 for e in events if e["inst"] == "lead"),
                           tuned_snare_hits=sum(1 for e in events if e["inst"] == "snare")),
    )
    return stems


# ---------------------------------------------------------------------------------------------- story analysis
def _story_spans(meta: dict) -> list:
    s = {d["name"]: d for d in meta["sections"]}
    return [("intro", 0.0, meta["drop1"]), ("groove_a", meta["drop1"], meta["breakdown"]),
            ("breakdown", meta["breakdown"], meta["drop2"]),
            ("breakdown_dark", *s["breakdown"]["dark"]), ("breakdown_build", *s["breakdown"]["build"]),
            ("drop2", meta["drop2"], meta["end_hit"]), ("end", meta["end_hit"], meta["duration"])]


def _kw(x: np.ndarray) -> np.ndarray:
    """BS.1770 K-weighting (48 kHz)."""
    b1, a1 = [1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585]
    b2, a2 = [1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621]
    return sps.lfilter(b2, a2, sps.lfilter(b1, a1, x, axis=0), axis=0)


def _max_momentary(x: np.ndarray) -> float:
    y = _kw(x)
    p = np.concatenate([[0.0], np.cumsum(np.sum(y ** 2, axis=1))])
    w, h = int(0.4 * SR), int(0.05 * SR)
    if len(y) < w:
        return -120.0
    idx = np.arange(0, len(y) - w + 1, h)
    return float(-0.691 + 10 * np.log10(np.max((p[idx + w] - p[idx]) / w) + 1e-20))


def story_section_report(x: np.ndarray, meta: dict) -> dict:
    """Per section: gated LUFS full band and through the phone model (4th-order HP 150 Hz), max momentary
    LUFS, spectral centroid and % of energy above 2 kHz (darkness).  Plus the contrasts in LU."""
    x = np.asarray(x, np.float64)
    hp = _filt(x, "highpass", 150.0, 4)
    rows = {}
    for name, a, b in _story_spans(meta):
        i0, i1 = int(round(a * SR)), int(round(b * SR))
        seg = x[i0:i1]
        f, P = sps.welch(seg.mean(axis=1), SR, nperseg=8192)
        rows[name] = dict(t0=round(a, 3), t1=round(b, 3), LUFS=round(lufs(seg), 2), LUFS_HP150=round(lufs(hp[i0:i1]), 2),
                          maxM=round(_max_momentary(seg), 2), maxM_HP150=round(_max_momentary(hp[i0:i1]), 2),
                          centroid_Hz=round(float(np.sum(f * P) / (np.sum(P) + 1e-30)), 0),
                          above2k_pct=round(float(100 * np.sum(P[f >= 2000]) / (np.sum(P) + 1e-30)), 2))
    c = {}
    for key in ("LUFS", "LUFS_HP150"):
        c[key] = dict(groove_a_minus_intro=round(rows["groove_a"][key] - rows["intro"][key], 2),
                      drop2_minus_groove_a=round(rows["drop2"][key] - rows["groove_a"][key], 2),
                      breakdown_minus_groove_a=round(rows["breakdown"][key] - rows["groove_a"][key], 2),
                      breakdown_dark_minus_groove_a=round(rows["breakdown_dark"][key] - rows["groove_a"][key], 2),
                      build_minus_dark=round(rows["breakdown_build"][key] - rows["breakdown_dark"][key], 2))
    return dict(sections=rows, contrast_LU=c)


def _onset(x: np.ndarray, t: float, lp=None, frac=0.3):
    """First sample within [t-20 ms, t+60 ms] above `frac` of the window peak: (offset ms, breath dB), where
    breath = peak of the 30 ms before t re the window peak."""
    a = np.asarray(x, np.float64)
    if lp:
        a = sps.sosfiltfilt(_sos("lowpass", lp, 4), a, axis=0)        # zero-phase: no filter delay
    e = np.abs(a).sum(axis=1)
    i0, i1 = max(0, int(round((t - 0.02) * SR))), min(len(e), int(round((t + 0.06) * SR)))
    seg = e[i0:i1]
    k = int(np.argmax(seg > frac * seg.max()))
    pre = e[max(0, int(round((t - 0.03) * SR))):int(round((t - 0.001) * SR))]
    return round(((i0 + k) / SR - t) * 1000.0, 2), round(_db(pre.max() / seg.max()), 1)


def _discontinuities(x: np.ndarray, k=12.0, floor=1e-3) -> list:
    """Times where |2nd difference| > k x its local (10 ms) RMS and > floor (the audit's click detector)."""
    a = np.asarray(x, np.float64).sum(axis=1)
    d2 = np.abs(np.diff(a, 2))
    loc = np.sqrt(np.maximum(uniform_filter1d(d2 ** 2, size=481), 0.0)) + 1e-12
    idx = np.nonzero((d2 > k * loc) & (d2 > floor))[0]
    out = []
    for i in idx:
        if not out or i - out[-1][1] > 240:
            out.append([i, i])
        else:
            out[-1][1] = i
    return [round(c[0] / SR, 4) for c in out]


def _out_of_key_pct(x: np.ndarray, lo: float, hi: float) -> float:
    """% of mono-sum spectral energy in lo..hi Hz whose nearest pitch class is outside F minor.  Saw
    harmonics of in-key notes land partly outside the key (5th harmonic of F = A, of C = E ...), so this is a
    cross-check against a reference, not zero."""
    s = np.asarray(x, np.float64).mean(axis=1)
    nfft = 1 << int(math.ceil(math.log2(len(s))))
    S = np.abs(np.fft.rfft(s * np.hanning(len(s)), nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / SR)
    m = (f >= lo) & (f <= hi)
    pc = np.round(69 + 12 * np.log2(f[m] / 440.0)).astype(int) % 12
    c = np.bincount(pc, weights=S[m], minlength=12)
    return round(float(100 * sum(c[i] for i in range(12) if i not in SCALE_PCS) / (c.sum() + 1e-30)), 2)


def render_story_demo(out_dir=DEMO_DIR, seed=1, **anchors):
    """Render render_story(**STORY_DEFAULTS | anchors), write demo/music_techhouse_story30_demo.wav (stems
    summed, preview master -14 LUFS / <= -1 dBTP), _spectrogram.png and _report.json (section loudness full
    band + phone model HP150, anchors, transients, tail, clicks, stems, checks)."""
    import time
    os.makedirs(out_dir, exist_ok=True)
    kw = dict(STORY_DEFAULTS, **anchors)
    t0 = time.time()
    st = render_story(seed=seed, **kw)
    render_s = time.time() - t0
    meta = st["meta"]
    mix = sum(st[k].astype(np.float64) for k in STEMS)
    y = master_preview(mix)
    y = (y - np.mean(y, axis=0) * _fades(len(y), 0.005, STORY_TAIL_FADE)[:, None])
    base = os.path.join(out_dir, "music_techhouse_story30_demo")
    sf.write(base + ".wav", y.astype(np.float32), SR, subtype="FLOAT")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", base + ".wav", "-lavfi",
                    "showspectrumpic=s=1800x600:legend=1", base + "_spectrogram.png"], check=True)
    eb = _ebur128(base + ".wav")
    sec_stems = story_section_report(mix, meta)
    sec_demo = story_section_report(y, meta)
    srep = stem_report(st)
    mono = stereo_report(st["music"])
    first100 = _db(np.sqrt(np.mean(mix[:int(0.1 * SR)] ** 2)))
    first100_demo = _db(np.sqrt(np.mean(y[:int(0.1 * SR)] ** 2)))

    # anchors: kick (LP 200) and broadband onsets; drums stem
    anchors_rep = {}
    for name in ("drop1", "breakdown", "drop2", "end_hit"):
        t = meta[name]
        anchors_rep[name] = dict(t=t, is_downbeat=any(abs(b - t) < 1e-9 for b in meta["downbeat_times"]),
                                 broadband_onset_ms_breath_dB=_onset(st["drums"], t),
                                 mix_onset_ms=_onset(mix, t)[0])
    # tail: RMS (dBFS) of the master in windows before the end, re the 400 ms after the hit
    hit, dur = meta["end_hit"], meta["duration"]

    def rms(a, t0_, t1_):
        s = a[int(round(t0_ * SR)):int(round(t1_ * SR))]
        return _db(np.sqrt(np.mean(s ** 2)) + 1e-15)
    tail = dict(hit_0_400ms_dBFS=round(rms(y, hit, hit + 0.4), 1),
                **{f"{a:.2f}-{b:.2f}s_dBFS": round(rms(y, a, b), 1)
                   for a, b in ((hit + 1, hit + 2), (dur - 2, dur - 1), (dur - 1, dur - 0.5), (dur - 0.5, dur - 0.3),
                                (dur - 0.3, dur - 0.15), (dur - 0.15, dur - 0.05), (dur - 0.05, dur - 0.01),
                                (dur - 0.01, dur))},
                last_sample_master=float(np.max(np.abs(y[-1]))),
                last_sample_stems=max(float(np.max(np.abs(st[k][-1]))) for k in STEMS))
    # clicks: the audit's 2nd-difference detector on the tonal stems.  A detection 0..8 ms after a note / hit
    # onset rendered into that stem is "explained" (any attack after a quiet moment trips it); anything else
    # is a candidate click.  Boundaries = section downbeats + build bar; +-50 ms covers the 50 ms breaths.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        events = arrange_story(story_grid(**kw), np.random.default_rng(seed + 1000))
    stem_of = dict(bass=("bass",), music=("arp", "stab", "pad", "lead", "chord"),
                   fx=("riser", "downlifter", "revcym", "boom", "impact"))
    bnd = meta["section_boundaries"][1:-1] + [meta["sections"][2]["build"][0]]
    disc, unexpl, near = {}, {}, {}
    for k, insts in stem_of.items():
        on = np.array(sorted(e["t"] for e in events if e["inst"] in insts))
        ds = _discontinuities(st[k])
        disc[k] = len(ds)
        unexpl[k] = [t for t in ds if not np.any((t - on >= -0.001) & (t - on <= 0.008))]
        near[k] = [t for t in unexpl[k] if any(abs(t - b) < 0.05 for b in bnd)]
    hf = {k: len([s for s in _hf_spikes(st[k], fc) if s[1] > -80]) for k, fc in (("bass", 3000.0), ("music", 12000.0))}
    # notes: symbolic check + an audio cross-check per section (music 150-2000 Hz).  Bass fundamentals
    # (40-100 Hz) only where notes are long enough to resolve a semitone there (dark-bar pulses, final F1):
    # 105 ms groove notes smear ~10 Hz, wider than a semitone at 40-100 Hz.
    ook = {}
    for name, a, b in _story_spans(meta):
        if name == "breakdown":
            continue
        i0, i1 = int(a * SR), int(b * SR)
        ook[name] = dict(music=_out_of_key_pct(st["music"][i0:i1], 150.0, 2000.0))
        if name in ("breakdown_dark", "end"):
            ook[name]["bass"] = _out_of_key_pct(st["bass"][i0:i1], 40.0, 100.0)
    # loudness curve: 1 s windows, 0.5 s hop, ungated K-weighted (full band / phone model)
    curve = []
    hp150 = _filt(mix, "highpass", 150.0, 4)
    for t in np.arange(0.0, meta["duration"] - 0.999, 0.5):
        i0, i1 = int(round(t * SR)), int(round((t + 1.0) * SR))
        kw_ = [round(-0.691 + 10 * np.log10(np.sum(np.mean(_kw(z[i0:i1]) ** 2, axis=0)) + 1e-20), 1) for z in (mix, hp150)]
        curve.append([round(float(t), 2)] + kw_)
    S, C = sec_stems["sections"], sec_stems["contrast_LU"]
    checks = dict(
        stems_true_peak_le_minus6=all(v["true_peak_dBTP"] <= -6.0 for v in srep.values()),
        stems_sample_peak_le_minus6=all(v["peak_dBFS"] <= -6.0 for v in srep.values()),
        stems_dc_lt_1e4=all(max(abs(d) for d in v["dc"]) < 1e-4 for v in srep.values()),
        stems_finite=all(v["finite"] for v in srep.values()),
        stems_all_have_signal=all(v["lufs"] > -50.0 for v in srep.values()),
        stems_start_end_zero=all(max(v["first_last_sample"]) < 1e-6 for v in srep.values()),
        notes_in_key=meta["musical_check"]["violations"] == 0,
        anchors_are_downbeats=all(v["is_downbeat"] for v in anchors_rep.values()),
        hit_onsets_on_anchors_within_2ms=all(abs(anchors_rep[n]["broadband_onset_ms_breath_dB"][0]) <= 2.0
                                             for n in ("drop1", "breakdown", "drop2", "end_hit")),
        end_hit_transient=abs(anchors_rep["end_hit"]["broadband_onset_ms_breath_dB"][0]) <= 2.0
        and anchors_rep["end_hit"]["broadband_onset_ms_breath_dB"][1] <= -20.0,
        order_full_intro_lt_grooveA_lt_drop2=S["intro"]["LUFS"] < S["groove_a"]["LUFS"] < S["drop2"]["LUFS"],
        order_HP150_intro_lt_grooveA_lt_drop2=S["intro"]["LUFS_HP150"] < S["groove_a"]["LUFS_HP150"] < S["drop2"]["LUFS_HP150"],
        drop2_ge_grooveA_plus1_full=C["LUFS"]["drop2_minus_groove_a"] >= 1.0,
        drop2_ge_grooveA_plus1_HP150=C["LUFS_HP150"]["drop2_minus_groove_a"] >= 1.0,
        breakdown_le_grooveA_minus3_both=C["LUFS"]["breakdown_minus_groove_a"] <= -3.0
        and C["LUFS_HP150"]["breakdown_minus_groove_a"] <= -3.0,
        breakdown_darker=S["breakdown_dark"]["centroid_Hz"] < 0.7 * S["groove_a"]["centroid_Hz"]
        and S["breakdown_dark"]["above2k_pct"] < 0.5 * S["groove_a"]["above2k_pct"],
        audible_from_t0=first100 > -35.0,
        tail_fades_to_zero=tail["last_sample_master"] < 1e-6 and tail["last_sample_stems"] < 1e-6
        and tail[f"{dur - 0.01:.2f}-{dur:.2f}s_dBFS"] < -60.0,
        no_unexplained_clicks_at_boundaries=all(len(v) == 0 for v in near.values()),
        demo_I_minus14_pm05=abs(eb["I_LUFS"] + 14.0) <= 0.5,
        demo_true_peak_le_minus1=eb["TP_dBTP"] <= -1.0,
        demo_LRA_le_8=eb["LRA_LU"] <= 8.0,
        music_mono_corr_ge_045=mono["corr"] >= 0.45,
        music_no_window_corr_lt_02=mono["worst_window"]["corr"] >= 0.2)
    rep = dict(checks=checks, all_checks_pass=all(checks.values()),
               params=dict(kw, seed=seed), render_s=round(render_s, 1),
               section_loudness_stems_sum=sec_stems, section_loudness_demo_master=sec_demo,
               anchors=anchors_rep, tail=tail,
               clicks=dict(boundaries=bnd, discontinuities_total=disc, unexplained=unexpl,
                           unexplained_near_boundaries=near, hf_spikes_gt_minus80dBFS=hf),
               loudness_curve_1s=dict(columns=["t0", "LUFS_full", "LUFS_HP150"], rows=curve),
               out_of_key_energy_pct=ook,
               meta={k: v for k, v in meta.items() if k != "beat_times"}, n_beats=len(meta["beat_times"]),
               stems=srep, music_stereo=mono,
               unmastered_sum=dict(lufs=round(lufs(mix), 2), true_peak_dBTP=round(_db(_true_peak(mix)), 2),
                                   first100ms_rms_dBFS=round(first100, 1)),
               demo=dict(ebur128=eb, python_lufs=round(lufs(y), 2), true_peak_dBTP=round(_db(_true_peak(y)), 2),
                         first100ms_rms_dBFS=round(first100_demo, 1),
                         dc=[float(f"{v:.2e}") for v in np.mean(y, axis=0)]))
    with open(base + "_report.json", "w") as fh:
        json.dump(rep, fh, indent=2)
    return rep


def _hf_spikes(x, fc, win=0.002, thresh_db=20.0):
    """Broadband discontinuities: energy above fc in 2 ms windows > thresh_db over the 200 ms median.
    Returns [(time, level dBFS)]."""
    from scipy.ndimage import median_filter
    y = _filt(np.asarray(x, np.float64), "highpass", fc, 4)
    p = np.sum(y ** 2, axis=1)
    w = int(win * SR)
    nb = len(p) // w
    e = p[:nb * w].reshape(nb, w).mean(axis=1) + 1e-24
    ex = 10 * np.log10(e / (median_filter(e, size=101, mode="nearest") + 1e-24))
    return [(i * w / SR, float(10 * np.log10(e[i]))) for i in np.nonzero(ex > thresh_db)[0]]


STORY_SELF_TEST_CASES = (
    dict(),                                                         # the Story anchors (2 / 14 / 18 / 26 / 30)
    dict(breakdown=14.04, drop2=17.97, end_hit=26.02),              # cue sheet drift -> snapped with warnings
    dict(duration=24.0, drop1=0.0, breakdown=8.0, drop2=14.0, end_hit=20.0),   # drop on sample 0, 2 dark bars
)


def story_self_test(cases=STORY_SELF_TEST_CASES, seed=1, verbose=True) -> list:
    """Render each anchor set and assert the contract: shapes/dtype, finite, stem true peak <= -6 dBTP,
    |DC| < 1e-4, zero first/last samples, every anchor a downbeat in meta and (snapped) equal to the request
    rounded to a bar, a warning exactly when an anchor had to move, audible from t=0."""
    results, failed = [], []
    for case in cases:
        kw = dict(STORY_DEFAULTS, **case)
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            st = render_story(seed=seed, **kw)
        m = st["meta"]
        N = int(round(kw["duration"] * SR))
        xs = {k: st[k].astype(np.float64) for k in STEMS}
        mix = sum(xs.values())
        first100 = _db(np.sqrt(np.mean(mix[:int(0.1 * SR)] ** 2)))
        want = {n: kw["drop1"] + round((kw[n] - kw["drop1"]) / BAR) * BAR for n in ("breakdown", "drop2", "end_hit")}
        moved = any(abs(want[n] - kw[n]) > 1e-6 for n in want)
        chk = dict(
            shape_dtype=all(st[k].shape == (N, 2) and st[k].dtype == np.float32 for k in STEMS),
            finite=all(np.all(np.isfinite(v)) for v in xs.values()),
            true_peak_le_minus6=all(_db(_true_peak(v)) <= -6.0 for v in xs.values()),
            dc_lt_1e4=all(np.max(np.abs(np.mean(v, axis=0))) < 1e-4 for v in xs.values()),
            ends_zero=all(np.max(np.abs(v[[0, -1]])) < 1e-6 for v in xs.values()),
            anchors_snapped=all(abs(m[n] - want[n]) < 1e-9 for n in want) and abs(m["drop1"] - kw["drop1"]) < 1e-12,
            anchors_downbeats=all(any(abs(b - m[n]) < 1e-9 for b in m["downbeat_times"])
                                  for n in ("drop1", "breakdown", "drop2", "end_hit")),
            warning_iff_snapped=(len(wl) > 0) == moved,
            first100ms_gt_minus35dBFS=first100 > -35.0)
        r = dict(case=kw, anchors=[m["drop1"], m["breakdown"], m["drop2"], m["end_hit"]],
                 first100ms_dBFS=round(first100, 1), ok=all(chk.values()), failed=[k for k, v in chk.items() if not v])
        results.append(r)
        if verbose:
            print(json.dumps(r), flush=True)
        if not r["ok"]:
            failed.append(r)
    if failed:
        raise AssertionError("story self-test failed: " + json.dumps(failed))
    return results


if __name__ == "__main__":
    import sys
    import time
    t0 = time.time()
    if "--selftest" in sys.argv[1:]:
        self_test()
        print(f"self-test passed in {time.time() - t0:.1f}s")
    elif "--story-selftest" in sys.argv[1:]:
        story_self_test()
        print(f"story self-test passed in {time.time() - t0:.1f}s")
    elif "--story" in sys.argv[1:]:
        r = render_story_demo()
        print(json.dumps({k: r[k] for k in ("checks", "all_checks_pass", "section_loudness_stems_sum", "demo")}, indent=1))
        print(f"rendered in {time.time() - t0:.1f}s")
    else:
        r = render_demo()
        print(json.dumps(r, indent=2))
        print(f"rendered in {time.time() - t0:.1f}s")
