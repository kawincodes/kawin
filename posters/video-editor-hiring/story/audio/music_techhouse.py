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
                                          each peaking <= -6 dBFS (true peak),
        'meta' : {'bpm', 'key', 'beat_times', 'drop_time', 'end_hit_time_snapped', ...extras}}
    check_arrangement(...)  -> list of musical-rule violations (render() raises if any)
    render_demo()           -> writes demo/music_techhouse_demo.wav + spectrogram PNG + stem report

Run:  python3 -I music_techhouse.py      (renders the demo)
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from scipy import signal as sps
from pedalboard import Chorus, Compressor, LadderFilter, Limiter, Reverb

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
STEM_PEAK_DBTP = -6.5      # stems are scaled (all together) so the hottest true peak sits here

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


# ----------------------------------------------------------------------------------------------
# beat grid
# ----------------------------------------------------------------------------------------------
def beat_grid(duration: float, drop_time: float, end_hit_time: float) -> dict:
    """Beat k is at drop_time + k*BEAT; k = 0 is the drop downbeat, k % 4 == 0 are bar lines."""
    k_first = -int(math.floor(drop_time / BEAT + 1e-9))                    # first beat >= 0 s
    k_last = int(math.ceil((duration - drop_time) / BEAT - 1e-9)) - 1     # last beat < duration
    hk = int(math.floor((end_hit_time - drop_time) / BEAT + 0.5))          # nearest beat (ties up)
    hk_max = int(math.floor((duration - 1.0 - drop_time) / BEAT))          # leave >= 1 s of tail
    hk = max(3, min(hk, hk_max))
    beats = [drop_time + k * BEAT for k in range(k_first, k_last + 1)]
    return dict(k_first=k_first, k_last=k_last, hk=hk, beats=beats,
                t_hit=drop_time + hk * BEAT, drop=drop_time, duration=duration)


def chord_of_beat(k: int, hk: int) -> str:
    if k >= hk:
        return FINAL_CHORD
    if k >= hk - 2:
        return TURNAROUND_CHORD
    return PROGRESSION[(k // 4) % 4]


def chord_segments(g: dict) -> list:
    """[(t0, t1, chord)] -- merges beats with the same chord; starts at the intro bar line (< 0 s)."""
    drop, hk = g["drop"], g["hk"]
    k0 = (g["k_first"] // 4) * 4 if g["k_first"] < 0 else 0
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
    drop, hk, t_hit = g["drop"], g["hk"], g["t_hit"]
    k0 = (g["k_first"] // 4) * 4 if g["k_first"] < 0 else 0
    ev = []

    def jit(v, amt=0.05):
        return float(np.clip(v * (1.0 + rng.uniform(-amt, amt)), 0.05, 1.2))

    # ---------------- arp (+ octave lift layer in the second half) ----------------
    for k in range(k0 - 1, hk):
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
        ev.append(dict(inst="snare", t=tn, tn=tn, vel=0.30 + 0.70 * p ** 1.3, pitch=1.0 + 0.45 * p,
                       hp=250 + 900 * p))
    # small fill at the end of every 4th bar (not inside the turnaround)
    for k in range(15, hk - 2, 16):
        for s, v in ((1, 0.45), (2, 0.6), (3, 0.8)):
            tn = drop + k * BEAT + s * S16
            ev.append(dict(inst="snare", t=tn + _swing(s), tn=tn, vel=v, pitch=1.0 + 0.1 * s, hp=300))
        ev.append(dict(inst="crash", t=drop + (k + 1) * BEAT, tn=drop + (k + 1) * BEAT, vel=0.55))
    # turnaround fill: beat hk-2 = 16ths on the 2nd half, beat hk-1 = 32nds crescendo (no kick)
    for s in (2, 3):
        tn = drop + (hk - 2) * BEAT + s * S16
        ev.append(dict(inst="snare", t=tn + _swing(s), tn=tn, vel=0.45 + 0.1 * s, pitch=1.05, hp=300))
    for i in range(8):
        tn = drop + (hk - 1) * BEAT + i * S16 / 2
        if tn >= t_hit - GAP:
            continue
        p = i / 7.0
        ev.append(dict(inst="snare", t=tn, tn=tn, vel=0.55 + 0.45 * p, pitch=1.1 + 0.35 * p, hp=400 + 800 * p))
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
def check_arrangement(events: list, g: dict) -> list:
    segs = chord_segments(g)
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
            if e["inst"] in ("arp", "pad", "stab") and m % 12 not in pcs(ch):
                issues.append(f"{e['inst']} @ {e['t']:.3f}s: MIDI {m} not in chord {ch}")
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


def _additive(f0, n, amps, fc, q=0.8, poles=4, fmax=19000.0, block=16384):
    """Band-limited oscillator with a (time-varying) resonant low-pass applied per harmonic.
    amps[k-1] = amplitude of harmonic k; fc = cutoff (scalar or per-sample array). Alias-free."""
    K = int(min(len(amps), fmax // f0))
    fc = np.broadcast_to(np.asarray(fc, dtype=np.float64), (n,))
    kmax = int(min(K, (fc.max() * 14.0) // f0))      # beyond 14x cutoff a 24 dB/oct LP is < -90 dB
    if kmax < 1:
        return np.zeros(n)
    k = np.arange(1, kmax + 1, dtype=np.float64)[:, None]
    a = np.asarray(amps[:kmax], dtype=np.float64)
    out = np.empty(n)
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        t = np.arange(i0, i1) / SR
        r2 = (k * f0 / fc[None, i0:i1]) ** 2
        H = 1.0 / np.sqrt((1.0 - r2) ** 2 + r2 / (q * q))
        if poles == 4:
            H = H * H
        ph = np.mod(2.0 * np.pi * f0 * k * t[None, :], 2.0 * np.pi)
        out[i0:i1] = np.einsum("k,kn->n", a, H * np.sin(ph))
    return out


_K = np.arange(1, 1200, dtype=np.float64)
SAW_AMPS = 1.0 / _K
SQR_AMPS = np.where(_K % 2 == 1, 1.0 / _K, 0.0)


def _ladder(x: np.ndarray, cutoff, mode="LPF24", res=0.1, drive=1.0, block=256) -> np.ndarray:
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


def _pingpong(x: np.ndarray, delay_s: float, fb: float, taps: int = 7) -> np.ndarray:
    """Ping-pong echoes of the mono sum, each repeat a little darker; first repeat on the left."""
    src = x.mean(axis=1)
    src = _filt(src, "highpass", 280.0)
    wet = np.zeros_like(x)
    cur = src
    g = 1.0
    lp = _sos("lowpass", 5200.0, 1)
    for i in range(1, taps + 1):
        g *= fb
        cur = sps.sosfilt(lp, cur)
        wet[:, (i - 1) % 2] += g * _delay(cur[:, None], delay_s * i)[:, 0]
    return wet


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
def _kick(vel, rng, big=False):
    dur = 1.6 if big else 0.42
    n = int(dur * SR)
    t = np.arange(n) / SR
    f_end = hz(nm("F1"))                                          # body tuned to the tonic, 43.7 Hz
    f = f_end + ((210.0 if big else 175.0) - f_end) * np.exp(-t / 0.028) + 380.0 * np.exp(-t / 0.0035)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR)
    hold, tau = (0.06, 0.45) if big else (0.035, 0.17)
    amp = np.where(t < hold, 1.0, np.exp(-(t - hold) / tau))
    body = np.tanh(1.7 * body * amp) / np.tanh(1.7)
    click = _filt(rng.standard_normal(n), "bandpass", [2600.0, 5200.0], 2) * np.exp(-t / 0.0030)
    click /= np.max(np.abs(click[: int(0.01 * SR)])) + 1e-9
    y = body + 0.22 * click
    return vel * y * _fades(n, 0.002, 0.03 if not big else 0.3)


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
    y = (body + air) * env
    y /= np.max(np.abs(y)) + 1e-9
    return vel * y * _fades(n, 0.002, 0.05)


_METAL = (205.3, 304.4, 369.6, 522.7, 540.0, 800.0)


def _metal(n, rng, scale=1.0, os=4):
    t = np.arange(n * os) / (SR * os)
    x = np.zeros(n * os)
    for f in _METAL:
        x += np.sign(np.sin(2 * np.pi * f * scale * t + rng.uniform(0, 2 * np.pi)))
    return sps.resample_poly(x, 1, os)


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


def _pluck(midi, dur, vel, base_fc, env_amt=3.2, env_tau=0.045, amp_tau=0.11, detune_c=7.0, cache=None):
    key = (midi, round(dur, 3), round(vel, 2), int(base_fc / 25) * 25, round(env_amt, 2), round(amp_tau, 3))
    if cache is not None and key in cache:
        return cache[key]
    n = int(dur * SR)
    t = np.arange(n) / SR
    f0 = hz(midi)
    bfc = base_fc * (0.65 + 0.35 * vel)
    fc = np.minimum(bfc * (1.0 + env_amt * np.exp(-t / env_tau)), 17000.0)
    sq = _additive(f0, n, SQR_AMPS, fc, q=0.9) * 0.45
    L = _additive(f0 * 2 ** (-detune_c / 1200), n, SAW_AMPS, fc, q=0.9) + sq
    R = _additive(f0 * 2 ** (+detune_c / 1200), n, SAW_AMPS, fc, q=0.9) + sq
    amp = np.exp(-t / amp_tau) * _fades(n, 0.002, min(0.06, dur * 0.3))
    y = np.stack([L, R], axis=1) * (amp * vel ** 1.2)[:, None]
    if cache is not None:
        cache[key] = y
    return y


def _bass_note(midi, dur, vel, cache, final=False):
    key = (midi, round(dur, 3), round(vel, 2), final)
    if key in cache:
        return cache[key]
    n = int(dur * SR)
    t = np.arange(n) / SR
    f0 = hz(midi)
    if final:                                     # the long tonic note under the end card
        amp = np.exp(-t / 1.1) * _fades(n, 0.003, 0.2)
        fc = 160.0 + 900.0 * np.exp(-t / 0.08)
    else:
        amp = _fades(n, 0.003, 0.014) * (0.75 + 0.25 * np.exp(-t / 0.05))
        fc = 260.0 + 1100.0 * np.exp(-t / 0.035) * vel
    sub = np.sin(2 * np.pi * f0 * t) * amp
    mid = _additive(f0, n, SAW_AMPS, fc, q=1.1) * amp
    mid = np.tanh(2.0 * mid) / 2.0
    y = (sub, mid, vel)
    cache[key] = y
    return y


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
# automation curves (video time, seconds)
# ----------------------------------------------------------------------------------------------
def _pad_cutoff(t, g):
    drop, t_hit, dur = g["drop"], g["t_hit"], g["duration"]
    t0 = min(-0.2, drop - BAR)
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
    t0 = min(-0.2, drop - BAR)
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


def _gap_gate(n_tot, times):
    """Silence the last GAP seconds before each given time (4 ms ramps)."""
    g = np.ones(n_tot)
    r = int(0.004 * SR)
    for t in times:
        i1 = int(round((t + PRE) * SR))
        i0 = i1 - int(GAP * SR)
        g[i0 - r:i0] = np.minimum(g[i0 - r:i0], 0.5 + 0.5 * np.cos(np.pi * np.arange(r) / r))
        g[i0:i1] = 0.0
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
LV = dict(kick=-7.0, clap=-15.5, clap_plate=-21.0, hat=-23.0, ohat=-24.0, snare=-17.0, crash=-23.0,
          sub=-11.0, midbass=-15.5, pad=-17.0, arp=-19.0, lift=-19.0, stab=-15.0, delay=-25.0,
          riser=-22.0, riser_tone=-27.0, boom=-12.0, downlifter=-27.0, revcym=-19.0, impact=-21.0)


def render(duration, drop_time, end_hit_time, seed=1):
    rng = np.random.default_rng(seed)
    N = int(round(duration * SR))
    n_tot = int(PRE * SR) + N + int(0.5 * SR)
    g = beat_grid(duration, drop_time, end_hit_time)
    events = arrange(g, np.random.default_rng(seed + 1000))
    issues = check_arrangement(events, g)
    if issues:
        raise AssertionError("musical check failed:\n  " + "\n  ".join(issues))

    drop, t_hit = g["drop"], g["t_hit"]
    tl = np.arange(n_tot) / SR - PRE                   # video time of every buffer sample
    kicks = [e["t"] for e in events if e["inst"] == "kick"]
    ghost = [drop + k * BEAT for k in range((g["k_first"] // 4) * 4 - 4, 0)]   # intro pulse, no kick

    # ------------------------------------------------ DRUMS
    drums = _Bus(n_tot)
    plate_send = _Bus(n_tot)
    hat_bank = [_hat(1.0, rng) for _ in range(6)]
    ohat_bank = [_hat(1.0, rng, open_=True) for _ in range(3)]
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
    plate = _pb(Reverb(room_size=0.32, damping=0.6, wet_level=1.0, dry_level=0.0, width=0.9),
                _delay(plate_send.x, 0.012))
    plate = _filt(_filt(plate, "highpass", 450.0, 2), "lowpass", 7500.0, 2)
    D = drums.x + _db2(LV["clap_plate"]) * plate * 2.0
    D = D * _gap_gate(n_tot, [drop, t_hit])[:, None]

    # ------------------------------------------------ BASS
    sub_bus, mid_bus = _Bus(n_tot), _Bus(n_tot)
    bcache = {}
    for e in events:
        if e["inst"] != "bass":
            continue
        sub, mid, vel = _bass_note(e["midi"], e["dur"], e["vel"], bcache, final=e.get("final", False))
        sub_bus.add(e["t"], sub, vel * _db2(LV["sub"]))
        mid_bus.add(e["t"], mid, vel * _db2(LV["midbass"]))
    sub_x = _filt(sub_bus.x, "lowpass", 140.0, 2)
    mid_x = _filt(mid_bus.x, "highpass", 110.0, 2)
    mid_x = _peq(mid_x, 900.0, 2.0, 0.8)                                # phone-speaker audibility
    B = sub_x + mid_x
    B *= _pump(kicks, n_tot, 0.85, rel=0.16)[:, None]
    B *= _gap_gate(n_tot, [drop, t_hit])[:, None]

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
            q = np.clip(tt / max(e["dur"] - TAIL_FADE, 0.1), 0, 1)
            amp = (1.0 + 0.35 * np.exp(-tt / 0.25)) * (1.0 - 0.6 * q ** 1.5)    # swell on hit, ring out
            y *= (amp * _fades(n, 0.004, 0.05))[:, None]
        else:
            n = int((e["dur"] + rel) * SR)
            y = _supersaw_chord(e["midis"], n, rng)
            y *= _fades(n, 0.03 if e["t"] > 0 else 0.002, rel)[:, None]
        pad_bus.add(e["t"], y, _db2(LV["pad"]))
    P = _ladder(pad_bus.x, _pad_cutoff(tl, g), res=0.12)
    P = _filt(P, "highpass", 150.0, 4)
    P = _peq(P, 320.0, -2.0, 1.0)
    P = _pb(Chorus(rate_hz=0.33, depth=0.22, centre_delay_ms=7.0, feedback=0.0, mix=0.45), P)
    P = _pb(Reverb(room_size=0.78, damping=0.55, wet_level=0.24, dry_level=0.85, width=1.0), P)
    pump_pad = np.minimum(_pump(kicks, n_tot, 0.55, rel=0.2), _pump(ghost, n_tot, 0.4, rel=0.24))
    P *= pump_pad[:, None]

    # ------------------------------------------------ MUSIC: arp + stabs
    arp_bus = _Bus(n_tot)
    pcache = {}
    for e in events:
        if e["inst"] == "arp":
            fc = float(_arp_cutoff(np.array([e["t"]]), g)[0])
            y = _pluck(e["midi"], e["dur"], e["vel"], fc, cache=pcache)
            lvl = LV["lift"] if e.get("layer") == "lift" else LV["arp"]
            arp_bus.add(e["t"], y, _db2(lvl))
        elif e["inst"] == "stab":
            fin = e.get("final", False)
            for j, m in enumerate(e["midis"]):
                y = _pluck(m, e["dur"], 0.95, 3200.0 if fin else 2800.0, env_amt=2.5, env_tau=0.08,
                           amp_tau=0.9 if fin else 0.28, cache=pcache)
                arp_bus.add(e["t"] + 0.004 * j, y, _db2(LV["stab"]) / np.sqrt(len(e["midis"])))
    A = _filt(arp_bus.x, "highpass", 220.0, 2)
    A = _peq(A, 2800.0, -2.5, 1.2)
    A_d = _pingpong(A, 0.375, 0.42) * _db2(LV["delay"] - LV["arp"]) * 4.0
    A = A + A_d
    A = _pb(Reverb(room_size=0.5, damping=0.6, wet_level=0.13, dry_level=1.0, width=1.0), A)
    A *= _pump(kicks, n_tot, 0.30, rel=0.16)[:, None]
    M = P + A
    M *= _gap_gate(n_tot, [drop, t_hit])[:, None]

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
            f = hz(nm("F1")) + (hz(nm("F2")) * 1.25 - hz(nm("F1"))) * np.exp(-tt / 0.07)
            tau = 0.9 if e.get("final") else 0.45
            y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / tau)
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
    imp_wet = _pb(Reverb(room_size=0.9, damping=0.45, wet_level=1.0, dry_level=0.0, width=1.0), send.x)
    F = fx.x + 0.35 * imp_wet
    F = _filt(F, "highpass", 28.0, 2)

    # ------------------------------------------------ crop, clean, fades, level
    stems = {}
    s0 = int(PRE * SR)
    n_fade = int(TAIL_FADE * SR)
    out_env = np.ones(N)
    out_env[:int(0.005 * SR)] = 0.5 - 0.5 * np.cos(np.pi * np.arange(int(0.005 * SR)) / int(0.005 * SR))
    out_env[N - n_fade:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(1, n_fade + 1) / n_fade)
    for name, X in (("drums", D), ("bass", B), ("music", M), ("fx", F)):
        X = _filt(X, "highpass", 22.0, 2)                               # DC / rumble
        X = X[s0:s0 + N] * out_env[:, None]
        X = X - (np.mean(X, axis=0) * out_env[:, None])                 # residual DC, keeps the 0 ends
        stems[name] = X
    tps = {k: _true_peak(v) for k, v in stems.items()}
    scale = _db2(STEM_PEAK_DBTP) / max(tps.values())
    for k in stems:
        stems[k] = np.nan_to_num(stems[k] * scale).astype(np.float32)

    beats = [float(b) for b in g["beats"]]
    segs = chord_segments(g)
    stems["meta"] = dict(
        bpm=BPM, key=KEY, beat_times=beats, drop_time=float(drop_time),
        end_hit_time_snapped=float(t_hit),
        # extras
        downbeat_times=[float(drop + k * BEAT) for k in range(g["k_first"], g["k_last"] + 1) if k % 4 == 0],
        scale=list(SCALE), progression=list(PROGRESSION), turnaround_chord=TURNAROUND_CHORD,
        final_chord=FINAL_CHORD,
        chord_timeline=[(max(0.0, float(a)), float(b), c) for (a, b, c) in segs if b > 0],
        stem_scale_db=float(_db(scale)), sample_rate=SR, swing_s=SWING,
        n_events=len(events),
    )
    return stems


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


def master_preview(mix: np.ndarray, target_lufs=-14.0, ceiling_dbtp=-1.0) -> np.ndarray:
    """Quick preview master: glue comp, gain to target, limiter, true-peak safety trim."""
    x = _pb(Compressor(threshold_db=-20.0, ratio=2.0, attack_ms=15.0, release_ms=150.0), mix)
    x = x * _db2(target_lufs - lufs(x))
    for _ in range(3):
        y = _pb(Limiter(threshold_db=ceiling_dbtp - 1.0, release_ms=80.0), x)
        tp = _db(_true_peak(y))
        if tp <= ceiling_dbtp - 0.1:
            break
        x = x * _db2(ceiling_dbtp - 0.2 - tp)
    tp = _true_peak(y)
    if _db(tp) > ceiling_dbtp - 0.1:
        y = y * _db2(ceiling_dbtp - 0.15) / tp
    return y


def render_demo(duration=17.5, drop_time=1.8, end_hit_time=14.5, seed=1, out_dir=DEMO_DIR):
    os.makedirs(out_dir, exist_ok=True)
    st = render(duration, drop_time, end_hit_time, seed=seed)
    mix = sum(st[k].astype(np.float64) for k in ("drums", "bass", "music", "fx"))
    y = master_preview(mix)
    wav = os.path.join(out_dir, "music_techhouse_demo.wav")
    sf.write(wav, y.astype(np.float32), SR, subtype="FLOAT")
    png = os.path.join(out_dir, "music_techhouse_demo_spectrogram.png")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", wav, "-lavfi",
                    "showspectrumpic=s=1600x600:legend=1", png], check=True)
    rep = dict(params=dict(duration=duration, drop_time=drop_time, end_hit_time=end_hit_time, seed=seed),
               meta={k: v for k, v in st["meta"].items() if k != "beat_times"},
               n_beats=len(st["meta"]["beat_times"]),
               stems=stem_report(st),
               unmastered_sum=dict(lufs=round(lufs(mix), 2), true_peak_dBTP=round(_db(_true_peak(mix)), 2)),
               demo=dict(ebur128=_ebur128(wav), python_lufs=round(lufs(y), 2),
                         true_peak_dBTP=round(_db(_true_peak(y)), 2),
                         dc=[float(f"{v:.2e}") for v in np.mean(y, axis=0)]))
    with open(os.path.join(out_dir, "music_techhouse_demo_report.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    return rep


if __name__ == "__main__":
    import time
    t0 = time.time()
    r = render_demo()
    print(json.dumps(r, indent=2))
    print(f"rendered in {time.time() - t0:.1f}s")
