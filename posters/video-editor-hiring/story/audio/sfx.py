"""
sfx.py - original, fully synthesised sound effects for the KRS Tech Solutions
"WE'RE HIRING - VIDEO EDITOR" Instagram Story (1080x1920, 30 fps, ~18 s).

Everything is generated from oscillators and seeded noise in this file: no
samples, loops or downloaded material (copyright-safe).

Shared contract (agrees with the music module):
    * 48 kHz, stereo, float32 numpy arrays shaped (N, 2)
    * key of F minor (F G Ab Bb C Db Eb), tonal centre F, tempo 120 BPM
    * every tuned element is taken from the F-minor scale (checked by
      ``verify_key()``), rhythmic subdivisions are locked to 120 BPM
    * each function returns a sound that is true-peak normalised to -3 dBTP
      (sample peak about -3 dBFS), DC-free, with raised-cosine fades
      (>= 2 ms) at both ends.  Fully deterministic (fixed RNG seeds).

Public API
    whoosh(dur=0.45, direction='lr', intensity=1.0)
    slash(dur=0.35)
    impact(size='big')            # 'big' | 'medium' | 'small'
    reject()
    check()
    pop(pitch_semitones=0)
    tick()
    scrub(dur)
    riser(dur=1.5)
    downlifter(dur=1.0)
    shimmer(dur=1.2)
    reverse_swell(dur=0.8)
    anchor(name, **kwargs)        # sync point (s) inside the returned array
    render_demo(out_dir=DEMO_DIR) # demo WAV + spectrogram + report
    verify_key()                  # asserts every tuned note is in F minor
"""
from __future__ import annotations

import os
import re
import subprocess
import warnings
import zlib

import numpy as np
import soundfile as sf
from scipy import signal
from pedalboard import Compressor, Reverb

SR = 48000
BPM = 120.0
BEAT = 60.0 / BPM            # 0.5 s
SIXTEENTH = BEAT / 4         # 0.125 s
PEAK_DB = -3.0               # output normalisation (true peak)

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(HERE, "demo")

# ----------------------------------------------------------------------------
# key / tuning
# ----------------------------------------------------------------------------
KEY_NAME = "F minor"
_PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
       "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
       "B": 11}
KEY_PCS = frozenset({5, 7, 8, 10, 0, 1, 3})        # F G Ab Bb C Db Eb
_PC_NAME = {0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F", 6: "Gb", 7: "G",
            8: "Ab", 9: "A", 10: "Bb", 11: "B"}

# Every tuned note any sound uses is recorded here: {sound: set(note names)}
NOTES_USED: dict[str, set] = {}


def note_midi(name: str) -> int:
    m = re.fullmatch(r"([A-G](?:#|b)?)(-?\d)", name)
    if not m:
        raise ValueError(f"bad note name {name!r}")
    return _PC[m.group(1)] + 12 * (int(m.group(2)) + 1)


def midi_name(m: int) -> str:
    return f"{_PC_NAME[m % 12]}{m // 12 - 1}"


def midi_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def in_key(m: int) -> bool:
    return (int(round(m)) % 12) in KEY_PCS


def _note(name: str, sound: str) -> float:
    """Hz of a note that must be in F minor; records it for verify_key()."""
    m = note_midi(name)
    if not in_key(m):
        raise ValueError(f"{sound}: {name} is not in {KEY_NAME}")
    NOTES_USED.setdefault(sound, set()).add(midi_name(m))
    return midi_hz(m)


# ----------------------------------------------------------------------------
# small DSP toolkit
# ----------------------------------------------------------------------------
def _n(sec: float) -> int:
    return max(1, int(round(sec * SR)))


def _t(n: int) -> np.ndarray:
    return np.arange(n) / SR


def _rng(tag: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(tag.encode("utf-8")))


def _ramp_in(n: int, sec: float) -> np.ndarray:
    """Raised-cosine 0 -> 1 over `sec`, then 1."""
    t = _t(n)
    a = np.clip(t / max(sec, 1e-6), 0.0, 1.0)
    return np.sin(0.5 * np.pi * a) ** 2


def _decay(n: int, tau: float, attack: float = 0.001) -> np.ndarray:
    t = _t(n)
    return np.exp(-t / tau) * _ramp_in(n, attack)


def _place(dst: np.ndarray, src: np.ndarray, start: int) -> None:
    """Mix src into dst at sample `start` (both mono or both stereo)."""
    if start >= len(dst):
        return
    end = min(len(dst), start + len(src))
    dst[start:end] += src[: end - start]


def _phase(freq, n: int) -> np.ndarray:
    f = np.broadcast_to(np.asarray(freq, dtype=float), (n,))
    ph = np.empty(n)
    ph[0] = 0.0
    ph[1:] = np.cumsum(f[:-1])
    return 2.0 * np.pi * ph / SR


def _sine(freq, n: int, phase0: float = 0.0) -> np.ndarray:
    return np.sin(_phase(freq, n) + phase0)


def _bl_wave(freq, n: int, kind: str = "saw", max_h: int | None = None,
             phase0: float = 0.0) -> np.ndarray:
    """Band-limited (additive) saw or square; harmonics fade out near Nyquist."""
    f = np.broadcast_to(np.asarray(freq, dtype=float), (n,))
    ph = _phase(f, n) + phase0
    k_max = int(0.45 * SR / max(float(f.min()), 1.0))
    if max_h:
        k_max = min(k_max, max_h)
    ks = range(1, k_max + 1) if kind == "saw" else range(1, k_max + 1, 2)
    y = np.zeros(n)
    for k in ks:
        g = np.clip((0.45 * SR - k * f) / (0.05 * SR), 0.0, 1.0)
        y += g * np.sin(k * ph) / k
    return y * (2.0 / np.pi if kind == "saw" else 4.0 / np.pi)


def _sos(kind: str, fc, order: int = 2):
    return signal.butter(order, fc, btype=kind, fs=SR, output="sos")


def _filt(x: np.ndarray, kind: str, fc, order: int = 2) -> np.ndarray:
    return signal.sosfilt(_sos(kind, fc, order), x, axis=0)


def _svf(x: np.ndarray, fc, q=0.707, mode: str = "lp") -> np.ndarray:
    """Time-varying TPT state-variable filter (Zavalishin).  Stable under fast
    cutoff modulation.  mode: 'lp' | 'bp' (unity peak gain) | 'hp'.
    x is mono; fc/q may be scalars or per-sample arrays."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        return np.stack([_svf(x[:, c], fc, q, mode) for c in range(x.shape[1])], 1)
    n = len(x)
    fc = np.broadcast_to(np.asarray(fc, dtype=float), (n,))
    q = np.broadcast_to(np.asarray(q, dtype=float), (n,))
    g = np.tan(np.pi * np.clip(fc, 15.0, 0.46 * SR) / SR)
    k = 1.0 / q
    a1 = 1.0 / (1.0 + g * (g + k))
    a2 = g * a1
    a3 = g * a2
    xs, A1, A2, A3, K = x.tolist(), a1.tolist(), a2.tolist(), a3.tolist(), k.tolist()
    out = [0.0] * n
    ic1 = ic2 = 0.0
    if mode == "lp":
        for i in range(n):
            v3 = xs[i] - ic2
            v1 = A1[i] * ic1 + A2[i] * v3
            v2 = ic2 + A2[i] * ic1 + A3[i] * v3
            ic1 = 2.0 * v1 - ic1
            ic2 = 2.0 * v2 - ic2
            out[i] = v2
    elif mode == "bp":
        for i in range(n):
            v3 = xs[i] - ic2
            v1 = A1[i] * ic1 + A2[i] * v3
            v2 = ic2 + A2[i] * ic1 + A3[i] * v3
            ic1 = 2.0 * v1 - ic1
            ic2 = 2.0 * v2 - ic2
            out[i] = K[i] * v1
    elif mode == "hp":
        for i in range(n):
            xi = xs[i]
            v3 = xi - ic2
            v1 = A1[i] * ic1 + A2[i] * v3
            v2 = ic2 + A2[i] * ic1 + A3[i] * v3
            ic1 = 2.0 * v1 - ic1
            ic2 = 2.0 * v2 - ic2
            out[i] = xi - K[i] * v1 - v2
    else:
        raise ValueError(mode)
    return np.asarray(out)


def _pan(m: np.ndarray, pan) -> np.ndarray:
    """Equal-power pan of a mono signal; pan in [-1, 1] (scalar or array)."""
    th = (np.clip(np.asarray(pan, dtype=float), -1, 1) + 1.0) * np.pi / 4.0
    return np.stack([m * np.cos(th), m * np.sin(th)], axis=1) * np.sqrt(2.0)


def _center(m: np.ndarray) -> np.ndarray:
    return np.stack([m, m], axis=1)


def _stereo_noise(n: int, rng: np.random.Generator, width=0.5) -> np.ndarray:
    """White noise pair with controllable L/R correlation (width 0 = mono,
    1 = uncorrelated).  Width may be a per-sample array."""
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    w = np.asarray(width, dtype=float)
    norm = 1.0 / np.sqrt(1.0 + w * w)
    return np.stack([(a + w * b) * norm, (a - w * b) * norm], axis=1)


def _pinkish(x: np.ndarray) -> np.ndarray:
    """-3 dB/oct tilt (Kellet-style 3-pole approximation) along axis 0."""
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    return signal.lfilter(b, a, x, axis=0) * 8.0


def _mono_low(x: np.ndarray, fc: float = 140.0) -> np.ndarray:
    """Collapse everything below fc to mono (keeps phones / mono sum solid)."""
    mid = 0.5 * (x[:, 0] + x[:, 1])
    side = 0.5 * (x[:, 0] - x[:, 1])
    side = _filt(side, "highpass", fc, 2)
    return np.stack([mid + side, mid - side], axis=1)


def _sat(x: np.ndarray, drive: float) -> np.ndarray:
    """Symmetric soft clip (adds odd harmonics so sub energy reads on phones)."""
    return np.tanh(drive * x) / np.tanh(drive)


def _reverb(x: np.ndarray, room=0.5, damp=0.5, width=1.0, hp=200.0, lp=8000.0,
            predelay=0.0, tail=0.0) -> np.ndarray:
    """Wet-only Freeverb (pedalboard) return, band-limited, length len(x)+tail."""
    pad = _n(tail) if tail > 0 else 0
    pre = _n(predelay) if predelay > 0 else 0
    xin = np.concatenate([np.zeros((pre, 2)), x, np.zeros((pad, 2))], axis=0)
    rv = Reverb(room_size=room, damping=damp, wet_level=1.0, dry_level=0.0,
                width=width)
    w = rv(np.ascontiguousarray(xin.T, dtype=np.float32), SR).T.astype(float)
    w = w[: len(x) + pad]
    w = _filt(w, "highpass", hp, 2)
    w = _filt(w, "lowpass", lp, 2)
    return w


def _with_tail(x: np.ndarray, extra: int) -> np.ndarray:
    return np.concatenate([x, np.zeros((extra, 2))], axis=0)


def _compress(x: np.ndarray, thr=-16.0, ratio=2.5, attack=8.0, release=150.0):
    pk = np.max(np.abs(x)) + 1e-12
    y = x / pk * 0.7
    comp = Compressor(threshold_db=thr, ratio=ratio, attack_ms=attack,
                      release_ms=release)
    y = comp(np.ascontiguousarray(y.T, dtype=np.float32), SR).T.astype(float)
    return y


def true_peak(x: np.ndarray) -> float:
    """Linear true-peak estimate (4x oversampled, as BS.1770 / ffmpeg)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    pad = np.zeros((64, x.shape[1]))
    up = signal.resample_poly(np.concatenate([pad, x, pad]), 4, 1, axis=0)
    return float(max(np.max(np.abs(up)), np.max(np.abs(x))))


def _fade(x: np.ndarray, fin: float, fout: float) -> np.ndarray:
    n = len(x)
    ni = min(max(_n(fin), _n(0.002)), n // 2)
    no = min(max(_n(fout), _n(0.002)), n // 2)
    g = np.ones(n)
    g[:ni] = np.sin(0.5 * np.pi * np.arange(ni) / ni) ** 2
    g[n - no:] = np.cos(0.5 * np.pi * (np.arange(no) + 1) / no) ** 2
    return x * g[:, None]


_DC_SOS = None


def _finish(x: np.ndarray, length: int | None = None, fade_in=0.002,
            fade_out=0.004, trim_db=-60.0, peak_db=PEAK_DB) -> np.ndarray:
    """Common mastering for every sound: NaN guard, DC block, optional tail
    trim (or exact length), raised-cosine fades, mean removal, true-peak
    normalisation, float32 (N, 2)."""
    global _DC_SOS
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim == 1:
        x = _center(x)
    if _DC_SOS is None:
        _DC_SOS = _sos("highpass", 16.0, 2)
    x = signal.sosfilt(_DC_SOS, x, axis=0)
    if length is not None:
        if len(x) >= length:
            x = x[:length]
        else:
            x = np.concatenate([x, np.zeros((length - len(x), 2))])
    else:
        env = np.max(np.abs(x), axis=1)
        thr = env.max() * 10 ** (trim_db / 20.0)
        idx = np.nonzero(env > thr)[0]
        last = (idx[-1] if len(idx) else len(x) - 1) + _n(0.005)
        x = x[: min(len(x), last)]
        fade_out = max(fade_out, min(0.06, 0.2 * len(x) / SR))
    x = _fade(x, fade_in, fade_out)
    # exact DC removal with a smooth (zero-ended) window -> no steps, mean = 0
    n = len(x)
    if n > 16:
        w = np.hanning(n)
        w *= n / w.sum()
        x = x - np.mean(x, axis=0)[None, :] * w[:, None]
    tp = true_peak(x)
    if tp > 0:
        x = x * (10 ** (peak_db / 20.0) / tp)
    return x.astype(np.float32)


# ----------------------------------------------------------------------------
# reusable sound elements
# ----------------------------------------------------------------------------
def _modal_click(modes, n: int, rng: np.random.Generator | None = None,
                 noise_amp=0.0, noise_hp=6000.0, noise_tau=0.0007) -> np.ndarray:
    """Struck-object click: decaying sine modes (+ optional HF noise snap).
    modes = [(freq, amp, tau), ...]; sines start at phase 0 -> no step."""
    t = _t(n)
    y = np.zeros(n)
    att = _ramp_in(n, 0.0004)
    for f, a, tau in modes:
        y += a * np.sin(2 * np.pi * f * t) * np.exp(-t / tau)
    y *= att
    if noise_amp > 0 and rng is not None:
        nz = _filt(rng.standard_normal(n), "highpass", noise_hp, 2)
        y += noise_amp * nz * np.exp(-t / noise_tau) * att
    return y


# ----------------------------------------------------------------------------
# 1. whoosh
# ----------------------------------------------------------------------------
_WHOOSH_PEAK = {"lr": 0.56, "rl": 0.56, "up": 0.72, "down": 0.30}


def whoosh(dur: float = 0.45, direction: str = "lr", intensity: float = 1.0) -> np.ndarray:
    """Filtered-noise swoosh with doppler-like filter sweep and pan motion.

    direction 'lr' / 'rl': object passes left->right (or right->left); the
    filter opens as it approaches and closes as it leaves.
    'up': rising sweep, builds to a late peak; 'down': early peak, falling.
    intensity (0.2-2): brightness, low 'whomp' body and air hiss.  Output is
    always normalised to -3 dBTP; intensity changes character, not level.
    Length is exactly `dur` seconds; sync point = anchor('whoosh', ...)."""
    if direction not in _WHOOSH_PEAK:
        raise ValueError("direction must be 'lr', 'rl', 'up' or 'down'")
    dur = max(0.08, float(dur))
    I = float(np.clip(intensity, 0.2, 2.0))
    n = _n(dur)
    u = np.arange(n) / n
    rng = _rng(f"whoosh|{direction}|{dur:.4f}|{I:.3f}")
    up = _WHOOSH_PEAK[direction]

    # proximity curve: asymmetric Lorentzian (distance of a passing object)
    if direction in ("lr", "rl"):
        wr, wf = 0.17, 0.13
    elif direction == "up":
        wr, wf = 0.24, 0.09
    else:
        wr, wf = 0.07, 0.22
    w = np.where(u < up, wr, wf)
    prox = 1.0 / (1.0 + ((u - up) / w) ** 2)
    taper = (np.sin(0.5 * np.pi * np.clip(u / 0.06, 0, 1)) ** 2 *
             np.sin(0.5 * np.pi * np.clip((1 - u) / 0.10, 0, 1)) ** 2)
    amp = prox ** 1.4 * taper

    # filter trajectories
    if direction in ("lr", "rl"):
        bright = prox
    elif direction == "up":
        bright = 0.15 + 0.85 * u ** 1.3 * (0.5 + 0.5 * prox)
    else:
        bright = 0.15 + 0.85 * (1 - u) ** 1.3 * (0.5 + 0.5 * prox)
    bright = np.clip(bright, 0, 1)
    fc_body = 500.0 * (1.0 + (6.0 + 3.0 * I) * bright ** 1.2)      # ~500 -> 5k
    fc_swish = 380.0 * (1.0 + (3.2 + 1.2 * I) * bright ** 1.1)     # ~380 -> 2k

    # pan / width
    if direction == "lr":
        pan = -0.85 + 1.7 / (1.0 + np.exp(-(u - up) / 0.085))
    elif direction == "rl":
        pan = 0.85 - 1.7 / (1.0 + np.exp(-(u - up) / 0.085))
    else:
        pan = np.zeros(n)
    width = 0.25 + 0.55 * prox

    nz = _stereo_noise(n, rng, width)
    body = _svf(_pinkish(nz), fc_body, 0.75, "lp") * 0.8
    swish = _svf(nz, fc_swish, 1.6, "bp") * 1.1
    air = _filt(_stereo_noise(n, rng, 0.8), "highpass", 6500.0, 4)
    air *= (prox ** 3)[:, None] * 0.10 * I
    low = _filt(_pinkish(rng.standard_normal(n)), "lowpass", 170.0, 4)
    low = _center(low * prox ** 2 * 0.9 * I ** 1.5)

    sig = (body + swish) * amp[:, None] + air * taper[:, None]
    # equal-power pan applied on mid; keep some of the side for width
    mid = 0.5 * (sig[:, 0] + sig[:, 1])
    side = 0.5 * (sig[:, 0] - sig[:, 1])
    th = (pan + 1.0) * np.pi / 4.0
    gl, gr = np.cos(th) * np.sqrt(2), np.sin(th) * np.sqrt(2)
    out = np.stack([mid * gl + side, mid * gr - side], axis=1)
    out += low * taper[:, None]
    out += 0.16 * _reverb(out, room=0.28, damp=0.6, hp=300, lp=7000)
    out = _mono_low(out, 150)
    return _finish(out, length=n, fade_in=0.004, fade_out=0.012)


# ----------------------------------------------------------------------------
# 2. slash
# ----------------------------------------------------------------------------
_SLASH_HIT = 0.065


def slash(dur: float = 0.35) -> np.ndarray:
    """Sharp diagonal blade swipe: fast rising bright swish panned L->R that
    culminates in a cut transient + a short metallic 'shing' tuned to F minor
    (F6 C7 C8 F8 Ab8 C9 partials).  Exact length `dur`; sync at 0.065 s."""
    dur = max(0.15, float(dur))
    n = _n(dur)
    t = _t(n)
    rng = _rng(f"slash|{dur:.4f}")
    th = _SLASH_HIT

    # swish: band-pass noise sweeping 1.5k -> 7k (into the cut) -> 9k
    fc = np.where(t < th, 1500.0 * (7000.0 / 1500.0) ** (t / th),
                  7000.0 + 2000.0 * (1 - np.exp(-(t - th) / 0.04)))
    env = np.where(t < th, (t / th) ** 2.5, np.exp(-(t - th) / 0.032))
    pan = np.clip(-0.75 + 1.5 * t / 0.12, -0.75, 0.75)
    nz = _stereo_noise(n, rng, 0.35)
    swish = _svf(nz, fc, 2.2, "bp") * 1.3
    body = _svf(nz, np.minimum(fc * 0.45, 3500.0), 0.7, "lp") * 0.55
    sw = (swish + body) * env[:, None]
    mid = 0.5 * (sw[:, 0] + sw[:, 1])
    side = 0.5 * (sw[:, 0] - sw[:, 1])
    a = (pan + 1) * np.pi / 4
    out = np.stack([mid * np.cos(a) * 1.414 + side, mid * np.sin(a) * 1.414 - side], 1)

    # cut transient: 3 ms bright noise snap at the hit
    k = _n(0.012)
    snap = _filt(rng.standard_normal(k), "highpass", 3500.0, 2)
    snap *= np.exp(-_t(k) / 0.0015) * _ramp_in(k, 0.0003)
    _place(out, _pan(snap * 0.55, 0.35), _n(th))

    # shing: tuned metallic partials with slow beating pairs (ring)
    ring_n = n - _n(th - 0.004)
    tr = _t(ring_n)
    ring = np.zeros((ring_n, 2))
    partials = [("F6", 0.30, 0.090), ("C7", 0.16, 0.070), ("C8", 0.07, 0.070),
                ("F8", 0.12, 0.090), ("Ab8", 0.09, 0.075), ("C9", 0.06, 0.060)]
    for i, (nm, amp, tau) in enumerate(partials):
        f = _note(nm, "slash")
        for det, p in ((0.9985, -0.5), (1.0018, 0.5)):
            s = np.sin(2 * np.pi * f * det * tr + 0.7 * i) * np.exp(-tr / tau)
            ring += _pan(s * amp * 0.5, p * 0.6 + 0.25)
    ring *= _ramp_in(ring_n, 0.003)[:, None]
    _place(out, ring * 0.55, _n(th - 0.004))

    # displaced-air low whump (gives the slice some weight)
    low = _filt(rng.standard_normal(n), "lowpass", 260.0, 4)
    lenv = np.where(t < th, (t / th) ** 2, np.exp(-(t - th) / 0.045))
    out += _center(low * lenv * 0.9)

    out += 0.22 * _reverb(out, room=0.45, damp=0.35, hp=600, lp=10000)
    out = _mono_low(out, 150)
    return _finish(out, length=n, fade_in=0.003, fade_out=0.05)


# ----------------------------------------------------------------------------
# 3. impact
# ----------------------------------------------------------------------------
_IMPACT = {
    "big": dict(f0=120.0, f1=38.0, glide=0.075, sub_tau=0.40, sub_amp=1.0,
                body="F2", body_tau=0.17, body_amp=0.75, punch_tau=0.045,
                crack_tau=0.030, crack_tail=0.16, crack_amp=0.55, click=0.45,
                drive=1.8, room=0.78, wet=0.34, length=2.0),
    "medium": dict(f0=140.0, f1=48.0, glide=0.05, sub_tau=0.24, sub_amp=0.85,
                   body="C3", body_tau=0.11, body_amp=0.75, punch_tau=0.035,
                   crack_tau=0.022, crack_tail=0.10, crack_amp=0.50, click=0.40,
                   drive=1.5, room=0.62, wet=0.26, length=1.2),
    "small": dict(f0=190.0, f1=70.0, glide=0.03, sub_tau=0.11, sub_amp=0.6,
                  body="F3", body_tau=0.065, body_amp=0.8, punch_tau=0.022,
                  crack_tau=0.014, crack_tail=0.05, crack_amp=0.45, click=0.35,
                  drive=1.3, room=0.40, wet=0.18, length=0.7),
}


def impact(size: str = "big") -> np.ndarray:
    """Slam: sub drop (sine 120->38 Hz for 'big'), tuned punchy body (F2 / C3 /
    F3), transient click (~3-5 kHz), short noise crack, reverb tail on
    everything but the sub.  Sub is soft-saturated so it reads on phones.
    Hit is at t=0; natural length (big ~1.8 s, medium ~1 s, small ~0.5 s)."""
    if size not in _IMPACT:
        raise ValueError("size must be 'big', 'medium' or 'small'")
    c = _IMPACT[size]
    n = _n(c["length"])
    t = _t(n)
    rng = _rng(f"impact|{size}")

    # sub drop
    f_sub = c["f1"] + (c["f0"] - c["f1"]) * np.exp(-t / c["glide"])
    sub = _sine(f_sub, n) * _decay(n, c["sub_tau"], 0.0015)
    sub = _sat(sub * 0.9, c["drive"]) * c["sub_amp"]

    # tuned body ("tom" with fast pitch drop) + low-mid thump noise
    fb = _note(c["body"], "impact")
    f_body = fb * (1.0 + 0.9 * np.exp(-t / 0.012))
    body = (_sine(f_body, n) + 0.25 * _sine(f_body * 1.5, n) * np.exp(-t / 0.05))
    body = _sat(body * _decay(n, c["body_tau"], 0.001) * 0.8, 1.4) * c["body_amp"]
    punch = _filt(_filt(rng.standard_normal(n), "lowpass", 420.0, 4), "highpass", 70.0, 2)
    punch *= _decay(n, c["punch_tau"], 0.001) * 0.9

    # crack (stereo noise) + click
    crack = _stereo_noise(n, rng, 0.6)
    crack = _filt(_filt(crack, "highpass", 450.0, 2), "lowpass", 7500.0, 2)
    cenv = np.exp(-t / c["crack_tau"]) + 0.12 * np.exp(-t / c["crack_tail"])
    crack *= (cenv * _ramp_in(n, 0.0008))[:, None] * c["crack_amp"]
    kn = _n(0.02)
    click = _filt(rng.standard_normal(kn), "bandpass", [2800.0, 5200.0], 2)
    click = click * np.exp(-_t(kn) / 0.0016) * _ramp_in(kn, 0.0003) * c["click"] * 2.2
    click_st = np.zeros((n, 2))
    _place(click_st, _center(click), 0)

    verb_in = _center(body * 0.6 + punch * 0.5) + crack + click_st
    wet = _reverb(verb_in, room=c["room"], damp=0.55, hp=220, lp=6000, predelay=0.012)
    wet = wet[:n]
    out = _center(sub + body + punch) + crack + click_st + c["wet"] * wet
    out = _compress(out, thr=-17.0, ratio=2.2, attack=10.0, release=180.0)
    out = _mono_low(out, 150)
    return _finish(out, fade_in=0.002, trim_db=-58.0)


# ----------------------------------------------------------------------------
# 4. reject
# ----------------------------------------------------------------------------
def reject() -> np.ndarray:
    """'Rejected clip' X-stamp: heavy low thud + rubber-stamp slap at t=0,
    then two short, muted square-ish blips C3 -> Ab2 (descending major third,
    both chord tones of F minor) a 16th note apart at 120 BPM."""
    n = _n(0.75)
    t = _t(n)
    rng = _rng("reject")

    f = 50.0 + 45.0 * np.exp(-t / 0.03)
    thud = _sat(_sine(f, n) * _decay(n, 0.13, 0.0015) * 0.9, 2.0)
    lowmid = _filt(rng.standard_normal(n), "bandpass", [110.0, 420.0], 2)
    lowmid *= _decay(n, 0.035, 0.001) * 1.1
    slap = _filt(rng.standard_normal(n), "bandpass", [700.0, 2200.0], 2)
    slap *= _decay(n, 0.009, 0.0005) * 0.55
    out = _center(thud + lowmid + slap)

    # muted error blips
    blip_len = 0.085
    bn = _n(blip_len)
    tb = _t(bn)
    benv = _ramp_in(bn, 0.004) * np.clip((blip_len - tb) / 0.022, 0, 1) ** 1.5
    for i, (nm, start, pan) in enumerate((("C3", 0.055, -0.12), ("Ab2", 0.055 + SIXTEENTH, 0.12))):
        fr = _note(nm, "reject")
        sq = _bl_wave(fr, bn, "square", max_h=31) + 0.7 * _bl_wave(fr * 2 ** (9 / 1200), bn, "square", max_h=31, phase0=0.4)
        sq = _filt(_filt(sq, "lowpass", 950.0, 4), "highpass", 80.0, 2)
        sq *= benv * (0.42 if i == 0 else 0.40)
        _place(out, _pan(sq, pan), _n(start))

    out += 0.15 * _reverb(out, room=0.32, damp=0.55, hp=250, lp=5000)
    out = _mono_low(out, 150)
    return _finish(out, fade_in=0.002, trim_db=-58.0)


# ----------------------------------------------------------------------------
# 5. check
# ----------------------------------------------------------------------------
def _bell_pluck(f: float, n: int, decay=0.42) -> np.ndarray:
    """Soft FM bell / mallet pluck (harmonic body + faint inharmonic strike)."""
    t = _t(n)
    idx = 1.1 * np.exp(-t / 0.035)
    car = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * t))
    y = car * np.exp(-t / decay)
    y += 0.16 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.15)
    y += 0.05 * np.sin(2 * np.pi * 3 * f * t) * np.exp(-t / 0.06)
    y += 0.035 * np.sin(2 * np.pi * 4.2 * f * t) * np.exp(-t / 0.012)
    return y * _ramp_in(n, 0.003)


def check() -> np.ndarray:
    """Positive confirm 'ding': two soft bell plucks F5 then C6 (a fifth up,
    tonic -> dominant of F minor) a 16th note apart, gentle plate + echo."""
    n = _n(1.3)
    out = np.zeros((n, 2))
    for nm, start, pan, amp in (("F5", 0.0, -0.22, 0.9), ("C6", SIXTEENTH, 0.22, 0.8)):
        f = _note(nm, "check")
        s = _bell_pluck(f, n - _n(start))
        _place(out, _pan(s * amp, pan), _n(start))
    echo = np.zeros_like(out)
    _place(echo, out[:, ::-1] * 0.14, _n(2 * SIXTEENTH))       # 8th-note echo, swapped
    out += _filt(echo, "lowpass", 4500.0, 2)
    out += 0.24 * _reverb(out, room=0.55, damp=0.45, hp=400, lp=8000, predelay=0.01)
    return _finish(out, fade_in=0.002, trim_db=-58.0)


# ----------------------------------------------------------------------------
# 6. pop
# ----------------------------------------------------------------------------
POP_BASE = "Ab5"


def pop(pitch_semitones: int = 0) -> np.ndarray:
    """UI card pop: rising 'bloop' sine + small harmonic + low thock + soft
    click.  Base pitch Ab5; offsets 0/+2/+4 -> Ab5/Bb5/C6 (all in F minor).
    In-key offsets are the Ab-major degrees (0 2 4 5 7 9 11 12, -1 -3 -5 ...)."""
    s = int(round(pitch_semitones))
    m = note_midi(POP_BASE) + s
    if not in_key(m):
        warnings.warn(f"pop({s}) -> {midi_name(m)} is outside {KEY_NAME}")
    else:
        NOTES_USED.setdefault("pop", set()).add(midi_name(m))
    f0 = midi_hz(m)
    n = _n(0.30)
    t = _t(n)
    rng = _rng(f"pop|{s}")

    f = f0 * (1.0 - 0.30 * np.exp(-t / 0.009))
    ph = _phase(f, n)
    body = np.sin(ph) * _decay(n, 0.055, 0.0008)
    body += 0.14 * np.sin(2 * ph) * _decay(n, 0.025, 0.0008)
    f_low = (f0 / 4) * (1.0 + 0.3 * np.exp(-t / 0.01))
    if in_key(m):
        NOTES_USED["pop"].add(midi_name(m - 24))
    thock = 0.30 * _sine(f_low, n) * _decay(n, 0.022, 0.0008)
    kn = _n(0.01)
    click = _filt(rng.standard_normal(kn), "bandpass", [2000.0, 6000.0], 2)
    click *= np.exp(-_t(kn) / 0.0008) * _ramp_in(kn, 0.0002) * 0.25
    y = body + thock
    y[:kn] += click
    out = _center(y)
    out += 0.12 * _reverb(out, room=0.3, damp=0.5, hp=350, lp=7000, predelay=0.006)
    return _finish(out, fade_in=0.002, trim_db=-56.0)


# ----------------------------------------------------------------------------
# 7. tick
# ----------------------------------------------------------------------------
def tick() -> np.ndarray:
    """Tiny UI click (timeline ruler / playhead start): short modal 'plastic'
    click tuned to F6/C7 with a faint HF snap.  ~40 ms."""
    n = _n(0.045)
    rng = _rng("tick")
    modes = [(_note("F6", "tick"), 1.0, 0.0085), (_note("C7", "tick"), 0.35, 0.0045),
             (_note("F8", "tick"), 0.12, 0.0022)]
    y = _modal_click(modes, n, rng, noise_amp=0.35, noise_hp=6500.0, noise_tau=0.0005)
    out = _pan(y, 0.0)
    out[:, 1] = np.roll(out[:, 1], 6)        # 0.12 ms skew: tiny width, mono-safe
    out[:6, 1] = 0.0
    return _finish(out, length=n, fade_in=0.002, fade_out=0.006)


# ----------------------------------------------------------------------------
# 8. scrub
# ----------------------------------------------------------------------------
def scrub(dur: float) -> np.ndarray:
    """Playhead scrub bed: soft rapid ruler ticks locked to 32nd notes at
    120 BPM (16 / s, accent every 4th = 8th notes) moving L->R, over a faint
    'tape-scrub' texture (band-passed noise whose pitch/level follow a
    wandering scrub speed).  Exact length `dur`."""
    dur = max(0.03, float(dur))
    n = _n(dur)
    t = _t(n)
    rng = _rng(f"scrub|{dur:.4f}")
    out = np.zeros((n, 2))

    rate = 2.0 / SIXTEENTH                # 16 ticks per second (32nd notes)
    kn = _n(0.03)
    f_a = _note("C7", "scrub")
    f_b = _note("F6", "scrub")
    times = np.arange(0.0, max(dur - 0.012, 0.0), 1.0 / rate)
    for i, tt in enumerate(times):
        acc = (i % 4 == 0)
        det = 1.0 + rng.uniform(-0.012, 0.012)
        amp = (1.0 if acc else rng.uniform(0.45, 0.65))
        modes = [((f_b if acc else f_a) * det, 1.0, 0.0035 if acc else 0.0025),
                 ((f_a if acc else f_a * 2) * det, 0.25, 0.0015)]
        k = _modal_click(modes, kn, rng, noise_amp=0.15, noise_hp=5000.0, noise_tau=0.0004)
        k = _filt(k, "lowpass", 7000.0, 2)
        pan = -0.4 + 0.8 * (tt / max(dur, 1e-6))
        _place(out, _pan(k * amp, pan), _n(tt))

    # tape texture: wandering 'speed' drives filter pitch and level
    m = max(4, int(dur * 40))
    walk = np.cumsum(rng.standard_normal(m))
    walk = (walk - walk.min()) / (np.ptp(walk) + 1e-9)
    speed = np.interp(t, np.linspace(0, dur, m), 0.45 + 0.75 * walk)
    speed = _filt(speed - speed.mean(), "lowpass", 8.0, 2) + speed.mean()
    nz = _stereo_noise(n, rng, 0.45)
    tex = _svf(nz, 650.0 * speed, 3.0, "bp") + 0.6 * _svf(nz, 1700.0 * speed, 4.0, "bp")
    flutter = 0.75 + 0.25 * np.cos(2 * np.pi * rate * t)
    tex *= (0.13 * speed * flutter)[:, None]
    out += tex
    out += 0.10 * _reverb(out, room=0.25, damp=0.6, hp=400, lp=6000)
    edge = min(0.02, dur / 4)
    return _finish(out, length=n, fade_in=edge, fade_out=min(0.03, dur / 4))


# ----------------------------------------------------------------------------
# 9. riser
# ----------------------------------------------------------------------------
def riser(dur: float = 1.5) -> np.ndarray:
    """Tension riser ending exactly at the last sample (align array end with
    the hit).  Layers: noise through an opening low-pass (220 Hz -> 10 kHz) +
    resonant whistle; an F power-fifth (F3+C4 -> F5+C6, 3 detuned saws per
    note, spread L/R) gliding up two octaves; tremolo accelerating 4 -> 16 Hz
    (8th -> 32nd notes at 120 BPM); widening stereo; reverb."""
    dur = max(0.3, float(dur))
    n = _n(dur)
    u = np.arange(n) / n
    t = _t(n)
    rng = _rng(f"riser|{dur:.4f}")

    # noise layer
    nz = _stereo_noise(n, rng, 0.2 + 0.7 * u)
    fc = 220.0 * 45.0 ** (u ** 1.35)
    noise = _svf(nz, fc, 0.9, "lp")
    whistle = _svf(nz, 300.0 * 20.0 ** (u ** 1.2), 6.0, "bp") * 0.45
    namp = u ** 2.2 * _ramp_in(n, 0.05)
    noise = (noise + whistle) * namp[:, None] * 0.9

    # tonal layer: power fifth gliding up two octaves, ending on F5 + C6
    F3 = _note("F3", "riser")
    _note("C4", "riser"); _note("F5", "riser"); _note("C6", "riser")
    semis = 24.0 * u ** 1.15
    tone = np.zeros((n, 2))
    for ratio, lvl in ((1.0, 1.0), (1.5, 0.7)):
        for cents, pan in ((-9.0, -0.6), (0.0, 0.0), (9.0, 0.6)):
            f = F3 * ratio * 2 ** ((semis + cents / 100.0) / 12.0)
            ph0 = rng.uniform(0, 2 * np.pi)
            v = _bl_wave(f, n, "saw", max_h=24, phase0=ph0) * lvl / 3.0
            tone += _pan(v, pan * (0.3 + 0.7 * u))
    tone = _svf(tone, 450.0 * 15.0 ** u, 1.0, "lp")
    trate = 4.0 * 4.0 ** u                                 # 4 -> 16 Hz
    tph = 2 * np.pi * np.cumsum(trate) / SR
    trem = 1.0 - (0.15 + 0.4 * u) * (0.5 + 0.5 * np.sin(tph))
    tone *= (u ** 1.8 * trem * 0.55)[:, None]

    out = noise + tone
    out += 0.25 * _reverb(out, room=0.7, damp=0.5, hp=300, lp=9000)
    out = _mono_low(out, 150)
    # finish but keep the end at full level (only a 3 ms click-guard fade)
    return _finish(out, length=n, fade_in=0.03, fade_out=0.003)


# ----------------------------------------------------------------------------
# 10. downlifter
# ----------------------------------------------------------------------------
def downlifter(dur: float = 1.0) -> np.ndarray:
    """Falling sweep/tail (place at a hit): wide noise through a closing
    low-pass (9 kHz -> 150 Hz) + an F4/C5 fifth gliding down two octaves
    (-> F2/C3), narrowing stereo, reverb.  Exact length `dur`."""
    dur = max(0.2, float(dur))
    n = _n(dur)
    u = np.arange(n) / n
    rng = _rng(f"downlifter|{dur:.4f}")

    nz = _stereo_noise(n, rng, 0.85 - 0.6 * u)
    fc = 9000.0 * (150.0 / 9000.0) ** (u ** 0.7)
    noise = _svf(_pinkish(nz), fc, 1.1, "lp")
    noise *= ((1 - u) ** 1.6 * _ramp_in(n, 0.004))[:, None] * 0.9

    F4 = _note("F4", "downlifter")
    _note("C5", "downlifter"); _note("F2", "downlifter"); _note("C3", "downlifter")
    semis = -24.0 * u ** 0.8
    tone = np.zeros((n, 2))
    for ratio, lvl, pan in ((1.0, 1.0, -0.25), (1.5, 0.6, 0.25)):
        f = F4 * ratio * 2 ** (semis / 12.0)
        v = _bl_wave(f, n, "saw", max_h=12) * lvl
        tone += _pan(v, pan)
    tone = _svf(tone, 3000.0 * (200.0 / 3000.0) ** u, 0.9, "lp")
    tone *= ((1 - u) ** 2.2 * _ramp_in(n, 0.004))[:, None] * 0.35

    out = noise + tone
    out += 0.28 * _reverb(out, room=0.65, damp=0.5, hp=250, lp=8000)
    out = _mono_low(out, 150)
    return _finish(out, length=n, fade_in=0.003, fade_out=min(0.08, dur * 0.1))


# ----------------------------------------------------------------------------
# 11. shimmer
# ----------------------------------------------------------------------------
def shimmer(dur: float = 1.2) -> np.ndarray:
    """Sparkly logo-reveal texture: a cloud of glassy bell grains drawn from
    F-minor(add9) chord tones (C6..F7, rising pool), soft F5/C6/F6 bell pad,
    glittering >8 kHz air, 16th / dotted-16th cross echoes and a bright room.
    Exact length `dur`; densest around 0.1-0.4 s."""
    dur = max(0.4, float(dur))
    n = _n(dur)
    t = _t(n)
    u = t / dur
    rng = _rng(f"shimmer|{dur:.4f}")
    out = np.zeros((n, 2))

    early = ["C6", "Eb6", "F6", "Ab6", "C7"]
    late = ["F6", "G6", "Ab6", "C7", "Eb7", "F7"]
    pool_e = [_note(x, "shimmer") for x in early]
    pool_l = [_note(x, "shimmer") for x in late]
    t_end = 0.72 * dur
    att = 0.10
    peak_rate = 60.0
    tt = 0.0
    while True:
        tt += rng.exponential(1.0 / peak_rate)
        if tt >= t_end:
            break
        dens = (np.sin(0.5 * np.pi * min(tt / att, 1.0)) ** 2 *
                np.exp(-max(tt - att, 0.0) / (0.3 * dur)))
        if rng.uniform() > dens:
            continue
        pool = pool_e if rng.uniform() > tt / t_end else pool_l
        f = pool[rng.integers(len(pool))]
        tau = rng.uniform(0.03, 0.14)
        gl = min(_n(5 * tau), n - _n(tt))
        if gl < 32:
            continue
        tg = _t(gl)
        a = rng.uniform(0.3, 1.0) * (1400.0 / f) ** 0.7
        g = (np.sin(2 * np.pi * f * tg) * np.exp(-tg / tau) +
             0.10 * np.sin(2 * np.pi * 2.0 * f * tg) * np.exp(-tg / (0.5 * tau)) +
             0.05 * np.sin(2 * np.pi * 2.76 * f * tg) * np.exp(-tg / (0.3 * tau)))
        g *= _ramp_in(gl, 0.0015) * a * 0.32
        _place(out, _pan(g, rng.uniform(-0.85, 0.85)), _n(tt))

    # soft bell pad
    env = _ramp_in(n, 0.06) * (1 - u) ** 1.8
    vib = 1.0 + 0.0025 * np.sin(2 * np.pi * 5.0 * t)
    for nm, lvl, pan in (("F5", 0.16, -0.2), ("C6", 0.12, 0.2), ("F6", 0.07, 0.0)):
        f = _note(nm, "shimmer")
        out += _pan(_sine(f * vib, n) * env * lvl, pan)

    # air glitter
    air = _filt(_stereo_noise(n, rng, 0.9), "highpass", 8000.0, 4)
    spk = _filt(np.maximum(rng.standard_normal(n), 0) ** 3, "lowpass", 60.0, 2)
    spk /= spk.max() + 1e-9
    dshape = _ramp_in(n, att) * np.exp(-np.maximum(t - att, 0) / (0.3 * dur))
    air *= (spk * dshape * 0.10)[:, None]
    out += air

    # cross echoes (16th / dotted 16th) and room
    echo = np.zeros_like(out)
    _place(echo[:, 0:1], out[:, 1:2] * 0.32, _n(SIXTEENTH))
    _place(echo[:, 1:2], out[:, 0:1] * 0.32, _n(1.5 * SIXTEENTH))
    out += _filt(echo, "lowpass", 6000.0, 2)
    out += 0.40 * _reverb(out, room=0.82, damp=0.35, hp=500, lp=10000, predelay=0.015)
    fade = np.cos(0.5 * np.pi * np.clip((u - 0.72) / 0.28, 0, 1)) ** 2
    out *= fade[:, None]
    return _finish(out, length=n, fade_in=0.004, fade_out=0.02)


# ----------------------------------------------------------------------------
# 12. reverse_swell
# ----------------------------------------------------------------------------
_CYM_RATIOS = (1.0, 1.4823, 1.8000, 2.5464, 2.6303, 3.8968)   # TR-808-like inharmonic set


def reverse_swell(dur: float = 0.8) -> np.ndarray:
    """Reverse-cymbal swell into a hit, ending exactly at the array end.
    A synthetic crash (inharmonic square cluster + HF noise) and a soft F-minor
    triad (F4 Ab4 C5) are reverberated forward, then reversed so the reverb
    blooms *into* the hit.  Length exactly `dur`."""
    dur = max(0.2, float(dur))
    n = _n(dur)
    t = _t(n)
    rng = _rng(f"reverse_swell|{dur:.4f}")

    # forward crash
    metal = np.zeros((n, 2))
    for ch, det in ((0, 1.0), (1, 1.0035)):
        for r in _CYM_RATIOS:
            metal[:, ch] += _bl_wave(337.0 * r * det, n, "square", max_h=41,
                                     phase0=rng.uniform(0, 6.28))
    metal = _filt(_filt(metal, "highpass", 5200.0, 4), "lowpass", 13000.0, 2) * 0.35
    hiss = _filt(_stereo_noise(n, rng, 0.85), "highpass", 6500.0, 4) * 0.7
    mid = _filt(_stereo_noise(n, rng, 0.7), "bandpass", [900.0, 2600.0], 2) * 0.18
    e_hi = np.exp(-t / (0.22 * dur))
    e_mid = np.exp(-t / (0.32 * dur))
    crash = (metal + hiss) * e_hi[:, None] + mid * e_mid[:, None]
    crash *= _ramp_in(n, 0.002)[:, None]

    # forward soft triad (reversed it becomes a swelling chord bloom)
    chord = np.zeros((n, 2))
    for nm, pan in (("F4", -0.3), ("Ab4", 0.3), ("C5", 0.0), ("F3", 0.0)):
        f = _note(nm, "reverse_swell")
        chord += _pan(_bell_pluck(f, n, decay=0.35 * dur), pan)
    chord *= 0.16

    fwd = crash + chord
    fwd = fwd + 0.6 * _reverb(fwd, room=0.8, damp=0.45, hp=300, lp=11000)[:n]
    rev = fwd[::-1].copy()
    # make the very start of the swell silent and smooth
    rev *= (np.sin(0.5 * np.pi * np.clip(t / (0.35 * dur), 0, 1)) ** 2)[:, None]
    rev = _mono_low(rev, 150)
    return _finish(rev, length=n, fade_in=0.01, fade_out=0.003)


# ----------------------------------------------------------------------------
# sync points
# ----------------------------------------------------------------------------
def anchor(name: str, **kw) -> float:
    """Seconds into the returned array where the sound 'lands' (align this
    with the visual event).  e.g. start = event_time - anchor('whoosh', dur=.45)."""
    if name == "whoosh":
        return _WHOOSH_PEAK[kw.get("direction", "lr")] * kw.get("dur", 0.45)
    if name == "slash":
        return _SLASH_HIT
    if name == "riser":
        return kw.get("dur", 1.5)
    if name == "reverse_swell":
        return kw.get("dur", 0.8)
    if name == "check":
        return 0.0              # second (higher) note at +0.125 s
    if name == "shimmer":
        return 0.0
    return 0.0                  # impact, reject, pop, tick, scrub, downlifter


# ----------------------------------------------------------------------------
# verification / demo
# ----------------------------------------------------------------------------
def verify_key() -> dict:
    """Assert every recorded tuned note is in F minor.  Returns NOTES_USED."""
    for snd, notes in NOTES_USED.items():
        for nm in notes:
            assert in_key(note_midi(nm)), f"{snd}: {nm} not in {KEY_NAME}"
    return {k: sorted(v, key=note_midi) for k, v in NOTES_USED.items()}


def _kweight(x: np.ndarray) -> np.ndarray:
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    return signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=0), axis=0)


def lufs_momentary_max(x: np.ndarray) -> float:
    """Max BS.1770 momentary loudness (400 ms window, 10 ms hop)."""
    w = _n(0.4)
    pad = np.zeros((w, 2))
    y = _kweight(np.concatenate([pad, np.asarray(x, float), pad]))
    p = np.sum(y ** 2, axis=1)
    c = np.concatenate([[0.0], np.cumsum(p)])
    idx = np.arange(0, len(p) - w + 1, _n(0.01))
    ms = (c[idx + w] - c[idx]) / w
    return float(-0.691 + 10 * np.log10(max(ms.max(), 1e-12)))


_OCT = (63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)


def _octave_levels(x: np.ndarray) -> list:
    mono = np.sum(np.asarray(x, float), axis=1)
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2
    fr = np.fft.rfftfreq(len(mono), 1 / SR)
    tot = spec.sum() + 1e-20
    out = []
    for fc in _OCT:
        m = (fr >= fc / np.sqrt(2)) & (fr < fc * np.sqrt(2))
        out.append(10 * np.log10(spec[m].sum() / tot + 1e-12))
    return out


def _dominant_note(x: np.ndarray, lo=100.0, hi=3000.0, seconds=0.25) -> str:
    mono = np.sum(np.asarray(x, float), axis=1)[: _n(seconds)]
    nfft = 1 << 17
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono)), nfft))
    fr = np.fft.rfftfreq(nfft, 1 / SR)
    m = (fr >= lo) & (fr <= hi)
    f = fr[m][np.argmax(spec[m])]
    midi = 69 + 12 * np.log2(f / 440.0)
    r = int(round(midi))
    return f"{f:7.1f} Hz = {midi_name(r)}{(midi - r) * 100:+.0f}c {'in-key' if in_key(r) else 'OUT-OF-KEY'}"


def demo_sequence():
    """(label, callable) pairs in demo order."""
    return [
        ("tick()", lambda: tick()),
        ("scrub(1.0)", lambda: scrub(1.0)),
        ("whoosh(0.45,'lr')", lambda: whoosh(0.45, "lr")),
        ("whoosh(0.45,'rl')", lambda: whoosh(0.45, "rl")),
        ("whoosh(0.6,'up')", lambda: whoosh(0.6, "up")),
        ("whoosh(0.6,'down')", lambda: whoosh(0.6, "down")),
        ("whoosh(0.45,'lr',0.5)", lambda: whoosh(0.45, "lr", 0.5)),
        ("slash(0.35)", lambda: slash(0.35)),
        ("impact('small')", lambda: impact("small")),
        ("impact('medium')", lambda: impact("medium")),
        ("impact('big')", lambda: impact("big")),
        ("reject()", lambda: reject()),
        ("check()", lambda: check()),
        ("pop(0)", lambda: pop(0)),
        ("pop(2)", lambda: pop(2)),
        ("pop(4)", lambda: pop(4)),
        ("riser(1.5)", lambda: riser(1.5)),
        ("downlifter(1.0)", lambda: downlifter(1.0)),
        ("shimmer(1.2)", lambda: shimmer(1.2)),
        ("reverse_swell(0.8)", lambda: reverse_swell(0.8)),
    ]


def analyse(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    pk = float(np.max(np.abs(x)))
    tp = true_peak(x)
    band = _octave_levels(x)
    side = 0.5 * (x[:, 0] - x[:, 1])
    mid = 0.5 * (x[:, 0] + x[:, 1])
    corr = float(np.sum(x[:, 0] * x[:, 1]) /
                 (np.sqrt(np.sum(x[:, 0] ** 2) * np.sum(x[:, 1] ** 2)) + 1e-20))
    return dict(
        dur=len(x) / SR,
        peak_db=20 * np.log10(pk + 1e-12),
        tp_db=20 * np.log10(tp + 1e-12),
        lufs_m=lufs_momentary_max(x),
        dc=float(np.max(np.abs(np.mean(x, axis=0)))),
        edge=float(max(np.max(np.abs(x[0])), np.max(np.abs(x[-1])))),
        finite=bool(np.all(np.isfinite(x))),
        bands=band,
        corr=corr,
        side_db=10 * np.log10((np.sum(side ** 2) + 1e-20) / (np.sum(mid ** 2) + 1e-20)),
    )


def render_demo(out_dir: str = DEMO_DIR, gap: float = 0.4) -> str:
    """Render every sound in turn with `gap` s of silence; writes
    sfx_demo.wav, sfx_demo_spectrogram.png, sfx_demo_waveform.png and
    sfx_report.txt in out_dir.  Returns the report text."""
    os.makedirs(out_dir, exist_ok=True)
    seq = demo_sequence()
    gapz = np.zeros((_n(gap), 2), dtype=np.float32)
    parts = [gapz]
    rows = []
    t0 = gap
    for label, fn in seq:
        x = fn()
        assert x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == 2, label
        a = analyse(x)
        a["label"] = label
        a["start"] = t0
        if label.startswith(("pop", "check", "tick")):
            a["pitch"] = _dominant_note(x, seconds=0.12 if label.startswith("check") else 0.25)
        rows.append(a)
        parts += [x, gapz]
        t0 += len(x) / SR + gap
    demo = np.concatenate(parts, axis=0)
    wav = os.path.join(out_dir, "sfx_demo.wav")
    sf.write(wav, demo, SR, subtype="FLOAT")

    lines = []
    lines.append(f"KRS story SFX report - {KEY_NAME}, {BPM:.0f} BPM, {SR} Hz stereo float32")
    lines.append(f"Normalisation: true peak {PEAK_DB:+.1f} dBTP per sound; fades >= 2 ms; DC removed")
    lines.append("")
    hdr = (f"{'sound':24s} {'start':>6s} {'dur':>6s} {'peak':>7s} {'TP':>7s} "
           f"{'LUFS-M':>7s} {'DC':>8s} {'edge':>8s} {'L/R r':>6s} {'S/M':>6s}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for a in rows:
        lines.append(
            f"{a['label']:24s} {a['start']:6.2f} {a['dur']:6.3f} {a['peak_db']:7.2f} "
            f"{a['tp_db']:7.2f} {a['lufs_m']:7.1f} {a['dc']:8.1e} {a['edge']:8.1e} "
            f"{a['corr']:6.2f} {a['side_db']:6.1f}")
    lines.append("")
    lines.append("units: start/dur s; peak dBFS (sample); TP dBTP (4x oversampled); "
                 "LUFS-M = max momentary loudness (400 ms);")
    lines.append("       DC = max |channel mean|; edge = max |first/last sample|; "
                 "L/R r = channel correlation; S/M = side/mid energy dB")
    lines.append("")
    lines.append("Octave-band energy, dB relative to total (centre Hz):")
    lines.append(f"{'sound':24s} " + " ".join(f"{fc:>6d}" for fc in _OCT))
    for a in rows:
        lines.append(f"{a['label']:24s} " + " ".join(f"{v:6.1f}" for v in a["bands"]))
    lines.append("")
    lines.append("Pitch check (dominant spectral peak):")
    for a in rows:
        if "pitch" in a:
            lines.append(f"  {a['label']:22s} {a['pitch']}")
    lines.append("")
    lines.append(f"Tuned notes declared per sound (all asserted in {KEY_NAME}):")
    for snd, notes in verify_key().items():
        lines.append(f"  {snd:14s} {' '.join(notes)}")
    checks = [
        ("all finite", all(a["finite"] for a in rows)),
        ("true peak <= -1 dBTP", all(a["tp_db"] <= -1.0 for a in rows)),
        ("|DC| < 1e-4", all(a["dc"] < 1e-4 for a in rows)),
        ("edges ~0 (fades)", all(a["edge"] < 1e-3 for a in rows)),
    ]
    lines.append("")
    lines.append("Checks: " + ", ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in checks))

    png = os.path.join(out_dir, "sfx_demo_spectrogram.png")
    wpng = os.path.join(out_dir, "sfx_demo_waveform.png")
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav, "-lavfi",
                        "showspectrumpic=s=1600x600:legend=1", png], check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav, "-lavfi",
                        "showwavespic=s=1600x300:split_channels=1", "-frames:v", "1", wpng],
                       check=True)
        r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", wav, "-af",
                            "ebur128=peak=true", "-f", "null", "-"],
                           capture_output=True, text=True)
        summ = r.stderr[r.stderr.rfind("Summary:"):]
        get = lambda k: re.search(k + r":\s+(-?[\d.]+|-inf)", summ)
        I = get(r"I"); LRA = get(r"LRA"); TPk = get(r"Peak")
        lines.append(f"ffmpeg ebur128 on whole demo: integrated {I.group(1)} LUFS, "
                     f"LRA {LRA.group(1)} LU, true peak {TPk.group(1)} dBFS")
    except Exception as e:  # ffmpeg missing -> report still written
        lines.append(f"(ffmpeg analysis skipped: {e})")
    lines.append(f"Demo: {wav} ({len(demo) / SR:.2f} s)")
    report = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "sfx_report.txt"), "w") as fh:
        fh.write(report)
    return report


__all__ = ["whoosh", "slash", "impact", "reject", "check", "pop", "tick", "scrub",
           "riser", "downlifter", "shimmer", "reverse_swell", "anchor",
           "render_demo", "verify_key", "true_peak", "lufs_momentary_max",
           "SR", "BPM", "KEY_NAME", "NOTES_USED"]

if __name__ == "__main__":
    print(render_demo())
