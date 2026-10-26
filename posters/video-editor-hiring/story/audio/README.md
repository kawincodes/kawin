
## Sound effects — `sfx.py`

All of these sounds are made in code from oscillators and seeded noise. The module uses no samples, loops or downloaded audio. Output is deterministic: the same call always returns the same samples.
Every function returns a `float32` array shaped `(N, 2)` at 48 kHz. Each sound is true-peak normalised to **-3 dBTP** (8×-oversampled estimate; sample peak about -3 dBFS), has its DC removed and has raised-cosine fades of at least 2 ms at both ends.
Shared contract: **F minor** (F G Ab Bb C Db Eb), **120 BPM**. `verify_key()` checks that every tuned note is in the key, and rhythmic spacing uses 16th/32nd notes at 120 BPM.

```python
import sfx                                   # python -I; needs numpy, scipy, soundfile, pedalboard
x = sfx.whoosh(0.45, 'lr')                   # (N, 2) float32
start = event_t - sfx.anchor('whoosh', dur=0.45, direction='lr')   # align the sound's peak with the event
gain = 10 ** (sfx.SUGGESTED_GAIN_DB['whoosh'] / 20)               # starting fader level
sfx.MIN_DUR                                  # shortest dur per sound; shorter requests are clamped + warnings.warn
hit = sfx.over_music_hit(sfx.impact('big'))  # slam that lands ON a music drop / end hit / kick (HP 90 Hz)
st['fx'] *= sfx.fx_duck_gain(len(st['fx']), [(t_hit - 1.0, t_hit)])  # -6 dB under an SFX riser(1.0)
sfx.render_demo()                            # writes demo/sfx_demo.wav, demo/sfx_demo_spectrogram.png,
                                             # demo/sfx_demo_waveform.png, demo/sfx_report.txt
```

| call | length | sync point (`anchor`) | what it is |
|---|---|---|---|
| `whoosh(dur=.45, direction='lr'\|'rl'\|'up'\|'down', intensity=1)` | exactly `dur` (≥ 0.08 s) | measured envelope peak of that exact render (about 0.56·dur lr/rl, 0.72·dur up, 0.32·dur down) | Pink and white noise through a time-varying SVF. **lr/rl**: doppler-style filter that opens as the sound approaches and closes as it leaves, with an equal-power pan sweep from -0.85 to +0.85. **up/down**: a resonant sweep from 280 Hz to about 3 kHz (or back down). `intensity` (0.2–2) sets brightness, the low "whomp" and the air hiss. It does not change the output level. A -3 dB presence cut at 2.2 kHz (deeper above intensity 1) keeps it out of the hissy band. |
| `slash(dur=.35)` | exactly `dur` (≥ 0.15 s) | 0.065 s | Blade swipe. A band-pass swish sweeps 1.5→7 kHz while panning L→R. At the cut it gives a 3 ms snap, then a short metallic "shing" from in-key partials (F6 C7 C8 F8 Ab8 C9, each a slowly beating pair) and a light air whump. Energy peaks in the 8 kHz octave. |
| `impact('big'\|'medium'\|'small')` | about 1.23 / 0.74 / 0.43 s | 0 | Sub drop, dry and mono: a sine from 120 Hz down to **F1** (43.65 Hz, the pitch of the music's kicks and booms) for big, 140 Hz → F1 for medium, 190 Hz → **Db2** (69.3 Hz) for small. A tuned "tom" body with a fast pitch drop (F2 / C3 / F3 plus a strong octave), a 90–700 Hz punch, a 2.5–6 kHz click, a 0.5–9 kHz crack and a reverb tail. **Phone layer:** a saturated copy of the sub kept only above 150 Hz, a 150–400 Hz thump and a harder-driven body (see design notes). |
| `reject()` | about 0.58 s | 0 | X-stamp. A 100→52 Hz thud (plus its harmonics above 150 Hz), an F3 knock and a rubber-stamp slap at 0. Then two muted square-ish blips, **C4 → Ab3** (falling major third, both F-minor chord tones), one 16th apart, low-passed at 2 kHz. |
| `check()` | about 1.3 s | 0 (2nd note at +0.125 s) | Two soft FM bell plucks, **F5 → C6** (tonic → fifth), a 16th apart. Adds an 8th-note echo with sides swapped and a small plate. |
| `pop(pitch_semitones=0)` | 0.30 s | 0 | Card pop: a rising "bloop" sine, its 2nd harmonic, a low "thock" two octaves down, and a soft click. The base note is **Ab5**, so 0/+2/+4 give **Ab5/Bb5/C6** for cards 01/02/03. Offsets stay in key when they are Ab-major degrees (…-5 -3 -1 0 2 4 5 7 9 11 12). Any other offset raises a warning. |
| `tick()` | 45 ms | 0 | Tiny modal "plastic" click tuned to F6/C7 with a faint HF snap. |
| `scrub(dur)` | exactly `dur` (≥ 0.03 s) | 0 | Ruler ticks at 16/s (32nd notes at 120 BPM), with every 4th one accented. They pan L→R as the playhead moves, over faint band-passed "tape-scrub" noise whose pitch follows a wandering scrub speed. |
| `riser(dur=1.5)` | exactly `dur` (≥ 0.3 s) | **end of array** | Pink noise through a low-pass opening 220 Hz→10 kHz, with a rising high-pass so it lifts instead of rumbling, plus a resonant whistle. An F power fifth (F3+C4) built from 3 detuned saws per note (spread up to ±0.4) glides up two octaves and lands on **F5+C6** at 88 %, then holds. The tremolo is locked to the grid **counted back from the end**: 8ths, then 16ths for the last second, then 32nds for the last half second, with a gain peak on the last sample. Only a 3 ms click guard at the end, so the hit lands at full level. |
| `downlifter(dur=1.0)` | exactly `dur` (≥ 0.2 s) | 0 | Wide noise through a closing low-pass (9 kHz → 150 Hz) plus an F4/C5 fifth gliding down two octaves to F2/C3. The stereo narrows as it falls. |
| `shimmer(dur=1.2)` | exactly `dur` (≥ 0.4 s) | 0 (densest 0.1–0.4 s) | A cloud of glassy bell grains on F-minor(add9) tones C6…F7; the pool rises over time. Adds a soft F5/C6/F6 bell pad, very-high glitter pings (F8 Ab8 C9), 8–13 kHz air sparkle, 16th / dotted-16th cross echoes and a 1.8 s bright room. |
| `reverse_swell(dur=0.8)` | exactly `dur` (≥ 0.2 s) | **end of array** | A synthetic crash (an inharmonic 808-style square cluster plus HF noise) and a soft **F4 Ab4 C5** triad are reverberated forward and then reversed, so the reverb blooms *into* the hit. |

**Design notes**
- Every sound is layered from a transient, a body and a tail. Envelopes are shaped analytically (exponential and raised-cosine). Time-varying filters use a TPT state-variable filter, which stays stable under fast sweeps.
- Pedalboard's Freeverb rang at about 1.85 and 2.4 kHz on short rooms, which you can see in the spectrograms. Reverb now goes through `pedalboard.Convolution` instead, with a **synthetic** impulse response made of decorrelated noise. Lows, mids and highs decay at different rates, it has a 15 ms build-up and soft early reflections. That gives smooth, non-metallic tails. The wet return is band-limited (HP 200–600 Hz) so the lows stay dry.
- **Slams on phone speakers.** Saturating the sub alone did not work: the old big impact had 99 % of its energy below 120 Hz and lost 10.7 LU through a 150 Hz high-pass (a phone-speaker model). Each impact now has a layer a phone can play. A parallel copy of the sub is driven hard (biased tanh, so 2nd *and* 3rd harmonics) and kept only between 150 Hz and 1.2 kHz, so its harmonic series carries the pitch drop (the missing-fundamental effect). The body has a full-level octave and drive 3, so F2 → F3 C4 F4. A 150–400 Hz band-passed noise thump decays in 60–75 ms. For the big impact, crack and click are +6 dB and the crack decays over 45 ms. A 2×-oversampled transient soft clip rounds the first-cycle peaks where all the layers coincide, which gains about 1 dB before true-peak normalisation. `reject()` gets the same thud harmonics, a knock and slap 5.5 dB louder, and blips an octave up with a 2 kHz low-pass.
- **Mono-safe stereo.** Everything below 150 Hz is folded to mono. The demo sounds have an L/R correlation of 0.40–0.98, and the lowest across all lengths and intensities is 0.35 (up/down whooshes at intensity 0.2). `reverse_swell` used to be 0.13: its metal cluster is now a shared mid with the detuned copy on the side only at half level, its noise width is 0.45 and its reverb return is at 0.45 with the side ×0.6, which brings it to 0.66. `riser` noise width is capped at 0.6 and its saws pan ±0.4, which gives 0.54–0.59.
- No 2–4 kHz pile-up. Whooshes keep 17–18 % of their energy in 2–5 kHz (21 % at intensity 2, down from 26–32 %), and their third-octave peak is at 1–1.3 kHz. Air and glitter layers are band-limited to 6–14 kHz.

**Verified** (`render_demo()` writes all of this to `demo/sfx_report.txt`, with PASS/FAIL checks):
- Every sound is finite, true peak -3.00 dBTP (8× oversampled), |DC| < 1e-10, first and last samples about 0, L/R r ≥ 0.35 (riser 3 s 0.54, reverse_swell 1.5 s 0.66). The demo measures -16.5 LUFS integrated with a true peak of -3.0 dBTP (ffmpeg ebur128).
- Measured pitches: check F5/C6, pops Ab5/Bb5/C6, tick F6, all within ±3 cents. Reject notes: F3 Ab3 C4.
- Phone model, at `SUGGESTED_GAIN_DB`, as max LUFS-M full-band / after the 150 Hz high-pass: big -14.2 / **-17.6** (was -14.9 / -25.6), medium -16.4 / -19.1, small -18.6 / -19.8, reject -17.2 / -19.0 (was -24.5). The big slam is now the loudest SFX on the phone model. The slams are ordered big ≥ medium ≥ small both ways, and every impact is within 3.5 LU of its full-band level (big 3.4, medium 2.7, small 1.2).
- The riser's tremolo gain on the last sample is 0.0 dB for every length (it used to range from -6.5 to -0.4 dB). `anchor('whoosh', …)` equals the 20 ms-envelope peak of the render and is within 25 ms of a 40 ms-smoothed peak. Durations below `MIN_DUR` are clamped and raise a `UserWarning`. Repeated calls are bit-identical.

**Mixing:** `SUGGESTED_GAIN_DB` gives starting fader offsets, all ≤ 0 dB. Each one is `ROLE_LUFS_M[k]` minus the sound's measured max momentary loudness, rounded to 0.5 dB. The targets run from -14.3 LUFS-M for the big slam down to -27 for the scrub bed, and the report re-derives them and flags any mismatch. Big, medium and small are 0 / -0.5 / -4 dB, so they stay ordered on a phone too.
Pure-tone sounds (`check`, `shimmer`) are loud for their peak level, so they get -8.5 dB. Use `riser` and `reverse_swell` by aligning the array end with the hit. Pair them with `impact('big')` or `downlifter()` at that same moment. On a music drop or end hit, follow the layering rules below.

**Layering with the music modules.** Both music beds already put a kick, an F1 sub boom, a crash and (cinematic) an F2 taiko on the drop and on the end hit. Their `fx` stem also already holds a riser, a reverse cymbal and (tech-house) a downlifter at those moments.
- **Slams on a music hit: `over_music_hit(impact(size))`.** Use it for the drop, the end hit and any slam that lands on a groove kick. It applies a zero-phase high-pass with `LAYER_HP_HZ = 90` Hz (|Butterworth-4|², -6 dB at 90 Hz, 48 dB/oct), so the music owns the low end. The output keeps the same length and sync point, its true peak never exceeds the input's and it gets no make-up gain, so `SUGGESTED_GAIN_DB` still applies. Energy below 60 Hz falls 27 dB, full-band loudness falls 2.5 LU (big) / 1.2 LU (medium), and the phone band changes by only -0.1 LU. A plain `impact()` (sub included, settling on F1) is for slams with no music hit under them.
- **Why.** Two low layers that are not phase-locked beat or partly cancel. Measured below 90 Hz, impact('big') against the summed stems of both modules on the demo drop (1.8 s) and end hit (14.3 s), plus six groove kicks per module (16 placements):

  | big slam | tech drop | tech hit | cine drop | cine hit | worst of 16 (r / energy) |
  |---|---|---|---|---|---|
  | old sub, 120→38 Hz | r -0.10 / -0.34 dB | -0.14 / -0.45 dB | +0.00 / +0.01 dB | +0.07 / +0.26 dB | -0.15 / -0.64 dB (-1.17 dB in a 150–600 ms window) |
  | sub tuned to F1 only | +0.01 / +0.03 dB | +0.09 / +0.28 dB | -0.10 / -0.35 dB | -0.23 / **-1.03 dB** | -0.23 / -1.03 dB |
  | `over_music_hit()` (90 Hz) | -0.04 / -0.03 dB | -0.01 / -0.01 dB | -0.03 / -0.03 dB | -0.14 / -0.18 dB | **-0.14 / -0.18 dB** (window -0.15 dB) |

  Tuning to F1 removes the 5.6 Hz beat but only moves the cancellation, because the phase offset depends on each module's kick. A 65 Hz corner, at the low end of the audit's 60–70 Hz suggestion, still lost 0.66 dB on the cinematic end hit, where the F2 body cancels against the F2 taiko. That is why the default corner is 90 Hz. For `over_music_hit(impact('medium'))` the worst of the 16 placements is r -0.04 / -0.03 dB. Below 150 Hz the worst layered big slam is -0.26 dB, against -0.71 dB for the old one. A causal high-pass was rejected because it rotated the sub start against the click, raised the true peak 1.0–1.35 dB and cost 1.0–1.4 LU on a phone once the peak was brought back.
- **Transitions: one source per transition.** `MUSIC_FX_DUPLICATES = ('riser', 'reverse_swell', 'downlifter')` duplicate what the music `fx` stem plays into and out of the drop and the end hit. For each of those transitions, use either the music's version or the SFX one, not both at full level. Use the SFX versions freely where the music has no transition, for example a visual build that does not end on the drop or the end hit. If a visual needs the SFX version on a music transition, duck the fx stem under it with `fx_duck_gain(n, spans, depth_db=MUSIC_FX_DUCK_DB)`, where `MUSIC_FX_DUCK_DB = -6`. The spans are where the SFX arrays sit: riser and reverse_swell use `(hit - dur, hit)`, downlifter uses `(t, t + dur)`. The returned curve only touches samples inside a span. Its raised-cosine ramps (50 ms attack, 30 ms release, in dB) reach 0 dB exactly at a span end, so the music's own boom at the hit stays whole after a riser span. A downlifter span that starts on the hit ducks the boom by about 2.5 dB over its first 50 ms, so start that span about 0.15 s after the hit. On both modules' demo stems the fx stem drops 5.1–6.0 dB inside each span, and samples outside the spans are bit-identical. An SFX `riser(1.0)` at its suggested gain then sits 11–12 dB over the music fx stem, up from 5–7 dB.

## Music bed — `music_techhouse.py` ("Driving tech pulse")

An original tech-house bed. Every sound is synthesized in the module: band-limited additive oscillators, polyBLEP saws, seeded noise, envelopes and filters. No samples, loops or downloaded audio. Output is deterministic for a given `seed`.
Contract: **F minor** (F G Ab Bb C Db Eb), **120 BPM**, 4/4. The off-16ths swing about 10 ms late. Output is 48 kHz stereo float32 `(N, 2)`.

```python
import music_techhouse as mt                 # python -I; needs numpy, scipy, soundfile, pedalboard
st = mt.render(duration=17.5, drop_time=1.8, end_hit_time=14.5, seed=1)
st['drums'], st['bass'], st['music'], st['fx']   # float32 (round(duration*48000), 2), unmastered
st['meta'] = {'bpm': 120.0, 'key': 'F minor', 'beat_times': [...], 'drop_time': 1.8,
              'end_hit_time_snapped': 14.3,
              # extras: downbeat_times, chord_timeline, progression, scale, musical_check, stem_scale_db, ...}
mt.render_demo()   # demo/music_techhouse_demo.wav (stems summed, preview master -14 LUFS / <= -1 dBTP),
                   # demo/music_techhouse_demo_spectrogram.png, demo/music_techhouse_demo_report.json (with a `checks` block)
mt.self_test()     # or: python3 -I music_techhouse.py --selftest  -- renders SELF_TEST_CASES and asserts the contract
```

**Timing.** Beat *k* is at `drop_time + k*0.5 s`. A bar line falls exactly on `drop_time`; bars before it can be partial. `drop_time` is clamped to `0 .. duration-3 s` with a `warnings.warn`, and `meta['drop_time']` reports the value actually used (a `duration` under 3 s or a non-finite time raises `ValueError`). `end_hit_time` snaps to the nearest beat (ties round up). The snapped hit is at least 3 beats after the drop and leaves at least 1 s of tail. For the demo, 14.5 → **14.3 s**, which is beat 2 of a bar. All events are placed by `round(t*48000)`, so the drop kick lands on sample 86400.
**Intro start:** the intro begins on the last bar line at or before 0 s (`k0 = 4*floor(-drop/2 s)`, always inside the 2.5 s pre-roll), so pad and arp are already sounding at t=0 for every drop time. With drop = 2.37 the intro starts at -1.63 s, with 1.8 at -0.2 s, and with 0.4 at -1.6 s.

**Arrangement (written as data at the top of the file).**
| span | harmony | what plays |
|---|---|---|
| 0 → drop (intro/build) | Eb (VII); a drop later than 2 s gets an Ab (III) bar first | Audible from t=0, no kick. A detuned-saw pad low-passed from 520 Hz up to 2.3 kHz, with a "ghost" sidechain pulse on each beat. A low-passed pluck arp opening from 650 Hz to 2.4 kHz, played 3 dB under its drop level. Closed-hat 16ths in a crescendo. A snare build (8ths → 16ths → 32nds) whose body steps up G3-Ab3-Bb3-C4-Db4, one degree per beat. A noise + F3→F5 saw riser. A 50 ms breath before the drop. |
| drop → hit-2 beats | **Fm – Db – Ab – Eb** (i–VI–III–VII), one bar each, cycling | Four-on-the-floor kick: a sine-sweep body tuned to F1, a 120–300 Hz "knock" layer settling on F3, and a 3–5 kHz click. Clap on 2 & 4 into a short plate. Swung closed-hat 16ths plus offbeat open hats. A rolling 16th bass (root/root/octave, fifth on beat 4) with deep kick sidechain. A 3-3-2-accented 16th pluck arp (patterns A/B per bar) with filter envelope, 2-bar filter ramps and a dotted-8th ping-pong delay. A warm 5-voice detuned-saw pad (LP-24 ladder automation, HP 150 Hz, chorus, hall, sidechain pump). At the drop: chord stab (+2 dB over the end stab), crash, sub boom and a downlifter. Snare fill (Ab3-Bb3-C4) plus crash every 4th bar. From bar 5 on, an octave "lift" layer is added to the arp. |
| last 2 beats before the hit | Eb (VII) turnaround | Snare 16ths → 32nds crescendo (Ab3 → Db4) with no kick on the last beat. The bass walks Bb1-C2-Db2-Eb2, the arp runs upward, a reverse crash swells into the hit, then a 50 ms breath. |
| end hit → duration | **Fm9** | Big kick, crash, sub boom and a hall-reverb impact. An Fm9 pluck chord stab whose echoes ring out. The Fm9 pad swells then its filter closes (3 kHz → 600 Hz). A long F1 bass decays under the end card. The last 0.25 s is a raised-cosine fade to exact zero. |

**Mix / design notes**
- Kick and bass are kept apart in two ways. The bass never plays on the kick's 16th, and the bass bus is ducked 85 % by a kick sidechain envelope. Every sub (bass notes, drop/end booms) starts **in phase with the ringing kick body** (shared pitch path / computed phase), so the sums are constructive.
- **Small-speaker translation.** The drop has to read on a phone, where nothing below ~150 Hz plays. The kick has a knock layer that follows the body's pitch sweep, shifted up to settle on F3. It is band-limited to 110–420 Hz and saturated together with the body, which adds 3.5 dB at 120–300 Hz without raising the kick's peak. A plain F3 sine knock cancelled against the sweep instead. The click is louder (0.75), the clap is +3 dB, the bass mid layer goes from -3 to -0.5 dB and the drop stab is +2 dB. The pad pumps 0.35 instead of 0.55. The build's top end comes down: riser -25 dB (was -22), riser tone -29 (was -27), snare-roll top velocity ×0.8, and the build arp sits 3 dB under its drop level. The sub is -2 dB (LV -6), and stems are high-passed at 30 Hz (4th order, was 22 Hz 2nd order). In the demo master, energy below 120 Hz falls from 74 % to 63 % and energy below 40 Hz from 0.75 % to 0.55 %. The 22–44 Hz octave still sits within 1 dB of the loudest band, because its top edge holds the F1 tonic (43.65 Hz) of the kick and bass.
- **Mono-safe music stem.** The arp/stab pluck plays one centred saw+square in both channels. The ±7-cent detuned pair is only a side layer at -6 dB (L = C + 0.5·D−, R = C + 0.5·D+). Chorus mix is 0.30 (was 0.45). Ping-pong taps are 70/30 (constant power) rather than hard L/R. The music bus has its side signal scaled ×0.7 above 1 kHz.
- **Tuned snare.** The snare's pitched body uses only F-minor degrees above G3 (`SNARE_STEPS` = G3 Ab3 Bb3 C4 Db4), and `check_arrangement()` checks this.
- **Alias-free metals.** Hats and crash are six band-limited squares (odd harmonics up to 20 kHz, additive via the sin(kx) recurrence). The old version used naive `sign(sin)` at 4× oversampling.
- All reverbs are **convolution with synthetic IRs**: decorrelated 3-band noise, highs decaying faster, build-up, early reflections. Pedalboard's Freeverb was measured ringing at 3.83 kHz and 5.77 kHz (+17 dB narrow peaks), so it is not used.
- Every event gets at least 2 ms raised-cosine fades (enforced centrally in `_Bus.add`). Gates use 2–4 ms ramps. Stems are high-passed at 30 Hz and DC-corrected.
- The four stems are scaled **together** (balance kept) so the hottest true peak is **-7 dBTP**. Gain staging lives in `LV` (dB per element) plus `KICK_KNOCK`, `KICK_CLICK`, `PUMP_PAD`, `MUSIC_SIDE_HI` and `STEM_HP`.
- **Speed.** `_additive` computes sin(k·θ) by recurrence, so there is one sin/cos per sample instead of one per harmonic. This is about 5× faster, and the result matches direct evaluation to within -240 dB. Harmonics are trimmed per 2048-sample block. The pluck cache keys on the cutoff quantised to 1/12 octave, with velocity applied outside the cache. The bass caches only its saw layer. Ladder automation blocks are 1024 samples. A render takes **about 9 s** (was 22 s). Four renders in parallel on this 4-core box take about 11 s each.

**Verified** (`render_demo()` → `checks` all true; `self_test()` passes; scratch re-runs of the audit scripts):
- `check_arrangement()` runs inside `render()` and raises on any violation. On the demo, 266 pitched notes and 28 tuned snare hits are all in F minor. Pad, arp and stab notes are chord tones of the chord sounding at their grid time. Every chord segment's first bass note is its root, and octave/fifth roles are correct. The drop is a beat and a downbeat, with a kick on its exact sample.
- **Audible from t=0:** the stem sum over the first 100 ms is -27.6 / -28.9 / -28.8 / -27.2 / -27.7 dBFS RMS for drop = 2.37 / 0.4 / 2.2 / 4.3 / 1.8 s. Before the fix these were -50.6 (2.37), -34.8 (0.4), -49.6 (2.2) and -52.0 dBFS (4.3), and the music stem was digitally silent until 0.37 / 0.40 / 0.20 / 0.30 s. The music stem is now non-zero from sample 0 in every case.
- **Drop vs the last second of the build** (stem sum, BS.1770 gated): full band **+5.0 LU** (was +3.6). HP150/4 **+2.4 LU** (was -0.8). HP400/4 +1.4 LU (was -1.9). Every groove bar is +2.2 LU on HP150 (was -0.7 to -1.2). After the preview master the figures are +4.9 / +2.3 / +1.3 LU.
- **Mono (music stem):** L/R correlation **0.78** (was 0.30). By band, 1–3 kHz is 0.80 (was 0.12) and 3–8 kHz is 0.90 (was 0.06). The worst 0.5 s window is 0.35 (was -0.01), and the mono fold-down loses 0.5 dB (was 1.9).
- **Metals:** inharmonic (alias) energy re harmonic energy is -79 to -86 dB in 7–24 kHz (was -13 to -20).
- **Robustness:** `render(17.5, 16.0, 17.0)` used to raise `AssertionError`. It now warns, clamps the drop to 14.5 and puts the hit at 16.5. The 12-case grid audit (drops 0 / 0.4 / 1.5 / 1.8 / 2.37 / 3.3, hits from 13.0 to 17.2, durations 15–18 s) shows no API problems. The same seed gives bit-identical stems.
- Demo stems: true peak drums -7.0 / bass -7.9 / music -9.5 / fx -14.5 dBTP. Loudness -21.3 / -24.6 / -21.3 / -28.9 LUFS. |DC| < 2e-9. First and last samples are 0, and all values are finite. The unmastered sum is -17.2 LUFS / -2.4 dBTP.
- Demo preview master (ffmpeg ebur128): **-14.0 LUFS integrated, LRA 1.4 LU, true peak -1.3 dBTP**, |DC| < 2e-6. Octave bands fall steadily above 500 Hz (2 kHz -11.6, 4 kHz -17.4, 8 kHz -20.6 dB re the loudest band), so nothing piles up at 2–4 kHz.

**For the mixer:** sum the stems and duck `music` (and some of `fx`) under voice or SFX as needed. `meta['beat_times']` / `downbeat_times` give the grid for snapping SFX. `fx` holds musical transitions (riser, booms, downlifter, reverse crash, end impact). It does not hold whooshes, which come from `sfx.py`. **Bus compression needs a side-chain high-pass of about 150 Hz and a threshold set relative to the programme loudness.** The old preview master used a full-band detector at a fixed -20 dBFS threshold. The kick pumped the whole bus, and the HP150 drop contrast went from +2.4 LU in the stems to -0.5 LU after mastering. `master_preview()` now uses `_glue_comp()` (detector HP150, threshold +3 dB re loudness, 2:1), which keeps +2.3 LU. Render time is about 9 s, so running it alongside the other modules is fine.

### 30 s story arrangement — `render_story()`

```python
st = mt.render_story(duration=30.0, drop1=2.0, breakdown=14.0, drop2=18.0, end_hit=26.0, seed=1)
st['drums'], st['bass'], st['music'], st['fx']   # same contract as render(): float32 (N, 2), unmastered, hottest true peak -7 dBTP
m = st['meta']   # render()'s keys (beat_times, downbeat_times, drop_time = drop1, end_hit_time_snapped = end_hit, chord_timeline, ...)
m['section_times']       # {'drop1': 2.0, 'breakdown': 14.0, 'drop2': 18.0, 'end_hit': 26.0, 'duration': 30.0} (same keys as music_cinematic)
m['sections']            # intro / grooveA / breakdown (+ 'dark', 'build' spans) / drop2 (+ 'turnaround') / end; m['section_boundaries']
m['fx_transitions']      # fx-stem spans for sfx.fx_duck_gain(): riser, downlifter, reverse_cymbal, sub_boom
mt.render_story_demo()   # demo/music_techhouse_story30_demo.wav (preview master), _spectrogram.png, _report.json (`checks` block)
mt.story_self_test()     # or: python3 -I music_techhouse.py --story   /   --story-selftest
```

`render()` is unchanged. The sha256 of all four stems and the meta is bit-identical to the pre-story module for (17.5, 1.8, 14.5), (17.5, 2.37, 14.5) and (16.0, 0.4, 13.0), and `self_test()` passes 7/7. Both functions now share one engine, `_render_events(events, N, seed, rng, auto)`. The story-only parts render only when their events exist, and story-only processing is switched on through `auto`. Each anchor snaps to the nearest bar line counted from `drop1` (2.0 s), with a `warnings.warn` if it moves. Each section must be at least 1 bar and the tail at least 1 s, otherwise `ValueError`. The arrangement is written as data: `STORY_SECTIONS` (per-section switches and levels for groove A and drop 2), `STORY_BREAKDOWN`, `BREAKDOWN_DARK` / `BREAKDOWN_BUILD`, `EXIT_FILL`, `STORY_LEAD`, `STORY_LV`, `STORY_END_HOLD` and `STORY_TAIL_PAD_FC`. It reuses the module's instruments. The new parts are a clock tick (`_clock`), a lead (`_lead`, `_lead_bus`), offbeat chord stabs (`_chord_bus`), a pulse shape in `_bass_note`, and an `Fmadd9` chord. `check_story()` runs on top of `check_arrangement()` inside `render_story()`.

| time (default anchors) | harmony | what plays |
|---|---|---|
| 0 → 2.0 intro | Eb (an Ab bar sits in the pre-roll) | `render()`'s intro, unchanged. Audible from t=0: pad, arp 3 dB down with its filter opening, hat crescendo, tuned snare roll, F3→F5 riser, 50 ms breath. |
| 2.0 → 14.0 groove A | Fm Db Ab Eb Fm Db | **Bars 1–4 (lighter):** open hats at 0.75, no arp lift, base filters. Fill and crash at 9.5 / 10.0. **Bars 5–6 (full):** octave lift on the arp, filters ×1.15 (pad) and ×1.2 (arp), open hats at 0.85. **Exit (13.5):** a falling Db4 C4 Ab3 G3 snare fill. |
| 14.0 → 16.0 breakdown, dark | Fm | Downbeat: one kick, a crash at 0.5 and a short sub boom (no music downlifter). Then no kick, clap, hats or arp. The pad glides from about 2.9 kHz down to a 480 Hz low-pass in 0.35 s, breathes ±0.15 octave per bar, +2 dB. **F1 sub pulse** in 8ths, accented on the off-8th, so the 14.5 / 15.0 / 15.5 stamps land in a dip. **Clock tick-tock:** F6 on the beat, C6 on the off-8th. The groove's arp echoes fade out. |
| 16.0 → 18.0 breakdown, build | Db → Eb (2 beats each) | The intro build again: pad 480 Hz → 2.3 kHz, arp back in 3 dB down with its filter opening 650 Hz → 2.4 kHz, hat crescendo, snare roll in 8ths → 16ths → 32nds, F3→F5 riser, reverse cymbal over the last 2 beats. Db2 / Eb2 bass pulses go from 8ths to 16ths in a crescendo. Ghost pump on the beats, 50 ms breath. |
| 18.0 → 26.0 drop 2 | Fm Db Ab Eb (last 2 beats: Eb turnaround) | Groove A's full setting from bar 1, plus: **lead counter-melody** C6 Ab5 C6 Eb6 / Db6 Ab5 F5 Ab5 / C6 Eb6 C6 Bb5 / Bb5 G5 Ab5 Bb5 (3-3-2 rhythm in 8ths, chord tones on beats 1 and 3; key-tracked saw with an octave-down body, dotted-8th ping-pong, hall). **Offbeat chord stabs** in the pad voicing, pumped by the kick. Snare layered on the claps, hats ×1.2, open hats at 0.9, pad +1.5 dB, filters ×1.3, kick and bass ×1.08. Ends with `render()`'s turnaround: bass walk, 32nd roll, reverse cymbal, breath. |
| 26.0 → 30.0 end | **Fm(add9)** | Big kick, crash, sub boom, impact and an Fm(add9) stab. The pad then **holds** the chord: +6 dB, envelope 1 − 0.45·q^2.5, low-pass 3.2 → 1.4 kHz. The F1 bass and reverb tails decay under it. The last 0.3 s is a raised-cosine fade to digital zero at 30.000 s. |

**For the mixer (story).** `meta['fx_transitions']` lists these spans:

- risers: 0–1.98 s and 16–17.98 s
- reverse cymbals: 17–18 s and 25–26 s
- downlifters: 2–4 s and 18–20 s
- sub booms at 2, 14, 18 and 26 s

The cue sheet's SFX `riser(1.0)` at 1 s and `riser(2.0)` at 16 s duplicate the music risers. Duck `fx` under them with `fx_duck_gain` over those spans, or leave the SFX risers out. The breakdown has no music downlifter, so the SFX `downlifter` at 13.75 s owns that transition, as in `music_cinematic`.

The `reject` stamps land in the sub's dips. Below 120 Hz the bass stem is -32.4 dBFS on the 14.5 / 15.0 / 15.5 s beats and -28.0 dBFS on the off-8ths. The SFX slams at 2.0, 18.0 and 26.0 s land on music kicks, so use `over_music_hit()` for them. The end chord holds under the end card. In the master it measures -18.2, -22.8 and -25.0 dBFS RMS over 27–28, 28–29 and 29–29.5 s.

**Verified** (`render_story_demo()` → 24 report `checks`, all true; `story_self_test()` passes 3/3). Section loudness is gated BS.1770 over each window, on the unmastered stem sum. HP150 is the 4th-order 150 Hz high-pass phone model. The centroid and the > 2 kHz share are from a Welch spectrum of the mono sum.

| section | window (s) | full LUFS | HP150 LUFS | centroid | energy > 2 kHz |
|---|---|---|---|---|---|
| intro | 0–2 | -25.9 | -26.0 | 1044 Hz | 8.5 % |
| groove A | 2–14 | -20.2 | -22.5 | 314 Hz | 2.1 % |
| breakdown | 14–18 | -25.3 | -26.8 | 334 Hz | 1.9 % |
| · dark bar / build bar | 14–16 / 16–18 | -26.4 / -24.2 | -29.8 / -24.9 | 135 / 661 Hz | 0.4 / 4.6 % |
| drop 2 | 18–26 | **-18.8** | **-20.4** | 387 Hz | 2.6 % |
| end hit + tail | 26–30 | -22.2 | -24.2 | 256 Hz | 1.8 % |

- **Section order (full / HP150):**
  - Groove A over the intro: +5.7 / +3.4 LU.
  - Drop 2 over groove A: **+1.48 / +2.12 LU** in the stems, +1.26 / +1.89 after the preview master.
  - Breakdown under groove A: -5.0 / -4.3 LU. Its dark bar: -6.2 / **-7.3 LU**, with a 135 Hz centroid and 0.4 % of its energy above 2 kHz.
  - The build climbs +2.2 / +5.0 LU over the dark bar.
  - In drop 2 the lead adds +0.54 / +0.81 LU and the offbeat stabs add +0.49 / +0.69 LU.
  - The end hit is the loudest moment: max momentary -15.9 LUFS, against -18.4 for groove A and -17.2 for drop 2.
- **Demo master:** ffmpeg ebur128 gives **-14.1 LUFS I, LRA 4.5 LU, -1.3 dBTP**. Octave bands re 63 Hz for drop 2: 1 kHz -5.2, 2 kHz -11.2, 4 kHz -17.6, 8 kHz -21.6 dB; groove A: -8.1 / -13.4 / -19.4 / -23.0 dB. No 2–4 kHz pile-up.
- **Anchors:** 2 / 14 / 18 / 26 s are all in `downbeat_times`, and `check_story()` puts a kick on each exact sample. Drums-stem broadband onsets are +1.44 / +1.33 / +1.42 / +1.46 ms, the 30 % point of every event's 2 ms raised-cosine edge. The breath before drop 1, drop 2 and the end hit is -85 / -84 / -84 dB. The breakdown has no breath (-25 dB, the exit fill).
- **Notes:** `check_arrangement()` + `check_story()` pass 584 pitched notes (481 events) and 56 tuned snare hits, all in F minor. Pad, arp, stab and chord-stab notes are chord tones, and bass roles are correct. In the audio, the out-of-key share of the music stem's 150 Hz–2 kHz energy is 0.1–3.8 % per section. The final F1 is 0.01 % out of key.
- **Tail:** the master goes from -9.6 dBFS RMS in the 400 ms after the hit, to -18.2 (27–28 s), -22.8 (28–29), -25.0 (29–29.5), -28.7 (29.5–29.7) and -88 in the last 10 ms. The last sample of the master and of every stem is 0.
- **Stems:** peak / true peak are drums -9.8 / -9.6, bass -11.5 / -11.4, music -7.0 / -7.0, fx -16.9 / -16.4 dBFS / dBTP. |DC| < 2e-10. Music L/R correlation is 0.77 and the worst 0.5 s window is 0.47 (the end chord). The first 100 ms are -30.8 dBFS RMS unmastered and -24.4 mastered, so the bed is audible from frame 0.
- **Clicks:** the audit's 2nd-difference scan finds 0 events on bass and fx. On music it finds 18, every one 0–8 ms after a note onset (arp and stab attacks after a quiet moment, the same as in `render()`). Nothing is unexplained, near the boundaries or anywhere else. Round 1 found kinks 69 ms into the breakdown pulses and in the final F1, at -32 dB re the sub in the 2–5 kHz range. Their cause is pedalboard's `Compressor`, whose peak ballistics switch at every waveform peak. The story bass bus now uses `_smooth_comp()` (25 ms RMS detector, C1-smooth gain) and `_peak_limit(smooth=True)`; `render()` keeps its old chain.
- **Rounds:**
  - Drop 2 over groove A, full band: +0.71 LU in R1 (louder lead and stabs) → +1.01 in R2 (kick and bass ×1.08, pad +1.5 dB) → +1.48 in R3.
  - R3 also caught a NaN in `_smooth_comp` that `nan_to_num` had turned into a silent bass stem. The report now checks that every stem has signal.
  - R4–R5: the end chord used to sink 11 LU under groove A by 29 s, so it now holds. The clock was 11 dB under the dark bar's content above 800 Hz. It is now 13.5 dB over the gaps between ticks in the 0.9–6 kHz band, and the dark bar's loudness is unchanged.
  - R6: the breakdown downlifter was removed and the pulse accent moved to the off-8th.
- **Robustness:** `story_self_test()` covers three cases:
  - the default anchors;
  - 14.04 / 17.97 / 26.02, snapped to 14 / 18 / 26 with warnings;
  - a 24 s render with drop1 = 0 and a 3-bar breakdown (2 dark bars and a build).

  The same seed gives bit-identical stems and meta. A render takes about 20 s.

## Music bed — `music_cinematic.py` ("Cinematic hybrid pulse")

An original trailer/showreel-style score. Everything is synthesized in the file: polyBLEP saws, sines, seeded noise, analytic envelopes, time-varying biquads, tanh drive and a convolution reverb built from a **synthetic** impulse response. It uses no samples or GM patches, and output is deterministic per `seed`.
**F minor, 120 BPM, 4/4.** The progression is **Fm – Db – Bbm – Csus4** (i–VI–iv–Vsus4, one chord per bar from the drop). The intro builds on a **Csus4** dominant pedal. The last bar before the end hit is the **Db → Eb** turnaround (VI–VII–i). The tail rings on **Fm(add9)**. The V is a sus4 chord, so every note stays in F natural minor with no E natural.

```python
import music_cinematic as mc                     # python -I; needs numpy, scipy, soundfile (+ ffmpeg for the demo)
st = mc.render(duration=17.5, drop_time=1.8, end_hit_time=14.5, seed=1)
st['drums'], st['bass'], st['music'], st['fx']  # float32 (N, 2), N = round(duration*48000), unmastered
st['meta']  # bpm, key, beat_times, drop_time, end_hit_time_snapped (+ downbeat_times, bar_times, meter, chord_timeline, ...)
mc.render_demo()   # demo/music_cinematic_demo.wav, _spectrogram.png, _waveform.png, _report.json
```

**Meta contract.** `drop_time` and `end_hit_time_snapped` are rounded to 9 decimals, the same as `beat_times`, so `meta['end_hit_time_snapped'] in meta['beat_times']` is an exact test (drop=2.37 gives 14.37, not 14.370000000000001). `downbeat_times` is the bar-line key shared with `music_techhouse.py`, so the mixer can swap styles without code changes. Here it equals `bar_times`, and it includes the 5/4 or 6/4 bar line of the last bar, so the drop and the end hit are both in it. A `drop_time` outside 0 .. duration-3 s, or an end hit whose nearest beat is out of range, is clamped with a `warnings.warn`. The clamped values are the ones reported in meta.

| time | what happens |
|---|---|
| 0 → drop | **Build.** Starts sounding at t=0: the first 8th note is at 0.05 s, and the pad is already sustaining from the pre-roll. A clock tick-tock plays 8ths (F8 on the beat panned left, C8 off the beat panned right). An 8th-note C pulse bass opens its low-pass from 550 to 850 Hz. Its octave-up saw and filter envelope put the pulse at 250 Hz–1.5 kHz from t=0, so phone speakers hear it. A Csus4 string pad is in at mid-range from t=0 (×0.85, LP 1.8 kHz) and opens to LP 3.6 kHz (×0.88). Heartbeat toms land on the beats, with a 16th tom fill and a hat crescendo over the last beat. The toms are choked 40 ms before the drop. A grid-locked noise riser (tremolo 8ths → 16ths → 32nds) and a reverse cymbal lead in. Sustained parts drop out for the last 16th (the "breath"). |
| **drop** | A bar line falls exactly on `drop_time`. **Hit:** big kick, F2 taiko, crash, F1 sub boom, and an Fm "braam" (4 detuned saws per note, pitch scoop, LP envelope). Then the full groove: kick (A/B patterns), processed claps on 2 and 4, 16th hats, the clock ticks, an 8th-note saw+sub ostinato bass on the chord roots (octave pickup on the last 8th), a 16th spiccato-style synth ostinato with a 3/16 ping-pong delay, and the supersaw string pad (high-passed at 150 Hz). Nothing pulls back at the drop: the pad goes to ×0.97 with LP 3.7 kHz, and the bass LP goes to 800 Hz (900 Hz from bar 5). Deep tom groove from bar 3. Tom fills close each 2-bar phrase. |
| drop + 4 bars | Second cycle: crash, a softer braam, a reverse-cymbal lead-in, and the bass, pad and ostinato filters open further. |
| last bar | **Turnaround** Db → Eb. If the hit is not 4·k beats after the drop, this bar becomes 5/4 or 6/4 (or a 3/4 bar is added), so the end hit is also a downbeat. The Eb braam swells, the bass drives in 16ths, a 16th tom fill plays with a clap roll, a riser and reverse cymbal build, then a one-16th breath. |
| **end hit** | `end_hit_time` snapped to the nearest beat (14.5 → **14.3** in the demo), reported in `meta['end_hit_time_snapped']`. Big kick, taiko, clap, crash, sub boom and braam. The arrangement then thins to a sustained **Fm(add9)** pad, a ringing F1 bass, the braam tail and a few fading clock ticks. Everything decays and fades to digital zero exactly at `duration` (the last 0.25 s is a raised-cosine fade). |

**Stems** share one common gain, so they are balanced as mixed. The hottest stem (drums) sits at -6.5 dBTP. Demo values: drums -6.5 / bass -11.1 / music -8.0 / fx -12.3 dBTP. Summed at unity they peak at -2.7 dBTP and measure -17.6 LUFS. `fx` holds musical transitions only (risers, reverse cymbals, sub booms). Whooshes, stamps and pops come from `sfx.py`.

**Mixing design.** The bass is sidechained to every kick (-14 dB, 120 ms release); the pad (-3 dB) and ostinato (-2.5 dB) duck lightly. The kick tail is tuned to Ab1 (52 Hz), clear of the F1 sub. Kick, toms, pad and braam are folded to mono below 120–150 Hz (mix side/mid below 150 Hz: -31 dB). Each bus is set by loudness while it plays (`BUS_LUFS`), not by peak. The ostinato is at -23 LUFS and the riser at -27 LUFS, so the groove sits above the end of the build. Tom fills, heartbeat toms and their room reverb are **choked** before the drop and the end hit. Each tom is cut with a 12 ms fade that ends 40 ms before the hit, and the processed fill bus is gated shut for those 40 ms. The toms that land on the hits are on a separate, ungated bus. The bass has an octave-up saw layer and drive, so the pulse still reads on phone speakers (energy from 250 Hz to 1 kHz).

**Verified** (`render_demo()` writes a `checks` block, and all are true):
- **Stems:** true peak ≤ -6 dBTP, |DC| < 1e-7 in the demo (< 1e-4 in every tested case), finite, first and last samples 0.
- **Demo:** ffmpeg ebur128 gives **-14.1 LUFS I, LRA 1.7 LU, -1.4 dBTP**.
- **Build → drop** (gated LUFS of the unmastered stem sum; the phone models are 4th-order high-passes at 150/250/400 Hz; values given as full / HP150 / HP250 / HP400). The groove is **+2.4 / +2.6 / +2.5 / +2.7 LU** over the last second of the build (it was +0.6 / +1.0 / +0.8 / +0.7). The drop bar is **+3.2 / +3.3 / +2.8 / +2.9 LU** over it (it was +1.7 / +2.2 / +1.7 / +1.5). The first second is **-21.4 / -25.4 / -26.5 / -28.7 LUFS** (it was -22.3 / -28.4 / -29.7 / -31.7), so on phones it sits 4.4 / 4.4 / 5.5 LU under the groove instead of 6.3 / 6.2 / 6.9. Full-band groove-over-build is +2.1 to +2.7 LU for drops at 1.5, 1.8, 2.37 and 3.3 s. The report has these under `section_loudness_LUFS` and `contrast_LU`.
- **Transients:** in the 30 ms before the drop, the drums stem is **-53 dB** (LP 200 Hz) and **-43 dB** (broadband) under the drop's peak. It was -10 dB, because the last fill tom rang into the drop. A 30 % onset detector now lands at +3.0 ms (LP 200, the kick's sine attack) and +0.9 ms (broadband) for every tested drop time. Before, it fired 13.7 ms early (LP 200) and 15.8 / 19.3 ms early (broadband) at drop = 1.5 / 3.3 s. At the end hit the breath is -27 dB (LP 200) and -19 dB (broadband, the clap roll's tail), with onsets at +2.8 / +0.9 ms. The report has these under `transients`.
- **Timing:** drop onset within 1 ms of `drop_time`. Drop and end hit are bar lines for drop/hit/duration combinations including drop=0, 0.4, 3.1 and 6.0 s and hits needing 4/4, 5/4 and 6/4 last bars.
- **Notes:** `check_score()` passed 523 notes. All pitches are in F minor. Pad, ostinato and braam notes are chord tones, and the bass is always the chord root.
- **Clicks:** none found. Bass and music HF above 14 kHz never exceed -70 dBFS, and every sound has ≥ 2 ms raised-cosine edges.
- **Spectrum:** demo octave bands fall steadily: 1 kHz -10.6, 2 kHz -15.6, 4 kHz -18.8, 8 kHz -19.6 dB re 63 Hz, so nothing piles up at 2–4 kHz. Hats sit above 7 kHz. The kick is 45–160 Hz plus a 3–5 kHz click.

### 30 s story arrangement — `render_story()`

```python
st = mc.render_story(duration=30.0, drop1=2.0, breakdown=14.0, drop2=18.0, end_hit=26.0, seed=1)
st['drums'], st['bass'], st['music'], st['fx']   # same stems contract as render(): float32 (N, 2), true peak <= -6 dBTP
m = st['meta']   # render()'s keys (beat_times, downbeat_times, drop_time = drop1, end_hit_time_snapped = end_hit, ...)
m['section_times']   # {'drop1': 2.0, 'breakdown': 14.0, 'drop2': 18.0, 'end_hit': 26.0, 'duration': 30.0}
m['sections']        # [{'name': 'intro', 't0': 0.0, 't1': 2.0}, ... grooveA, breakdown, drop2, end]
m['fx_transitions']  # where the fx stem has a riser / reverse cymbal / sub boom, for fx_duck_gain()
mc.render_story_demo()   # demo/music_cinematic_story30_demo.wav, _spectrogram.png, _waveform.png, _report.json
# or: python3 -I music_cinematic.py --story   (exit code 1 if any check fails)
```

`render()` is unchanged: its output is bit-identical to the version before `render_story()` was added, and `render_demo()` still passes all its checks. Each anchor snaps to the nearest bar line counted from `drop1` (every 2.0 s), with a `warnings.warn` if it moves. Every section keeps at least 1 bar and the tail keeps at least 1 s; a duration under 7 s raises `ValueError`. The arrangement is written as data at the top of the story block: `STORY_PLAN` (one dict per bar with kick pattern, toms, fill, energy, accent and layers, grouped in 4-bar phrases), `BREAKDOWN_*`, `SNARE_*`, `LEAD_MOTIF`, `OST_PATTERNS` and `STORY_AUTO` (filter and level automation as anchor breakpoints). It reuses the module's instruments. Four parts are new: a tuned snare (`_snare`), a sub drone (`_drone_bus`), a counter-melody lead (`_lead_bus`) and a wide octave pad (the `_pad_bus` voicing option). `check_story_score()` runs inside `render_story()` and raises on any violation.

| time (default anchors) | harmony | what plays |
|---|---|---|
| 0 → 2.0 intro | Csus4 | `render()`'s build, unchanged: audible from t=0, ticks, 8th pulse bass, heartbeat toms, riser, reverse cymbal, one-16th breath. |
| 2.0 → 14.0 groove A | Fm Db Bbm Csus4 Fm Db | **Phrase 1 (bars 1–4, lighter):** the drop hit, then no toms for 2 bars, ostinato ×0.85–0.95, pad 0.92–0.95, a small tom fill at the end of bar 2 and a **turnaround** in bar 4 (no kick on beat 4, 16th tom fill, clap roll, 16th bass). **Phrase 2 (from 10.0):** crash, braam and reverse-cymbal "lift", the tom groove, open hats, the ostinato's rising arpeggio pattern B, and the filters open further. **Exit (13.5):** a unison kick + clap + tom stop on beat 4. The pad filter falls 4.2 → 1.1 kHz, then kick and clap are choked 40 ms before the breakdown. |
| 14.0 → 16.0 breakdown (dark) | Fm | No kick, clap or hats. Downbeat: a taiko, a short F1 sub boom and a "dark" braam with its low-pass kept under about 0.7 kHz. Pad low-passed at 700 Hz, level 0.48. A **sub drone** on the chord root (F1) with a sidechain pump: -8 dB **on** each beat, swelling between beats, so stamps that land on the beat land in a dip. The clock tick-tock 8ths stay exposed. |
| 16.0 → 18.0 breakdown (build) | Db → Csus4 | Drone pumping in 8ths, heartbeat toms, then the 16th tom fill. A tuned **snare roll** in 8ths, 8ths, 16ths and 32nds (G3 Ab3 Bb3 C4, crescendo). Hats crescendo over the last 2 beats. The ostinato returns with its low-pass opening 420 Hz → 2 kHz. A 2 s riser and a reverse cymbal lead into 18.0. One-16th breath. |
| 18.0 → 26.0 drop 2 | Fm Db Bbm \| Db Eb | Groove A's full phrase pushed harder: kick ×1.1, clap and tom energy 1.12, bass +1.6 dB. Two layers are added: a **counter-melody lead** (brassy saw, doubled an octave below, 35 ms portamento, ping-pong delay; C5 F5 Ab5 G5 / F5 Eb5 Db5 / Db5 F5 Bb5 Ab5 / Ab5 F5 G5 Bb5) and a **wide octave pad** (the top 2 pad voices one octave up, ±24 cent spread). The last bar is `render()`'s Db → Eb turnaround with riser, rising braam, fill and clap roll. |
| 26.0 → 30.0 end | Fm(add9) | End hit as in `render()`: big kick, taiko, clap, crash, boom and braam. The lead lands on C6. Then the Fm9 pad, F1 bass, braam tail and fading ticks decay to about -9 dB. The **last 0.3 s** is a raised-cosine fade to digital zero at 30.000 s. |

**For the mixer (story).** `fx` already has a riser at 16–18 s and a reverse cymbal at 17–18 s into drop 2, and the intro riser at 0–2 s. The cue sheet's SFX `riser(2.0)` at 16 s and `riser(1.0)` at 1 s duplicate them, so duck `fx` under those SFX with `fx_duck_gain` using the spans in `meta['fx_transitions']` (one source per transition). There is no music downlifter at the breakdown, so the SFX `downlifter` at 13.75 s has that transition to itself. The breakdown is kept sparse for the `reject` and `check` stamps. Their thuds land in the drone's dips: below 120 Hz the bass stem is -34.6 dBFS on the 14.5 / 15.0 / 15.5 s beats and -29 dBFS between them.

**Verified** (`render_story_demo()` → report `checks`, all true; demo master = `master_preview`). Section loudness, gated BS.1770 over each window. HP150 is the 4th-order 150 Hz high-pass phone model; the report also has HP250 and HP400. The centroid is taken over 150 Hz–16 kHz.

| section | window (s) | full LUFS | HP150 LUFS | centroid | energy > 1 kHz |
|---|---|---|---|---|---|
| intro | 0–2 | -17.4 | -20.8 | 878 Hz | 3.8 % |
| groove A | 2–14 | -14.2 | -17.7 | 862 Hz | 4.5 % |
| · lighter phrase / full phrase | 2–10 / 10–14 | -14.3 / -14.2 | -17.8 / -17.7 | | |
| breakdown | 14–18 | -19.2 | -22.1 | 933 Hz | 5.1 % |
| · dark bar / build bar | 14–16 / 16–18 | -19.8 / -18.3 | -25.4 / -20.0 | 617 / 1034 Hz | 1.2 / 9.5 % |
| drop 2 | 18–26 | **-12.6** | **-15.5** | 968 Hz | 6.5 % |
| · without the turnaround bar | 18–24 | -12.7 | -15.6 | | |
| end hit + tail | 26–30 | -15.6 | -20.7 | 819 Hz | 2.7 % |

- **Section order:** groove A is +3.2 / +3.1 LU over the intro (full / HP150). Drop 2 is **+1.65 / +2.15 LU** over groove A, and +1.65 / +2.19 over groove A's full phrase. Without its turnaround bar it is +1.49 / +2.10. The breakdown is -5.0 / -4.5 LU under groove A, its dark bar -5.6 / **-7.7**, with about a quarter of groove A's share of energy above 1 kHz (1.2 % vs 4.5 %). The build bar then climbs +1.5 / +5.4 LU. The lighter phrase is lighter in arrangement but measures within 0.1 LU of the full one, because the drop hit sits in it. Round 1 had drop 2 only +0.8 to +1.0 LU full-band over groove A, because the lead and wide pad mostly added energy above 150 Hz; the kick, clap, tom and bass push fixed that. Round 1's dark bar was only -3.9 LU, and lowering the taiko, boom, dark braam and pad fixed it.
- **Demo master:** ffmpeg ebur128 gives **-14.1 LUFS I, LRA 5.5 LU, -1.4 dBTP**. Octave bands: 1 kHz -10.1, 2 kHz -14.6, 4 kHz -18.2, 8 kHz -20.0 dB re 63 Hz.
- **Anchors:** 2.0 / 14.0 / 18.0 / 26.0 are all in `downbeat_times` and `beat_times`. Drums-stem onsets (30 % detector) are +0.9 / +1.1 / +0.9 / +0.8 ms broadband and +3.0 / +3.0 / +3.0 / +2.75 ms LP 200. The breath before each hit is -43 / -39 / -44 / -19 dB broadband and -53 / -43 / -53 / -28 dB LP 200. Before the kick/clap choke, the breakdown's LP 200 breath was -14.3 dB.
- **Notes:** `check_story_score()` passes 914 notes: all in F minor, bass and drone on chord roots, lead chord tones on every beat, pad, ostinato, braam and wide-pad notes as chord tones. In the audio, every chord segment's strongest bass pitch class is its root, and the music stem's top 3 pitch classes are in key.
- **Stems:** true peak drums -6.5 / bass -11.4 / music -9.7 / fx -13.0 dBTP. |DC| < 2e-6, first and last samples are 0, and L/R correlation matches `render()` (music 0.75, worst 400 ms window 0.50).
- **Clicks:** none. The discontinuity scan finds 0 events on every stem over the whole 30 s, and the HF-burst scan finds 0 within ±0.3 s of each boundary on music and fx. Round 1 found two small kinks in the bass's un-ducked 16th drives (-46 dBFS steps, from the filter envelope jumping). `render_story` now ramps that envelope in over 3 ms (`_bass_bus(fenv_attack=0.003)`; `render()` keeps 0).
- **Tail:** the last sample is 0.0 and the last 10 ms is -150 dBFS. The last 0.3 s peaks at -34 LUFS momentary.
- **Robustness:** these runs render cleanly with all anchors on downbeats: anchors snapped from 13.9 / 18.1 / 26.05, a 1-bar breakdown, drop1 = 0, a whole grid offset to 2.37, a 20 s duration, and an end hit at 29.5 (moved to 28.0 with a warning). Same seed gives bit-identical stems and meta. A render takes about 45 s.

## Mixer — `mix.py`

`mix.py` is the re-recording mixer. It turns a cue sheet and a silent video into the delivery MP4. It renders the music bed and every SFX, mixes and masters them, and muxes the result without touching the video stream. With `--report` it also writes a QC report. Every step is deterministic: two runs give a bit-identical master, and a run from cached stems matches a fresh render exactly.

```bash
python3 -I mix.py --cues ../final/cues.json --video ../final/story-silent.mp4 --music techhouse \
                  --out ../final/story-techhouse.mp4 [--wav mix.wav] [--report <dir>] \
                  [--seed 1] [--cache-dir <dir>] [--whoosh-sync window|peak]
# --music cinematic for the other bed; exit code 1 if a --report check fails
```

**Options**
- `--wav` keeps the 48 kHz float32 master.
- `--cache-dir` stores the rendered stems, keyed by the music module's source and its arguments. Use it while iterating: cinematic takes about 45 s to render, tech-house about 20 s.
- `--seed` is the music seed (default 1).
- `--whoosh-sync` sets how whoosh cues are placed; see the sync rules below.

A full run takes about 50–90 s, including up to four master/encode passes.

**Cue sheet.** The cue sheet is a list of `{t, type, ...}` objects; a `{"cues": [...]}` wrapper also works.
- **Anchors:** `anchor` cues set the music anchors. The note text is matched loosely: `drop1` / `drop 1` / `DROP1`, `breakdown`, `drop2`, `end hit` / `end_hit` / `endhit`. An `end` anchor is checked against the video duration. A missing anchor falls back to 2 / 14 / 18 / 26 with a warning.
- **Duration:** the video's duration from ffprobe sets the mix length, N = round(D × 48000). `render_story(duration=D, drop1, breakdown, drop2, end_hit, seed)` renders the bed for exactly that length.
- **Robustness:** unknown cue types are skipped with a note. Out-of-key `pop` offsets are snapped to the nearest F-minor note (ties go up), so the draft's `pop(3)` (B5) becomes `pop(4)` (C6). `sfx.MIN_DUR` clamps and module warnings are collected into the report.

**Sync rules.** The cue `t` is the time in the sheet. The *target* is where the perceptual hit lands, and the QC measures against it.

| type | placement | measured in the SFX bus as |
|---|---|---|
| impact, reject, pop, tick, check, downlifter | onset at `t` (`anchor` = 0) | first crossing of -20 dB re max, 1 ms envelope |
| slash | the cut (`anchor('slash')` = 0.065 s) at `t` | 20 ms-envelope peak |
| scrub, shimmer | array starts at `t`, length `dur` | first crossing of -30 dB re max, 5 ms envelope |
| riser, reverse_swell | `t` is the start; the array **ends on the hit at `t + dur`** (notes: "Hit = 2.0", "ahead of the END HIT") | last sample above -20 dB re max |
| whoosh (`window`, default) | `(t, dur)` is the on-screen motion window, and the array spans it | centre of the -3 dB region of the 20 ms envelope |
| whoosh (`--whoosh-sync peak`) | envelope peak on `t` (start = `t - sfx.anchor('whoosh', …)`) | same |

Why whooshes use the window: the storyboard and the notes give whooshes as motion windows ("5.75-6.25 WHIP-PAN LEFT", "17.7-18.3 brand-slash band ... crossing the frame centre exactly on 18.0"). The whoosh is rendered at exactly `dur`, so its natural peak (0.55–0.68 × dur) lands on the fastest point of the motion: 18.03 s for the brand slash. Putting the *peak* on `t` instead would end the whoosh before the band crosses.

The peak is measured as the centre of the -3 dB region, not the argmax. A 0.5 s whoosh stays within 1 dB of its maximum for about 50 ms, and `sfx.anchor()` returns the argmax at the trailing edge of that plateau. The glue compressor's 1–2 dB under a coincident slam moved that argmax 36 ms in round 1. The -3 dB centre is 0–19 ms earlier than `anchor()` and stable.

**Gain staging (SFX).** Each SFX starts at `sfx.SUGGESTED_GAIN_DB`, and these changes are applied on top (all data at the top of `mix.py`):
- **Hero slams:** the big impact on drop1, drop2 and end_hit goes through `over_music_hit()` and gets **+2.5 dB** (`HERO_SLAM_DB`). That gives back the 2.5 LU full-band loss of the 90 Hz high-pass, so these three are the loudest SFX both full-band and on the HP150 phone model.
- **Other impacts:** a big slam off the anchors (EDITOR at 2.5) stays at 0 dB. Any impact whose music low end is within 12 dB of the kick level also uses `over_music_hit()`; this is measured on drums + bass + fx below 150 Hz, from 5 ms before the hit to 60 ms after it. On this draft that is every impact. A slam in a low-end gap would keep its F1 sub.
- **Type trims** (`TYPE_TRIM_DB`): pop -2, check -2.5, reject -1.5, shimmer -2 dB. UI pops sit 0.5–4.5 dB under the bed's momentary level. The breakdown stamps sit over the sparse breakdown bed (rejects +7.7 to +11.1 dB, check +8.4 to +8.7 dB) but about 3 dB under the hero slams.
- **Length compensation:** `SUGGESTED_GAIN_DB` was measured on one reference length. A longer, louder render is pulled down to its `ROLE_LUFS_M` target, by at most 3 dB, and never boosted; for example `riser(2.0)` gets -1.3 dB.
- **Stacking:** a slash, whoosh, shimmer, pop, tick or check whose sync point is within 60 ms of a hero slam gets -3 dB; on this draft that is the 18.0 slash, the 17.75 whoosh and the 26.0 shimmer.
- **Light cues:** a cue whose note starts with "(light)" gets -3 dB and whoosh intensity 0.7.

**Signal flow.**
1. **Music bus.** The four stems are summed. The `fx` stem is first ducked with `sfx.fx_duck_gain()` (-6 dB, README spans) wherever an SFX riser, reverse_swell or downlifter overlaps a matching transition in `meta['fx_transitions']`, so each transition has one source.
   - On the draft that means the SFX riser at 1.0–2.0 over the music riser (and, in cinematic, its reverse cymbal), and the riser at 16–18 over the music riser and reverse cymbal.
   - In the final cues it also covers the reverse_swell at 25.5–26.0 over the music's reverse cymbal.
   - The breakdown downlifter has no music twin, so nothing is ducked there.

   The bus is set to -20 LUFS integrated, so both styles sit at the same balance against the SFX. Then a **two-band sidechain duck** runs on a zero-phase, exactly complementary split at 150 Hz, keyed by the impacts, rejects and the check.
   - Attack: 6–8 ms of look-ahead, so the music is fully ducked when the transient lands.
   - Hold: 30–60 ms. Release: 150–250 ms. Ramps are raised-cosine in dB, and overlapping keys take the deeper value.

   | key | high band (>150 Hz) | low band on a music hit / otherwise |
   |---|---|---|
   | big impact | 4.5 dB | 1.5 / 5 dB |
   | medium impact | 4 dB | 1 / 4 dB |
   | small impact | 3 dB | 0.5 / 3 dB |
   | reject | 4 dB | 4 dB |
   | check | 2 dB | 0 dB |

   On a music hit the low band is barely ducked, so the music's kick and F1 boom keep the low end that `over_music_hit()` handed them.
2. **SFX bus.**
   - **Shared room:** a short convolution room with a synthetic IR from `sfx._make_ir` (RT 0.55 s, 12 ms predelay, return high-passed at 300 Hz and low-passed at 7 kHz, -4 dB). Sends: tick 0.35, pop 0.30, scrub 0.25, check / slash 0.20, reject 0.18, downlifter 0.15, whoosh / impact 0.12, shimmer 0.10. Risers and reverse swells get no send, so their ends stay sharp.
   - **Glue compression:** detector high-passed at 80 Hz, 10 ms RMS, threshold 6 dB under the bus's loudest moment, 1.5:1, 4 dB knee, 10 / 150 ms; at most 1.4 dB on the slams.
   - **Transient soft clip:** 4× oversampled, tanh above 0.6 of a ceiling set 1 dB above the level the master make-up will put at -1.5 dBTP. It rounds only the first-cycle peaks of the slams (about 0.12 s of samples in total, residual about -20 dB over the first 80 ms of a hero slam) and takes 4 dB off the bus true peak. Without it the master limiter needed 6.5 dB of gain reduction and pulled the music down with every slam. With it, master gain reduction is at most 4.4 dB, and over 3 dB for 0.09–0.16 s in total.
3. **Master.**
   - **EQ:** high-pass 25 Hz, -1 dB at 280 Hz (Q 0.9), +1 dB high shelf at 9 kHz.
   - **Bus compressor:** detector high-passed at 150 Hz (as the music README asks), threshold +3 dB re the detector's loudness, 1.5:1, 20 / 200 ms; at most 1.6–1.8 dB.
   - **Loudness:** gain to -14 LUFS integrated (BS.1770-4, iterated through the limiter).
   - **Limiter:** look-ahead true-peak limiter, 4× oversampled, 5 ms look-ahead, 80 ms release, -1.5 dBTP.
   - **Edges:** 2 ms fade-in (a click guard only; the hook plays from frame 0), then a 0.3 s raised-cosine fade to exact digital zero at the video end, and windowed DC removal.
4. **Mux.** `ffmpeg -i video -i mix.wav -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -profile:a aac_low -b:a 256k -ar 48000 -ac 2 -map_metadata 0 -movflags +faststart`. There is no `-shortest`: the WAV is exactly N samples, and the MP4 edit list trims the AAC priming and padding to 30.000 s.
5. **AAC true-peak loop.** The MP4's audio is decoded and measured with an 8× true peak. AAC overshoots the PCM by 0.1–1.0 dB here; the worst case is the tech-house end hit at 26.0 s, -1.46 dBTP in the PCM but -0.51 dBTP after the encode.
   - If the decoded peak is above -1.15 dBTP, the limiter ceiling is lowered **only around the overshooting spots**, by each spot's excess + 0.05 dB, ±60 ms with 20 ms ramps, and the master is redone.
   - After two local rounds the drop becomes global, because AAC rate control is global and a local fix can move an overshoot elsewhere. Up to five passes are allowed.
   - On the draft, tech-house needed one extra pass (-0.67 dB at 26.0 s). Cinematic needed two (0.5 / 6.5 / 18.0 / 24.5 s), and its lowest local ceiling is -1.95 dBTP.
   - On the final cue sheet, cinematic needed the global step (-0.16 dB).

**Report (`--report <dir>`).**

| file | contents |
|---|---|
| `cue_alignment.csv` | One row per SFX cue: cue `t`, type, parameters, sync rule, array start and end, target, detector, the time detected in the SFX bus (this cue's own contribution after the room, glue and soft clip), error in ms, PASS/FAIL at ±1 frame. Also the error against the raw cue `t`, the README `anchor()` time for whooshes, an onset found in the *summed* SFX bus for transient cues (time, error, rise in dB), gain, send and every processing decision. |
| `anchor_alignment.csv` | Each anchor against `meta['section_times']` and `downbeat_times`, plus the broadband kick onset in the drums stem. |
| `loudness.txt` | ffmpeg ebur128 of the master WAV, of the master through the phone model (4th-order 150 Hz high-pass) and of the MP4. The phone model's true peak is not a delivery figure, because a high-pass raises peaks. Also astats, the master-chain settings and gain-reduction figures, the master passes, per-section loudness (full band and HP150, with the breakdown split into its dark and build bars), each SFX's max momentary loudness against the ducked music, the loudest-SFX ranking and every check. |
| `master_spectrogram.png`, `master_waveform.png` | An own STFT on a log-frequency axis from 30 Hz to 20 kHz. The waveform view adds the SFX-bus RMS and the momentary and short-term loudness curves. Both mark the anchors (dashed), each SFX's sync point and its array span (PIL). |
| `ffprobe_output.json`, `video_md5.txt` | ffprobe of the output, and the input and output video-stream md5 (`ffmpeg -map 0:v -c copy -f md5 -`). |
| `mix_report.json` | Every check, the settings, the notes and the sha256 of the master samples. |

The checks are:
- every SFX within ±1 frame (33.3 ms)
- anchors on music downbeats, with kick onsets within one frame
- -14 ± 0.5 LUFS (WAV and MP4)
- true peak ≤ -1.0 dBTP (ebur128 and 8×, WAV and decoded MP4)
- LRA ≤ 8 LU
- no samples at or above -0.01 dBFS, |DC| < 1e-4
- audible from frame 0 (frame-0 RMS > -45 dBFS)
- last sample 0 with a falling fade
- the hero slams are the loudest SFX, full band and on HP150
- the video stream is bit-identical
- container, video and audio durations equal the input to within 1 ms
- AAC-LC 48 kHz stereo, about 256 kb/s
- `moov` before `mdat` (faststart)
- decoded MP4 audio lags the master WAV by 0 samples

**Verified on the draft** (`drafts/story30-cues.json` + `drafts/story30.mp4`, 39 SFX cues, no cache). The outputs are `test/story30-draft-techhouse.mp4` and `test/story30-draft-cinematic.mp4`, each with a `-report/` directory. All 14 checks pass for both styles. ffmpeg was re-run on the MP4s outside `mix.py` and confirmed the figures.

| | tech-house | cinematic |
|---|---|---|
| SFX alignment, max \|error\| (39 cues) | 12.5 ms (whoosh up, room tail); onsets 0.2–1.9 ms | 12.5 ms; onsets 0.2–1.9 ms |
| anchors 2 / 14 / 18 / 26: music section time, kick onset | exact, downbeats; +1.2 to +1.4 ms | exact, downbeats; +0.6 to +0.8 ms |
| MP4 (ebur128): I / LRA / TP | -14.1 LUFS / 2.7 LU / -1.3 dBTP (8× decoded -1.34) | -14.1 LUFS / 3.0 LU / -1.2 dBTP (8× decoded -1.24) |
| master WAV: I / TP 8× / sample peak / \|DC\| | -14.01 LUFS / -1.46 dBTP / -1.51 dBFS / 2e-12 | -14.02 / -1.48 / -1.51 / 2e-12 |
| phone model (HP150) integrated | -15.5 LUFS | -16.4 LUFS |
| frame 0 RMS / momentary 0–400 ms | -26.7 dBFS / -23.1 LUFS | -20.6 dBFS / -18.8 LUFS |
| last 0.3 s (thirds, RMS) / last 10 ms / last sample | -31 / -40 / -63 dBFS / -144 dBFS / 0 | -36 / -45 / -67 / -151 / 0 |
| loudest SFX, LUFS-M full / HP150 | hero slams -10.1 to -10.3 / -11.0 to -11.2; next is impact@2.5 -11.9 / -12.8 | -9.9 to -10.1 / -10.9 to -11.0; next is impact@2.5 -11.8 / -12.7 |
| sections, full / HP150 LUFS: intro, groove A, dark bar, build bar, drop 2, end | -15.8/-15.8, -14.0/-15.8, -14.9/-16.7, -16.6/-17.0, -13.0/-14.4, -17.0/-18.5 | -15.0/-16.2, -14.3/-16.9, -14.7/-16.9, -16.9/-17.8, -13.1/-15.6, -16.5/-21.0 |
| video md5 in = out | 584f3f16…c89c72b | same |
| container / video / audio duration | 30.000 / 30.000 / 30.000 s, 1,440,000 samples | same |

Drop 2 sits +1.0 / +1.3 LU (tech-house, full / HP150) and +1.2 / +1.2 LU (cinematic) over groove A. The breakdown bed drops out under the stamps: momentary loudness falls to about -20 LUFS between the stamps, which land at -13.3 LUFS-M. The dark bar's gated level is close to groove A only because of the stamps.

Further checks on the master WAVs, using the audit's `hf_spikes`:
- **Clicks:** 131 HF bursts (tech-house) and 984 (cinematic, ticks and hats). Every one is also present in the raw stem sum or the SFX bus, so mixing added **none**.
- **Mono:** L/R correlation 0.91, and the mono fold-down loses 0.21 dB.
- **Spectrum:** 2–5 kHz holds 2.4 % (tech-house) and 1.8 % (cinematic) of the energy, so there is no pile-up.
- **Final cues:** the run also passes on `final/cues.json` + `final/story-silent.mp4` (58 SFX cues, including reverse_swell, 0.1 s whooshes and a "(light)" whoosh, max error 12.5 ms), with `--whoosh-sync peak`, and with cached stems.

**For the final mix:** run the command above with `--music techhouse` or `--music cinematic` and `--report`. Cue times can move: everything is re-derived from the sheet. Anchors that move off the 2 s bar grid are snapped by the music module with a warning. If that happens, `anchor_alignment.csv` shows the offset between the cue and the music.
