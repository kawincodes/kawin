
## Sound effects — `sfx.py`

All of these sounds are made in code from oscillators and seeded noise. The module uses no samples, loops or downloaded audio. Output is deterministic: the same call always returns the same samples.
Every function returns a `float32` array shaped `(N, 2)` at 48 kHz. Each sound is true-peak normalised to **-3 dBTP** (sample peak about -3 dBFS), has its DC removed and has raised-cosine fades of at least 2 ms at both ends.
Shared contract: **F minor** (F G Ab Bb C Db Eb), **120 BPM**. `verify_key()` checks that every tuned note is in the key, and rhythmic spacing uses 16th/32nd notes at 120 BPM.

```python
import sfx                                   # python -I; needs numpy, scipy, soundfile, pedalboard
x = sfx.whoosh(0.45, 'lr')                   # (N, 2) float32
start = event_t - sfx.anchor('whoosh', dur=0.45, direction='lr')   # align the sound's peak with the event
gain = 10 ** (sfx.SUGGESTED_GAIN_DB['whoosh'] / 20)               # starting fader level
sfx.render_demo()                            # writes demo/sfx_demo.wav, demo/sfx_demo_spectrogram.png,
                                             # demo/sfx_demo_waveform.png, demo/sfx_report.txt
```

| call | length | sync point (`anchor`) | what it is |
|---|---|---|---|
| `whoosh(dur=.45, direction='lr'\|'rl'\|'up'\|'down', intensity=1)` | exactly `dur` | peak at 0.56·dur (lr/rl), 0.72·dur (up), 0.30·dur (down) | Pink and white noise through a time-varying SVF. **lr/rl**: doppler-style filter that opens as the sound approaches and closes as it leaves, with an equal-power pan sweep from -0.85 to +0.85. **up/down**: a resonant sweep from 280 Hz to about 3 kHz (or back down). `intensity` (0.2–2) sets brightness, the low "whomp" and the air hiss. It does not change the output level. |
| `slash(dur=.35)` | exactly `dur` | 0.065 s | Blade swipe. A band-pass swish sweeps 1.5→7 kHz while panning L→R. At the cut it gives a 3 ms snap, then a short metallic "shing" from in-key partials (F6 C7 C8 F8 Ab8 C9, each a slowly beating pair) and a light air whump. Energy peaks in the 8 kHz octave. |
| `impact('big'\|'medium'\|'small')` | about 1.2 / 0.8 / 0.4 s | 0 | Sub drop (sine 120→38 Hz for big, 140→48 Hz medium, 190→70 Hz small), soft-saturated so it still reads on phone speakers. A tuned "tom" body with a fast pitch drop (F2 / C3 / F3, plus the octave above), a 90–700 Hz punch, a 2.5–6 kHz click (about 2 ms), a 0.5–9 kHz crack and a reverb tail. The sub stays dry and mono. |
| `reject()` | about 0.58 s | 0 | X-stamp. A 100→52 Hz thud, an F3 knock and a rubber-stamp slap at 0, then two muted square-ish blips, **C3 → Ab2** (falling major third, both F-minor chord tones), one 16th apart, low-passed at 1.3 kHz. |
| `check()` | about 1.3 s | 0 (2nd note at +0.125 s) | Two soft FM bell plucks, **F5 → C6** (tonic → fifth), a 16th apart. Adds an 8th-note echo with sides swapped and a small plate. |
| `pop(pitch_semitones=0)` | 0.30 s | 0 | Card pop: a rising "bloop" sine, its 2nd harmonic, a low "thock" two octaves down, and a soft click. The base note is **Ab5**, so 0/+2/+4 give **Ab5/Bb5/C6** for cards 01/02/03. Offsets stay in key when they are Ab-major degrees (…-5 -3 -1 0 2 4 5 7 9 11 12). Any other offset raises a warning. |
| `tick()` | 45 ms | 0 | Tiny modal "plastic" click tuned to F6/C7 with a faint HF snap. |
| `scrub(dur)` | exactly `dur` | 0 | Ruler ticks at 16/s (32nd notes at 120 BPM), with every 4th one accented. They pan L→R as the playhead moves, over faint band-passed "tape-scrub" noise whose pitch follows a wandering scrub speed. |
| `riser(dur=1.5)` | exactly `dur` | **end of array** | Pink noise through a low-pass opening 220 Hz→10 kHz, with a rising high-pass so it lifts instead of rumbling, plus a resonant whistle. An F power fifth (F3+C4) built from 3 detuned saws per note, spread L/R, glides up two octaves and lands on **F5+C6** at 88 %, then holds. Tremolo speeds up from 4 to 16 Hz (8ths → 32nds) and the stereo widens. Only a 3 ms click guard at the end, so the hit lands at full level. |
| `downlifter(dur=1.0)` | exactly `dur` | 0 | Wide noise through a closing low-pass (9 kHz → 150 Hz) plus an F4/C5 fifth gliding down two octaves to F2/C3. The stereo narrows as it falls. |
| `shimmer(dur=1.2)` | exactly `dur` | 0 (densest 0.1–0.4 s) | A cloud of glassy bell grains on F-minor(add9) tones C6…F7; the pool rises over time. Adds a soft F5/C6/F6 bell pad, very-high glitter pings (F8 Ab8 C9), 8–13 kHz air sparkle, 16th / dotted-16th cross echoes and a 1.8 s bright room. |
| `reverse_swell(dur=0.8)` | exactly `dur` | **end of array** | A synthetic crash (an inharmonic 808-style square cluster plus HF noise) and a soft **F4 Ab4 C5** triad are reverberated forward and then reversed, so the reverb blooms *into* the hit. |

**Design notes**
- Every sound is layered from a transient, a body and a tail. Envelopes are shaped analytically (exponential and raised-cosine). Time-varying filters use a TPT state-variable filter, which stays stable under fast sweeps.
- Pedalboard's Freeverb rang at about 1.85 and 2.4 kHz on short rooms, which you can see in the spectrograms. Reverb now goes through `pedalboard.Convolution` instead, with a **synthetic** impulse response made of decorrelated noise. Lows, mids and highs decay at different rates, it has a 15 ms build-up and soft early reflections. That gives smooth, non-metallic tails. The wet return is band-limited (HP 200–600 Hz) so the lows stay dry.
- Everything below 150 Hz is folded to mono, which keeps phone speakers and the mono sum solid. Subs and bodies are soft-clipped (tanh) to add harmonics around 100–350 Hz, so the hits are still audible on small speakers.
- Nothing piles up at 2–4 kHz. That octave stays at or below the neighbouring bands for every sound (see the octave table in `demo/sfx_report.txt`). Air and glitter layers are band-limited to 6–14 kHz.

**Verified** (by `render_demo()`, re-checked over 4 rounds of spectrogram and report review): every sound is finite, true peak -3.0 dBTP, |DC| < 1e-9, first and last samples about 0. Measured pitches: check F5/C6, pops Ab5/Bb5/C6, tick F6, all within ±3 cents. Exact lengths and `anchor()` peaks were confirmed for durations from 0.03 s to 4 s. The whole demo measures -16.4 LUFS integrated with a true peak of -3.0 dBTP (ffmpeg ebur128).

**Mixing:** `SUGGESTED_GAIN_DB` gives starting fader offsets, all ≤ 0 dB. They are based on each sound's measured max momentary loudness (from -14 LUFS-M for the big slam down to about -27 for the scrub bed).
Pure-tone sounds (`check`, `shimmer`) are loud for their peak level, so they get -8.5 dB. Use `riser` and `reverse_swell` by aligning the array end with the hit. Pair them with `impact('big')` or `downlifter()` at that same moment.

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
                   # demo/music_techhouse_demo_spectrogram.png, demo/music_techhouse_demo_report.json
```

**Timing.** Beat *k* is at `drop_time + k*0.5 s`. A bar line falls exactly on `drop_time`; bars before it can be partial. `end_hit_time` snaps to the nearest beat (ties round up). The snapped hit is at least 3 beats after the drop and leaves at least 1 s of tail. For the demo, 14.5 → **14.3 s**, which is beat 2 of a bar. All events are placed by `round(t*48000)`, so the drop kick lands on sample 86400.

**Arrangement (written as data at the top of the file).**
| span | harmony | what plays |
|---|---|---|
| 0 → drop (intro/build) | Eb (VII) | Audible from t=0, no kick. A detuned-saw pad low-passed from 520 Hz up to 2.3 kHz, with a "ghost" sidechain pulse on each beat. A low-passed pluck arp opening from 650 Hz to 2.4 kHz. Closed-hat 16ths in a crescendo. A snare build (8ths → 16ths → 32nds, rising pitch and high-pass). A noise + F3→F5 saw riser. A 50 ms breath before the drop. |
| drop → hit-2 beats | **Fm – Db – Ab – Eb** (i–VI–III–VII), one bar each, cycling | Four-on-the-floor kick with a sine-sweep body tuned to F1 and a 3–5 kHz click. Clap on 2 & 4 into a short plate. Swung closed-hat 16ths plus offbeat open hats. A rolling 16th bass (root/root/octave, fifth on beat 4) with deep kick sidechain. A 3-3-2-accented 16th pluck arp (patterns A/B per bar) with filter envelope, 2-bar filter ramps and a dotted-8th ping-pong delay. A warm 5-voice detuned-saw pad (LP-24 ladder automation, HP 150 Hz, chorus, hall, sidechain pump). At the drop: chord stab, crash, sub boom and a downlifter. Snare fill plus crash every 4th bar. From bar 5 on, an octave "lift" layer is added to the arp. |
| last 2 beats before the hit | Eb (VII) turnaround | Snare 16ths → 32nds crescendo with no kick on the last beat. The bass walks Bb1-C2-Db2-Eb2, the arp runs upward, a reverse crash swells into the hit, then a 50 ms breath. |
| end hit → duration | **Fm9** | Big kick, crash, sub boom and a hall-reverb impact. An Fm9 pluck chord stab whose echoes ring out. The Fm9 pad swells then its filter closes (3 kHz → 600 Hz). A long F1 bass decays under the end card. The last 0.25 s is a raised-cosine fade to exact zero. |

**Mix / design notes**
- Kick and bass are kept apart in two ways. The bass never plays on the kick's 16th, and the bass bus is ducked 85 % by a kick sidechain envelope. Every sub (bass notes, drop/end booms) starts **in phase with the ringing kick body** (shared pitch path / computed phase). Measured low-band (<75 Hz) correlation kick↔boom is +0.5 and kick↔bass at the hit is +0.8, so the sums are constructive.
- The bass mid layer is a saw with a filter envelope and 4×-oversampled drive, HP 110 / LP 5 kHz. That puts about 14 % of the bass stem's energy above 120 Hz, which keeps the line audible on phone speakers.
- All reverbs are **convolution with synthetic IRs**: decorrelated 3-band noise, highs decaying faster, build-up, early reflections. Pedalboard's Freeverb was measured ringing at 3.83 kHz and 5.77 kHz (+17 dB narrow peaks), so it is not used.
- There is no 2–4 kHz pile-up: that band is 0.2 % of the drums stem, 0.1 % of bass and 4 % of music. Hats sit above 6 kHz (96 % of their energy).
- Every event gets at least 2 ms raised-cosine fades (enforced centrally in `_Bus.add`). Gates use 2–4 ms ramps. Stems are high-passed at 22 Hz and DC-corrected.
- The four stems are scaled **together** (balance kept) so the hottest true peak is **-7 dBTP**. Gain staging lives in `LV` (dB per element).

**Verified** (`render_demo()` plus extra scripts, 5 rounds of spectrogram, report and listening-proxy review):
- `check_arrangement()` runs inside `render()` and raises on any violation. On the demo, 266 pitched notes are all in F minor. Pad, arp and stab notes are all chord tones of the chord sounding at their grid time. Every chord segment's first bass note is its root, and octave/fifth roles are correct. The walk-up notes are in key. The drop is a beat and a downbeat with a kick on its exact sample, and the hit is on the grid.
- Demo stems: true peak drums -7.0 / bass -8.0 / music -9.5 / fx -10.8 dBTP. Loudness -19.0 / -20.3 / -20.6 / -25.1 LUFS. |DC| < 1e-9. First and last samples are 0. All values are finite. The music stem is audible from 0.5 ms.
- Demo preview master (ffmpeg ebur128): **-14.0 LUFS integrated, LRA 0.8 LU, true peak -1.5 dBTP**, |DC| < 1e-6. The unmastered stem sum is -14.9 LUFS / -1.7 dBTP.
- Robustness: drop at 0 s, drop at 3.3 s, a hit too close to the end (clamped) and a 12 s duration all render with valid checks. The same seed gives bit-identical output. Render time is about 20 s for 17.5 s of audio.

**For the mixer:** sum the stems and duck `music` (and some of `fx`) under voice or SFX as needed. `meta['beat_times']` / `downbeat_times` give the grid for snapping SFX. `fx` holds musical transitions (riser, booms, downlifter, reverse crash, end impact). It does not hold whooshes, which come from `sfx.py`.

## Music bed — `music_cinematic.py` ("Cinematic hybrid pulse")

An original trailer/showreel-style score. Everything is synthesized in the file: polyBLEP saws, sines, seeded noise, analytic envelopes, time-varying biquads, tanh drive and a convolution reverb built from a **synthetic** impulse response. It uses no samples or GM patches, and output is deterministic per `seed`.
**F minor, 120 BPM, 4/4.** The progression is **Fm – Db – Bbm – Csus4** (i–VI–iv–Vsus4, one chord per bar from the drop). The intro builds on a **Csus4** dominant pedal. The last bar before the end hit is the **Db → Eb** turnaround (VI–VII–i). The tail rings on **Fm(add9)**. The V is a sus4 chord, so every note stays in F natural minor with no E natural.

```python
import music_cinematic as mc                     # python -I; needs numpy, scipy, soundfile (+ ffmpeg for the demo)
st = mc.render(duration=17.5, drop_time=1.8, end_hit_time=14.5, seed=1)
st['drums'], st['bass'], st['music'], st['fx']  # float32 (N, 2), N = round(duration*48000), unmastered
st['meta']  # bpm, key, beat_times, drop_time, end_hit_time_snapped (+ bar_times, meter, chord_timeline, ...)
mc.render_demo()   # demo/music_cinematic_demo.wav, _spectrogram.png, _waveform.png, _report.json
```

| time | what happens |
|---|---|
| 0 → drop | **Build.** Starts sounding at t=0: the first 8th note is at 0.05 s, and the pad is already sustaining from the pre-roll. A clock tick-tock plays 8ths (F8 on the beat panned left, C8 off the beat panned right). An 8th-note C pulse bass opens its low-pass from 170 to 850 Hz. A Csus4 string pad swells (×0.62 → ×1.05, LP 1 → 3.6 kHz). Heartbeat toms land on the beats, with a 16th tom fill and a hat crescendo over the last beat. A grid-locked noise riser (tremolo 8ths → 16ths → 32nds) and a reverse cymbal lead in. Sustained parts drop out for the last 16th (the "breath"). |
| **drop** | A bar line falls exactly on `drop_time`. **Hit:** big kick, F2 taiko, crash, F1 sub boom, and an Fm "braam" (4 detuned saws per note, pitch scoop, LP envelope). Then the full groove: kick (A/B patterns), processed claps on 2 and 4, 16th hats, the clock ticks, an 8th-note saw+sub ostinato bass on the chord roots (octave pickup on the last 8th), a 16th spiccato-style synth ostinato with a 3/16 ping-pong delay, and the supersaw string pad (high-passed at 150 Hz). Deep tom groove from bar 3. Tom fills close each 2-bar phrase. |
| drop + 4 bars | Second cycle: crash, a softer braam, a reverse-cymbal lead-in, and the bass, pad and ostinato filters open further. |
| last bar | **Turnaround** Db → Eb. If the hit is not 4·k beats after the drop, this bar becomes 5/4 or 6/4 (or a 3/4 bar is added), so the end hit is also a downbeat. The Eb braam swells, the bass drives in 16ths, a 16th tom fill plays with a clap roll, a riser and reverse cymbal build, then a one-16th breath. |
| **end hit** | `end_hit_time` snapped to the nearest beat (14.5 → **14.3** in the demo), reported in `meta['end_hit_time_snapped']`. Big kick, taiko, clap, crash, sub boom and braam. The arrangement then thins to a sustained **Fm(add9)** pad, a ringing F1 bass, the braam tail and a few fading clock ticks. Everything decays and fades to digital zero exactly at `duration` (the last 0.25 s is a raised-cosine fade). |

**Stems** share one common gain, so they are balanced as mixed. The hottest stem (drums) sits at -6.5 dBTP. Demo values: drums -6.5 / bass -11.8 / music -9.2 / fx -12.7 dBTP. Summed at unity they peak at -3.4 dBTP and measure -18.2 LUFS. `fx` holds musical transitions only (risers, reverse cymbals, sub booms). Whooshes, stamps and pops come from `sfx.py`.

**Mixing design.** The bass is sidechained to every kick (-14 dB, 120 ms release); the pad (-3 dB) and ostinato (-2.5 dB) duck lightly. The kick tail is tuned to Ab1 (52 Hz), clear of the F1 sub. Kick, toms, pad and braam are folded to mono below 120–150 Hz (mix side/mid below 150 Hz: -31 dB). Each bus is set by loudness while it plays (`BUS_LUFS`), not by peak. The bass has an octave-up saw layer and drive, so the pulse still reads on phone speakers (energy from 250 Hz to 1 kHz).

**Verified** (`render_demo()` writes a `checks` block, and all are true):
- **Stems:** true peak ≤ -6 dBTP, |DC| < 1e-7 in the demo (< 1e-4 in every tested case), finite, first and last samples 0.
- **Demo:** ffmpeg ebur128 gives **-14.1 LUFS I, LRA 2.0 LU, -1.4 dBTP**.
- **Timing:** drop onset within 1 ms of `drop_time`. Drop and end hit are bar lines for drop/hit/duration combinations including drop=0, 0.4, 3.1 and 6.0 s and hits needing 4/4, 5/4 and 6/4 last bars.
- **Notes:** `check_score()` passed 523 notes. All pitches are in F minor. Pad, ostinato and braam notes are chord tones, and the bass is always the chord root.
- **Clicks:** none found. Bass and music HF above 14 kHz never exceed -70 dBFS, and every sound has ≥ 2 ms raised-cosine edges.
- **Spectrum:** demo octave bands fall steadily: 2 kHz -16.8, 4 kHz -18.9, 8 kHz -19.5 dB re 63 Hz, so nothing piles up at 2–4 kHz. Hats sit above 7 kHz. The kick is 45–160 Hz plus a 3–5 kHz click.
