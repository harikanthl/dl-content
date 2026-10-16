# Episode explainers: the Manim studio

This folder renders the short (60 to 90 s) animated explainer for each episode of *LLMs from First
Principles*. Each film has 7 to 10 narrated beats, a piper voiceover, burned-in headline and caption rails, a WebVTT
caption track and a poster frame. Optionally there is a 1080x1920 vertical cut for Shorts, Reels and TikTok.

The engine is ported from the pattern-explainers studio in `harikanth-site/studio/pattern-explainers`, which made
15 films. It is MuseDrop's host-TTS + prebaked-voice architecture:

1. Narration is synthesised **first**.
2. Each beat's true spoken length goes to `voice_manifest.json`.
3. Every animation is sized against `tracker.duration`.
4. The render writes back `beat_timings.json`, the beat boundaries it actually hit.
5. The mux lays each voice line at its measured start.

Picture and voice cannot drift.

## Layout

```
house.py            palette, rails (headline/caption), layout + safe box, VERTICAL switch
shots.py            animated primitives (pattern set + deep-learning set, see below)
prebaked_voice.py   voiceover/timing engine + deterministic layout QA
tts.py              narration synthesis (piper preferred, macOS `say` fallback)
build.py            TTS -> render -> mux -> captions -> poster -> explainers.json
smoke.py            fast authoring loop: preview render with stub narration + layout QA
validate.py         static checks over every scene (contract, beat/word budget, no em dashes)
scenes/<file>.py    one scene per episode; the file name is the episode's `explainer` field
                    in curriculum/curriculum.json
explainers.json     metadata rows keyed by episode code (committed)
work/  out/         render workdirs and finished masters (gitignored)
.venv-tts  voices/  symlinks into the harikanth-site studio (piper + models), gitignored
```

## Commands

All scripts run under manim's interpreter, because scene modules import manim:

```sh
PY=~/.local/share/uv/tools/manim/bin/python
cd ~/Documents/DL-content/explainers

$PY smoke.py M01E01                 # preview render + layout QA (about 10 s)
$PY smoke.py M01E01 --frames        # also dump one PNG per beat end into the smoke work dir
$PY smoke.py M01E01 --vertical      # check the vertical layout
$PY validate.py                     # static QA over every scene

$PY build.py M01E01                 # full 1080p30 build with piper narration
$PY build.py M01E01 --vertical      # 1080x1920 cut -> out/M01E01-vertical.*
$PY build.py --all --jobs 2         # every scene under scenes/
$PY build.py M01E01 --voice-only    # narration only
$PY build.py M01E01 --copy-to ~/Documents/harikanth-site/public/media/dl-explainers
```

You can target an episode by code (`M01E01`, looked up in `curriculum.json`) or by scene file stem.

### Outputs

Each build writes:

* `out/<CODE>.mp4`, `out/<CODE>.jpg` (poster at `POSTER_AT`) and `out/<CODE>.vtt`.
* A metadata row in `explainers.json` with `title`, `summary`, `duration`, `beats` and `chapters` (`[{at, label}]`).
  A vertical build adds `"vertical": {duration, chapters}` to the same row.

Asset URLs are not stored in the row. A site composes them from its own base URL plus `<CODE>.mp4`, so moving
hosts needs no re-render. To publish, use `--copy-to <dir>` to drop the three files into any folder (for example the
site's public media dir), or upload `out/` to a bucket.

After a successful build, `build.py` deletes the large intermediates in `work/` (manim media, per-beat WAVs, the
muxed copy) and keeps only the small JSON reports. Use `--keep-work` to keep everything.

### Time cost per film

Measured on M01E01 (88 s film, 9 beats, 232 words, Apple Silicon):

| step | wall time |
|---|---|
| `smoke.py` (480p15, stub voice) | ~10 s |
| `build.py` (piper TTS + 1080p30 render + mux) | ~41 s |
| `build.py --vertical` | ~38 s |

Outputs are about 6 MB for the 16:9 mp4 and 5 MB for the vertical one. Disk use during a build stays under 50 MB.

## Adding an episode explainer

1. Look up the episode in `curriculum/curriculum.json`. Its `explainer` field is the scene path, for example
   `explainers/scenes/m01e01_tensors_what_they_actually_are.py`.
2. Copy `scenes/m01e01_tensors_what_they_actually_are.py` to that path. Set `EPISODE_CODE`, `TITLE`, `SUMMARY`,
   `SCENE_CLASS`, `POSTER_AT` and `CHAPTERS` (beat index -> label).
3. Write `BEATS`:
   * If `curriculum/outlines/<module>.json` has the episode's `explainer` (narrative) and `explainer_visuals` fields,
     write the beats from those. Otherwise use the episode's `learn` and `build` fields and its prep note.
   * Aim for 7 to 10 beats and 170 to 240 words. The default pace is ~2.67 words/s, so that gives 60 to 90 s.
   * Use plain first-person-plural English ("we transpose", "let's open one up"), with no em dashes.
   * `validate.py` enforces the word and beat budgets and the no-dash rule.
4. Write one `_beatN()` per beat:
   * Each one is `with self.voiceover(text=BEATS[N]) as tr:`, then `swap_rails(...)`, then animations sized from
     `tr.duration`, ending with `self.wait(max(0.1, tr.get_remaining_duration(buff=...)))`.
   * Keep BEATS and the animation in the same file.
5. Run `$PY smoke.py <CODE> --frames` until it prints `ok layout clean`, then look at the frames.
   **Treat layout QA hits as bugs**: off-frame or overlapping text is the most common failure in generated manim.
6. Run `$PY build.py <CODE>`, and optionally `--vertical`.

### The style contract (house.py)

1. **Headline before symbols**: each beat opens with a plain-language claim, then the notation.
2. **Caption everything**: nothing appears on screen that the narration hasn't named.
3. **Camera into the exact term**: zoom on the thing being spoken about, then restore. Zoom tight enough that the
   rails sit fully outside the frame (see M01E01 beat 3).
4. **Depth only when the idea is 3D**: 2D only. A 3D tensor is drawn as offset 2D slices.
5. **Nothing keyframed**: motion comes from computed state. Offsets are computed from strides, the plot dot rides a
   `ValueTracker`, and cells travel to computed positions.
6. **Visual continuity**: each beat ends on the object the next beat opens with.

Scene gotcha: animating a sub-group (for example `Indicate` on three storage cells) makes manim promote that group to
the top level and split its parent. Clear the stage by family, not by identity. M01E01's `_clear` shows how.

## Shot library (shots.py)

The pattern set is kept as is: `ArrayRow`, `pointer`, `window_over`, `BarSet`, `node_chain`, `arrow_between`,
`hash_buckets`, `TreeViz`, `graph_viz`, `DPGrid`, `interval_track`, `chip`.

The deep-learning set:

| primitive | what it draws | key methods |
|---|---|---|
| `TensorGrid(data, shape=True, stride=(3,1), show_index, name)` | 2D or 3D grid of numbered cells with shape/stride labels (3D as offset slices) | `cell/row/col`, `focus`, `focus_row`, `focus_col`, `reset_all`, `set_value` |
| `stride_jump(row, src, dst, "+3")` | a curved hop over a storage `ArrayRow`: a stride as a jump in flat memory | |
| `MatMul(A, B, layout="row"\|"L")` | A x B = C with C filling in | `step(i, j)` -> (anims, dot-product readout), `clear_focus` |
| `Heatmap(w, row_labels, col_labels, causal, colorbar, show_values)` | attention weights, PANEL->accent scale | `mask()` animates the causal mask, `set_weights`, `focus_row` |
| `BlockDiagram(nodes, edges)` / `layer_stack(names, shapes)` | boxes + arrows, shape labels beside the arrows | `reveal`, `flow` (pulse along edges), `focus`, `reset` |
| `CurvePlot(fn or points, x_range, y_range, x_label, y_label)` | axes + curve with Text labels (no LaTeX) | `add_curve`, `tracking_dot(ValueTracker)` |
| `softmax(xs, temperature)` | helper for computing heatmap/bar targets | |
| `CompGraph(nodes, edges)`, `value_node`, `op_node` | autograd graph: value nodes (data/grad) and op nodes | `reveal`, `forward_flash`, `backward_flash(edge)`, `set_grad` |

## Vertical mode (`--vertical`)

Vertical mode works. With `EXPLAINER_VERTICAL=1` (which `build.py`/`smoke.py --vertical` set), `house.py` changes four things:

* **Frame:** it re-declares the frame as 9 x 16 units at 1080x1920. A unit is ~120 px versus 135 px in the master,
  so text keeps roughly the same physical size.
* **Rails:** the headline and caption move to the top and bottom of the tall frame.
* **Wrapping:** rail text wraps into balanced, centred lines.
* **Safe box:** it narrows to +-4.2 units across and grows to +-5.5 units tall.

Layout QA and the frame-fit safety net use the camera frame, so they work unchanged.

The scene's own composition is the only per-scene cost. A scene reads `house.VERTICAL` and picks positions from a
small layout dict, as `L = {...}` does in M01E01. Side-by-side compositions wider than ~8.4 units have to stack. The
16:9 master is still the priority. Author for landscape first, then check `smoke.py --vertical`.

## Narration

`tts.py` uses piper `en_US-ryan-high` at `--length-scale 1.3` (~160 wpm), the same as the source studio. It falls back
to macOS `say`. Every line is loudness-levelled to 48 kHz mono. A 0.28 s breath is added to each beat's slot in the
voice manifest, so the scene allots it.

Changing the pace (`--length-scale`) needs a re-render, because animation timing is derived from the measured
narration.

Fix over the source studio: the source mux appended a 0.28 s breath after every line whether or not the beat had
room. That drifted the voice later each beat, and `-shortest` clipped the final line. Here the breath lives in the
manifest, and the mux trims to the picture's length, never the audio's.

## Requirements

* manim 0.20.1 (`~/.local/share/uv/tools/manim`), ffmpeg. LaTeX is needed only if a scene uses `MathTex`; the DL
  primitives use plain `Text`.
* piper via the symlinked `.venv-tts` and `voices/` from `~/Documents/harikanth-site/studio/pattern-explainers`. If
  those move, re-point the symlinks, or install per that studio's README.
