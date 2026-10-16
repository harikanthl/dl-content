# The Production Playbook

How every episode of *LLMs from First Principles* gets made, from prep sheet to five platforms.
Prep sheets and scripts handle *what* you say. This file handles *how* it ships.

The recording setup (Meld scenes, Brio 300, mic filters, nvim `:RecordMode`, storage and
`archive`) is the same as the DSA series. Read `~/Documents/dsa-content/PRODUCTION.md` §2 once;
nothing here repeats it.

---

## 1. The format (locked)

**Target length: 10–15 min** (M01 episodes are 5–9 min so they stitch into *PyTorch in 1 Hour*).

| # | Beat | Time | On screen |
|---|---|---|---|
| 1 | **Hook** | 0:00–0:15 | Face, full frame (TALK). One sentence: the surprise or payoff |
| 2 | **Question** | 0:15–1:00 | Face. What breaks without this idea, tied to a named real model |
| 3 | **Intuition** | 1:00–3:00 | The **Manim explainer** plays (60–90 s), then you add one line |
| 4 | **Math** | 3:00–6:00 | Screen: derive in a markdown/LaTeX buffer or on the iPad. Shapes everywhere |
| 5 | **Code** | 6:00–10:00 | nvim, from scratch, narrate every line |
| 6 | **Receipt** | 10:00–11:00 | The test that proves it: `assert_close` vs PyTorch, gradcheck, or a benchmark. Green |
| 7 | **Measure** | 11:00–12:00 | One number or one plot |
| 8 | **Takeaway** | 12:00–12:30 | One sentence + the interview line |
| 9 | **Next** | last 10 s | "Next: …". Nothing else |

**The receipt is the series' signature.** Polished tutorials claim things work. This series
proves it on camera, every time. A failing receipt you fix on camera is better content than
one that passes first try. Never cut it.

**Never edit out a mistake in beats 5–6.** Same rule as the DSA series, same reason.

## 2. The daily pipeline

```
night before   dl today → read both prep sheets, dl test <code>, watch explainers/out/<code>.mp4
record         Meld → ~/Movies/dl/<code>-raw.mp4
               dl done <code>          (stages edit/<code>/)
edit (Claude)  "edit <code>"  →  edit/<code>/renders/{long.mp4, short.mp4, chapters.txt}
review         watch long + short at 1.5x, note fixes
publish        YouTube (title/desc/chapters from the sheet) → Shorts/Reels/TikTok/FB → X → LinkedIn
tick           sheet: Recorded, Edited, YouTube, Short, X, LinkedIn, IG, TikTok, FB
```

### 2a. Explainers (Manim, automated)

One 60–90 s narrated film per episode, rendered by the engine in `explainers/` (ported from
`harikanth-site/studio/pattern-explainers`, which made the LeetCode pattern films). Narration
is synthesised first and every animation is sized to the measured voice, so picture and voice
cannot drift. The narration lines come from the outline's `explainer` field, the pictures from
`explainer_visuals`. See `explainers/README.md`.

Each explainer is used three ways:
1. inserted at beat 3 of the long video,
2. posted standalone as a Short/Reel/TikTok (vertical render),
3. embedded on harikanth.site's learn page (reuse `PatternExplainer.astro` + `explainers.json`).

Render explainers in batches on Sunday for the coming week. Disk is tight: `explainers/work/`
is deleted after each build.

### 2b. The edit (HyperFrames)

Reuses what was proven in `~/Projects/IndiaOne/videos/`:
- `meld-walkthrough-captions/` is the template for a long Meld recording with a webcam bubble
  and captions only (frame `01-walkthrough.html` over the raw file).
- `videos/tools/subs.py` turns a word-level transcript into neat burned-in `.ass` captions
  (≤42 chars/line, keeps clear of the webcam circle with `--side`).
- `productvideo.md` §6–7 has the retime/revision rules and the known traps.

Per episode Claude:
1. transcribes `edit/<code>/raw.mp4` (word timings),
2. builds the long cut: captions + KaTeX equation overlays at the MATH beat + chapter title cards
   (`/talking-head-recut` style overlays; the footage itself is untouched) + the explainer
   spliced at the INTUITION beat,
3. writes `chapters.txt` from the beat timestamps for the YouTube description,
4. carves the Short: the beat named in the sheet's *Short* column, 9:16, burned captions,
   45–60 s, or uses the vertical explainer when that beat is weaker.

Rendering is local (headless Chrome + ffmpeg).

## 3. Titles, thumbnails, descriptions

**Titles**: search term first, then the promise. From the sheet's *YouTube title* column.
```
Self-Attention from Scratch in PyTorch (and proof it matches SDPA)
FlashAttention in Triton: within X% of FA2 on a 4090
GRPO from Scratch: Qwen 1.5B learns GSM8K on one GPU
```

**Thumbnails**: one Canva template per module colour (colours are in the Roadmap tab and on
every episode code cell). 3–5 huge words, face right third. After 20 videos the colour alone
tells a returning viewer the module.

**Description template**
```
{one-line insight}

Code: {repo}/{code_file}
Series: LLMs from First Principles · {module} · episode {mod_ep}/{n}
Playlist: {module playlist}

⏱ {chapters.txt}

Sources: {from the sheet's Sources column}
GPU for this episode: {RunPod / sponsor credit line, when used}

#pytorch #llm #deeplearning
```

## 4. Publishing

The cadence, times and caption rules per platform live in the sheet's **Publishing** tab.
Summary: 2 long videos a day on YouTube, 2 Shorts a day cross-posted to Shorts, Reels, TikTok
and Facebook, X twice a day, LinkedIn three times a week.

**Recruiters will see this.** One channel, one name, clean playlists per module. The
harikanth.site learn page is the hub that links each episode, its code and its explainer.

## 5. The first ten videos

Same rule as DSA: record episodes 1–5 before publishing any. Watch them back, fix your tics,
publish video 1 the day you record video 6.
