# HOW-TO: read this first

Orientation for Harikanth or a future Claude session picking this up cold.

## What this is

Harikanth is building every piece of an LLM on camera, 216 episodes, two a day, to get hired
at a frontier lab (Anthropic, OpenAI, DeepMind, the Anthropic Fellows program) as a research or
performance engineer. He is a U.S. Army veteran with an MS in EE who ships complete products
alone (OperatorOK and Vireeel on the App Store) but has no ML training work on his resume yet.
**Public, dated, verifiable evidence of skill is the whole point.** Bias every decision toward
work a lab engineer would respect: real numbers, comparisons against production systems,
reproductions of published results, merged upstream PRs.

Sister project: `~/Documents/dsa-content` (186 LeetCode-pattern videos). Same conventions.
Job search plan: `~/Documents/job-search/PLAN.md`.

## Decisions already made (don't relitigate)

| Decision | Value |
|---|---|
| Language | Python + PyTorch 2.x; CUDA C++ and Triton for kernels |
| Local hardware | Mac mini M4 (MPS) for M00–M06, most of M09–M10 (gloo on CPU) |
| GPU | RunPod: RTX 4090 for kernels/inference, H100 for post-training, multi-GPU for distributed. Set a spend cap every time |
| Format | 9 beats, locked (`PRODUCTION.md` §1). Every episode ends with a receipt |
| Explainers | Manim engine in `explainers/`, one 60–90 s film per episode |
| Edit | HyperFrames, reusing `~/Projects/IndiaOne/videos/` (captions + overlays) |
| Cadence | 2 videos/day Mon–Sat; slot B becomes a project build log during sprints |
| Tracking | Google Sheet "LLM Roadmap" for daily status; `curriculum.json` for content |

## How to write things

- **Outlines** (`curriculum/outlines/Mxx.json`): follow `templates/OUTLINE-SPEC.md` exactly.
- **Prep sheets**: follow `templates/PREP-TEMPLATE.md`; the bar is the M01 sheets. Code goes in
  `code/` with pytest receipts that actually pass; put measured numbers in the sheet.
- **Recent papers** (M15 DeepSeek, M16 frontier): read the paper before writing. If a detail
  can't be verified, write "verify in the paper" instead of guessing.
- **Tone**: smart friend who is tired. Short sentences. No hype, no "simply", no em dashes in
  spoken lines. Say the hard part is hard, then make it easy.

## Keeping the DeepSeek Watch current

DeepSeek ships ideas that work (MLA, DeepSeekMoE, MTP, FP8 training, GRPO, DSA, mHC, Engram,
V4). Weekly: check github.com/deepseek-ai and huggingface.co/deepseek-ai, add new papers to
the sheet's *DeepSeek Watch* tab and, if they matter, a new M15 episode in `seed_curriculum.py`
→ rerun `build_index.py` and `build_sheet.py`.

## Progress

Track honestly at the end of every session.

| Area | Status |
|---|---|
| Curriculum (216 eps, 18 modules, 9 projects) | ✅ |
| Outlines/scripts | ✅ 216/216 (curriculum/outlines) |
| Prep sheets + code | ✅ M00–M01 (16 eps, all tests pass); M02 onward: write a week ahead |
| Explainer engine | ✅ explainers/ (16:9 + --vertical); M01E01 built |
| Google Sheet | ✅ "LLM Roadmap (DL content)": https://docs.google.com/spreadsheets/d/1bzBoZvzsl55eVUd6ocMGG9ORmn_UJJp0PBdFmU77Cvc |
| GPU credits | researched (funding/); nothing sent yet, needs YouTube URL |

The Google Sheet is the live status tracker. Rebuilding the xlsx and re-importing would wipe
ticked status, so after the first import, edit the live sheet directly (new tabs/rows by hand
or via Chrome), not by re-import.
