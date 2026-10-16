# LLMs from First Principles

**216 episodes. Every piece of a modern LLM, built from scratch on camera and checked against
the real thing.** PyTorch → the math → a transformer → training → CUDA/Triton → FlashAttention →
modern architectures → distributed training → SFT/LoRA/DPO/GRPO → quantization → inference
engines (vLLM, SGLang) → evals and interpretability → the DeepSeek playbook → research
reproductions → capstones.

Plus **9 flagship projects** with hard bars (FlashAttention in Triton within 70% of FA2, an
inference engine at 60% of vLLM, a 4D-parallel trainer at ≥35% MFU, a published GRPO
reproduction with an original ablation, DeepSeek-V3 in miniature, AI-control safety research,
merged upstream PRs). The episodes are the path. The projects are the resume.

## Layout

```
curriculum/
  curriculum.json     SOURCE OF TRUTH: modules, episodes, sources registry
  projects.json       the 9 flagship projects and their bars
  outlines/Mxx.json   per-episode scripts (hook, timed beats, math, receipt, explainer lines, posts)
  schedule.csv        day-by-day calendar (generated)
  progress.csv        per-episode state (generated, merged, never loses state)
  CURRICULUM.md       human index (generated)
prep/Mxx-*/           full prep sheets, read the night before
code/Mxx-*/           the code typed on camera + its receipt tests (pytest)
explainers/           Manim explainer engine + one scene per episode
edit/<code>/          per-episode HyperFrames edit (raw, renders)
funding/              GPU credit programs and outreach drafts
sheets/               LLM-Roadmap.xlsx (imported into Google Sheets)
scripts/              dl CLI, build_index.py, build_sheet.py, seed_curriculum.py
templates/            prep sheet template, outline spec
PRODUCTION.md         format, pipeline, publishing
HOW-TO.md             read first when picking this up cold
```

## Daily loop

```bash
dl today              # today's two slots + tomorrow's (read those tonight)
dl start              # nvim | pytest-on-save | prep sheet, in tmux
dl script M05E06      # the timed beats
dl test               # the receipt
dl done M05E06        # recorded → ask Claude "edit M05E06"
dl stats
```

Add to `~/.zshrc` (done): `alias dl='~/Documents/DL-content/scripts/dl'`

## Rebuild generated files

```bash
python3 scripts/build_index.py           # CURRICULUM.md, schedule.csv, progress.csv
.venv/bin/python scripts/build_sheet.py   # sheets/LLM-Roadmap.xlsx → re-import into Google Sheets
```
