# Outline spec: one JSON object per episode

Files: `curriculum/outlines/<MODULE>.json`, a JSON array, one object per episode of that module,
in episode order. These become the "Scripts" in the Google Sheet, so they must be usable by
a person standing in front of a camera with no other notes.

```json
{
  "code": "M05E06",
  "hook": "One spoken sentence, <= 25 words. The surprising fact or payoff. Never 'today we will'.",
  "beats": [
    "0:00 HOOK — exact line to say (face cam)",
    "0:15 QUESTION — 2-3 sentences: why this matters, tied to a named real model",
    "1:00 INTUITION — the analogy and the one picture; [EXPLAINER plays here]",
    "3:00 MATH — each equation in order, written in plain text/LaTeX, with the reason for each step",
    "6:00 CODE — what to type, in order: function signatures, key lines, shapes",
    "10:00 RECEIPT — the exact assert/benchmark that proves it works and the expected output",
    "11:00 MEASURE — the number or plot to show",
    "12:00 TAKEAWAY — one sentence + the interview line",
    "12:30 NEXT — 'Next: <next episode title>'"
  ],
  "key_math": "The 1-4 equations of the episode in LaTeX-ish plain text. Empty string if none.",
  "code_task": "What the from-scratch code is, with function names and tensor shapes.",
  "receipt": "Exact check: e.g. torch.testing.assert_close(my_attn(q,k,v), F.scaled_dot_product_attention(q,k,v,is_causal=True)).",
  "gotchas": "2-3 real bugs viewers will hit, ';'-separated.",
  "interview_q": "One real interview question this episode answers, then a 1-2 sentence model answer.",
  "explainer": ["6-10 narration lines (each 15-30 words) for the 60-90 s Manim explainer. Plain spoken English, first person plural, each line names exactly what is on screen. Headline claim before symbols."],
  "explainer_visuals": ["one visual per narration line: what Manim draws (tensor grid, heatmap, arrows, plot)."],
  "short": "Which beat becomes the 45-60 s vertical Short, plus its on-screen caption.",
  "yt_title": "<= 70 chars. Search term first, then the promise. e.g. 'Self-Attention from Scratch in PyTorch — and why it matches SDPA exactly'",
  "thumbnail": "3-5 huge words",
  "x_post": "<= 270 chars. One insight + 'Code + video:' placeholder. No hashtags spam (max 1).",
  "linkedin": "2-4 sentence post: what I built today, the number, what I learned.",
  "prereq": "Episode codes this depends on."
}
```

## Quality bar (read this twice)

- **Technically exact.** Correct shapes, correct equations, correct API names for current PyTorch
  (2.x: `torch.nn.functional.scaled_dot_product_attention`, `torch.nn.RMSNorm`, `torch.compile`,
  FSDP2 `fully_shard`). If unsure of an API, describe the operation instead of inventing a name.
- **Hands-on and research-grade.** Every episode builds or measures something real. The receipt
  compares against the production implementation or a published number.
- **Timings sum to 10-15 min.** Adjust beat timestamps to the content; a derivation-heavy
  episode can spend 5 min on MATH.
- **Tone:** smart friend who is tired. Short sentences. No hype, no "simply", no emojis in lines.
  Say the hard part is hard, then make it easy.
- **No em dashes inside spoken lines** (the channel owner's style); use commas or periods.
  The "0:00 HOOK —" beat label separator is fine.
