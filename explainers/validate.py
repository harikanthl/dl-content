#!/usr/bin/env python3
"""
Static QA across every scene module: run this before a batch render.

    ~/.local/share/uv/tools/manim/bin/python validate.py

Checks the contract that `build.py` depends on, plus authoring rules that are cheap to verify
without rendering:

  * required module constants present; EPISODE_CODE unique, present in
    curriculum/curriculum.json, and the scene file name matches that episode's `explainer`;
  * BEATS length in the 7-14 band and ~55-100 s of speech at the default pace
    (piper ryan-high, length-scale 1.3, ~2.67 words/sec);
  * no em dashes in narration (house rule: plain first-person-plural English);
  * one `voiceover` call per beat, and no narration text duplicated into a headline/caption;
  * CHAPTERS indices in range;
  * no banned 3D / default-palette usage.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENES = HERE / "scenes"
CURRICULUM = HERE.parent / "curriculum" / "curriculum.json"

WORDS_PER_SECOND = 2.67
MIN_BEATS, MAX_BEATS = 7, 14
MIN_SECONDS, MAX_SECONDS = 55, 100

REQUIRED = ("EPISODE_CODE", "TITLE", "SUMMARY", "SCENE_CLASS", "POSTER_AT", "BEATS", "CHAPTERS")

BANNED = [
    (r"\bThreeDScene\b", "3D scene (charter rule 4: 2D only)"),
    (r"\bSphere\b|\bCube\b|\bSurface\b", "3D mobject (charter rule 4)"),
    (r"\bfrom manim import[^\n]*\bRED\b", "default RED palette"),
]


def load(stem: str):
    sys.path.insert(0, str(HERE))
    path = SCENES / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[stem] = mod
    spec.loader.exec_module(mod)
    return mod


def curriculum() -> dict[str, dict]:
    try:
        eps = json.loads(CURRICULUM.read_text())["episodes"]
    except Exception:
        return {}
    return {e["code"]: e for e in eps}


def main() -> int:
    files = sorted(p for p in SCENES.glob("*.py") if not p.name.startswith("_"))
    if not files:
        print("no scene files found")
        return 1
    eps = curriculum()

    failures = 0
    seen: dict[str, str] = {}
    print(f"{'scene':<46} {'code':<8} {'beats':>5} {'~sec':>6}  status")
    print("-" * 96)

    for path in files:
        stem = path.stem
        src = path.read_text()
        problems: list[str] = []
        try:
            mod = load(stem)
        except Exception as exc:  # noqa: BLE001
            print(f"{stem:<46} {'-':<8} {'-':>5} {'-':>6}  x import failed: {exc}")
            failures += 1
            continue

        for name in REQUIRED:
            if not hasattr(mod, name):
                problems.append(f"missing {name}")

        code = getattr(mod, "EPISODE_CODE", "?")
        beats = list(getattr(mod, "BEATS", []))

        if code in seen:
            problems.append(f"EPISODE_CODE duplicates {seen[code]}")
        seen[code] = stem
        if eps:
            ep = eps.get(code)
            if ep is None:
                problems.append(f"code '{code}' not in curriculum.json")
            elif Path(ep.get("explainer", "")).stem != stem:
                problems.append(f"file name should be {Path(ep.get('explainer', '')).name}")

        if not (MIN_BEATS <= len(beats) <= MAX_BEATS):
            problems.append(f"{len(beats)} beats (want {MIN_BEATS}-{MAX_BEATS})")

        words = sum(len(b.split()) for b in beats)
        seconds = words / WORDS_PER_SECOND
        if not (MIN_SECONDS <= seconds <= MAX_SECONDS):
            problems.append(f"~{seconds:.0f}s of speech (want {MIN_SECONDS}-{MAX_SECONDS})")

        if any("—" in b or "–" in b for b in beats):
            problems.append("em/en dash in narration")

        vo_calls = len(re.findall(r"self\.voiceover\(", src))
        loop_vo = re.search(r"for .* in .*:\s*\n\s*with self\.voiceover\(", src)
        if vo_calls != len(beats) and not loop_vo:
            problems.append(f"{vo_calls} voiceover calls for {len(beats)} beats")

        chapters = getattr(mod, "CHAPTERS", {})
        bad_ch = [k for k in chapters if not (0 <= k < len(beats))]
        if bad_ch:
            problems.append(f"CHAPTERS out of range: {bad_ch}")

        for pattern, why in BANNED:
            if re.search(pattern, src):
                problems.append(why)

        dupe = 0
        for b in beats:
            head = b.strip().rstrip(".")
            if len(head) > 40 and re.search(
                r'(headline|Text)\(\s*["\']' + re.escape(head[:40]), src
            ):
                dupe += 1
        if dupe:
            problems.append(f"{dupe} narration line(s) duplicated into on-screen text")

        status = "ok" if not problems else "x " + "; ".join(problems)
        if problems:
            failures += 1
        print(f"{stem:<46} {code:<8} {len(beats):>5} {seconds:>6.0f}  {status}")

    print("-" * 96)
    print(f"{len(files)} scene(s), {failures} with problems")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
