#!/usr/bin/env python3
"""
Smoke-test one scene without synthesizing narration — the fast authoring loop.

    $PY smoke.py M01E01                # episode code or scene file stem
    $PY smoke.py M01E01 --vertical     # check the 1080x1920 layout
    $PY smoke.py M01E01 --frames       # also dump one PNG per beat end for eyeballing

Copies the scene + shared modules into an isolated work dir, writes a stub narration
manifest sized to the scene's own beat count, renders at preview quality, and prints the
deterministic layout QA (off-frame / text-overlap) the render recorded. Iterate on the
scene until this exits 0 with no layout issues.

Must be run with manim's interpreter, e.g.

    ~/.local/share/uv/tools/manim/bin/python smoke.py M01E01
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENES = HERE / "scenes"
SMOKE_ROOT = HERE / "work" / "_smoke"

STUB_BEAT_SECONDS = 5.0  # representative of the real piper narration


def load_module(slug: str):
    path = SCENES / f"{slug}.py"
    if not path.exists():
        raise SystemExit(f"no scene module at {path}")
    spec = importlib.util.spec_from_file_location(slug, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[slug] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="episode code (M01E01) or scene file stem")
    ap.add_argument("--vertical", action="store_true", help="smoke the 1080x1920 layout")
    ap.add_argument("--frames", action="store_true",
                    help="write one PNG at the end of every beat to <work>/frames/")
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    from build import resolve_stem  # noqa: E402

    slug = resolve_stem(args.target)
    mod = load_module(slug)
    beats = list(getattr(mod, "BEATS", []))
    scene_class = getattr(mod, "SCENE_CLASS", None)
    if not beats or not scene_class:
        raise SystemExit(f"{slug}: needs both BEATS and SCENE_CLASS")

    work = SMOKE_ROOT / (f"{slug}-vertical" if args.vertical else slug)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    for name in ("house.py", "shots.py", "prebaked_voice.py"):
        shutil.copy2(HERE / name, work / name)
    shutil.copy2(SCENES / f"{slug}.py", work / "scene.py")
    (work / "voice_manifest.json").write_text(
        json.dumps([STUB_BEAT_SECONDS] * len(beats))
    )

    words = sum(len(b.split()) for b in beats)
    print(f"smoke: {slug} - {len(beats)} beats, {words} words "
          f"(~{words / 2.67:.0f}s at the default pace), class {scene_class}"
          f"{' [vertical]' if args.vertical else ''}")
    env = dict(os.environ, EXPLAINER_VERTICAL="1" if args.vertical else "0")
    cmd = ["manim", "render", "-ql", "--disable_caching", "--media_dir", "media"]
    if args.vertical:
        cmd += ["--resolution", "480,854"]
    res = subprocess.run(cmd + ["scene.py", scene_class],
                         cwd=str(work), capture_output=True, text=True, env=env)
    if res.returncode != 0:
        tail = (res.stdout + res.stderr)[-3000:]
        print(tail)
        raise SystemExit(f"x render failed for {slug}")

    clean = True
    report = work / "layout_report.json"
    if report.exists():
        issues = json.loads(report.read_text())
        if issues:
            clean = False
            print(f"! layout QA flagged {len(issues)} beat(s):")
            for entry in issues:
                for iss in entry["issues"][:5]:
                    print(f"    beat {entry['index']} @{entry['start']}s: {iss}")
            print("  (fix these, then re-run: off-frame or overlapping text is a bug)")
    if clean:
        print("ok layout clean")

    vids = [p for p in (work / "media").rglob("*.mp4") if "partial_movie_files" not in p.parts]
    if vids:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(vids[0])],
            capture_output=True, text=True,
        )
        print(f"ok rendered {out.stdout.strip()}s of video: {vids[0]}")
        if args.frames:
            fdir = work / "frames"
            fdir.mkdir(exist_ok=True)
            timings = json.loads((work / "beat_timings.json").read_text())
            for t in timings:
                at = max(0.0, float(t["end"]) - 0.3)
                subprocess.run(
                    ["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.2f}", "-i", str(vids[0]),
                     "-frames:v", "1", str(fdir / f"beat{t['index']:02d}.png")],
                    capture_output=True,
                )
            print(f"ok frames in {fdir}")
    print(f"  work dir: {work}")
    if not clean:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
