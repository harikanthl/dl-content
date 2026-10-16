#!/usr/bin/env python3
"""
Build one episode explainer end to end.

    $PY build.py M01E01                       # by episode code (looked up in curriculum.json)
    $PY build.py m01e01_tensors_what_they_actually_are   # or by scene file stem
    $PY build.py M01E01 --vertical            # 1080x1920 cut for Shorts / Reels / TikTok
    $PY build.py M01E01 --voice-only
    $PY build.py --all                        # every scene file under scenes/

($PY is manim's interpreter: ~/.local/share/uv/tools/manim/bin/python)

Pipeline (MuseDrop's host-TTS + prebaked-voice architecture, ported from the
harikanth-site pattern-explainers studio):

  1. import the scene module and read its `BEATS` narration list (single source of truth:
     the words and the animation live in one file and cannot drift apart);
  2. synthesize each beat's line to a levelled 48 kHz mono WAV and measure its true spoken
     duration -> `voice_manifest.json`;
  3. render the scene with manim at 1920x1080/30 (or 1080x1920 with --vertical), locked to
     those durations;
  4. read `beat_timings.json` (the ACTUAL rendered beat boundaries) and lay the voice lines
     onto the video at their measured starts, so sync survives a beat whose animation
     overran its narration;
  5. emit out/<CODE>.mp4, a WebVTT caption track, a poster frame, and merge a metadata row
     into explainers.json keyed by episode code.

Large intermediates in work/ are deleted after a successful build unless --keep-work.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCENES = HERE / "scenes"
WORK = HERE / "work"
OUT = HERE / "out"
MANIFEST = HERE / "explainers.json"
CURRICULUM = REPO / "curriculum" / "curriculum.json"

ENGINE_FILES = ("house.py", "shots.py", "prebaked_voice.py")

# Render tiers: (landscape resolution, vertical resolution, fps). `preview` is for fast
# visual iteration; narration durations are real, so the layout you inspect is the layout
# you ship.
QUALITY = {
    "preview": ("854,480", "480,854", 15),
    "720p30": ("1280,720", "720,1280", 30),
    "1080p30": ("1920,1080", "1080,1920", 30),
}

sys.path.insert(0, str(HERE))
from tts import Voice, synth_beats  # noqa: E402


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def curriculum_episodes() -> list[dict]:
    try:
        return json.loads(CURRICULUM.read_text()).get("episodes", [])
    except Exception:
        return []


def resolve_stem(target: str) -> str:
    """Accept an episode code (M01E01), a scene file stem, or a path to the scene file."""
    p = Path(target)
    if p.suffix == ".py":
        return p.stem
    if (SCENES / f"{target}.py").exists():
        return target
    code = target.upper()
    for ep in curriculum_episodes():
        if ep.get("code", "").upper() == code and ep.get("explainer"):
            return Path(ep["explainer"]).stem
    # Fall back to a scene file whose name starts with the lower-cased code.
    hits = sorted(SCENES.glob(f"{target.lower()}_*.py"))
    if hits:
        return hits[0].stem
    raise SystemExit(f"cannot resolve '{target}' to a scene under {SCENES}")


def episode_code(mod, stem: str) -> str:
    code = getattr(mod, "EPISODE_CODE", None)
    if code:
        return code
    for ep in curriculum_episodes():
        if Path(ep.get("explainer", "")).stem == stem:
            return ep["code"]
    return stem.split("_", 1)[0].upper()


def load_scene_module(stem: str):
    path = SCENES / f"{stem}.py"
    if not path.exists():
        raise SystemExit(f"no scene module for '{stem}' at {path}")
    spec = importlib.util.spec_from_file_location(stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[stem] = mod
    spec.loader.exec_module(mod)
    if not getattr(mod, "BEATS", None):
        raise SystemExit(f"{path} defines no BEATS narration list")
    cls = getattr(mod, "SCENE_CLASS", None)
    if cls is None:
        raise SystemExit(f"{path} defines no SCENE_CLASS")
    return mod, cls


def ffprobe_duration(path: Path) -> float:
    out = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path),
    ])
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def ts(seconds: float) -> str:
    """WebVTT timestamp."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# --------------------------------------------------------------------------------------
# pipeline stages
# --------------------------------------------------------------------------------------
def workdir_for(stem: str, vertical: bool) -> Path:
    return WORK / (f"{stem}-vertical" if vertical else stem)


def prepare(stem: str, voice: Voice | None = None,
            vertical: bool = False) -> tuple[Path, list[str], list[float]]:
    mod, _ = load_scene_module(stem)
    beats = list(mod.BEATS)
    work = workdir_for(stem, vertical)
    work.mkdir(parents=True, exist_ok=True)

    for name in ENGINE_FILES:
        shutil.copy2(HERE / name, work / name)
    shutil.copy2(SCENES / f"{stem}.py", work / "scene.py")

    for name in ("voice_manifest.json", "beat_timings.json", "layout_report.json"):
        (work / name).unlink(missing_ok=True)

    durations = synth_beats(beats, work, voice=voice)
    total = sum(durations)
    words = sum(len(b.split()) for b in beats)
    print(f"  narration: {len(beats)} beats, {words} words, {total:.1f}s "
          f"({voice.describe if voice else 'auto'})")
    return work, beats, durations


def render(work: Path, scene_class: str, quality: str = "1080p30",
           vertical: bool = False) -> Path:
    media = work / "media"
    if media.exists():
        shutil.rmtree(media)
    land, vert, fps = QUALITY[quality]
    resolution = vert if vertical else land
    cmd = [
        "manim", "render", "--disable_caching",
        "--resolution", resolution, "--frame_rate", str(fps),
        "--media_dir", str(media),
        str(work / "scene.py"), scene_class,
    ]
    env = dict(os.environ, EXPLAINER_VERTICAL="1" if vertical else "0")
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(work), env=env)
    if res.returncode != 0:
        print(res.stdout[-2500:])
        print(res.stderr[-2500:])
        raise SystemExit(f"manim render failed for {scene_class}")

    # The render is done; surface the deterministic layout QA if it flagged anything.
    report = work / "layout_report.json"
    if report.exists():
        try:
            issues = json.loads(report.read_text())
            if issues:
                print(f"  ! layout QA flagged {len(issues)} beat(s):")
                for entry in issues[:8]:
                    for iss in entry["issues"][:4]:
                        print(f"      beat {entry['index']}: {iss}")
        except Exception:
            pass

    vids = [p for p in media.rglob("*.mp4") if "partial_movie_files" not in p.parts]
    if not vids:
        raise SystemExit("manim produced no mp4 (see output above)")
    vids.sort(key=lambda p: p.stat().st_size, reverse=True)
    return vids[0]


def build_audio(work: Path, beats: list[str], video: Path) -> Path | None:
    """Lay each beat's voice line at its MEASURED rendered start and concat to one track."""
    timing_file = work / "beat_timings.json"
    if not timing_file.exists():
        print("  ! no beat_timings.json, skipping narration mux")
        return None
    timings = json.loads(timing_file.read_text())

    vdir = work / "voice"
    clips = []
    for i, _text in enumerate(beats):
        wav = vdir / f"{i:03d}.wav"
        if not wav.exists():
            continue
        entry = next((t for t in timings if t["index"] == i), None)
        start = float(entry["start"]) if entry else 0.0
        clips.append((start, wav, ffprobe_duration(wav)))

    if not clips:
        return None

    seq = work / "audio_seq"
    if seq.exists():
        shutil.rmtree(seq)
    seq.mkdir(parents=True, exist_ok=True)

    lines = []
    cursor = 0.0
    pad_index = 0
    for start, wav, dur in clips:
        gap = start - cursor
        if gap > 0.02:
            pad = seq / f"pad{pad_index:03d}.wav"
            _run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                  "anullsrc=r=48000:cl=mono", "-t", f"{gap:.3f}", str(pad)])
            lines.append(f"file '{pad.name}'")
            pad_index += 1
        lines.append(f"file '{wav.name}'")
        cursor = start + dur
        # A short breath between beats so lines don't butt together.
        tail = seq / f"tail{pad_index:03d}.wav"
        _run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
              "anullsrc=r=48000:cl=mono", "-t", "0.28", str(tail)])
        lines.append(f"file '{tail.name}'")
        pad_index += 1
        cursor += 0.28

    for _, wav, _d in clips:
        shutil.copy2(wav, seq / wav.name)
    (seq / "list.txt").write_text("\n".join(lines) + "\n")

    track = work / "narration.wav"
    res = subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
        "-i", str(seq / "list.txt"), "-c:a", "pcm_s16le", str(track),
    ], capture_output=True, text=True, cwd=str(seq))
    if res.returncode != 0:
        print(f"  ! audio concat failed: {res.stderr[-400:]}")
        return None

    final = work / "final.mp4"
    res = subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(track),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
        "-movflags", "+faststart", str(final),
    ], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  ! mux failed: {res.stderr[-400:]}")
        return None
    return final


def write_captions(work: Path, beats: list[str], dest: Path) -> None:
    timing_file = work / "beat_timings.json"
    timings = json.loads(timing_file.read_text()) if timing_file.exists() else []
    rows = ["WEBVTT", ""]
    for i, text in enumerate(beats):
        entry = next((t for t in timings if t["index"] == i), None)
        if entry is None:
            continue
        start, end = float(entry["start"]), float(entry["end"])
        # Split long lines into readable caption cues of ~64 chars on word boundaries.
        words, line, chunks = text.split(), "", []
        for w in words:
            if len(line) + len(w) + 1 > 64:
                chunks.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
        if line:
            chunks.append(line)
        total = sum(len(c) for c in chunks) or 1
        t = start
        for c in chunks:
            share = (len(c) / total) * max(end - start, 0.8)
            rows.append(f"{ts(t)} --> {ts(min(t + share, end))}")
            rows.append(c)
            rows.append("")
            t += share
    dest.write_text("\n".join(rows))


def write_poster(video: Path, dest: Path, at: float) -> None:
    _run([
        "ffmpeg", "-y", "-v", "error", "-ss", f"{at:.2f}", "-i", str(video),
        "-frames:v", "1", "-q:v", "3", str(dest),
    ])


def chapter_rows(mod, timings: list[dict]) -> list[dict]:
    labels = getattr(mod, "CHAPTERS", {})
    rows = []
    for idx, label in sorted(labels.items()):
        entry = next((t for t in timings if t["index"] == idx), None)
        if entry is None:
            continue
        rows.append({"at": round(float(entry["start"]), 2), "label": label})
    return rows


def update_manifest(code: str, mod, work: Path, video: Path, vertical: bool) -> None:
    """Merge this episode's metadata row into explainers.json (keyed by episode code).

    Asset URLs are deliberately NOT stored: assets are named <CODE>.{mp4,jpg,vtt}
    (<CODE>-vertical.* for the vertical cut), so a site composes URLs from its own base.
    """
    timing_file = work / "beat_timings.json"
    timings = json.loads(timing_file.read_text()) if timing_file.exists() else []
    duration = round(ffprobe_duration(video), 1)

    lock_path = MANIFEST.with_suffix(".lock")
    lock = open(lock_path, "w")
    try:
        if fcntl:
            fcntl.flock(lock, fcntl.LOCK_EX)
        data = {}
        if MANIFEST.exists():
            try:
                data = json.loads(MANIFEST.read_text())
            except Exception:
                data = {}
        row = data.get(code, {})
        if vertical:
            row.setdefault("title", mod.TITLE)
            row["vertical"] = {"duration": duration, "chapters": chapter_rows(mod, timings)}
        else:
            row.update({
                "title": mod.TITLE,
                "summary": getattr(mod, "SUMMARY", ""),
                "duration": duration,
                "beats": len(mod.BEATS),
                "chapters": chapter_rows(mod, timings),
            })
        data[code] = row
        MANIFEST.write_text(json.dumps(dict(sorted(data.items())), indent=2) + "\n")
    finally:
        if fcntl:
            fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def clean_work(work: Path) -> None:
    """Delete the large intermediates; keep the small JSON reports for debugging."""
    for name in ("media", "audio_seq", "voice", "__pycache__"):
        shutil.rmtree(work / name, ignore_errors=True)
    for name in ("final.mp4", "narration.wav"):
        (work / name).unlink(missing_ok=True)


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def build(stem: str, voice: Voice | None = None, quality: str = "1080p30",
          vertical: bool = False, copy_to: str | None = None,
          keep_work: bool = False) -> str:
    mod, scene_class = load_scene_module(stem)
    code = episode_code(mod, stem)
    print(f"\n> {code}  ({stem}){'  [vertical]' if vertical else ''}")
    work, beats, _durations = prepare(stem, voice=voice, vertical=vertical)

    land, vert, fps = QUALITY[quality]
    print(f"  rendering {vert if vertical else land} @ {fps}fps ...")
    video = render(work, scene_class, quality, vertical=vertical)
    print(f"  video: {video.name} ({ffprobe_duration(video):.1f}s, "
          f"{video.stat().st_size / 1e6:.1f} MB)")

    final = build_audio(work, beats, video) or video

    name = f"{code}-vertical" if vertical else code
    OUT.mkdir(parents=True, exist_ok=True)
    dest_video = OUT / f"{name}.mp4"
    shutil.copy2(final, dest_video)

    poster = OUT / f"{name}.jpg"
    poster_at = getattr(mod, "POSTER_AT", ffprobe_duration(final) * 0.35)
    write_poster(dest_video, poster, poster_at)

    vtt = OUT / f"{name}.vtt"
    write_captions(work, beats, vtt)

    update_manifest(code, mod, work, dest_video, vertical)

    if copy_to:
        dest = Path(copy_to).expanduser()
        dest.mkdir(parents=True, exist_ok=True)
        for f in (dest_video, poster, vtt):
            shutil.copy2(f, dest / f.name)
        print(f"  copied assets to {dest}")

    if not keep_work:
        clean_work(work)
    print(f"  done: {ffprobe_duration(dest_video):.1f}s -> {dest_video}")
    return code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="*", help="episode code(s) (M01E01) or scene file stem(s)")
    ap.add_argument("--all", action="store_true", help="every scene file under scenes/")
    ap.add_argument("--vertical", action="store_true",
                    help="render the 1080x1920 cut (Shorts/Reels/TikTok) as <CODE>-vertical.*")
    ap.add_argument("--voice-only", action="store_true",
                    help="synthesize narration only, skip rendering")
    ap.add_argument("--voice", default=None, help="piper voice name override")
    ap.add_argument("--length-scale", type=float, default=1.3,
                    help="narration pace: 1.0 is native (~193 wpm), 1.3 is ~160 wpm "
                         "(the default), 1.4 is ~151 wpm")
    ap.add_argument("--quality", default="1080p30", choices=sorted(QUALITY),
                    help="render tier (preview is fast, for layout iteration)")
    ap.add_argument("--copy-to", default=None,
                    help="also copy <CODE>.{mp4,jpg,vtt} into this directory "
                         "(e.g. the site's public media folder)")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep the large intermediates in work/ (default: delete on success)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="render N scenes concurrently (manifest writes are locked)")
    args = ap.parse_args()

    if args.all:
        stems = sorted(p.stem for p in SCENES.glob("*.py") if not p.name.startswith("_"))
    elif args.target:
        stems = [resolve_stem(t) for t in args.target]
    else:
        raise SystemExit("pass one or more episode codes / scene stems, or --all")

    voice = Voice(args.voice, length_scale=args.length_scale)
    print(f"voice: {voice.describe}")
    if args.voice_only:
        for stem in stems:
            _work, _beats, durations = prepare(stem, voice=voice, vertical=args.vertical)
            print(f"  narration total {sum(durations):.1f}s")
        return

    def one(stem):
        return build(stem, voice=voice, quality=args.quality, vertical=args.vertical,
                     copy_to=args.copy_to, keep_work=args.keep_work)

    jobs = max(1, args.jobs)
    if jobs == 1:
        for stem in stems:
            try:
                one(stem)
            except SystemExit as exc:
                # One bad scene should not abandon the rest of a batch.
                print(f"  x {stem}: {exc}")
    else:
        print(f"rendering {len(stems)} scene(s), {jobs} at a time")
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(one, s): s for s in stems}
            for future in concurrent.futures.as_completed(futures):
                stem = futures[future]
                try:
                    future.result()
                except SystemExit as exc:
                    print(f"  x {stem}: {exc}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  x {stem}: unexpected {type(exc).__name__}: {exc}")
    print("\ndone.")


if __name__ == "__main__":
    main()
