"""
Plays a scene in lockstep with pre-measured per-beat narration durations.

Vendored from MuseDrop's `PrebakedVoiceScaffold.swift`
(~/Desktop/All/MuseDrop/MuseDrop/Services/PrebakedVoiceScaffold.swift) — same host-TTS +
timing-manifest architecture, lifted out of the Swift string into a real module so the
explainer studios can use it directly.

The contract:

  1. The host synthesizes each beat's narration and writes the exact spoken durations to
     `voice_manifest.json` as a JSON array of seconds.
  2. A scene subclasses `PrebakedVoiceMovingCameraScene` and wraps each beat:
         with self.voiceover(text="...") as tracker:
             self.play(..., run_time=tracker.duration)
  3. On beat exit the helper waits until this beat's CUMULATIVE narration end time, so a
     beat that overran its line is absorbed by later slack instead of pushing every
     following beat late (per-beat-only waiting drifts).
  4. The helper also writes `beat_timings.json` — the ACTUAL rendered [start, end] of each
     beat — which the host mux reads to place each voice line at its measured start. Sync
     therefore survives a beat whose animation overran its narration.

`tracker.duration` is the exact spoken length of the beat, so every `run_time=` written
against it lands the visual change exactly when the words happen.
"""

import json
import os

from manim import Scene, config

try:
    from manim import MovingCameraScene
except Exception:  # pragma: no cover - fall back to the submodule path, never to Scene
    from manim.scene.moving_camera_scene import MovingCameraScene


def _load_durations():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_manifest.json")
    try:
        with open(path) as handle:
            data = json.load(handle)
        return [float(value) for value in data]
    except Exception:
        return []


class _Tracker:
    """manim-voiceover-compatible tracker: `duration` plus `get_remaining_duration()`."""

    def __init__(self, duration, scene, start):
        self.duration = duration
        self._scene = scene
        self._start = start

    def get_remaining_duration(self, buff=0.0):
        elapsed = self._scene.renderer.time - self._start
        return max(0.0, self.duration - elapsed - buff)


def _fit_frame(scene):
    """Beat-boundary safety net: if what's VISIBLE leaks outside the frame, downscale the
    union (only when oversize) and shift it back inside, preserving relative layout.
    Camera-aware (an intentional zoom-in is respected). Deliberately conservative:
    fully-offscreen mobjects are intentional staging; a full-bleed background is ignored;
    tolerances ensure a stable in-bounds layout never re-triggers. Never raises."""
    try:
        from manim import Group

        frame = getattr(scene.camera, "frame", None)
        if frame is not None:
            cx, cy = float(frame.get_center()[0]), float(frame.get_center()[1])
            fw, fh = float(frame.width), float(frame.height)
        else:
            cx, cy = 0.0, 0.0
            fw, fh = float(config.frame_width), float(config.frame_height)
        left_e, right_e = cx - fw / 2, cx + fw / 2
        bot_e, top_e = cy - fh / 2, cy + fh / 2

        def bbox(m):
            return (
                float(m.get_left()[0]),
                float(m.get_right()[0]),
                float(m.get_bottom()[1]),
                float(m.get_top()[1]),
            )

        included = []
        for m in scene.mobjects:
            if not (len(getattr(m, "points", [])) or len(m.submobjects)):
                continue
            l, r, b, t = bbox(m)
            if r <= left_e or l >= right_e or t <= bot_e or b >= top_e:
                continue  # fully offscreen: staged for a later slide-in
            if (r - l) >= 0.95 * fw and (t - b) >= 0.95 * fh:
                continue  # full-bleed background
            included.append(m)
        if not included:
            return
        union = Group(*included)
        l, r, b, t = bbox(union)
        if (r - l) > fw * 1.02 or (t - b) > fh * 1.02:
            factor = min(fw / max(r - l, 1e-6), fh / max(t - b, 1e-6)) * 0.96
            union.scale(factor)
            l, r, b, t = bbox(union)

        def axis_shift(lo, hi, lo_edge, hi_edge, pad=0.1, leak=0.1):
            if (hi - lo) > (hi_edge - lo_edge) - 2 * pad:
                return (lo_edge + hi_edge) / 2.0 - (lo + hi) / 2.0
            if lo < lo_edge - leak:
                return (lo_edge + pad) - lo
            if hi > hi_edge + leak:
                return (hi_edge - pad) - hi
            return 0.0

        dx = axis_shift(l, r, left_e, right_e)
        dy = axis_shift(b, t, bot_e, top_e)
        if dx or dy:
            union.shift([dx, dy, 0.0])
    except Exception:
        pass


def _record_beat_timing(scene, index, start):
    """Append this beat's ACTUAL rendered [start, end] and rewrite beat_timings.json, so the
    host mux can align each voice line to its measured start. Never raises."""
    try:
        boundaries = getattr(scene, "_vt_boundaries", None)
        if boundaries is None:
            boundaries = []
            scene._vt_boundaries = boundaries
        t0 = getattr(scene, "_vt_t0", None)
        if t0 is None:
            t0 = start
        boundaries.append(
            {"index": index, "start": float(start - t0), "end": float(scene.renderer.time - t0)}
        )
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beat_timings.json")
        with open(p, "w") as h:
            json.dump(boundaries, h)
    except Exception:
        pass


_TEXTISH = {
    "Text",
    "MarkupText",
    "Tex",
    "MathTex",
    "Title",
    "Paragraph",
    "DecimalNumber",
    "Integer",
    "SingleStringMathTex",
    "Code",
}


def _describe(m):
    name = type(m).__name__
    for attr in ("text", "tex_string"):
        val = getattr(m, attr, None)
        if isinstance(val, str) and val.strip():
            snippet = " ".join(val.split())
            if len(snippet) > 24:
                snippet = snippet[:24] + "..."
            return name + " '" + snippet + "'"
    return name


def _is_textish(m):
    if type(m).__name__ in _TEXTISH:
        return True
    subs = [
        s for s in getattr(m, "submobjects", []) if len(getattr(s, "points", [])) or s.submobjects
    ]
    return bool(subs) and all(type(s).__name__ in _TEXTISH for s in subs)


def _report_layout(scene, index, start):
    """Deterministic, vision-free layout QA at each beat boundary: off-frame + text-overlap
    problems go to layout_report.json so the host can repair layout without pixels. Text
    sitting fully inside a shape (a card/badge) is intentional and never flagged.
    Never raises."""
    try:
        frame = getattr(scene.camera, "frame", None)
        if frame is not None:
            cx, cy = float(frame.get_center()[0]), float(frame.get_center()[1])
            fw, fh = float(frame.width), float(frame.height)
        else:
            cx, cy = 0.0, 0.0
            fw, fh = float(config.frame_width), float(config.frame_height)
        left_e, right_e = cx - fw / 2, cx + fw / 2
        bot_e, top_e = cy - fh / 2, cy + fh / 2

        def bbox(m):
            return (
                float(m.get_left()[0]),
                float(m.get_right()[0]),
                float(m.get_bottom()[1]),
                float(m.get_top()[1]),
            )

        items = []
        for m in scene.mobjects:
            if not (len(getattr(m, "points", [])) or len(m.submobjects)):
                continue
            l, r, b, t = bbox(m)
            if r <= left_e or l >= right_e or t <= bot_e or b >= top_e:
                continue
            if (r - l) >= 0.95 * fw and (t - b) >= 0.95 * fh:
                continue
            items.append((m, l, r, b, t, _is_textish(m)))

        issues = []
        for (m, l, r, b, t, _tx) in items:
            for side, over in (
                ("left", left_e - l),
                ("right", r - right_e),
                ("bottom", bot_e - b),
                ("top", t - top_e),
            ):
                if over > 0.15:
                    issues.append(
                        {
                            "type": "offframe",
                            "a": _describe(m),
                            "side": side,
                            "leak": round(over, 2),
                        }
                    )
                    break

        def inside(inner, outer):
            il, ir, ib, it = inner
            ol, orr, ob, ot = outer
            e = 0.05
            return il >= ol - e and ir <= orr + e and ib >= ob - e and it <= ot + e

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                mi, li, ri, bi, ti, txi = items[i]
                mj, lj, rj, bj, tj, txj = items[j]
                if not (txi or txj):
                    continue
                ow = min(ri, rj) - max(li, lj)
                oh = min(ti, tj) - max(bi, bj)
                if ow <= 0 or oh <= 0:
                    continue
                area_i = max((ri - li) * (ti - bi), 1e-6)
                area_j = max((rj - lj) * (tj - bj), 1e-6)
                if (ow * oh) / min(area_i, area_j) < 0.2:
                    continue
                if txi and not txj and inside((li, ri, bi, ti), (lj, rj, bj, tj)):
                    continue
                if txj and not txi and inside((lj, rj, bj, tj), (li, ri, bi, ti)):
                    continue
                issues.append({"type": "overlap", "a": _describe(mi), "b": _describe(mj)})

        if not issues:
            return
        report = getattr(scene, "_vt_layout", None)
        if report is None:
            report = []
            scene._vt_layout = report
        t0 = getattr(scene, "_vt_t0", None)
        report.append(
            {
                "index": index,
                "start": round(float(start - (t0 if t0 is not None else start)), 2),
                "issues": issues,
            }
        )
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "layout_report.json")
        with open(p, "w") as h:
            json.dump(report, h)
    except Exception:
        pass


class _VoiceTiming:
    """Mixin providing `voiceover`. Usage:

        with self.voiceover(text="...") as tracker:
            self.play(Create(x), run_time=tracker.duration)
    """

    def _voice_durations(self):
        if not hasattr(self, "_vt_durations"):
            self._vt_durations = _load_durations()
            self._vt_index = 0
            self._vt_t0 = None
            self._vt_expected = 0.0
        return self._vt_durations

    def voiceover(self, text="", duration=None):
        durations = self._voice_durations()
        index = self._vt_index
        self._vt_index += 1
        if duration is not None:
            beat = float(duration)
        elif index < len(durations):
            beat = durations[index]
        else:
            beat = 2.0
        scene = self
        start = scene.renderer.time
        if scene._vt_t0 is None:
            scene._vt_t0 = start
        scene._vt_expected += beat
        expected_end = scene._vt_t0 + scene._vt_expected

        class _Ctx:
            def __enter__(self_inner):
                return _Tracker(beat, scene, start)

            def __exit__(self_inner, *exc):
                _fit_frame(scene)
                _report_layout(scene, index, start)
                remaining = expected_end - scene.renderer.time
                if remaining > 0.04:
                    scene.wait(remaining)
                _record_beat_timing(scene, index, start)
                return False

        return _Ctx()


class PrebakedVoiceScene(_VoiceTiming, Scene):
    pass


class PrebakedVoiceMovingCameraScene(_VoiceTiming, MovingCameraScene):
    pass
