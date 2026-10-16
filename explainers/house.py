"""
House style for the deep-learning episode explainers (ported from the pattern-explainers
studio in harikanth-site).

The style contract is MuseDrop's `ManimCinematicCharter` (see
~/Desktop/All/MuseDrop/MuseDrop/Services/ManimCinematicCharter.swift) grafted onto the
layout/API correctness rules from `ManimCEGenerationGuide`. The six charter rules, which
every scene in this studio obeys:

  1. HEADLINE BEFORE SYMBOLS - open each beat with a plain-language claim, then notation.
  2. CAPTION EVERYTHING      - no mobject appears that the narration hasn't named.
  3. CAMERA INTO THE TERM    - zoom the exact thing being spoken about, then restore.
  4. DEPTH ONLY WHEN 3D      - 2D only; draw depth as a 2D sketch when the idea needs it.
  5. NOTHING KEYFRAMED       - motion comes from computed state (ValueTracker updaters).
  6. VISUAL CONTINUITY       - each beat ends on the object the next beat opens with.

Palette is GitHub-dark-adjacent so notation reads cleanly, with a single amber accent for
"the thing the narration is talking about right now" — the eye always knows where to look.
"""

import os

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    AnimationGroup,
    Create,
    FadeIn,
    FadeOut,
    Line,
    ManimColor,
    RoundedRectangle,
    Text,
    VGroup,
    config,
)

# --------------------------------------------------------------------------------------
# Palette — one primary, one accent, one "good", one "gone". Nothing else.
# --------------------------------------------------------------------------------------
BG = ManimColor("#0E1116")  # deep slate ground
PANEL = ManimColor("#161B22")  # card / cell fill
STROKE = ManimColor("#30363D")  # cell border
INK = ManimColor("#E6EDF3")  # primary text
MUTED = ManimColor("#8B949E")  # secondary text
PRIMARY = ManimColor("#6EA8FE")  # the structure itself (array, tree, graph)
ACCENT = ManimColor("#F0B72F")  # the current focus (pointers, live cell)
GOOD = ManimColor("#3FB950")  # found / accepted / win
GONE = ManimColor("#F85149")  # eliminated / rejected
WINDOW = ManimColor("#388BFD")  # an active span/range

# --------------------------------------------------------------------------------------
# Layout contract (from ManimCEGenerationGuide).
#
# Landscape master (1920x1080): frame is x in [-7.1, 7.1], y in [-4, 4].
# Vertical (1080x1920, EXPLAINER_VERTICAL=1, set by `build.py --vertical`): the frame is
# re-declared as 9 x 16 units so a scene unit is about the same number of pixels as in the
# landscape master (120 vs 135 px), i.e. text stays the same physical size on a phone. The
# rails move to the top/bottom of the tall frame, headlines wrap to two lines, and the stage
# is narrower (SAFE_W 4.2) but much taller. Scenes read `VERTICAL` to switch side-by-side
# compositions to stacked ones; everything else (rails, fit_stage, layout QA) adapts here.
# --------------------------------------------------------------------------------------
VERTICAL = os.environ.get("EXPLAINER_VERTICAL", "") == "1"

if VERTICAL:
    config.frame_height = 16.0
    config.frame_width = 9.0
    SAFE_W = 4.2  # half-width of the safe box
    SAFE_H = 7.4
    TITLE_Y = 6.55
    CAPTION_Y = -6.7
    STAGE_TOP = 5.4
    STAGE_BOTTOM = -5.6
else:
    SAFE_W = 6.8
    SAFE_H = 3.8
    TITLE_Y = 3.15  # headline rail baseline
    CAPTION_Y = -3.35  # caption rail baseline
    STAGE_TOP = 2.35  # where content may start (below the headline)
    STAGE_BOTTOM = -2.75  # where content must end (above the caption)

FONT = "Helvetica Neue"
MONO = "Menlo"

TITLE_FS = 34
HEAD_FS = 26
BODY_FS = 22
SMALL_FS = 18
MONO_FS = 22


def _wrap(text: str, max_chars: int) -> str:
    """Greedy word wrap, used for the rails in vertical mode (max two-ish lines)."""
    words, lines, line = text.split(), [], ""
    for w in words:
        if line and len(line) + 1 + len(w) > max_chars:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    return "\n".join(lines)


def bg(scene) -> None:
    """Charter rule: set the ground first, always."""
    scene.camera.background_color = BG


# --------------------------------------------------------------------------------------
# Rails — the two persistent bands. A beat's headline is the plain-language claim
# (charter rule 1); the caption is the verbatim line being spoken, so the video is
# fully readable muted and the WebVTT track matches the frame.
# --------------------------------------------------------------------------------------
def headline(text: str, color=INK, fs: int = TITLE_FS) -> Text:
    if VERTICAL:
        text = _wrap(text, 24)
    t = Text(text, font=FONT, font_size=fs, color=color, weight="MEDIUM")
    if t.width > 2 * SAFE_W:
        t.scale_to_fit_width(2 * SAFE_W)
    t.move_to([0, TITLE_Y, 0])
    return t


def caption(text: str, color=MUTED, fs: int = BODY_FS) -> Text:
    if VERTICAL:
        text = _wrap(text, 32)
    t = Text(text, font=FONT, font_size=fs, color=color)
    if t.width > 2 * SAFE_W:
        t.scale_to_fit_width(2 * SAFE_W)
    t.move_to([0, CAPTION_Y, 0])
    return t


def label(text: str, color=INK, fs: int = BODY_FS, mono: bool = False) -> Text:
    return Text(text, font=MONO if mono else FONT, font_size=fs, color=color)


def rule(width: float = 2 * SAFE_W, color=STROKE) -> Line:
    """A hairline under the headline — a cheap, consistent sense of design."""
    ln = Line(LEFT * width / 2, RIGHT * width / 2, color=color, stroke_width=1.4)
    ln.move_to([0, TITLE_Y - (0.95 if VERTICAL else 0.45), 0])
    return ln


# --------------------------------------------------------------------------------------
# Beat plumbing. Every beat is: swap the headline+caption rails, then animate the stage.
# Keeping this in one helper is what makes 15 films feel like one series.
# --------------------------------------------------------------------------------------
def rails_of(scene) -> list:
    """The rail mobjects currently on screen (tracked on the scene, not on the mobject)."""
    return list(getattr(scene, "_rails", []))


def swap_rails(
    scene,
    head: Text | None,
    cap: Text | None,
    *,
    run_time: float = 0.5,
):
    """Fade the headline/caption rails to a new pair.

    Sequential, not simultaneous: cross-fading two headlines renders both strings on top of
    each other for the length of the transition, which is unreadable. Out first, then in.

    Rails are tracked in `scene._rails` rather than by tagging the mobject, so nothing
    depends on manim internals.
    """
    old = rails_of(scene)
    new = [m for m in (head, cap) if m is not None]
    if old:
        scene.play(*[FadeOut(m, run_time=run_time * 0.45) for m in old])
    if new:
        scene.play(*[FadeIn(m, run_time=run_time * 0.55) for m in new])
    scene._rails = new


def clear_stage(scene, *, run_time: float = 0.4):
    """Fade out everything that is not a rail — the end-of-beat cleanup."""
    rails = {id(m) for m in rails_of(scene)}
    stage = [m for m in scene.mobjects if id(m) not in rails]
    if stage:
        scene.play(*[FadeOut(m, run_time=run_time) for m in stage])


def drop_stage(scene, *mobs):
    """Fade out specific stage mobjects (keeps the rails)."""
    if mobs:
        scene.play(*[FadeOut(m, run_time=0.35) for m in mobs])


def fit_stage(group: VGroup, *, max_w: float = 2 * SAFE_W - 0.4, max_h: float | None = None):
    """Downscale-only fit so a stack never leaks past the safe box (guide rule)."""
    if max_h is None:
        max_h = STAGE_TOP - STAGE_BOTTOM
    if group.width > max_w:
        group.scale_to_fit_width(max_w)
    if group.height > max_h:
        group.scale_to_fit_height(max_h)
    return group


def stage_center(y: float = -0.15):
    """Where a beat's main visual lives (slightly below centre: the headline owns the top).

    In vertical mode the same call is fine: the stage is centred on y=0 in both frames."""
    return [0, y, 0]


def card(width: float, height: float, color=STROKE, fill=PANEL, fill_opacity: float = 1.0):
    return RoundedRectangle(
        corner_radius=0.12,
        width=width,
        height=height,
        stroke_color=color,
        stroke_width=1.6,
        fill_color=fill,
        fill_opacity=fill_opacity,
    )


# Convenience re-exports so scene files need a single import.
__all__ = [
    "VERTICAL", "BG", "PANEL", "STROKE", "INK", "MUTED", "PRIMARY", "ACCENT", "GOOD", "GONE", "WINDOW",
    "SAFE_W", "SAFE_H", "TITLE_Y", "CAPTION_Y", "STAGE_TOP", "STAGE_BOTTOM",
    "FONT", "MONO", "TITLE_FS", "HEAD_FS", "BODY_FS", "SMALL_FS", "MONO_FS",
    "bg", "headline", "caption", "label", "rule", "swap_rails", "clear_stage",
    "drop_stage", "rails_of", "fit_stage", "stage_center", "card",
    "Text", "VGroup", "ORIGIN", "UP", "DOWN", "LEFT", "RIGHT",
    "Create", "FadeIn", "FadeOut", "AnimationGroup", "config", "ManimColor",
]
