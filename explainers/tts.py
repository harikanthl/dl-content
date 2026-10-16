"""
Narration synthesis. Pluggable backends, picked by availability:

  * `piper`  — neural TTS (preferred when a voice model is present under `voices/`)
  * `say`    — macOS built-in, always available, standard-quality voices

Every backend writes 48 kHz mono PCM WAV so the mux never has to resample mid-pipeline.
A `loudnorm` pass levels every beat, so beats cut together without audible jumps.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOICES = HERE / "voices"

PIPER_BIN = HERE / ".venv-tts" / "bin" / "piper"
# Preference order: a good neural voice, then the best built-in macOS voices.
PIPER_VOICE_CANDIDATES = [
    "en_US-ryan-high",
    "en_US-lessac-high",
    "en_US-amy-medium",
    "en_US-lessac-medium",
]
SAY_VOICE_CANDIDATES = ["Samantha", "Alex", "Daniel"]


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _duration(path: Path) -> float:
    out = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path),
    ])
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def _wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


class Voice:
    """The active narration backend, resolved once per run.

    `length_scale` is the pace knob for piper: 1.0 is the model's native rate (~193 wpm,
    which is brisk for technical narration), and larger values slow it down. Measured on a
    full 355-word film with `en_US-ryan-high`:

        1.0 -> 193 wpm (1:50)     1.3 -> 160 wpm (2:13)
        1.2 -> 170 wpm (2:05)     1.4 -> 151 wpm (2:21)

    1.3 is the default: 160 wpm is the usual pace for an explainer, and the films still land
    near the two-minute target. For the `say` fallback the equivalent knob is words per
    minute (`rate`), which is slower when lower.
    """

    def __init__(self, voice: str | None = None, rate: int = 172, length_scale: float = 1.3):
        self.rate = rate
        self.length_scale = length_scale
        self.backend, self.model, self.name = self._resolve(voice)

    def _resolve(self, requested: str | None):
        # 1. piper, if the binary + a voice model both landed.
        if PIPER_BIN.exists():
            names = [requested] if requested else PIPER_VOICE_CANDIDATES
            for n in names:
                if n and (VOICES / f"{n}.onnx").exists():
                    return "piper", VOICES / f"{n}.onnx", n
        # 2. macOS say.
        available = _run(["say", "-v", "?"]).stdout
        for v in ([requested] if requested else SAY_VOICE_CANDIDATES):
            if v and v in available:
                return "say", None, v
        return "say", None, "Samantha"

    @property
    def describe(self) -> str:
        if self.backend == "piper":
            return f"{self.backend}:{self.name} (length-scale {self.length_scale})"
        return f"{self.backend}:{self.name} (rate {self.rate})"

    def synth(self, text: str, out_wav: Path) -> float:
        """Synthesize `text` to a levelled 48 kHz mono WAV. Returns its duration in seconds."""
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        raw = out_wav.with_suffix(".raw.wav")

        if self.backend == "piper":
            res = subprocess.run(
                [
                    str(PIPER_BIN),
                    "--model", str(self.model),
                    "--length-scale", str(self.length_scale),
                    "--output_file", str(raw),
                ],
                input=text, capture_output=True, text=True,
            )
            if res.returncode != 0 or not raw.exists():
                raise RuntimeError(f"piper failed: {res.stderr[-500:]}")
        else:
            aiff = out_wav.with_suffix(".aiff")
            # `say` takes words per minute; scale the rate inversely so a length_scale above
            # 1 slows `say` the same way it slows piper.
            wpm = max(90, int(self.rate / max(self.length_scale, 0.1)))
            res = _run([
                "say", "-v", self.name, "-r", str(wpm),
                "-o", str(aiff), "--data-format=LEF32@22050", text,
            ])
            if res.returncode != 0 or not aiff.exists():
                raise RuntimeError(f"say failed: {res.stderr[-500:]}")
            raw = aiff

        # Level every beat identically, then land on 48 kHz mono PCM.
        res = subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", str(raw),
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000",
            "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(out_wav),
        ], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg level failed: {res.stderr[-500:]}")
        if raw != out_wav and raw.exists():
            raw.unlink()
        return _wav_seconds(out_wav)


def synth_beats(beats: list[str], workdir: Path, voice: Voice | None = None) -> list[float]:
    """Synthesize every beat's line, returning the measured duration of each."""
    voice = voice or Voice()
    vdir = workdir / "voice"
    if vdir.exists():
        shutil.rmtree(vdir)
    vdir.mkdir(parents=True, exist_ok=True)
    durations = []
    for i, text in enumerate(beats):
        wav = vdir / f"{i:03d}.wav"
        durations.append(voice.synth(text, wav))
    (workdir / "voice_manifest.json").write_text(
        json.dumps([round(d, 3) for d in durations])
    )
    (workdir / "voice_settings.json").write_text(
        json.dumps({"voice": voice.describe, "length_scale": voice.length_scale})
    )
    return durations
