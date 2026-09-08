"""Wake-word hotword engines (openWakeWord / sherpa-onnx KWS / Porcupine).

All three run fully on-device. Config, platform probes and sensitivity accessors
live in :mod:`tools.wake_word`; engines read them lazily through that module (import cycle).
"""

from __future__ import annotations

import logging
import os
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("tools.wake_word")


def _ww():
    from tools import wake_word
    return wake_word


def _ensure_dep(feature: str) -> None:
    from tools import lazy_deps
    lazy_deps.ensure(feature, prompt=False)


class _Engine:
    """Minimal hotword-engine contract: feed int16 frames, get a bool. Subclasses set ``feature``
    (lazy_deps name, ensured before ``_build``) and their own ``cfg`` sub-section ``section``."""

    feature: str = ""
    section: str = ""
    frame_length: int = 1280  # 80 ms at 16 kHz

    #: (matched phrase, profile name) of the most recent fire. Multi-phrase engines
    #: (sherpa) set this for profile routing; single-phrase engines leave it None.
    last_match: Optional[tuple[str, str]] = None

    def __init__(self, cfg: Dict[str, Any]):
        _ensure_dep(self.feature)
        self._build(cfg, _sub(cfg, self.section), _ww())

    def _build(self, cfg: Dict[str, Any], sub: Dict[str, Any], ww) -> None:
        raise NotImplementedError

    def process(self, frame) -> bool:  # frame: 1-D int16 ndarray
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any internal audio/feature buffer (called on every (re)start)."""

    def close(self) -> None:
        """Release engine resources (called once on stop)."""


def _looks_like_path(value: str) -> bool:
    return os.sep in value or value.endswith((".onnx", ".tflite", ".ppn")) or os.path.exists(value)


def _sub(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    sub = cfg.get(key)
    return sub if isinstance(sub, dict) else {}


# ---------------------------------------------------------------------------
# Speaker-verification gate (voice-keyed wake word) — custom VM audio stack.
# A custom openWakeWord phrase model is trained voice-invariant (many voices)
# so it fires on the phrase in ANY voice — the only way to keep false-fires low.
# To honor "only my voice," the openWakeWord engine additionally runs a
# speaker-verification gate: on a phrase fire it takes the last ~1.5 s of
# buffered audio, computes a resemblyzer d-vector, and compares it to enrolled
# reference embeddings. The wake only fires when the speaker is confirmed.
# The gate runs as a subprocess (resemblyzer/torch live in the `oc` conda env,
# not the Hermes venv). Fail-closed: any gate error means "no fire."
# ---------------------------------------------------------------------------
_GATE_SAMPLE_RATE = 16000  # frames arrive at the 16 kHz capture rate

_GATE_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    # Path to speaker_gate.py (reads a 16 kHz mono int16 WAV from stdin,
    # prints a JSON verdict). Lives in the wakeword project dir.
    "gate_script": "/home/o/.hermes/projects/2026-08-17-wakeword/speaker_gate.py",
    # Persistent gate worker (newline-delimited JSON; encoder loaded once, so a
    # fire is ~10 ms instead of a ~1.75 s cold subprocess). Used when available.
    "worker_script": "/home/o/.hermes/projects/2026-08-17-wakeword/speaker_gate_worker.py",
    # Interpreter that has resemblyzer + torch (the `oc` conda env).
    "python": "/home/o/.conda/envs/oc/bin/python",
    # Enrolled-voice reference audio (16 kHz mono WAV). A d-vector cosine
    # >= threshold against ANY reference passes.
    "refs": ["/home/o/.hermes/projects/2026-08-17-wakeword/enrolled_ref.wav"],
    # Validated: enrolled user's clips score >= 0.549, stranger clips <= 0.460.
    "threshold": 0.50,
    # How many seconds of recent audio to keep for the gate (phrase ~1.2 s).
    "lookback_seconds": 1.5,
}


class _OpenWakeWordEngine(_Engine):
    """openWakeWord — free, local ONNX/tflite hotword detection. Scores one ~80 ms frame at a time;
    ``sensitivity`` IS the raw 0..1 threshold (higher = stricter). A real utterance holds the score
    high across frames while a stray phoneme spikes one, so ``confirmation_frames`` hits are required."""

    feature, section = "wake.openwakeword", "openwakeword"
    frame_length = 1280  # openWakeWord recommends 80 ms frames.

    def _build(self, cfg, sub, ww) -> None:
        import openwakeword
        from openwakeword.model import Model
        model_ref = str(sub.get("model") or ww._BUNDLED_MODEL_NAME).strip()
        framework = self._usable_framework(ww.resolve_inference_framework(cfg))
        self._threshold = ww._sensitivity(cfg)
        self._confirm_needed = ww._confirmation_frames(cfg)
        self._confirm_streak = 0
        # Default (or explicit "hey_hermes") → the bundled model; built-in names / paths as-is.
        if model_ref.lower() in ww._BUNDLED_MODEL_ALIASES:
            model_ref = ww._bundled_wakeword_path(framework)
        # download_models() also fetches the shared feature models (melspectrogram +
        # embedding) needed for ANY model, so a custom path must call it too.
        try:
            openwakeword.utils.download_models([model_ref])
        except Exception as e:  # pragma: no cover - network/path dependent
            logger.debug("openwakeword model download skipped: %s", e)
        self._model = Model(wakeword_models=[model_ref], inference_framework=framework)
        self._labels = list(self._model.models.keys())
        # Speaker-verification gate (voice-keyed). Fail-closed, disabled by
        # default. When enabled, a rolling buffer of recent audio is kept and,
        # on a phrase fire, the resemblyzer gate confirms the speaker before the
        # wake actually fires (see _GATE_DEFAULTS above for the layout).
        self._gate_enabled = False
        self._gate_buf = None
        self._gate_worker = None
        self._gate_worker_lock = None
        try:
            raw_gate = cfg.get("speaker_gate")
            gate_sub = raw_gate if isinstance(raw_gate, dict) else {}
            self._gate = dict(_GATE_DEFAULTS)
            self._gate.update({k: v for k, v in gate_sub.items() if v is not None})
            self._gate_enabled = bool(self._gate.get("enabled"))
            if self._gate_enabled:
                from collections import deque
                import threading
                lookback = int(float(self._gate.get("lookback_seconds", 1.5)) * _GATE_SAMPLE_RATE)
                self._gate_buf = deque(maxlen=max(1, lookback // self.frame_length))
                self._gate_worker_lock = threading.Lock()
                # Pre-spawn so the FIRST fire is also fast (non-blocking).
                self._prewarm_gate_worker()
        except Exception as e:  # noqa: BLE001 - never break engine build on gate init
            logger.debug("wake word: speaker gate init failed (gate stays off): %s", e)
            self._gate_enabled = False

    @staticmethod
    def _usable_framework(framework: str) -> str:
        """Refuse openWakeWord's silent tflite→onnx downgrade: without a tflite runtime it falls back
        to onnx, which on macOS ARM64 never fires (armed but deaf). Install + bridge the runtime first
        (gate lives here because dep specs can't carry PEP 508 markers); on that Mac raise instead."""
        ww = _ww()
        if framework != "tflite" or ww.ensure_tflite_runtime():
            return framework
        try:
            _ensure_dep("wake.openwakeword.tflite")
        except Exception as e:
            logger.debug("wake word: tflite runtime install failed: %s", e)
        if ww.ensure_tflite_runtime():
            return framework
        if ww._is_macos_arm64():
            raise RuntimeError("The wake word needs the tflite backend on this Mac, but its "
                               "runtime is missing. Install it with: pip install ai-edge-litert")
        logger.warning("wake word: no tflite runtime available — falling back to onnx")
        return "onnx"

    def process(self, frame) -> bool:
        # Keep a rolling buffer of recent audio for the speaker gate.
        if self._gate_enabled and self._gate_buf is not None:
            self._gate_buf.append(frame)
        hit = any(score >= self._threshold for score in self._model.predict(frame).values())
        self._confirm_streak = self._confirm_streak + 1 if hit else 0
        if self._confirm_streak < self._confirm_needed:
            return False
        self._confirm_streak = 0
        # Voice-keyed: confirm the speaker before firing. Fail-closed.
        if self._gate_enabled and not self._speaker_gate_pass():
            logger.info(
                "wake word: phrase detected but speaker gate rejected — not firing"
            )
            return False
        return True

    def reset(self) -> None:
        # Clears openWakeWord's rolling feature buffer so stale audio captured before a
        # pause can't re-fire the moment we resume.
        self._confirm_streak = 0
        with suppress(Exception):
            self._model.reset()

    def close(self) -> None:
        self.reset()
        with suppress(Exception):
            if self._gate_enabled and self._gate_worker_lock is not None:
                with self._gate_worker_lock:
                    self._kill_gate_worker_locked()

    # ---- Speaker-verification gate internals (fail-closed) ----

    def _speaker_gate_pass(self) -> bool:
        """Run the speaker-verification gate on the buffered audio. Fail-closed.

        Takes the last ~lookback seconds of captured audio, encodes it as a 16
        kHz mono int16 WAV, and asks the gate for a verdict. The gate prefers a
        persistent worker (encoder pre-loaded, ~10 ms) and falls back to a
        one-shot subprocess (~1.75 s cold-start) if the worker is unavailable.
        Returns True only when the enrolled speaker is confirmed; any error,
        missing file, or timeout means "no fire".
        """
        import io
        import wave

        import numpy as np

        try:
            frames = list(self._gate_buf or [])
            if not frames:
                return False
            audio = np.concatenate([np.asarray(f, dtype=np.int16) for f in frames])
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(_GATE_SAMPLE_RATE)
                w.writeframes(audio.tobytes())
            wav_bytes = buf.getvalue()
            verdict = self._gate_verdict_worker(wav_bytes)
            if verdict is None:  # worker unavailable — fall back to one-shot
                verdict = self._gate_verdict_oneshot(wav_bytes)
            if not verdict.get("pass"):
                if verdict.get("error"):
                    logger.warning("wake word: speaker gate error: %s", verdict["error"])
                return False
            logger.info("wake word: speaker gate passed (sim=%.3f)", verdict.get("sim", -1.0))
            return True
        except Exception as e:  # noqa: BLE001 - gate must never crash the wake loop
            logger.warning("wake word: speaker gate failed (fail-closed): %s", e)
            return False

    def _gate_worker_cmd(self) -> Optional[list]:
        """Command to spawn the persistent gate worker, or None if unavailable."""
        worker = str(self._gate.get("worker_script") or "").strip()
        python = str(self._gate.get("python") or "").strip()
        refs = [str(r) for r in (self._gate.get("refs") or [])]
        if not (worker and python and refs):
            return None
        if not (os.path.exists(worker) and os.path.exists(python)
                and all(os.path.exists(r) for r in refs)):
            return None
        return [python, worker, "--refs", *refs]

    def _prewarm_gate_worker(self) -> None:
        """Spawn the gate worker now (at engine build) so the first fire is fast."""
        import subprocess
        import threading

        lock = self._gate_worker_lock
        if lock is None:
            return
        with lock:
            if self._gate_worker is not None:
                return
            cmd = self._gate_worker_cmd()
            if cmd is None:
                return
            try:
                proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("wake word: gate worker prewarm failed: %s", e)
                return
            if proc.stdin is None or proc.stdout is None:
                proc.kill()
                return
            self._gate_worker = proc
            logger.info("wake word: speaker gate worker prewarmed (pid %s)", proc.pid)

    def _gate_verdict_worker(self, wav_bytes: bytes) -> Optional[Dict[str, Any]]:
        """Ask the persistent worker for a verdict. None if the worker is down."""
        import base64
        import json
        import subprocess

        lock = self._gate_worker_lock
        if lock is None:
            return None
        with lock:
            proc = self._gate_worker
            if proc is None or proc.poll() is not None:
                cmd = self._gate_worker_cmd()
                if cmd is None:
                    return None
                try:
                    proc = subprocess.Popen(
                        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("wake word: gate worker spawn failed: %s", e)
                    return None
                if proc.stdin is None or proc.stdout is None:
                    self._kill_gate_worker_locked()
                    logger.warning("wake word: gate worker pipes unavailable")
                    return None
                self._gate_worker = proc
            try:
                req = json.dumps({
                    "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
                    "threshold": float(self._gate.get("threshold", 0.50)),
                }) + "\n"
                proc.stdin.write(req.encode("ascii"))
                proc.stdin.flush()
                line = proc.stdout.readline()
            except Exception as e:  # noqa: BLE001
                self._kill_gate_worker_locked()
                logger.warning("wake word: gate worker I/O failed: %s", e)
                return None
            if not line:
                self._kill_gate_worker_locked()
                return None
            try:
                return json.loads(line.decode("ascii", "replace"))
            except Exception:  # noqa: BLE001
                self._kill_gate_worker_locked()
                return None

    def _kill_gate_worker_locked(self) -> None:
        """Terminate the worker. Caller must hold _gate_worker_lock."""
        proc = self._gate_worker
        self._gate_worker = None
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                pass

    def _gate_verdict_oneshot(self, wav_bytes: bytes) -> Dict[str, Any]:
        """One-shot gate subprocess (cold start, ~1.75 s). Fail-closed."""
        import json
        import subprocess

        refs = [str(r) for r in (self._gate.get("refs") or [])]
        cmd = [
            str(self._gate.get("python")),
            str(self._gate.get("gate_script")),
            "--refs", *refs,
            "--threshold", str(self._gate.get("threshold", 0.50)),
        ]
        proc = subprocess.run(cmd, input=wav_bytes, capture_output=True, timeout=10)
        lines = proc.stdout.decode().strip().splitlines()
        if not lines:
            return {"pass": False, "error": "no gate output"}
        try:
            return json.loads(lines[-1])
        except Exception:  # noqa: BLE001
            return {"pass": False, "error": "bad gate output"}


# sherpa-onnx open-vocabulary KWS model: small streaming zipformer transducer (English,
# GigaSpeech), downloaded once under HERMES_HOME. Keywords are tokenized at RUNTIME.
_SHERPA_KWS_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
    "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2"
)
_SHERPA_KWS_MODEL_DIR = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"


def _sherpa_model_root() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / "wakewords"


def _ensure_sherpa_model(root: Optional[Path] = None) -> Path:
    """Download + unpack the sherpa KWS model once; return its directory."""
    root = root or _sherpa_model_root()
    target = root / _SHERPA_KWS_MODEL_DIR
    if (target / "tokens.txt").exists():
        return target
    import tarfile
    import urllib.request
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{_SHERPA_KWS_MODEL_DIR}.tar.bz2"
    logger.info("wake word: downloading sherpa KWS model (one-time, ~13 MB)")
    urllib.request.urlretrieve(_SHERPA_KWS_MODEL_URL, archive)  # noqa: S310
    with tarfile.open(archive, "r:bz2") as tf:
        tf.extractall(root, filter="data")
    archive.unlink(missing_ok=True)
    if not (target / "tokens.txt").exists():
        raise RuntimeError(f"sherpa KWS model unpack failed: {target}")
    return target


class _SherpaKwsEngine(_Engine):
    """sherpa-onnx open-vocabulary keyword spotting — any typed phrase, zero training. ``wake_word.phrase``
    is BPE-tokenized at runtime against the model's vocabulary: DETECTION config, not a cosmetic label."""

    feature, section = "wake.sherpa", "sherpa"
    frame_length = 1280  # streaming zipformer accepts any chunk; match capture path.

    def _build(self, cfg, sub, ww) -> None:
        import sherpa_onnx
        import tempfile
        from sherpa_onnx import text2token
        model_dir = str(sub.get("model_dir") or "").strip()
        d = Path(model_dir) if model_dir else _ensure_sherpa_model()
        if not (d / "tokens.txt").exists():
            raise RuntimeError(f"sherpa KWS model not found at {d}")

        # Phrase set: this profile's phrase plus — with profile routing on — every other
        # wake-enabled profile's phrase, so ONE listener can wake any profile.
        phrase = str(ww._get(cfg, "phrase") or "hey hermes").strip()
        phrase_map: Dict[str, str] = {phrase: ww._active_profile_name()}
        if bool(cfg.get("profile_routing", True)):
            for prof, p in ww.enrolled_profile_phrases().items():
                phrase_map.setdefault(p.strip(), prof)
        phrases = list(phrase_map)
        tokens = text2token([p.upper() for p in phrases], tokens=str(d / "tokens.txt"), tokens_type="bpe",
                            bpe_model=str(d / "bpe.model"))
        # sherpa keyword entries reject spaces in the @display-name; underscore them and
        # map display → profile for match routing.
        self._display_to_profile: Dict[str, str] = {}
        kw = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", prefix="hermes-kws-", delete=False,
                                         encoding="utf-8")
        for p, toks in zip(phrases, tokens):
            display = p.upper().replace(" ", "_")
            self._display_to_profile[display] = phrase_map[p]
            kw.write(" ".join(toks) + f" @{display}\n")
        kw.close()
        self._keywords_file = kw.name

        # Shared 0..1 sensitivity → sherpa keywords_threshold. 0.5 lands on sherpa's
        # recommended 0.25; a stricter 0.35 missed ~12% of true positives in live TTS
        # matrix tests while 0.25 held zero false fires.
        threshold = 0.05 + 0.4 * ww._sensitivity(cfg)

        def _model_file(part: str) -> str:
            hits = sorted(d.glob(f"{part}-*[!8].onnx"))
            if not hits:
                raise RuntimeError(f"sherpa KWS model file missing: {d}/{part}-*[!8].onnx")
            return str(hits[0])

        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(d / "tokens.txt"), encoder=_model_file("encoder"), decoder=_model_file("decoder"),
            joiner=_model_file("joiner"), keywords_file=self._keywords_file, keywords_threshold=threshold,
            num_threads=1,
        )
        self._stream = self._spotter.create_stream()

    def process(self, frame) -> bool:
        import numpy as np
        self._stream.accept_waveform(_ww().SAMPLE_RATE, np.asarray(frame, dtype=np.float32) / 32768.0)
        fired = False
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            result = self._spotter.get_result(self._stream)
            if result:
                fired, display = True, str(result)
                self.last_match = (display.replace("_", " ").lower(),
                                   self._display_to_profile.get(display, ""))
                self._spotter.reset_stream(self._stream)  # one utterance must not fire repeatedly
        return fired

    def reset(self) -> None:
        # Fresh stream drops buffered audio/decoder state (pause → resume must not re-fire).
        with suppress(Exception):
            self._stream = self._spotter.create_stream()

    def close(self) -> None:
        with suppress(OSError):
            os.unlink(self._keywords_file)


class _PorcupineEngine(_Engine):
    """Picovoice Porcupine — premium, on-device, needs an access key."""

    feature, section = "wake.porcupine", "porcupine"

    def _build(self, cfg, sub, ww) -> None:
        import pvporcupine
        access_key = (os.getenv("PORCUPINE_ACCESS_KEY") or "").strip()
        if not access_key:
            raise RuntimeError("Porcupine wake word requires PORCUPINE_ACCESS_KEY "
                               "(get a free key at https://console.picovoice.ai).")
        keyword = str(sub.get("keyword") or "jarvis").strip()
        # Porcupine's `sensitivities` runs the OPPOSITE way to our shared knob (higher =
        # looser); invert so "higher = stricter" holds for every engine.
        kwargs: Dict[str, Any] = {"access_key": access_key, "sensitivities": [1.0 - ww._sensitivity(cfg)]}
        kwargs["keyword_paths" if _looks_like_path(keyword) else "keywords"] = [keyword]
        self._porcupine = pvporcupine.create(**kwargs)
        self.frame_length = self._porcupine.frame_length

    def process(self, frame) -> bool:
        return self._porcupine.process(frame) >= 0  # pvporcupine wants a plain sequence of int16

    def close(self) -> None:
        with suppress(Exception):
            self._porcupine.delete()
