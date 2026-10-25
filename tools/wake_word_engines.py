"""Wake-word hotword engines (pyopen-wakeword / sherpa-onnx KWS / Porcupine).

All three run fully on-device. Config, platform probes and sensitivity accessors
live in :mod:`tools.wake_word`; engines read them lazily through that module (import cycle).
This module is the SINGLE owner of the engine implementations — ``tools.wake_word``
imports these classes and must not shadow them with copies.

Dependency admission: constructing an engine ensures its ``wake-*`` extra. The
``audio-io`` extra (sounddevice + numpy) is ensured only when the resolved
capture mode is ``local`` — client capture (desktop/TUI streaming PCM via
``wake.feed``) never needs, and must never trigger installation of, local
audio libraries.
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


def _ensure_dep(feature: str, cfg: dict[str, Any]) -> None:
    import pm

    pm.ensure_import(feature)
    # Only local capture needs the microphone dependencies.
    if _ww().resolve_capture_mode(cfg) == "local" and not pm.available("audio-io"):
        pm.ensure_import("audio-io")


class _Engine:
    """Minimal hotword-engine contract: feed int16 frames, get a bool. Subclasses set ``feature``
    (the pm extra ensured before ``_build``) and their own ``cfg`` sub-section ``section``."""

    feature: str = ""
    section: str = ""
    frame_length: int = 1280  # 80 ms at 16 kHz

    #: (matched phrase, profile name) of the most recent fire. Multi-phrase engines
    #: (sherpa) set this for profile routing; single-phrase engines leave it None.
    last_match: Optional[tuple[str, str]] = None

    def __init__(self, cfg: dict[str, Any]):
        _ensure_dep(self.feature, cfg)
        self._build(cfg, _sub(cfg, self.section), _ww())

    def _build(self, cfg: dict[str, Any], sub: dict[str, Any], ww) -> None:
        raise NotImplementedError

    def process(self, frame) -> bool:  # frame: 1-D int16 ndarray
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any internal audio/feature buffer (called on every (re)start)."""

    def close(self) -> None:
        """Release engine resources (called once on stop)."""


def _looks_like_path(value: str) -> bool:
    return os.sep in value or value.endswith((".onnx", ".tflite", ".ppn")) or os.path.exists(value)


def _sub(cfg: dict[str, Any], key: str) -> dict[str, Any]:
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


# ONNX model backend (custom): openWakeWord-architecture network exported to
# ONNX. Same geometry as the pyopen-wakeword TFLite path — 16 embedding
# windows of 96 features, 80-window history — so a custom model trained on
# openWakeWord embeddings keeps its scores.
_ONNX_WINDOWS = 16
_ONNX_FEATURES = 96
_ONNX_HISTORY = 80


class _OpenWakeWordEngine(_Engine):
    """pyopen-wakeword — free, local hotword detection (TFLite via a bundled
    tensorflowlite_c lib; no runtime download, no framework choice). Scores one
    ~80 ms frame at a time; ``sensitivity`` IS the raw 0..1 threshold (higher =
    stricter). A real utterance holds the score high across frames while a stray
    phoneme spikes one, so ``confirmation_frames`` hits are required."""

    feature, section = "wake-openwakeword", "openwakeword"

    def _build(self, cfg, sub, ww) -> None:
        from pyopen_wakeword import OpenWakeWord, OpenWakeWordFeatures

        model_ref = str(sub.get("model") or ww._BUNDLED_MODEL_NAME).strip()
        # Default (or explicit "hey_hermes") → the bundled model; a custom path
        # is used as-is. pyopen-wakeword bundles the shared feature models
        # (melspectrogram + embedding — byte-identical to the openWakeWord
        # v0.5.1 files) inside its wheel, so there is no download_models step.
        if model_ref.lower() in ww._BUNDLED_MODEL_ALIASES:
            model_ref = ww._bundled_wakeword_path()
        # pyopen-wakeword returns a 0..1 score per completed window; sensitivity
        # IS the raw threshold a score must clear. Higher = stricter (fewer
        # false fires). Default 0.6 sits above openWakeWord's permissive 0.5
        # baseline, which let near-misses like "hey hor" through.
        self._threshold = ww._sensitivity(cfg)
        self._confirm_needed = ww._confirmation_frames(cfg)
        self._confirm_streak = 0
        self._features = OpenWakeWordFeatures.from_builtin()
        # Model backend: pyopen-wakeword ships TFLite only, but this
        # installation's production phrase model is a custom openWakeWord
        # architecture network exported to ONNX (wake_word.openwakeword.model +
        # inference_framework: onnx). Keep that backend: the same feature
        # pipeline (mel + embedding, identical to openWakeWord v0.5.1) feeds an
        # ONNX session through the same 16x96 sliding window the TFLite path
        # uses, so scores stay comparable to the pre-v0.22 engine.
        self._onnx = None
        self._onnx_input = ""
        self._onnx_output = ""
        self._onnx_emb = None
        self._onnx_new = 0
        self._model = None
        _ref_l = str(model_ref).lower()
        _framework = str(sub.get("inference_framework") or "").lower()
        if _ref_l.endswith(".onnx") or (_framework == "onnx" and not _looks_like_path(_ref_l)):
            if not _ref_l.endswith(".onnx"):
                raise RuntimeError(
                    f"wake word: openwakeword model {model_ref!r} is not a TFLite "
                    "model; set wake_word.openwakeword.model to the .onnx model path"
                )
            import numpy as _np
            import onnxruntime as _ort

            self._onnx = _ort.InferenceSession(
                str(model_ref), providers=["CPUExecutionProvider"]
            )
            self._onnx_input = self._onnx.get_inputs()[0].name
            self._onnx_output = self._onnx.get_outputs()[0].name
            self._onnx_emb = _np.zeros((_ONNX_HISTORY, _ONNX_FEATURES), dtype=_np.float32)
            self._labels = [Path(model_ref).stem]
        else:
            self._model = OpenWakeWord.from_model(model_ref)
            self._labels = [self._model.id]
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

    def process(self, frame) -> bool:
        # Keep a rolling buffer of recent audio for the speaker gate.
        if self._gate_enabled and self._gate_buf is not None:
            self._gate_buf.append(frame)
        # frame is a 1-D int16 ndarray; the features pipeline consumes int16
        # bytes. process_streaming() yields embeddings as the window fills and
        # the model yields one 0..1 score per completed window.
        over = False
        for emb in self._features.process_streaming(frame.tobytes()):
            if self._onnx is not None:
                for score in self._onnx_scores(emb):
                    if score >= self._threshold:
                        over = True
            else:
                for score in self._model.process_streaming(emb):
                    if score >= self._threshold:
                        over = True
        # Require N consecutive over-threshold frames: a real phrase holds the
        # score high across frames, a stray ambient phoneme spikes just one.
        if not over:
            self._confirm_streak = 0
            return False
        self._confirm_streak += 1
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

    def _onnx_scores(self, embeddings):
        """Slide the 16x96 embedding window and score it with the ONNX model.

        Mirrors pyopen-wakeword's OpenWakeWord.process_streaming precisely (same
        ring buffer, same window advance), so a custom ONNX model sees exactly
        the input geometry it was trained on."""
        import numpy as np

        num = embeddings.shape[2]
        self._onnx_emb[:-num] = self._onnx_emb[num:]
        self._onnx_emb[-num:] = embeddings[0, 0, :, :]
        self._onnx_new = min(len(self._onnx_emb), self._onnx_new + num)
        while self._onnx_new >= _ONNX_WINDOWS:
            window = np.zeros((1, _ONNX_WINDOWS, _ONNX_FEATURES), dtype=np.float32)
            window[0, :] = self._onnx_emb[
                -self._onnx_new: len(self._onnx_emb) - self._onnx_new + _ONNX_WINDOWS
            ]
            self._onnx_new = max(0, self._onnx_new - 1)
            out = self._onnx.run([self._onnx_output], {self._onnx_input: window})[0]
            yield float(np.asarray(out).reshape(-1)[0])

    def reset(self) -> None:
        # Clears pyopen-wakeword's rolling feature/prediction buffer so stale
        # audio captured before a pause can't re-fire the moment we resume.
        self._confirm_streak = 0
        if self._onnx is not None:
            import numpy as np
            self._onnx_emb = np.zeros((_ONNX_HISTORY, _ONNX_FEATURES), dtype=np.float32)
            self._onnx_new = 0
        with suppress(Exception):
            self._features.reset()
            if self._model is not None:
                self._model.reset()

    def close(self) -> None:
        self.reset()
        with suppress(Exception):
            if self._gate_enabled and self._gate_worker_lock is not None:
                with self._gate_worker_lock:
                    self._kill_gate_worker_locked()
            self._features.close()
            if self._model is not None:
                self._model.close()

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
    urllib.request.urlretrieve(_SHERPA_KWS_MODEL_URL, archive)
    with tarfile.open(archive, "r:bz2") as tf:
        tf.extractall(root, filter="data")
    archive.unlink(missing_ok=True)
    if not (target / "tokens.txt").exists():
        raise RuntimeError(f"sherpa KWS model unpack failed: {target}")
    return target


class _SherpaKwsEngine(_Engine):
    """sherpa-onnx open-vocabulary keyword spotting — any typed phrase, zero training. ``wake_word.phrase``
    is BPE-tokenized at runtime against the model's vocabulary: DETECTION config, not a cosmetic label."""

    feature, section = "wake-sherpa", "sherpa"
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
        phrase_map: dict[str, str] = {phrase: ww._active_profile_name()}
        if bool(cfg.get("profile_routing", True)):
            for prof, p in ww.enrolled_profile_phrases().items():
                phrase_map.setdefault(p.strip(), prof)
        phrases = list(phrase_map)
        tokens = text2token([p.upper() for p in phrases], tokens=str(d / "tokens.txt"), tokens_type="bpe",
                            bpe_model=str(d / "bpe.model"))
        # sherpa keyword entries reject spaces in the @display-name; underscore them and
        # map display → profile for match routing.
        self._display_to_profile: dict[str, str] = {}
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

    feature, section = "wake-porcupine", "porcupine"

    def _build(self, cfg, sub, ww) -> None:
        import pvporcupine
        access_key = (os.getenv("PORCUPINE_ACCESS_KEY") or "").strip()
        if not access_key:
            raise RuntimeError("Porcupine wake word requires PORCUPINE_ACCESS_KEY "
                               "(get a free key at https://console.picovoice.ai).")
        keyword = str(sub.get("keyword") or "jarvis").strip()
        # Porcupine's `sensitivities` runs the OPPOSITE way to our shared knob (higher =
        # looser); invert so "higher = stricter" holds for every engine.
        kwargs: dict[str, Any] = {"access_key": access_key, "sensitivities": [1.0 - ww._sensitivity(cfg)]}
        kwargs["keyword_paths" if _looks_like_path(keyword) else "keywords"] = [keyword]
        self._porcupine = pvporcupine.create(**kwargs)
        self.frame_length = self._porcupine.frame_length

    def process(self, frame) -> bool:
        return self._porcupine.process(frame) >= 0  # pvporcupine wants a plain sequence of int16

    def close(self) -> None:
        with suppress(Exception):
            self._porcupine.delete()
