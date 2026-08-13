"""Hands-free voice input: segment speech, transcribe with Whisper.

A background thread reads the machine's microphone, a voice-activity detector
marks each frame speech or silence, and ``Segmenter`` groups frames into
utterances (speech ended by a run of silence). Each utterance is transcribed
with faster-whisper and handed back as text.

The heavy dependencies (``sounddevice``, ``webrtcvad``, ``faster_whisper``) are
imported lazily inside the functions that need them, so this module and the
``Segmenter`` state machine import and test without them installed. Install them
with ``poetry install --with voice``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

_SAMPLE_RATE = 16000  # what webrtcvad and Whisper expect
_FRAME_MS = 30  # webrtcvad accepts 10, 20, or 30 ms frames
_MODEL_NAME = "base.en"
# Bias transcription toward the vocabulary this app hears.
_VOCAB_PROMPT = (
    "Edit a molecule: residues like VAL67, ALA37, PHE; groups like isopropyl, "
    "methoxy, trifluoromethyl, hydroxy; positions ortho, meta, para."
)


class Segmenter:
    """Group per-frame speech decisions into complete utterances.

    Speech starts after ``start_frames`` consecutive speech frames (ignoring
    blips) and ends after ``end_silence_ms`` of trailing silence. ``push``
    returns the utterance's raw audio when it ends, else None.
    """

    _in_speech: bool
    _speech_run: int
    _silence_run: int
    _buffer: bytearray

    def __init__(
        self,
        *,
        sample_rate: int = _SAMPLE_RATE,
        frame_ms: int = _FRAME_MS,
        start_frames: int = 3,
        end_silence_ms: int = 600,
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._start_frames = start_frames
        self._end_frames = max(1, end_silence_ms // frame_ms)
        self._reset()

    def push(self, frame: bytes, is_speech: bool) -> bytes | None:
        """Add one frame and its speech decision; return an utterance if it ends.

        Args:
            frame: Raw 16-bit mono PCM for one frame.
            is_speech: Whether the VAD marked this frame as speech.

        Returns:
            The utterance's audio bytes when trailing silence ends it, else None.
        """
        if self._in_speech:
            return self._accumulate(frame, is_speech)
        self._await_speech(frame, is_speech)
        return None

    def _await_speech(self, frame: bytes, is_speech: bool) -> None:
        """Buffer candidate speech; enter the utterance once it is sustained."""
        if is_speech:
            self._speech_run += 1
            self._buffer += frame
            if self._speech_run >= self._start_frames:
                self._in_speech = True
        else:
            self._speech_run = 0
            self._buffer = bytearray()  # drop leading silence and blips

    def _accumulate(self, frame: bytes, is_speech: bool) -> bytes | None:
        """Extend the current utterance; end it after enough trailing silence."""
        self._buffer += frame
        if is_speech:
            self._silence_run = 0
            return None
        self._silence_run += 1
        if self._silence_run < self._end_frames:
            return None
        utterance = bytes(self._buffer)
        self._reset()
        return utterance

    def _reset(self) -> None:
        """Return to waiting for the next utterance."""
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._buffer = bytearray()


def transcribe(audio: bytes, sample_rate: int = _SAMPLE_RATE) -> str:
    """Transcribe one utterance's PCM audio to text.

    Args:
        audio: Raw 16-bit mono PCM samples.
        sample_rate: Sample rate of ``audio`` in Hz.

    Returns:
        The recognized text, stripped (empty if nothing was recognized).
    """
    import numpy as np

    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = _model().transcribe(
        samples, language="en", initial_prompt=_VOCAB_PROMPT
    )
    return " ".join(segment.text for segment in segments).strip()


def listen(on_text: Callable[[str], None], stop: threading.Event) -> None:
    """Capture the mic and call ``on_text`` with each transcribed utterance.

    Blocks until ``stop`` is set; meant to run in a background thread.

    Args:
        on_text: Called with the text of each completed utterance.
        stop: Setting this event ends the capture loop.
    """
    import sounddevice as sd
    import webrtcvad

    vad = webrtcvad.Vad(2)  # 0..3, higher is more aggressive at rejecting non-speech
    segmenter = Segmenter()
    frame_len = int(_SAMPLE_RATE * _FRAME_MS / 1000)

    with sd.RawInputStream(
        samplerate=_SAMPLE_RATE, blocksize=frame_len, dtype="int16", channels=1
    ) as stream:
        while not stop.is_set():
            data, _ = stream.read(frame_len)
            frame = bytes(data)
            utterance = segmenter.push(frame, vad.is_speech(frame, _SAMPLE_RATE))
            if utterance is not None:
                text = transcribe(utterance)
                if text:
                    on_text(text)


_MODEL = None


def _model():  # type: ignore[no-untyped-def]
    """The lazily loaded, cached faster-whisper model (CPU, int8)."""
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel

        _MODEL = WhisperModel(_MODEL_NAME, device="cpu", compute_type="int8")
    return _MODEL
