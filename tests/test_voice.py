"""Utterance segmentation for hands-free voice input.

Only the pure state machine is tested here: given a per-frame speech/silence
decision, it must emit exactly one utterance when speech is followed by enough
trailing silence. The mic capture and Whisper transcription are integration and
run on a machine with an audio device.
"""

from chemistree.app.voice import Segmenter

_FRAME = b"\x00\x01" * 480  # one 30 ms frame at 16 kHz, 16-bit mono


def _run(segmenter: Segmenter, decisions: list[bool]) -> list[bytes]:
    """Push a frame per decision, collecting the utterances that complete."""
    out = []
    for is_speech in decisions:
        utterance = segmenter.push(_FRAME, is_speech)
        if utterance is not None:
            out.append(utterance)
    return out


def test_speech_then_silence_emits_one_utterance():
    seg = Segmenter(frame_ms=30, start_frames=2, end_silence_ms=90)  # 3 silent frames
    utterances = _run(seg, [True] * 5 + [False] * 3)
    assert len(utterances) == 1
    assert len(utterances[0]) == 8 * len(_FRAME)  # 5 speech + 3 trailing silence frames


def test_pure_silence_emits_nothing():
    seg = Segmenter(frame_ms=30, start_frames=2, end_silence_ms=90)
    assert _run(seg, [False] * 20) == []


def test_a_blip_shorter_than_start_frames_is_ignored():
    seg = Segmenter(frame_ms=30, start_frames=3, end_silence_ms=90)
    assert _run(seg, [True, False] * 10) == []  # never 3 speech frames in a row


def test_two_utterances_are_segmented_separately():
    seg = Segmenter(frame_ms=30, start_frames=2, end_silence_ms=90)
    decisions = [True] * 4 + [False] * 3 + [True] * 4 + [False] * 3
    assert len(_run(seg, decisions)) == 2
