"""Word-level speaker assignment and voiceprint mapping (no model downloads).

    python tests/test_speaker_alignment.py
"""
import os
import sys
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import predict as p


def W(word, s, e):
    return NS(word=word, start=s, end=e, probability=0.9)


def seg(words, logprob=-0.2):
    return NS(start=words[0].start, end=words[-1].end, text="".join(w.word for w in words),
              avg_logprob=logprob, words=words)


# Host's vocative and the candidate's answer inside ONE whisper segment.
s1 = seg([W(" Mr.", 0.0, 0.3), W(" Fontes,", 0.3, 0.8), W(" your", 0.8, 1.0), W(" response.", 1.0, 1.5),
          W(" Thank", 1.7, 1.9), W(" you.", 1.9, 2.1), W(" My", 2.2, 2.4), W(" name", 2.4, 2.6),
          W(" is", 2.6, 2.7), W(" Adrian", 2.7, 3.0), W(" Fontes.", 3.0, 3.5)])
turns = [(0.0, 1.6, "SPEAKER_00"), (1.65, 3.6, "SPEAKER_01")]
out = p.split_by_speaker([s1], turns)
assert [o["speaker"] for o in out] == ["SPEAKER_00", "SPEAKER_01"], out
assert out[0]["text"] == "Mr. Fontes, your response.", out[0]["text"]
assert out[1]["text"].startswith("Thank you. My name is Adrian Fontes"), out[1]["text"]
assert all(w["speaker"] == o["speaker"] for o in out for w in o["words"])

# A word in a gap between turns takes the nearest turn.
s2 = seg([W(" so", 10.0, 10.2), W(" yes", 11.9, 12.1)])
out = p.split_by_speaker([s2], [(9.0, 10.5, "A"), (12.0, 14.0, "B")])
assert [o["speaker"] for o in out] == ["A", "B"], out

# Word after all turns -> last turn; cursor stays linear across segments.
segs = [seg([W(" a", 0.1, 0.2)]), seg([W(" b", 5.1, 5.3)]), seg([W(" c", 20.0, 20.4)])]
out = p.split_by_speaker(segs, [(0.0, 1.0, "A"), (5.0, 6.0, "B")])
assert [o["speaker"] for o in out] == ["A", "B", "B"], out

# No diarization -> SPEAKER_00, segment kept whole.
out = p.split_by_speaker([s1], [])
assert len(out) == 1 and out[0]["speaker"] == "SPEAKER_00"

# Segment without words keeps its own text and gets a speaker.
out = p.split_by_speaker([NS(start=2.0, end=3.0, text=" hi ", avg_logprob=-0.1, words=[])], turns)
assert out[0]["speaker"] == "SPEAKER_01" and out[0]["text"] == "hi"

# Voiceprints: rows follow speaker_diarization.labels(); NaN rows dropped.
from pyannote.core import Annotation, Segment

ann = Annotation()
ann[Segment(0, 1.6)] = "SPEAKER_00"
ann[Segment(1.65, 3.6)] = "SPEAKER_01"
ann[Segment(4, 5)] = "SPEAKER_02"
emb = np.vstack([np.ones(256), np.full(256, 2.0), np.full(256, np.nan)])
diar = NS(speaker_diarization=ann, exclusive_speaker_diarization=ann, speaker_embeddings=emb)
final = p.split_by_speaker([s1], turns)
vp = p.build_speaker_voiceprints(diar, final)
assert set(vp) == {"SPEAKER_00", "SPEAKER_01"}, vp.keys()
assert vp["SPEAKER_01"]["embedding"][0] == 2.0 and vp["SPEAKER_01"]["dim"] == 256
assert vp["SPEAKER_00"]["model_version"] == "pyannote-speaker-diarization-community-1"
assert vp["SPEAKER_00"]["segment_count"] == 1 and abs(vp["SPEAKER_00"]["duration_s"] - 1.5) < 1e-9
print("all split/voiceprint checks passed")

# Zero-length word on a turn boundary's far side sits inside the later turn.
out = p.split_by_speaker([seg([W(" make", 63.07, 63.07)])], [(56.0, 63.0, "A"), (63.0, 70.0, "B")])
assert out[0]["speaker"] == "B", out
print("zero-length word check passed")
