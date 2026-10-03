"""Select a wake word's labelled clips from the puck, by what was actually SAID.

The puck tags every reviewed clip with `said`: the wake word really spoken, or
"none". For a model named X:

  * positives = clips where X was said                      (?said=X)
  * negatives = reviewed clips where X was NOT said          (?not_said=X)
                — another wake word, or no wake word at all

Selecting by the puck's `label` instead is wrong with several wake words loaded:
`label` is relative to whichever model FIRED, so a clip where `stop` fired but
"hey tars" was said is `false`, and pulling all `false` clips as hey_tars
negatives would train hey_tars to ignore its own wake word.

Shared by pull_negatives.py, evaluate.py and the web UI so all three agree on
which clips belong to which model.
"""
import io
import json
import tarfile
import urllib.parse
import urllib.request
import wave

import numpy as np

RATE = 16000


def model_name(wake_word: str) -> str:
    """The model stem train.sh derives from the wake-word phrase."""
    return wake_word.strip().replace(" ", "_")


class PuckTooOld(RuntimeError):
    """The puck ignored the said/not_said filter (no `said` field), so it would
    have returned every clip, unreviewed ones included."""


def _check_said(events: list) -> None:
    if events and "said" not in events[0]:
        raise PuckTooOld(
            "puck predates said/not_said filtering — update voice-orchestrator; "
            "refusing to use unfiltered clips")


def list_events(orch: str, **query: str) -> list:
    """Event metadata (no audio) for e.g. said="hey_tars" or not_said="hey_tars"."""
    url = f"{orch}/events?{urllib.parse.urlencode(query)}"
    events = json.loads(urllib.request.urlopen(url, timeout=8).read()).get("events", [])
    _check_said(events)
    return events


def fetch_clips(orch: str, clip_len: int, **query: str) -> list:
    """[(clip_id, int16 array padded/trimmed to clip_len)] for the query."""
    url = f"{orch}/events/export?{urllib.parse.urlencode(query)}"
    data = urllib.request.urlopen(url, timeout=30).read()
    out = []
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        manifest = tar.extractfile("events.json")
        _check_said(json.loads(manifest.read()) if manifest else [])
        for m in tar.getmembers():
            if not m.name.endswith(".wav"):
                continue
            clip_id = m.name.rsplit("/", 1)[-1][: -len(".wav")]
            with wave.open(tar.extractfile(m)) as w:
                a = np.frombuffer(w.readframes(w.getnframes()), np.int16)
            a = a[:clip_len] if len(a) >= clip_len else np.pad(a, (0, clip_len - len(a)))
            out.append((clip_id, a))
    return out
