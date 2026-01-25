import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from symusic import Note, Score, Track

PITCH_MIN = 0
PITCH_MAX = 95
BAR_ID = 96
POS_OFFSET = 97
MAX_TOKEN_ID = int(np.iinfo(np.uint16).max)


def load_meta(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"meta not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_tensor(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".pt":
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch is required for .pt input") from exc
        tensor = torch.load(path, map_location="cpu")
        return tensor.detach().cpu().numpy()
    raise ValueError("input must be .npy or .pt")


def resolve_special_ids(
    bar_ticks: int,
    bos_id: Optional[int],
    eof_id: Optional[int],
    pad_id: Optional[int],
) -> tuple[int, int, int]:
    if bos_id is None:
        bos_id = POS_OFFSET + int(bar_ticks)
    if eof_id is None:
        eof_id = int(bos_id) + 1
    if pad_id is None:
        pad_id = int(bos_id) + 2
    if int(pad_id) > MAX_TOKEN_ID:
        raise ValueError("bar_ticks too large: token id exceeds dtype range")
    return int(bos_id), int(eof_id), int(pad_id)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Decode REMI96 (BOS/BAR/POS/PITCH/EOF) tokens back to MIDI."
    )
    p.add_argument("tensor", type=Path, help="Input .npy or .pt path")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("from_tensor.mid"),
        help="Output MIDI file path",
    )
    p.add_argument("--meta", type=Path, default=None, help="Optional meta JSON")
    p.add_argument("--tpq", type=int, default=None, help="Ticks per quarter")
    p.add_argument("--dur", type=int, default=None, help="Fixed duration")
    p.add_argument("--velocity", type=int, default=64, help="Fixed velocity")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    meta = load_meta(args.meta)

    tpq = args.tpq if args.tpq is not None else int(meta.get("tpq", 8))
    dur = int(meta.get("min_dur", 1)) if args.dur is None else int(args.dur)
    velocity = int(args.velocity)
    time_shift = int(meta.get("time_shift", 0))
    bar_ticks = meta.get("bar_ticks_reltime")
    if bar_ticks is None:
        raise ValueError("bar_ticks_reltime is required for REMI decoding")
    bar_ticks = int(bar_ticks)

    bos_id = meta.get("bos_id")
    eof_id = meta.get("eof_id")
    pad_id = meta.get("pad_id")
    bos_id, eof_id, pad_id = resolve_special_ids(
        bar_ticks,
        int(bos_id) if bos_id is not None else None,
        int(eof_id) if eof_id is not None else None,
        int(pad_id) if pad_id is not None else None,
    )

    data = load_tensor(args.tensor)
    if data.size == 0:
        score = Score()
        score.tpq = int(tpq)
        score.dump_midi(str(args.out))
        return
    if data.ndim != 1:
        raise ValueError("tensor must be 1D")

    if not np.isfinite(data).all():
        raise ValueError("tensor contains NaN or inf")
    if np.issubdtype(data.dtype, np.floating):
        rounded = np.rint(data)
        if np.max(np.abs(data - rounded)) > 1e-6:
            raise ValueError("tensor must contain integer-like values")
        data = rounded.astype(np.int64)
    else:
        data = data.astype(np.int64, copy=False)

    tokens = data.ravel().tolist()
    idx = 0
    while idx < len(tokens) and tokens[idx] == pad_id:
        idx += 1
    if idx >= len(tokens):
        raise ValueError("missing BOS token")
    if tokens[idx] != bos_id:
        raise ValueError("BOS token must appear first")
    idx += 1

    times_list: list[int] = []
    pitches_list: list[int] = []
    current_bar = -1
    current_pos = None
    eof_seen = False

    while idx < len(tokens):
        tok = tokens[idx]
        idx += 1
        if tok == pad_id:
            continue
        if tok == eof_id:
            eof_seen = True
            break
        if tok == bos_id:
            raise ValueError("unexpected BOS token in stream")
        if tok == BAR_ID:
            current_bar += 1
            current_pos = None
            continue
        if tok >= POS_OFFSET and tok < bos_id:
            current_pos = int(tok) - POS_OFFSET
            if current_pos < 0 or current_pos >= bar_ticks:
                raise ValueError("POS out of range")
            continue
        if tok < PITCH_MIN or tok > PITCH_MAX:
            raise ValueError("PITCH out of range")
        if current_bar < 0 or current_pos is None:
            raise ValueError("PITCH requires prior BAR and POS tokens")
        time_tick = current_bar * bar_ticks + current_pos
        times_list.append(int(time_tick))
        pitches_list.append(int(tok))

    if not eof_seen:
        raise ValueError("EOF token is required but was not found")

    score = Score()
    score.tpq = int(tpq)
    track = Track()
    track.is_drum = False

    if times_list:
        times = np.asarray(times_list, dtype=np.int64)
        pitches = np.asarray(pitches_list, dtype=np.int64)
        if time_shift:
            times = times + int(time_shift)
        durations = np.full_like(times, int(dur))
        velocities = np.full_like(times, int(velocity))
        note_list = None
        if hasattr(Note, "from_numpy"):
            try:
                note_list = Note.from_numpy(
                    time=times,
                    duration=durations,
                    pitch=pitches,
                    velocity=velocities,
                    ttype=score.ttype,
                )
            except Exception:
                note_list = None
        if note_list is not None and hasattr(track.notes, "extend"):
            track.notes.extend(note_list)
        elif note_list is not None:
            for n in note_list:
                track.notes.append(n)
        else:
            for t, p in zip(times.tolist(), pitches.tolist()):
                track.notes.append(Note(int(t), int(dur), int(p), int(velocity)))

    track.sort()
    score.tracks.append(track)
    score.sort()
    score.dump_midi(str(args.out))


if __name__ == "__main__":
    main()
