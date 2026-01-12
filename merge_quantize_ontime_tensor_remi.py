import argparse
import inspect
import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import chain, islice
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Optional

import numpy as np
from symusic import Score

# Fixed tokenizer design
PITCH_MIN = 0
PITCH_MAX = 127
BAR_ID = 128
POS_OFFSET = 129
BOS_ID = 512
ABS_END_ID = 513
EOF_ID = 514
PAD_ID = 515
TOKEN_DTYPE = np.uint16
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4
HEADER_MAGIC = 20240520
HEADER_VERSION = 1


def load_score(path: Path) -> Score:
    try:
        return Score(str(path))
    except Exception:
        if hasattr(Score, "from_midi"):
            return Score.from_midi(path.read_bytes())
        if hasattr(Score, "read_midi"):
            return Score.read_midi(str(path))
        raise


def collect_pitch_time_arrays(tracks, include_drums: bool) -> tuple[np.ndarray, np.ndarray]:
    times_parts: list[np.ndarray] = []
    pitches_parts: list[np.ndarray] = []
    for tr in tracks:
        if (not include_drums) and tr.is_drum:
            continue
        notes = tr.notes
        if hasattr(notes, "numpy"):
            arr = notes.numpy()
            if len(arr["time"]) == 0:
                continue
            times_parts.append(np.asarray(arr["time"], dtype=np.int64))
            pitches_parts.append(np.asarray(arr["pitch"], dtype=np.int64))
        else:
            if len(notes) == 0:
                continue
            times_parts.append(np.asarray([n.time for n in notes], dtype=np.int64))
            pitches_parts.append(np.asarray([n.pitch for n in notes], dtype=np.int64))
    if not times_parts:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty
    times = np.concatenate(times_parts)
    pitches = np.concatenate(pitches_parts)
    return pitches.astype(np.int64, copy=False), times.astype(np.int64, copy=False)


def sort_and_dedup(times: np.ndarray, pitches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if times.size == 0:
        return times, pitches
    order = np.lexsort((-pitches, times))
    times = times[order]
    pitches = pitches[order]
    if times.size > 1:
        mask = np.ones(times.shape[0], dtype=bool)
        mask[1:] = (times[1:] != times[:-1]) | (pitches[1:] != pitches[:-1])
        times = times[mask]
        pitches = pitches[mask]
    return times, pitches


def list_midi_files(path: Path):
    if path.is_file():
        yield path
        return
    exts = {".mid", ".midi"}
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            if Path(name).suffix.lower() in exts:
                yield Path(root) / name


def load_processed_paths(out_dir: Path) -> set[str]:
    processed: set[str] = set()
    for idx_path in sorted(out_dir.glob("shard_*.jsonl")):
        try:
            with idx_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        midi = record.get("midi")
                        if midi:
                            processed.add(str(midi))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            continue
    return processed


def clean_shard_files(out_dir: Path) -> None:
    for path in out_dir.glob("shard_*.bin"):
        path.unlink(missing_ok=True)
    for path in out_dir.glob("shard_*.jsonl"):
        path.unlink(missing_ok=True)


def next_shard_index(out_dir: Path) -> int:
    indices: list[int] = []
    for path in out_dir.glob("shard_*.jsonl"):
        stem = path.stem
        if stem.startswith("shard_"):
            try:
                indices.append(int(stem.split("_", 1)[1]))
            except ValueError:
                continue
    return (max(indices) + 1) if indices else 0


def last_shard_index(out_dir: Path) -> int:
    indices: list[int] = []
    for path in out_dir.glob("shard_*.bin"):
        stem = path.stem
        if stem.startswith("shard_"):
            try:
                indices.append(int(stem.split("_", 1)[1]))
            except ValueError:
                continue
    for path in out_dir.glob("shard_*.jsonl"):
        stem = path.stem
        if stem.startswith("shard_"):
            try:
                indices.append(int(stem.split("_", 1)[1]))
            except ValueError:
                continue
    return max(indices) if indices else -1


class DatasetWriter:
    def __init__(
        self,
        out_dir: Path,
        shard_size_bytes: int,
        start_index: int = 0,
        resume: bool = False,
        flush_interval: int = 0,
        fsync: bool = False,
    ) -> None:
        self.out_dir = out_dir
        self.shard_size_bytes = max(0, int(shard_size_bytes))
        self.shard_index = int(start_index)
        self.offset = 0
        self.bin_fp = None
        self.idx_fp = None
        self.flush_interval = max(0, int(flush_interval))
        self.fsync = bool(fsync)
        self.write_count = 0
        self.token_count = 0
        self._open_shard(resume=resume)

    def _shard_paths(self) -> tuple[Path, Path]:
        name = f"shard_{self.shard_index:03d}"
        return self.out_dir / f"{name}.bin", self.out_dir / f"{name}.jsonl"

    def _open_shard(self, resume: bool = False) -> None:
        if self.bin_fp:
            self._write_header(self.token_count)
            self.bin_fp.close()
        if self.idx_fp:
            self.idx_fp.close()
        bin_path, idx_path = self._shard_paths()
        if resume and bin_path.exists():
            self.bin_fp = bin_path.open("r+b")
            file_size = bin_path.stat().st_size
            if file_size < HEADER_BYTES:
                raise ValueError("invalid shard: missing header")
            header = np.frombuffer(self.bin_fp.read(HEADER_BYTES), dtype=np.int32)
            if header[0] != HEADER_MAGIC or header[1] != HEADER_VERSION:
                raise ValueError("invalid shard header")
            data_bytes = file_size - HEADER_BYTES
            if data_bytes % np.dtype(TOKEN_DTYPE).itemsize != 0:
                raise ValueError("invalid shard: token bytes not aligned")
            self.token_count = int(data_bytes // np.dtype(TOKEN_DTYPE).itemsize)
            self._write_header(self.token_count)
            self.bin_fp.seek(0, os.SEEK_END)
            self.offset = file_size
            self.idx_fp = idx_path.open("a", encoding="utf-8")
        else:
            self.bin_fp = bin_path.open("w+b")
            self.token_count = 0
            self.bin_fp.write(self._make_header(self.token_count))
            self.idx_fp = idx_path.open("w", encoding="utf-8")
            self.offset = HEADER_BYTES

    def _rotate_if_needed(self, nbytes: int) -> None:
        if self.shard_size_bytes <= 0:
            return
        if self.offset <= HEADER_BYTES:
            return
        if self.offset + nbytes <= self.shard_size_bytes:
            return
        self.shard_index += 1
        self._open_shard(resume=False)

    def _make_header(self, ntok: int) -> bytes:
        header = np.zeros(HEADER_INTS, dtype=np.int32)
        header[0] = HEADER_MAGIC
        header[1] = HEADER_VERSION
        header[2] = int(ntok)
        return header.tobytes()

    def _write_header(self, ntok: int) -> None:
        if self.bin_fp is None:
            return
        current_pos = self.bin_fp.tell()
        self.bin_fp.seek(0)
        self.bin_fp.write(self._make_header(ntok))
        self.bin_fp.seek(current_pos)

    def write(self, tokens: np.ndarray, midi_path: Path) -> None:
        data = np.ascontiguousarray(tokens, dtype=TOKEN_DTYPE)
        nbytes = int(data.nbytes)
        self._rotate_if_needed(nbytes)
        if self.bin_fp is None or self.idx_fp is None:
            raise RuntimeError("DatasetWriter is closed")
        self.bin_fp.write(data.tobytes(order="C"))
        record = {
            "midi": str(midi_path),
            "offset_bytes": int(self.offset),
            "length": int(data.size),
            "dtype": str(data.dtype),
            "nbytes": int(data.nbytes),
        }
        self.idx_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.offset += nbytes
        self.write_count += 1
        self.token_count += int(data.size)
        if self.flush_interval > 0 and (self.write_count % self.flush_interval == 0):
            self._write_header(self.token_count)
            self.bin_fp.flush()
            self.idx_fp.flush()
            if self.fsync:
                os.fsync(self.bin_fp.fileno())
                os.fsync(self.idx_fp.fileno())

    def close(self) -> None:
        if self.bin_fp:
            self._write_header(self.token_count)
            self.bin_fp.close()
            self.bin_fp = None
        if self.idx_fp:
            self.idx_fp.close()
            self.idx_fp = None


def tokenize_midi_worker(
    midi_path: str, tpq: int, min_dur: int, include_drums: bool
) -> tuple[str, np.ndarray, int]:
    path = Path(midi_path)
    score = load_score(path)
    score_q = score.resample(tpq=int(tpq), min_dur=int(min_dur))
    pitches, times = collect_pitch_time_arrays(score_q.tracks, include_drums)
    times, pitches = sort_and_dedup(times, pitches)
    tokens, _, _, _, _, _ = encode_tokens(times, pitches, score_q)
    return str(path), tokens, int(pitches.size)


def tokenize_midi_worker_safe(
    midi_path: str, tpq: int, min_dur: int, include_drums: bool
) -> tuple[str, Optional[np.ndarray], int, Optional[str]]:
    try:
        path_str, tokens, note_count = tokenize_midi_worker(
            midi_path, tpq, min_dur, include_drums
        )
        return path_str, tokens, note_count, None
    except Exception as exc:
        return str(midi_path), None, 0, f"{type(exc).__name__}: {exc}"


def resolve_time_signature(score: Score, default_beats: int = 4, default_denom: int = 4) -> tuple[int, int]:
    del score
    return int(default_beats), int(default_denom)


def compute_bar_ticks(score: Score, default_beats: int = 4, default_denom: int = 4) -> tuple[int, int, int]:
    num, denom = resolve_time_signature(score, default_beats, default_denom)
    tpq = int(getattr(score, "tpq", 0))
    if tpq <= 0:
        raise ValueError("tpq must be > 0 to compute bar ticks")
    ticks_per_beat_num = tpq * 4
    if ticks_per_beat_num % denom != 0:
        raise ValueError("tpq not divisible by time signature denominator")
    ticks_per_beat = ticks_per_beat_num // denom
    bar_ticks = ticks_per_beat * num
    if bar_ticks <= 0:
        raise ValueError("bar_ticks must be > 0")
    return int(bar_ticks), int(num), int(denom)


def compute_end_time(score: Score, times_sorted: np.ndarray, time_shift: int) -> int:
    end_time = int(getattr(score, "end", lambda: int(times_sorted[-1]))())
    end_time = max(end_time, int(times_sorted[-1] + time_shift)) - time_shift
    if end_time < int(times_sorted[-1]):
        end_time = int(times_sorted[-1])
    return end_time


def compute_token_ids(bar_ticks: int) -> tuple[int, int, int, int, int, int]:
    bar_id = BAR_ID
    pos_offset = POS_OFFSET
    bos_id = pos_offset + int(bar_ticks)
    abs_end_id = bos_id + 1
    eof_id = bos_id + 2
    pad_id = bos_id + 3
    max_id = int(np.iinfo(TOKEN_DTYPE).max)
    if pad_id > max_id:
        raise ValueError("bar_ticks too large: token id exceeds dtype range")
    return bar_id, pos_offset, bos_id, abs_end_id, eof_id, pad_id


def encode_tokens(
    times: np.ndarray, pitches: np.ndarray, score: Score
) -> tuple[np.ndarray, int, int, int, int, int]:
    if times.size == 0:
        bar_ticks, ts_num, ts_denom = compute_bar_ticks(score)
        bar_id, pos_offset, bos_id, abs_end_id, eof_id, _ = compute_token_ids(bar_ticks)
        tokens = np.asarray(
            [bos_id, 0, abs_end_id, bar_id, pos_offset, eof_id],
            dtype=TOKEN_DTYPE,
        )
        return tokens, 0, 0, bar_ticks, ts_num, ts_denom

    time_shift = int(np.min(times))
    times_shifted = times - time_shift
    pitches_sorted = pitches

    global_max = int(np.max(pitches_sorted))
    if global_max < PITCH_MIN or global_max > PITCH_MAX:
        raise ValueError("pitch must be in 0..127")

    if times_shifted.size == 1:
        starts = np.array([0], dtype=np.int64)
    else:
        boundaries = np.flatnonzero(times_shifted[1:] != times_shifted[:-1]) + 1
        if boundaries.size == 0:
            starts = np.array([0], dtype=np.int64)
        else:
            starts = np.empty(boundaries.size + 1, dtype=np.int64)
            starts[0] = 0
            starts[1:] = boundaries
    unique_times = times_shifted[starts]
    counts = np.empty(starts.size, dtype=np.int64)
    if starts.size > 1:
        counts[:-1] = starts[1:] - starts[:-1]
    counts[-1] = times_shifted.size - starts[-1]

    if unique_times.size > 1:
        if np.any(unique_times[1:] < unique_times[:-1]):
            raise ValueError("times must be non-decreasing after sort()")

    bar_ticks, ts_num, ts_denom = compute_bar_ticks(score)
    if bar_ticks <= 0:
        raise ValueError("bar_ticks must be > 0")
    bar_id, pos_offset, bos_id, abs_end_id, eof_id, _ = compute_token_ids(bar_ticks)

    gpi_vals = int(global_max) - pitches_sorted
    if gpi_vals.min() < 0 or gpi_vals.max() > 127:
        raise ValueError("GPI out of range")
    gpi_vals = gpi_vals.astype(np.int64, copy=False)

    n_groups = int(counts.size)
    bars = unique_times // int(bar_ticks)
    pos_vals = unique_times - (bars * int(bar_ticks))

    tokens_list: list[int] = [bos_id, global_max, abs_end_id]
    current_bar = -1
    gpi_index = 0
    for i in range(n_groups):
        bar = int(bars[i])
        pos = int(pos_vals[i])
        if bar > current_bar:
            tokens_list.append(bar_id)
            current_bar = bar
        tokens_list.append(pos_offset + pos)
        count = int(counts[i])
        if count:
            tokens_list.extend(gpi_vals[gpi_index : gpi_index + count].tolist())
            gpi_index += count
    tokens_list.append(eof_id)

    tokens = np.asarray(tokens_list, dtype=TOKEN_DTYPE)
    return tokens, time_shift, global_max, bar_ticks, ts_num, ts_denom


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Tokenize MIDI file(s) into globalmax_reltime_gpi_barpos token shards (4/4 fixed, packed ids)."
        )
    )
    p.add_argument("input", type=Path, help="Input MIDI file or directory path")
    p.add_argument("--tpq", type=int, default=8, help="Ticks per quarter")
    p.add_argument("--min-dur", type=int, default=1, help="Minimum duration")
    p.add_argument(
        "--include-drums",
        action="store_true",
        help="Include drum tracks in the merge (default: excluded)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dataset"),
        help="Output directory for dataset shards",
    )
    p.add_argument(
        "--shard-size-mb",
        type=int,
        default=2048,
        help="Shard size limit in MB (0 disables sharding)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes (default: cpu_count-1 for directories)",
    )
    p.add_argument(
        "--queue-size",
        type=int,
        default=0,
        help="Writer queue size (0 = auto)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip already indexed files and continue into new shards",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort immediately on the first failed MIDI",
    )
    p.add_argument(
        "--error-log",
        type=Path,
        default=None,
        help="Optional error log path (one JSON per line)",
    )
    p.add_argument(
        "--timeout-sec",
        type=float,
        default=None,
        help="Per-MIDI timeout in seconds (parallel only)",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Retries for failed/timeout MIDI files",
    )
    p.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=0,
        help="Recycle worker processes after N tasks (0 disables)",
    )
    p.add_argument(
        "--flush-interval",
        type=int,
        default=0,
        help="Flush shard files every N samples (0 disables)",
    )
    p.add_argument(
        "--fsync",
        action="store_true",
        help="Also fsync shard files on flush",
    )
    p.add_argument(
        "--failed-list",
        type=Path,
        default=None,
        help="Optional path to write failed MIDI list (one path per line)",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N processed files (0 disables)",
    )
    return p.parse_args()


def peek_iter(it):
    iterator = iter(it)
    try:
        first = next(iterator)
    except StopIteration:
        return None, iter(())
    return first, iterator


def supports_max_tasks_per_child() -> bool:
    try:
        return "max_tasks_per_child" in inspect.signature(ProcessPoolExecutor).parameters
    except (TypeError, ValueError):
        return False


def main() -> None:
    args = parse_args()
    if args.tpq <= 0:
        raise ValueError("--tpq must be > 0")
    if args.min_dur <= 0:
        raise ValueError("--min-dur must be > 0")
    if args.timeout_sec is not None and args.timeout_sec <= 0:
        raise ValueError("--timeout-sec must be > 0")
    if args.retries < 0:
        raise ValueError("--retries must be >= 0")
    if args.max_tasks_per_child < 0:
        raise ValueError("--max-tasks-per-child must be >= 0")
    if args.flush_interval < 0:
        raise ValueError("--flush-interval must be >= 0")

    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"input path not found: {input_path}")

    out_dir = args.out_dir
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"out-dir must be a directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        processed = load_processed_paths(out_dir)
        last_index = last_shard_index(out_dir)
        resume_writer = last_index >= 0
        start_index = last_index if resume_writer else 0
    else:
        clean_shard_files(out_dir)
        processed = set()
        start_index = 0
        resume_writer = False

    total_files = 0
    found_any = False
    for path in list_midi_files(input_path):
        found_any = True
        if processed and str(path) in processed:
            continue
        total_files += 1
    if not found_any:
        raise ValueError("no MIDI files found")
    if total_files == 0:
        return

    def iter_filtered_paths():
        for path in list_midi_files(input_path):
            if processed and str(path) in processed:
                continue
            yield path

    midi_paths = iter_filtered_paths()

    shard_size_bytes = int(args.shard_size_mb) * 1024 * 1024
    writer = DatasetWriter(
        out_dir,
        shard_size_bytes,
        start_index=start_index,
        resume=resume_writer,
        flush_interval=args.flush_interval,
        fsync=args.fsync,
    )

    workers = args.workers
    if workers is None:
        if input_path.is_dir():
            cpu = os.cpu_count() or 1
            workers = max(1, cpu - 1)
        else:
            workers = 0

    error_fh = None
    if args.error_log is not None:
        error_path = args.error_log
        if not error_path.is_absolute():
            error_path = out_dir / error_path
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_fh = error_path.open("a", encoding="utf-8")

    failed_fh = None
    if args.failed_list is not None:
        failed_path = args.failed_list
        if not failed_path.is_absolute():
            failed_path = out_dir / failed_path
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        failed_fh = failed_path.open("a", encoding="utf-8")

    error_count = 0
    start_time = time.perf_counter()
    progress_every = max(0, int(args.progress_every))
    queue_size = int(args.queue_size)
    if queue_size <= 0:
        if workers and workers > 1:
            queue_size = max(32, int(workers) * 4)
        else:
            queue_size = 32

    write_queue: Queue = Queue(maxsize=queue_size)
    writer_error: list[BaseException] = []
    writer_failed = threading.Event()
    stats = {"processed": 0, "tokens": 0, "notes": 0}

    def report_progress(processed: int, token_total: int) -> None:
        elapsed = time.perf_counter() - start_time
        rate = processed / elapsed if elapsed > 0 else 0.0
        tok_rate = token_total / elapsed if elapsed > 0 else 0.0
        pct = (processed / total_files * 100) if total_files else 0.0
        print(
            f"[progress] {processed}/{total_files} ({pct:.1f}%) "
            f"| {rate:.2f} files/s | {tok_rate:.1f} tok/s",
            flush=True,
        )

    def writer_loop() -> None:
        processed = 0
        token_total = 0
        note_total = 0
        try:
            while True:
                item = write_queue.get()
                if item is None:
                    write_queue.task_done()
                    break
                tokens, midi_path, note_count = item
                writer.write(tokens, Path(midi_path))
                processed += 1
                token_total += int(tokens.size)
                note_total += int(note_count)
                if progress_every and processed % progress_every == 0:
                    report_progress(processed, token_total)
                write_queue.task_done()
        except Exception as exc:
            writer_error.append(exc)
            writer_failed.set()
            while True:
                try:
                    write_queue.get_nowait()
                except Empty:
                    break
                else:
                    write_queue.task_done()
        finally:
            stats["processed"] = processed
            stats["tokens"] = token_total
            stats["notes"] = note_total

    writer_thread = threading.Thread(target=writer_loop, name="dataset-writer")
    writer_thread.start()

    retries = max(0, int(args.retries))
    timeout_sec = args.timeout_sec if args.timeout_sec and args.timeout_sec > 0 else None
    supports_mtpc = supports_max_tasks_per_child()

    def handle_error(midi_path: str, err: str) -> None:
        nonlocal error_count
        error_count += 1
        if error_fh is not None:
            record = {"midi": midi_path, "error": err}
            error_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        if failed_fh is not None:
            failed_fh.write(midi_path + "\n")
        if args.fail_fast:
            raise RuntimeError(err)

    def enqueue_result(tokens: np.ndarray, midi_path: str, note_count: int) -> None:
        if writer_failed.is_set():
            raise RuntimeError("Writer thread failed")
        while True:
            try:
                write_queue.put((tokens, midi_path, note_count), timeout=0.1)
                break
            except Full:
                if writer_failed.is_set():
                    raise RuntimeError("Writer thread failed")

    def handle_success(midi_path: str, tokens: np.ndarray, note_count: int) -> None:
        enqueue_result(tokens, midi_path, note_count)

    try:
        if workers and workers > 1:
            workers = int(workers)
            prefetch = max(2, workers * 2)
            use_mtpc = supports_mtpc and args.max_tasks_per_child > 0
            batch_limit = None
            if args.max_tasks_per_child > 0 and not supports_mtpc:
                batch_limit = int(args.max_tasks_per_child) * workers
                if batch_limit <= 0:
                    batch_limit = None

            def run_batch(batch_iter, use_max_tasks: bool) -> None:
                executor_kwargs = {"max_workers": workers}
                if use_max_tasks:
                    executor_kwargs["max_tasks_per_child"] = int(args.max_tasks_per_child)
                executor = ProcessPoolExecutor(**executor_kwargs)
                pending = set()
                future_info: dict = {}
                attempts: dict[str, int] = {}
                batch_iter = iter(batch_iter)

                def submit_path(path_str: str) -> None:
                    if path_str not in attempts:
                        attempts[path_str] = 0
                    fut = executor.submit(
                        tokenize_midi_worker_safe,
                        path_str,
                        args.tpq,
                        args.min_dur,
                        args.include_drums,
                    )
                    pending.add(fut)
                    future_info[fut] = {"path": path_str, "start": time.monotonic()}

                def maybe_resubmit(path_str: str, err_msg: str) -> None:
                    if attempts.get(path_str, 0) < retries:
                        attempts[path_str] = attempts.get(path_str, 0) + 1
                        submit_path(path_str)
                        return
                    handle_error(path_str, err_msg)

                try:
                    while len(pending) < prefetch:
                        try:
                            path = next(batch_iter)
                        except StopIteration:
                            break
                        submit_path(str(path))

                    poll_interval = 0.2 if timeout_sec is not None else None
                    while pending:
                        done, _ = wait(
                            pending, timeout=poll_interval, return_when=FIRST_COMPLETED
                        )
                        if not done:
                            if timeout_sec is not None:
                                now = time.monotonic()
                                timed_out = [
                                    fut
                                    for fut in pending
                                    if now - future_info[fut]["start"] > timeout_sec
                                ]
                                for fut in timed_out:
                                    info = future_info.pop(fut)
                                    pending.remove(fut)
                                    fut.cancel()
                                    err_msg = f"TimeoutError: exceeded {timeout_sec:.3f}s"
                                    maybe_resubmit(info["path"], err_msg)
                            while len(pending) < prefetch:
                                try:
                                    path = next(batch_iter)
                                except StopIteration:
                                    break
                                submit_path(str(path))
                            continue

                        for fut in done:
                            pending.remove(fut)
                            info = future_info.pop(fut)
                            try:
                                midi_path, tokens, note_count, err = fut.result()
                            except Exception as exc:
                                midi_path = info["path"]
                                tokens = None
                                note_count = 0
                                err = f"{type(exc).__name__}: {exc}"
                            if err is not None or tokens is None:
                                maybe_resubmit(midi_path, err or "Unknown error")
                            else:
                                handle_success(midi_path, tokens, note_count)
                            while len(pending) < prefetch:
                                try:
                                    path = next(batch_iter)
                                except StopIteration:
                                    break
                                submit_path(str(path))
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)

            if batch_limit is None:
                run_batch(midi_paths, use_mtpc)
            else:
                while True:
                    batch_iter = islice(midi_paths, batch_limit)
                    first, rest = peek_iter(batch_iter)
                    if first is None:
                        break
                    run_batch(chain([first], rest), False)
        else:
            for path in midi_paths:
                path_str = str(path)
                attempt = 0
                while True:
                    midi_path, tokens, note_count, err = tokenize_midi_worker_safe(
                        path_str, args.tpq, args.min_dur, args.include_drums
                    )
                    if err is None and tokens is not None:
                        handle_success(midi_path, tokens, note_count)
                        break
                    if attempt < retries:
                        attempt += 1
                        continue
                    handle_error(midi_path, err or "Unknown error")
                    break
    finally:
        if writer_thread is not None:
            if not writer_failed.is_set():
                while True:
                    try:
                        write_queue.put(None, timeout=0.1)
                        break
                    except Full:
                        if writer_failed.is_set():
                            break
                if not writer_failed.is_set():
                    write_queue.join()
            writer_thread.join()
        writer.close()
        if error_fh is not None:
            error_fh.close()
        if failed_fh is not None:
            failed_fh.close()

    if writer_error:
        raise RuntimeError(f"WriterError: {writer_error[0]}")

    processed_count = int(stats.get("processed", 0))
    token_total = int(stats.get("tokens", 0))
    note_total = int(stats.get("notes", 0))
    elapsed = time.perf_counter() - start_time
    rate = processed_count / elapsed if elapsed > 0 else 0.0
    tok_rate = token_total / elapsed if elapsed > 0 else 0.0
    avg_tokens = token_total / processed_count if processed_count else 0.0
    avg_notes = note_total / processed_count if processed_count else 0.0
    print(
        f"[done] processed={processed_count} errors={error_count} "
        f"| {elapsed:.2f}s | {rate:.2f} files/s "
        f"| {tok_rate:.1f} tok/s | avg_tokens={avg_tokens:.1f} "
        f"| avg_notes={avg_notes:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
