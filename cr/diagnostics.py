"""Bounded background diagnostic capture; never waits for disk in the decision loop."""
import gzip
import hashlib
import json
import queue
import threading
import time

import numpy as np


def encode(value):
    if isinstance(value, np.ndarray):
        return dict(dtype=str(value.dtype), shape=list(value.shape), values=value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class Capture:
    def __init__(self, path, *, max_queue=64):
        self.path = str(path)
        self.queue = queue.Queue(max_queue)
        self.dropped = self.written = 0
        self.error = None
        self.closed = False
        self.previous = None
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        try:
            with gzip.open(self.path, 'wt', compresslevel=1) as output:
                while True:
                    row = self.queue.get()
                    if row is None:
                        break
                    output.write(json.dumps(row, default=encode, separators=(',', ':'), allow_nan=False) + '\n')
                    self.written += 1
        except Exception as exc:
            self.error = repr(exc)

    def emit(self, row):
        if self.closed or self.error:
            self.dropped += 1
            return
        try:
            self.queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def frame(self, frame):
        now = time.monotonic_ns()
        tick = frame.get('game_tick')
        previous = self.previous
        self.previous = (tick, now)
        self.emit(dict(event='reader_frame', host_monotonic_ns=now,
                       tick_delta=None if previous is None else tick - previous[0],
                       host_gap_ms=None if previous is None else (now - previous[1]) / 1e6,
                       frame=frame))

    def tensors(self, tick, x, state=None):
        # CPU copies detach the capture from subsequent model operations.
        arrays = {k: v.detach().cpu().numpy().copy() for k, v in x.items()}
        if state is not None:
            arrays.update({f'recurrent_{i}': v.detach().cpu().numpy().copy() for i, v in enumerate(state)})
        bad = {k: int((~np.isfinite(v)).sum()) for k, v in arrays.items() if not np.isfinite(v).all()}
        # JSON has no portable NaN/Inf representation. Preserve them explicitly as strings.
        values = {k: v if np.isfinite(v).all() else v.astype(str) for k, v in arrays.items()}
        self.emit(dict(event='model_inputs', tick=tick, host_monotonic_ns=time.monotonic_ns(),
                       recurrent_initial=state is None, original_dtypes={k: str(v.dtype) for k, v in arrays.items()},
                       nonfinite=bad, tensors=values))

    def audit(self, frame, records, vocab, inputs):
        missing_characters, missing_effects, unknown = set(), set(), set()

        def check(kind, value):
            # Inspect without counting diagnostic lookups as inference misses.
            if value is not None and value not in (0, -1):
                value = int(value) & 0xffffffff
                if value not in vocab.index:
                    unknown.add((kind, value))

        for entity in frame.get('entities', ()):
            cid = int(entity.get('data_id', 0)) & 0xffffffff
            cid = records.aliases.get(cid, cid)
            if entity.get('kind') != 0 and entity.get('hp', -1) >= 0 and cid not in records.characters:
                missing_characters.add(cid)
            check('data', cid)
            check('card', records.card(entity.get('card_id', 0)))
            for buff in (entity.get('us') or {}).get('buffs', ()):
                effect = int(buff[0]) & 0xffffffff
                if effect not in records.effects:
                    missing_effects.add(effect)
                check('effect', effect)
        for player in frame.get('players', ()):
            for cid in player.get('deck_card_ids', ()):
                check('deck_card', cid)
        self.emit(dict(event='input_audit', tick=frame['game_tick'],
                       missing_characters=sorted(missing_characters), missing_effects=sorted(missing_effects),
                       unknown_vocab=sorted(unknown),
                       input_shapes={k: list(v.shape) for k, v in inputs.items()},
                       raw_objects=len(frame.get('entities', ())),
                       model_objects=int((inputs['obj_side'] > 0).sum())))

    def close(self):
        self.closed = True
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=.1)
                break
            except queue.Full:
                pass
        self.thread.join(timeout=10)
        return dict(event='diagnostic_summary', path=self.path, written=self.written, dropped=self.dropped,
                    error=self.error, writer_alive=self.thread.is_alive())


def provenance(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
