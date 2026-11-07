"""Placement-search labels for distillation (5 Nov 2026; the user's target: at least 1M a day). One model plays itself
on random pool deck pairs; both seats search their card plays as search_arena.py's fair teacher does: rollouts with both
sides playing (--rollout policy), judged by the searching seat's own value head (--judge value), no search before the
opponent's 3rd card play (--after 3) or while the seat's value says it has won or lost (--won 0.95, --lost 0.05).

One inference server per GPU serves all of its workers' games and their forked copies (bigger batches than a server
per game); --workers-per-gpu game processes each play games until --hours have passed.

Each game -> OUT/w<worker>-<game>.npz, per seat o that searched at least once:
    s<o>_<key>     its packed inputs (policy.pack) at every decision, stacked [T, ...], zero-padded along objects
    s<o>_action    the action its policy sampled at each decision [T] (a search may have moved the tile: see the labels)
    s<o>_t         the decision of each search [n]; s<o>_slot its hand slot [n]
    s<o>_cells     the candidate tiles [n, K] in the seat's own frame (the policy's sampled tile first; -1: none)
    s<o>_scores    each candidate's judged logit of P(win) [n, K] (NaN: none)
    meta           JSON: pair, seed, winner, ticks, both seats' search stats

    python experiments/search_labels.py MODEL.pt --engine ~/clapha-engine-conv --engine-lib LIB --engine-assets DATA \\
        --pool pool.json --out DIR --gpus 4 --servers-per-gpu 2 --workers-per-gpu 4 --hours 23
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr import observation as C, policy  # noqa: E402
from cr.engine import Engine, Game, tower_hp  # noqa: E402
from search_arena import Remote, Searcher, serve  # noqa: E402


def play(engine, remote, vocab, pair, seed, args):
    engine.overlap = [True, True]
    game = Game(engine, pair, seed, {0: vocab, 1: vocab})
    start = tower_hp(game.state)
    seats = [Searcher(remote, 'a', o, engine, args) for o in (0, 1)]
    rows, actions = ([], []), ([], [])
    while not game.over:
        views = game.views()
        choices = remote(*(seats[o].ask(views[o], args.k) for o in (0, 1)))  # one round trip for both seats
        for o in (0, 1):
            rows[o].append(seats[o].row)
            actions[o].append(seats[o].action)
        for o in (0, 1):  # seat 1 searches against seat 0's final choice
            n = len(seats[o].log)
            choices[o] = seats[o].decide(choices[o], start, game, seats[1 - o], choices[1 - o], None)
            if len(seats[o].log) > n:
                seats[o].log[-1]['t'] = len(rows[o]) - 1
        game.step({0: choices[0], 1: choices[1]})
        for o in (0, 1):
            seats[o].seen += sum(p.owner != o for p in game.state.plays)
    s = game.state
    if not s.finalized:
        raise RuntimeError('native battle did not finalize')
    out = {}
    for o, seat in enumerate(seats):
        found = [e for e in seat.log if None not in e['scores'] and len(e['cells']) > 1]
        if not found:
            continue
        K = max(len(e['cells']) for e in found)
        out.update({f's{o}_{k}': policy.pad_stack([r[k] for r in rows[o]]) for k in rows[o][0]})
        out[f's{o}_action'] = np.array(actions[o], np.int32)
        out[f's{o}_t'] = np.array([e['t'] for e in found], np.int32)
        out[f's{o}_slot'] = np.array([e['slot'] for e in found], np.int8)
        out[f's{o}_cells'] = np.array([e['cells'] + [-1] * (K - len(e['cells'])) for e in found], np.int16)
        out[f's{o}_scores'] = np.array([e['scores'] + [np.nan] * (K - len(e['scores'])) for e in found], np.float32)
    meta = dict(seed=seed, winner=s.result, crowns=list(s.crowns), ticks=s.tick, stats=[seat.stats for seat in seats])
    return out, meta


def worker(index, address, args):
    import torch  # the inputs need the model's vocabulary (only that: the model lives in the server)
    vocab = C.Vocab(torch.load(args.a, map_location='cpu', weights_only=False)['vocab'])
    engine = Engine(args.engine, lib=args.engine_lib, assets=args.engine_assets)
    pool = json.loads(args.pool.read_text())
    remote = Remote(address)
    rng = np.random.default_rng([args.seed, index])
    deadline, n = time.time() + 3600 * args.hours, 0
    while time.time() < deadline:
        pair, seed = int(rng.integers(len(pool))), int(rng.integers(1 << 30))
        t0 = time.monotonic()
        try:
            out, meta = play(engine, remote, vocab, pool[pair], seed, args)
        except Exception:  # noqa: BLE001  one bad game must not end a day-long worker (the next game resets the engine)
            traceback.print_exc()
            continue
        meta.update(pair=pair, seconds=round(time.monotonic() - t0, 1))
        np.savez_compressed(args.out / f'w{index:03d}-{n:05d}.npz', meta=np.frombuffer(json.dumps(meta).encode(), np.uint8),
                            **out)
        print(json.dumps(dict(worker=index, game=n, seconds=meta['seconds'], ticks=meta['ticks'],
                              labels=sum(len(v) for k, v in out.items() if k.endswith('_t')))), flush=True)
        n += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', type=Path)
    for key in ('engine', 'engine-lib', 'engine-assets', 'pool', 'out'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--gpus', type=int, default=1, help='one server on each of the first N visible GPUs')
    ap.add_argument('--servers-per-gpu', type=int, default=1, help='one Python server cannot keep a GPU busy with small '
                    'batches (5 Nov: 99%% busy, GPU 35%%); several share it')
    ap.add_argument('--workers-per-gpu', type=int, default=4)
    ap.add_argument('--hours', type=float, default=1.0)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--horizon', type=int, default=160)
    ap.add_argument('--rollout', choices=('passive', 'policy'), default='policy')
    ap.add_argument('--judge', choices=('formula', 'value'), default='value')
    ap.add_argument('--material', type=float, default=0.3, help="--judge formula's unit-hp weight")
    ap.add_argument('--won', type=float, default=0.95)
    ap.add_argument('--lost', type=float, default=0.05)
    ap.add_argument('--after', type=int, default=3)
    ap.add_argument('--timeout', type=float, default=120)
    ap.add_argument('--temperature', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    args.a = args.b = args.model  # what serve() reads
    args.trace = False
    args.out.mkdir(parents=True, exist_ok=True)
    spawn = mp.get_context('spawn')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(args.gpus)))).split(',')
    addresses, ready = [], []
    for g in [g for g in range(args.gpus) for _ in range(args.servers_per_gpu)]:  # a spawned process inherits the
        os.environ['CUDA_VISIBLE_DEVICES'] = visible[g]                             # environment: one GPU each
        addresses.append(os.path.join(tempfile.mkdtemp(), 'server'))
        ready.append(spawn.Event())
        spawn.Process(target=serve, args=(addresses[-1], args, ready[-1]), daemon=True).start()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(visible)
    if not all(r.wait(600) for r in ready):
        raise RuntimeError('an inference server did not start')
    workers = [spawn.Process(target=worker, args=(w, addresses[w % len(addresses)], args))
               for w in range(args.gpus * args.workers_per_gpu)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()


if __name__ == '__main__':
    main()
