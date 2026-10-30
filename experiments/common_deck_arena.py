"""Frozen-policy round robin of frequent human decks, with batched GPU inference.

Each distinct deck pair gets ten seeds, each played in both seats (20 games).
Run preparation separately to produce decks.json; workers never change training files.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import queue
import random
import sys
import time
import traceback


def tasks(meta):
    if 'tasks' in meta:
        return meta['tasks']
    result = []
    for a in range(len(meta['decks'])):
        for b in range(a + 1, len(meta['decks'])):
            for seed_index in range(10):
                seed = 29102900 + (a * len(meta['decks']) + b) * 10 + seed_index
                for seat in (0, 1):
                    result.append(dict(id=len(result), a=a, b=b, seed=seed, a_owner=seat))
    # Spread every matchup through the run, instead of finishing one deck first.
    random.Random(1029).shuffle(result)
    return result


def match(meta, task):
    if 'pair' in task:
        return task['pair']
    decks = []
    for index in (task['a'], task['b']):
        deck = list(meta['decks'][index]['deck'])
        random.Random(task['seed'] + index * 7919).shuffle(deck)
        decks.append(deck)
    if task['a_owner']:
        decks.reverse()
    return dict(decks=decks, sc=[meta['tower'], meta['tower']], kt=[16, 16],
                exp=[16, 16], lvlcap=16, cardlvlmin=16)


def collect(index, conn, work, results, args):
    import numpy as np
    from cr import policy
    from cr.actors import layout, wire
    from cr.engine import Engine, Game
    try:
        meta = json.loads(args.decks.read_text())
        vocab = policy.C.Vocab(json.loads((args.out / 'vocab.json').read_text()))
        sim = Engine(args.engine, lib=args.engine_lib, assets=args.engine_assets)
        while True:
            try:
                task = work.get_nowait()
            except queue.Empty:
                break
            random.seed(task['seed']); np.random.seed(task['seed'] % (2**32))
            started = time.monotonic()
            sim.overlap = [True, True]
            game = Game(sim, match(meta, task), task['seed'], {0: vocab, 1: vocab})
            shape = None
            refused = [0, 0]
            while not game.over:
                rows = {o: policy.pack(x) for o, x in game.views().items()}
                if shape is None:
                    shape = layout(rows[0]); conn.send(('reset', index, shape))
                conn.send(('act', index, [(o, 'baseline' if task.get('opponent') and o != task['a_owner'] else 'frozen',
                                          task.get('opponent') if o != task['a_owner'] else task.get('candidate'),
                                          wire(rows[o], shape)) for o in (0, 1)]))
                replies = conn.recv()
                actions = {o: policy.decode(reply[0], o) for o, reply in enumerate(replies)}
                submitted, transition = game.step(actions)
                for action, accepted in zip(submitted, transition.executed):
                    refused[action.owner] += not accepted
            s = game.state
            if not s.finalized:
                raise RuntimeError('native battle did not finalize')
            score = .5 if s.result not in (0, 1) else float(s.result == task['a_owner'])
            results.put(dict(**task, score=score, crowns=list(s.crowns), tick=s.tick,
                             refused=refused, seconds=round(time.monotonic()-started, 3)))
        conn.send(('done', index, None))
        conn.close()
    except BaseException:
        traceback.print_exc()
        results.put(dict(error=traceback.format_exc(), worker=index))
    finally:
        results.close(); results.join_thread()
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


def serve(gpu, conns, results, args):
    import torch
    from cr import policy
    from cr.actors import Decider
    try:
        torch.set_num_threads(1)
        device = torch.device(f'cuda:{gpu}')
        torch.cuda.set_device(device)
        torch.manual_seed(291029 + args.shard * 100 + gpu)
        model, _, _ = policy.load(args.checkpoint, device)
        model.eval(); model.temperature = .05; model.mix = 0.; model.overlap = True
        decider = Decider({'frozen': model}, device, model.config.core, overlap=True, frozen_temperature=.05)
        while conns:
            pending = []
            for conn in wait(conns, timeout=1):
                try:
                    kind, index, payload = conn.recv()
                except EOFError:
                    conns.remove(conn); continue
                if kind == 'done':
                    conns.remove(conn); conn.close()
                elif kind == 'reset':
                    decider.shape = payload; decider.reset(index)
                else:
                    pending.append((conn, [((index, o), name, path, row) for o, name, path, row in payload]))
            if pending:
                replies = decider.act([r for _, rs in pending for r in rs])
                offset = 0
                for conn, rs in pending:
                    conn.send(replies[offset:offset+len(rs)]); offset += len(rs)
    except BaseException:
        traceback.print_exc()
        results.put(dict(error=traceback.format_exc(), server=gpu))
    finally:
        results.close(); results.join_thread()
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


def run(args):
    import torch
    from cr import policy
    meta = json.loads(args.decks.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    selected = [t for t in tasks(meta) if t['id'] % args.shards == args.shard]
    path = args.out / f'games-{args.shard}.jsonl'
    done = {json.loads(s)['id'] for s in path.read_text().splitlines()} if path.exists() else set()
    selected = [t for t in selected if t['id'] not in done]
    if args.limit:
        selected = selected[:args.limit]
    _, vocab, _ = policy.load(args.checkpoint, 'cpu')
    vocab_path = args.out / 'vocab.json'
    # Identical content for all shards, replaced atomically to avoid partial reads.
    tmp = args.out / f'vocab-{args.shard}.tmp'
    tmp.write_text(json.dumps(list(vocab.index)))
    # Vocab.index maps native IDs to token indices; its keys are the saved vocab.
    os.replace(tmp, vocab_path)
    gpus = torch.cuda.device_count()
    if not gpus:
        raise RuntimeError('GPU allocation required')
    provenance = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      decks_sha256=hashlib.sha256(args.decks.read_bytes()).hexdigest(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      temperature=.05, overlap=True, ability_explore=0, shard=args.shard, shards=args.shards,
                      games=len(selected), workers=args.workers, seed_note='Native and deck-order seeds paired across seats; batched policy RNG stochastic')
    (args.out / f'metadata-{args.shard}.json').write_text(json.dumps(provenance, indent=2))
    ctx = mp.get_context('spawn'); work = ctx.Queue(); results = ctx.Queue()
    for t in selected:
        work.put(t)
    servers, workers = [], []
    pipes = [ctx.Pipe() for _ in range(args.workers)]
    for gpu in range(gpus):
        p = ctx.Process(target=serve, args=(gpu, [a for i,(a,b) in enumerate(pipes) if i%gpus==gpu], results, args))
        p.start(); servers.append(p)
    for i, (a, b) in enumerate(pipes):
        p = ctx.Process(target=collect, args=(i, b, work, results, args)); p.start(); workers.append(p)
        a.close(); b.close()
        time.sleep(.08)
    completed = 0; start = time.monotonic()
    with path.open('a') as f:
        while completed < len(selected):
            try:
                row = results.get(timeout=20)
            except queue.Empty:
                if all(not p.is_alive() for p in servers) or all(not p.is_alive() for p in workers):
                    raise RuntimeError('workers exited before all games finished')
                continue
            if 'error' in row:
                raise RuntimeError(row['error'])
            f.write(json.dumps(row)+'\n'); f.flush(); completed += 1
            if completed % 20 == 0 or completed == len(selected):
                print(json.dumps(dict(completed=completed, total=len(selected), games_hour=round(completed*3600/(time.monotonic()-start)))), flush=True)
    print('COMPLETE', args.shard, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source','decks','out','checkpoint','engine','engine-lib','engine-assets'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    p.add_argument('--workers', type=int, default=40); p.add_argument('--limit', type=int, default=0)
    args = p.parse_args(); sys.path.insert(0,str(args.source))
    try:
        run(args)
    except BaseException:
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
    sys.stdout.flush(); os._exit(0)


if __name__ == '__main__':
    main()
