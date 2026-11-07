"""How much does placement search add to a model? (5 Nov 2026, port of the 26 Oct search arena to obs-v2 and the token
network.) Paired-seat games, A against B, each deck pair played with A in both seats. With --search, A's card plays go
to the best-scoring of its --k best tiles and the one it sampled, each tried by cr.search in a forked copy of the game
for --horizon ticks (opponent plays A cannot see yet are taken out). The copy is played out by --rollout: 'passive' (no
further plays) or 'policy' (both sides keep playing, A and B's own networks; the opponent sees its true hand: an offline
teacher only). It is scored by --judge: 'formula' (tower damage + --material x unit hp), 'critic' (the PPO critic's
P(win) logit for A at the horizon, --critic: a ppo.py checkpoint with 'critic'; the critic follows A's whole game; it
sees the opponent's hand) or 'value' (A's own value head: only what A sees).
Both models play at --temperature with overlap, as the gates do.

Every network call (both players, the critic, inside the copies too) goes to this shard's own inference server, a
spawned process on the GPU that batches what arrives together: a CPU forward of the 24.5M token network is ~60 ms, and a
policy rollout makes ~64 of them per candidate. Linux, inside a Slurm allocation.

    python experiments/search_arena.py A.pt B.pt --engine ~/clapha-engine-conv --engine-lib LIB --engine-assets DATA \\
        --pool pool.json --games 400 --out out/shard-0.jsonl --search [--rollout policy] [--judge critic] --shard 0/8

Resumable: games already in --out are skipped. The same --seed gives the same deck pairs and game seeds in every run.
"""
import argparse
import collections
import json
import multiprocessing as mp
from multiprocessing.connection import Client, Listener, wait
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cr import observation as C, policy, search  # noqa: E402
from cr.actors import PRIVATE  # noqa: E402
from cr.engine import Engine, Game, tower_hp  # noqa: E402


# ---- the inference server ------------------------------------------------------------------------------------------------
def serve(address, args, ready):
    """Requests ('act', model, row, state, top) -> (action, top tiles or None, state) and ('critic', row, opp, state) ->
    (P(win) logit, state); rows are policy.pack()ed views, states numpy (h, c) or None. One batch per model per wakeup."""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(int.from_bytes(os.urandom(4), 'little'))
    models = {}  # name -> model; A and B from the same checkpoint share one (one forward for both seats' rows)
    for side in ('a', 'b'):
        same = [s for s in models if getattr(args, s).resolve() == getattr(args, side).resolve()]
        models[side] = models[same[0]] if same else policy.load(getattr(args, side), device)[0]
        models[side].temperature, models[side].mix, models[side].overlap = args.temperature, 0.0, True
    critic = None
    if args.judge == 'critic':
        import ppo  # the repository's learner module: the Critic class
        saved = torch.load(args.critic, map_location='cpu', weights_only=False)
        critic = ppo.Critic(policy.build(policy.Config(**saved['config'])))
        critic.load_state_dict(saved['critic'])
        critic = critic.to(device).eval()
    listener = Listener(address, family='AF_UNIX')
    conns, arrivals = [], []
    lock = threading.Lock()

    def accept():
        while True:
            conn = listener.accept()
            with lock:
                arrivals.append(conn)
    threading.Thread(target=accept, daemon=True).start()
    ready.set()

    def batch(rows, states, hidden):
        x = policy.tensors({k: policy.pad_stack([r[k] for r in rows])[:, None] for k in rows[0]}, device)
        zero = np.zeros((1, hidden), np.float32)
        h, c = (torch.from_numpy(np.stack([(s[j] if s is not None else zero)[0] for s in states])[None]).to(device)
                for j in (0, 1))
        return x, (h.contiguous(), c.contiguous())

    def unstate(state):
        """The new LSTM states [1, B, core] -> one (h, c) per row, numpy [1, core]: one readback for the whole batch."""
        h, c = (s.float().cpu().numpy() for s in state)
        return [(h[:, i].copy(), c[:, i].copy()) for i in range(h.shape[1])]

    clock, seen = collections.Counter(), time.monotonic()  # where the server's time goes, printed each minute

    def lap(key, t):
        now = time.monotonic()
        clock[key] += now - t
        return now

    with torch.no_grad(), torch.autocast(device.type, torch.bfloat16, enabled=device.type == 'cuda'):  # as cr.actors
        while True:
            with lock:
                conns += arrivals
                arrivals.clear()
            if not conns:
                time.sleep(0.001)
                continue
            t = time.monotonic()
            ready_conns = wait(conns, timeout=0.01)
            t = lap('wait', t)
            requests = []  # (conn, its message's index, sub-request)
            for conn in ready_conns:
                try:
                    requests += [(conn, j, r) for j, r in enumerate(conn.recv())]
                except (EOFError, OSError):
                    conns.remove(conn)
            t = lap('recv', t)
            clock['rows'] += len(requests)
            clock['wakeups'] += bool(requests)
            answers = {}
            for model in {id(m): m for m in models.values()}.values():
                group = [(conn, j, r) for conn, j, r in requests if r[0] == 'act' and models[r[1]] is model]
                if not group:
                    continue
                x, state = batch([r[2] for *_, r in group], [r[3] for *_, r in group], model.config.core)
                t = lap('batch', t)
                out = model(x, state)
                top = max(r[4] for *_, r in group)
                sampled = policy.sample(model, out, x, top=top) if top else policy.sample(model, out, x)
                actions = sampled[0].cpu().tolist()  # one readback each, not one per row
                tops = sampled[2].cpu().tolist() if top else None
                value = out['value'].reshape(-1).float().cpu().tolist()
                states = unstate(out['state'])
                t = lap('forward', t)
                grid = None
                if top and args.trace:  # --trace: the sampled card's whole untempered tile distribution
                    fo, fx = policy.flat(out, x)
                    slot = (sampled[0] - policy.CARD).clamp(min=0) // policy.CELLS
                    grid = policy.tile_logp(model, fo, fx, slot).float().exp().cpu().numpy()
                for i, (conn, j, r) in enumerate(group):
                    answers[conn, j] = (actions[i], tops[i] if r[4] else None, states[i],
                                        grid[i].round(5).tolist() if grid is not None else None, value[i])
            group = [(conn, j, r) for conn, j, r in requests if r[0] == 'critic']
            if group:
                x, state = batch([r[1] for *_, r in group], [r[3] for *_, r in group], critic.trunk.config.core)
                opp = policy.tensors({k: policy.pad_stack([r[2][k] for *_, r in group])[:, None] for k in PRIVATE}, device)
                card = critic.trunk.card
                private = torch.cat([card(opp['hand_card']).flatten(2), card(opp['mirror_card']).flatten(2),
                                     opp['hand_float'].flatten(2), card(opp['cycle_card']).flatten(2),
                                     opp['ability'].flatten(2)], -1)  # as ppo.Critic.forward, with the trunk's state
                out = critic.trunk(x, state)
                h = torch.cat([out['core'].reshape(len(group), 1, -1), x['scalars'], F.relu(critic.private(private))], -1)
                logits = critic.win(h).reshape(-1).float().cpu().tolist()
                states = unstate(out['state'])
                for i, (conn, j, r) in enumerate(group):
                    answers[conn, j] = (logits[i], states[i])
            for conn in {conn for conn, *_ in requests}:
                conn.send([answers[conn, j] for j in range(sum(c is conn for c, *_ in requests))])
            t = lap('send', t)
            if t - seen > 60:
                busy = sum(v for k, v in clock.items() if k not in ('wait', 'rows', 'wakeups'))
                print('server', json.dumps({k: round(v, 1) for k, v in clock.items()}), f'busy {busy / (t - seen):.0%}',
                      f'rows/wakeup {clock["rows"] / max(clock["wakeups"], 1):.1f}', flush=True)
                clock.clear()
                seen = t


class Remote:
    """A connection to the server; a forked copy opens its own (two processes must not share one socket)."""

    def __init__(self, address):
        self.address, self.pid, self.conn = address, None, None

    def __call__(self, *parts):
        """parts: (request, apply) pairs -> [apply(answer)], all in one round trip."""
        if self.pid != os.getpid():
            self.conn, self.pid = Client(self.address, family='AF_UNIX'), os.getpid()
        self.conn.send([request for request, _ in parts])
        return [apply(answer) for (_, apply), answer in zip(parts, self.conn.recv())]


class Player:
    """One side's network through the server: act(view, top) -> policy.decode()'s choice; self.tiles: with top, the
    card's best tiles."""

    def __init__(self, remote, name, owner):
        self.remote, self.name, self.owner, self.state = remote, name, owner, None

    def ask(self, view, top=0):
        """-> (request, apply): act(), to send together with others (Remote)."""
        def apply(answer):
            self.action, self.tiles, self.state, self.grid, self.value = answer
            return policy.decode(self.action, self.owner)
        self.row = policy.pack(view)  # what was sent (search_labels.py keeps it)
        return ('act', self.name, self.row, self.state, top), apply

    def act(self, view, top=0):
        return self.remote(self.ask(view, top))[0]


class Judge:
    """The PPO critic following one seat through the game: observe(views) -> the logit of P(win) for `owner`."""

    def __init__(self, remote, owner):
        self.remote, self.owner, self.state = remote, owner, None

    def ask(self, views):
        opp = policy.pack(views[1 - self.owner])

        def apply(answer):
            logit, self.state = answer
            return logit
        return ('critic', policy.pack(views[self.owner]), {k: opp[k] for k in PRIVATE}, self.state), apply

    def observe(self, views):
        return self.remote(self.ask(views))[0]


def logit(value):
    """A value head's output (2 P(win) - 1) -> the logit of P(win), as the critic gives."""
    p = min(max((value + 1) / 2, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


# ---- the search player ---------------------------------------------------------------------------------------------------
class Searcher(Player):
    """A Player whose card plays go to the best-scoring candidate tile (same card, same moment). Its choices come from
    ask(view, top=--k) like any Player's (sent with the other seat's, one round trip); decide() then searches them."""

    def __init__(self, remote, name, owner, engine, args):
        super().__init__(remote, name, owner)
        self.engine, self.args = engine, args
        self.stats = dict(searches=0, changed=0, gain=0.0, failed=0, seconds=0.0, skipped=0, early=0, lost=0)
        self.seen = 0  # the opponent's card plays so far
        self.log = []  # every search: --trace shows them, search_labels.py saves them

    def win(self, critic):
        """P(win) now for --won: the critic's, or with --judge value this seat's own value head (from its last answer)."""
        if self.args.judge == 'value':
            return (self.value + 1) / 2
        return None if critic is None else 1 / (1 + np.exp(-critic))

    def decide(self, choice, start, game, other, other_choice, judge, critic=None):
        """choice: this seat's sampled choice now; other_choice: the other seat's, which the copies play first."""
        if choice[0] != 'card':
            return choice
        win = self.win(critic)
        if win is not None and win >= self.args.won:  # won already: placement no longer matters
            self.stats['skipped'] += 1
            return choice
        if win is not None and win <= self.args.lost:  # lost already: nor here (the user, 5 Nov)
            self.stats['lost'] += 1
            return choice
        if self.seen < self.args.after:  # the opponent's deck still unknown: rollouts would show it (the user, 5 Nov)
            self.stats['early'] += 1
            return choice
        t0 = time.perf_counter()
        sampled = (self.action - policy.CARD) % policy.CELLS
        cells = list(dict.fromkeys([sampled] + [c for c in self.tiles if c >= 0]))
        candidates = search.plays(self.owner, choice[1], cells)
        rollout = None
        if self.args.rollout == 'policy' or self.args.judge != 'formula':
            rollout = lambda play: self.rollout(play, start, game, other, other_choice, judge)  # noqa: E731
        scores = search.search(self.engine, candidates, self.owner, start, self.args.horizon, self.args.material,
                               timeout=self.args.timeout, rollout=rollout)
        self.stats['searches'] += 1
        self.stats['seconds'] += time.perf_counter() - t0
        self.log.append(dict(tick=game.state.tick, slot=choice[1], card=next((h.card for h in game.state.players[self.owner].hand
                                                                              if h.slot == choice[1]), -1),
                             grid=self.grid, cells=cells, scores=scores, action=self.action))
        if any(s is None for s in scores):
            self.stats['failed'] += 1
            return choice
        best = int(np.argmax(scores))
        if best and scores[best] > scores[0] + 1e-6:
            self.stats['changed'] += 1
            self.stats['gain'] += scores[best] - scores[0]
            p = candidates[best]
            return ('card', p.slot, p.x, p.y)
        return choice

    def rollout(self, play, start, game, other, other_choice, judge):
        """In the forked copy: play `play` now (the other seat its choice of this moment), then --horizon ticks by
        --rollout, scored by --judge (as a logit of P(win); 'value': this seat's own value head at the end)."""
        a, b, now = self.owner, 1 - self.owner, self.engine.state.tick
        self.engine.pending[:] = [p for p in self.engine.pending if p['owner'] == a or p['due'] - now <= C.OPP_VISIBLE]
        respond, value = self.args.rollout == 'policy', self.args.judge == 'value'
        choices = {a: ('card', play.slot, play.x, play.y), b: other_choice if respond else ('wait',)}
        while True:
            game.step(choices)
            if game.state.tick - now >= self.args.horizon or game.over:
                break
            choices = {a: ('wait',), b: ('wait',)}
            if not (respond or judge or value):
                continue
            views = game.views()  # one round trip: the critic's step and both players' choices
            parts = ([judge.ask(views)] if judge else []) + ([self.ask(views[a])] if respond or value else []) \
                + ([other.ask(views[b])] if respond else [])
            answers = self.remote(*parts)[1 if judge else 0:]
            if respond:
                choices = {a: answers[0], b: answers[1]}
        if judge:
            return judge.observe(game.views())
        if value:
            self.remote(self.ask(game.views()[a]))
            return logit(self.value)
        return search.score(game.state, start, a, search.princess_hp(start), self.args.material)


def play(engine, remote, vocabs, pair, seed, a_owner, args):
    engine.overlap = [True, True]
    game = Game(engine, pair, seed, {a_owner: vocabs['a'], 1 - a_owner: vocabs['b']})
    start = tower_hp(game.state)
    a = Searcher(remote, 'a', a_owner, engine, args) if args.search else Player(remote, 'a', a_owner)
    b = Player(remote, 'b', 1 - a_owner)
    judge = Judge(remote, a_owner) if args.search and args.judge == 'critic' else None
    frames, landed = [], []  # --trace
    while not game.over:
        rows = game.views()
        parts = ([judge.ask(rows)] if judge else []) + [a.ask(rows[a_owner], args.k if args.search else 0),
                                                        b.ask(rows[1 - a_owner])]
        answers = remote(*parts)  # one round trip: the critic's step and both seats' choices
        critic, (choice_a, choice_b) = (answers[0] if judge else None), answers[-2:]
        if args.trace:
            s = game.state
            win = critic if judge else logit(a.value) if args.judge == 'value' else None
            frames.append(dict(tick=s.tick, win=win, elixir=[p.elixir / 10000 for p in s.players],
                               units=[(e.owner, e.card, e.x, e.y, e.hp, e.max_hp, e.entity_type[0]) for e in s.entities]))
        if args.search:
            choice_a = a.decide(choice_a, start, game, b, choice_b, judge, critic)
        game.step({a_owner: choice_a, 1 - a_owner: choice_b})
        if args.search:
            a.seen += sum(p.owner != a_owner for p in game.state.plays)
        landed += [dict(tick=p['tick'], tap=p['tap'], owner=p['owner'], card=p['card'], x=p.get('x'), y=p.get('y'))
                   for p in engine.played] if args.trace else []
    s = game.state
    if not s.finalized:
        raise RuntimeError('native battle did not finalize')
    score = .5 if s.result not in (0, 1) else float(s.result == a_owner)
    result = dict(score=score, a_owner=a_owner, crowns_a=s.crowns[a_owner], crowns_b=s.crowns[1 - a_owner], tick=s.tick,
                  **(a.stats if args.search else {}))
    if args.trace:
        result.update(frames=frames, plays=landed, searches=a.log, decks=[[c['d'] for c in game.match['battle'][f'deck{o}']['sp']]
                                                                            for o in (0, 1)])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('a', type=Path)
    ap.add_argument('b', type=Path)
    for key in ('engine', 'engine-lib', 'engine-assets', 'pool', 'out'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--search', action='store_true', help='A searches its tiles')
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--horizon', type=int, default=160)
    ap.add_argument('--material', type=float, default=0.3)
    ap.add_argument('--rollout', choices=('passive', 'policy'), default='passive')
    ap.add_argument('--judge', choices=('formula', 'critic', 'value'), default='formula',
                    help="value: A's own value head (no hidden information, unlike the critic)")
    ap.add_argument('--critic', type=Path, help="--judge critic: a ppo.py checkpoint with 'critic' (latest.pt)")
    ap.add_argument('--won', type=float, default=0.95, help="--judge critic/value: no search while the judge's P(win) is "
                    "at least this (the user's rule, 5 Nov: labels no longer matter in a won game)")
    ap.add_argument('--lost', type=float, default=0.05, help="--judge critic/value: no search while the judge's P(win) is "
                    "at most this (the user, 5 Nov)")
    ap.add_argument('--after', type=int, default=0, help="no search before the opponent's N-th card play (the user's "
                    "choice, 5 Nov: 3; earlier, policy rollouts reveal its unseen deck)")
    ap.add_argument('--timeout', type=float, default=30, help='seconds for one search (its copies run in parallel)')
    ap.add_argument('--games', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--temperature', type=float, default=0.05, help='both models, card and tile')
    ap.add_argument('--shard', default='0/1', help='i/n: this process plays deck pairs i, i+n, ...')
    ap.add_argument('--trace', action='store_true', help="one game only (this shard's first, A in seat 0) -> --out as one "
                    "JSON: every frame's units and critic logit, every play, every search's tile distribution and scores")
    args = ap.parse_args()
    i, n = map(int, args.shard.split('/'))
    pool = json.loads(args.pool.read_text())
    pairs = np.random.default_rng(args.seed).integers(len(pool), size=args.games // 2)
    address = os.path.join(tempfile.mkdtemp(), 'server')
    ready = mp.get_context('spawn').Event()
    mp.get_context('spawn').Process(target=serve, args=(address, args, ready), daemon=True).start()
    if not ready.wait(300):
        raise RuntimeError('the inference server did not start')
    remote = Remote(address)
    import torch  # the games' inputs need each model's vocabulary (only that: the models live in the server)
    vocabs = {side: C.Vocab(torch.load(getattr(args, side), map_location='cpu', weights_only=False)['vocab'])
              for side in ('a', 'b')}
    engine = Engine(args.engine, lib=args.engine_lib, assets=args.engine_assets)
    if args.trace:
        result = play(engine, remote, vocabs, pool[pairs[i]], args.seed * 100000 + i, 0, args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(pair=int(pairs[i]), k=args.k, horizon=args.horizon, rollout=args.rollout,
                                            judge=args.judge, temperature=args.temperature, **result)))
        return
    done = set()
    if args.out.exists():
        done = {json.loads(line)['game'] for line in args.out.read_text().splitlines() if line.strip()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('a') as stream:
        for pair_index in range(i, len(pairs), n):
            for seat in (0, 1):
                g = 2 * pair_index + seat
                if g in done:
                    continue
                t0 = time.monotonic()
                result = play(engine, remote, vocabs, pool[pairs[pair_index]], args.seed * 100000 + pair_index, seat, args)
                stream.write(json.dumps(dict(game=g, pair=int(pairs[pair_index]), search=args.search, rollout=args.rollout,
                                             judge=args.judge, game_s=round(time.monotonic() - t0, 1), **result)) + '\n')
                stream.flush()


if __name__ == '__main__':
    main()
