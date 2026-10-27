"""Placement search: candidate tiles for one card play, each tried in a forked copy of the native game, played `horizon`
ticks with no further plays, scored by tower damage (crowns + tower hp fractions, theirs minus ours) + `material` x
(our units' hp - theirs, in princess towers). Opponent plays the player can't see yet are taken out of the copies; the
opponent's hand never enters (nobody plays in a copy). Beat the plain policy 65.6% (gen-h v40, 26 Oct) and halved its
losses to the IL model; PPO's collectors label tiles with it (ppo.py "search_frac"). Linux (fork).
"""
import os
import pickle
import select
import signal
import time

from cr import observation as C
from cr.engine import PLAY_DELAY, Play, potential

DELAY = round(sum(PLAY_DELAY) / len(PLAY_DELAY))  # copies land plays at the mean delay (~22 ticks)


def score(state, start, owner, princess, material):
    mine = sum(e.hp for e in state.entities if e.card != -1 and e.hp and e.owner == owner)
    theirs = sum(e.hp for e in state.entities if e.card != -1 and e.hp and e.owner != owner)
    return potential(state, start, owner) + material * (mine - theirs) / princess


def princess_hp(start):
    """A princess tower's full hp, from tower_hp at the start (the material unit)."""
    return max(hp for _, hp, king in start.values() if not king)


def plays(owner, slot, cells):
    """Tile indices (in `owner`'s view) -> Plays of hand slot `slot`."""
    out = []
    for cell in cells:
        row, col = divmod(int(cell), C.W)
        out.append(Play(owner, slot, *C.rotate(col * 1000 + 500, row * 1000 + 500, owner)))
    return out


def _rollout(engine, play, owner, start, princess, horizon, material):
    now = engine.state.tick
    engine.pending[:] = [p for p in engine.pending if p['owner'] == owner or p['due'] - now <= C.OPP_VISIBLE]
    state = engine.step([play], 5, delay=DELAY).state
    while state.tick - now < horizon and not state.finalized:
        state = engine.step((), 5).state
    return score(state, start, owner, princess, material)


def search(engine, candidates, owner, start, horizon=160, material=0.3, timeout=30):
    """Each candidate Play in its own forked child, all at once -> [score or None (failed / timed out)]."""
    princess = princess_hp(start)
    kids = []
    for play in candidates:
        rd, wr = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(rd)
            try:
                res = _rollout(engine, play, owner, start, princess, horizon, material)
            except BaseException as e:  # noqa: BLE001
                res = repr(e)
            os.write(wr, pickle.dumps(res))
            os._exit(0)
        os.close(wr)
        kids.append((pid, rd))
    out, deadline = [], time.time() + timeout
    for pid, rd in kids:
        buf = b''
        while True:
            ready, _, _ = select.select([rd], [], [], max(0, deadline - time.time()))
            if not ready:
                os.kill(pid, signal.SIGKILL)
                break
            chunk = os.read(rd, 1 << 16)
            if not chunk:
                break
            buf += chunk
        os.close(rd)
        os.waitpid(pid, 0)
        res = pickle.loads(buf) if buf else None
        out.append(res if isinstance(res, float) else None)
    return out
