"""Do live inputs look like training inputs? Run before deploying a model on a client (1 Nov 2026: this found a
reader's zero deploy and spawn timers, Dagger Duchess's daggers never read on either client, and the opponent's plays
missing wherever their hand is hidden).

    python tools/live_census.py CLIENT 'CAPTURES*.inputs.jsonl.gz' 'ARCHIVES*.npz'     # CLIENT: a ~/cr-clients name

CAPTURES are play.py's per-battle captures (every reader frame); they are replayed through this checkout's live path
(LiveRecords, LivePlays, PlayerView) exactly as play_battle does, so the comparison is of the current code's inputs.
ARCHIVES are converted training games. Prints, per input column (object columns per object kind and per unit type), the
share of nonzero values and their mean, training vs live, with '<--' on large gaps. A gap is either the decks (only what
was played live shows up: compare unit types) or a reader / adapter bug. Also lists ids the live tables lack."""
import collections
import glob
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import play as P  # noqa: E402
from cr import observation as C  # noqa: E402

KINDS = {1: 'unit', 2: 'tower', 3: 'area', 4: 'projectile', 5: 'timed'}
FLOATS = {'hand_float': C.HAND_FLOAT_NAMES, 'scalars': C.SCALAR_NAMES, 'ability': C.ABILITY_FLOAT_NAMES,
          'own_flight_float': ('x', 'y', 'eta', 'cost'), 'own_flight2_float': ('x', 'y', 'eta', 'cost'), 'opp_flight_float': ('x', 'y', 'eta', 'cost'),
          'opp_seen_float': ('back_in', 'recency', 'cost')}
IDS = ('hand_card', 'cycle_card', 'ability_card', 'opp_hero_card', 'own_flight_card', 'own_flight2_card', 'opp_flight_card', 'opp_seen_card',
       'obj_card', 'obj_data', 'eff_id')


def live_inputs(client, captures, vocab):
    """Each capture -> {input: array} in the archives' layout (objects concatenated), as play_battle builds them."""
    records = P.LiveRecords(client)
    out = collections.defaultdict(list)
    for cap in captures:
        start = [json.loads(l) for l in open(cap.replace('.inputs.jsonl.gz', '.jsonl')) if '"start"' in l]
        try:
            frames = [r['frame'] for r in map(json.loads, gzip.open(cap, 'rt')) if r['event'] == 'reader_frame']
        except (EOFError, OSError, ValueError):
            print('skipped (unreadable):', cap)
            continue
        frames = [f for f in frames if f.get('battle_active') and f['failure'] == 'none' and f.get('coherent', True)]
        if not start or len(frames) < 200:
            continue
        owner = start[0]['owner']
        records.reset()
        view = C.PlayerView(owner, vocab, C.cached_legal(), next(p['deck_form_flags'] for p in frames[0]['players']
                                                                 if p['side'] == owner))
        plays, decided = P.LivePlays(owner), None
        for fr in frames:
            if not (fr['game_tick'] < P.FIRST_DECISION or fr.get('queue_ok')):
                continue
            plays.update(fr)
            t = fr['game_tick']
            if t >= P.FIRST_DECISION and (decided is None or t >= decided + 5):
                decided = t
                x = view.step(records(fr), plays.commands(tick=t))
                m = int((x['obj_side'] > 0).sum())
                for k, v in x.items():
                    out[k].append(v[:m] if k.startswith(('obj_', 'eff_')) else v[None])
    print('live tables lack these data ids (rows):', records.missing.most_common(10) or 'none')
    return {k: np.concatenate(v) for k, v in out.items()}


def archives(files):
    out = collections.defaultdict(list)
    for f in files:
        z = np.load(f, allow_pickle=False)
        for p in (0, 1):
            for k in z.files:
                if k.startswith(f'p{p}_x_'):
                    out[k[len(f'p{p}_x_'):]].append(z[k])
    return {k: np.concatenate(v) for k, v in out.items()}


def share_mean(a):
    nz = np.abs(a.astype(np.float64)) > 1e-9
    n = nz.sum(0)
    return nz.mean(0), np.where(n > 0, (a * nz).sum(0) / np.maximum(n, 1), 0)


def flag(st, sl, mt, ml, floor=.005):
    if (st > floor and sl < st / 4) or (sl > floor and st < sl / 4):
        return '<-- share'
    if min(st, sl) > floor and abs(mt - ml) > .5 * max(abs(mt), abs(ml), .05):
        return '<-- value'
    return ''


def compare(train, live, names):
    inv = {v: k for k, v in names.index.items()}
    print(f"decisions: training {len(train['scalars'])}, live {len(live['scalars'])}")
    print('columns: share nonzero training / live, mean of the nonzero values training / live')
    for kind, kname in KINDS.items():
        t, l = train['obj_kind'] == kind, live['obj_kind'] == kind
        print(f'== objects: {kname} ({t.sum()} / {l.sum()} rows)')
        if t.sum() and l.sum():
            (st, mt), (sl, ml) = share_mean(train['obj_float'][t]), share_mean(live['obj_float'][l])
            for i, n in enumerate(C.OBJ_FLOAT_NAMES):
                if max(st[i], sl[i]) > 0:
                    print('   %-20s %7.2f%% %7.2f%%  %8.3f %8.3f  %s' % (n, 100 * st[i], 100 * sl[i], mt[i], ml[i],
                                                                     flag(st[i], sl[i], mt[i], ml[i])))
    for k, cols in FLOATS.items():
        if k not in train or k not in live:          # an input the archives predate (own_flight2_*, 1 Nov)
            continue
        print('==', k)
        (st, mt), (sl, ml) = share_mean(train[k].reshape(-1, len(cols))), share_mean(live[k].reshape(-1, len(cols)))
        for i, n in enumerate(cols):
            print('   %-20s %7.2f%% %7.2f%%  %8.3f %8.3f  %s' % (n, 100 * st[i], 100 * sl[i], mt[i], ml[i],
                                                             flag(st[i], sl[i], mt[i], ml[i])))
    print('== ids: padding / unknown / known (%), training | live')
    for k in IDS:
        if k not in train or k not in live:
            continue
        f = lambda x: [100 * (x == 0).mean(), 100 * (x == 1).mean(), 100 * (x > 1).mean()]  # noqa: E731
        print('   %-16s %5.1f %5.1f %5.1f | %5.1f %5.1f %5.1f' % (k, *f(train[k].ravel()), *f(live[k].ravel())))
    print('== per unit type (>= 30 rows on both sides): object columns with a gap, and kinds that disagree')
    for i in np.unique(live['obj_data']):
        a, b = train['obj_data'] == i, live['obj_data'] == i
        if a.sum() < 30 or b.sum() < 30:
            continue
        ka, kb = np.bincount(train['obj_kind'][a]).argmax(), np.bincount(live['obj_kind'][b]).argmax()
        if ka != kb:
            print(f'   data {inv.get(i, i)}: kind {KINDS.get(ka)} in training, {KINDS.get(kb)} live')
        (st, mt), (sl, ml) = share_mean(train['obj_float'][a]), share_mean(live['obj_float'][b])
        for c, n in enumerate(C.OBJ_FLOAT_NAMES):
            if n not in ('x', 'y', 'dx', 'dy', 'own', 'hp_frac', 'target_dx', 'target_dy', 'aim_dx', 'aim_dy') and \
                    flag(st[c], sl[c], mt[c], ml[c], .02):
                print('   data %-12s %-18s %6.1f%% %7.3f | %6.1f%% %7.3f' % (inv.get(i, i), n, 100 * st[c], mt[c],
                                                                          100 * sl[c], ml[c]))


if __name__ == '__main__':
    client, captures, train = sys.argv[1:4]
    vocab = C.Vocab.load()
    live = live_inputs(P.Client.load(client), sorted(glob.glob(os.path.expanduser(captures))), vocab)
    compare(archives(sorted(glob.glob(os.path.expanduser(train), recursive=True))), live, vocab)
    print('live ids outside the vocabulary (lookups):', vocab.misses.most_common(10) or 'none')
