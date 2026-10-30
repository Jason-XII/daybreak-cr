"""How often a model presses a ready hero / champion ability (23 Oct): arena.py games, both seats under "overlap" (the
general run's rule), each decision counted per seat: an ability ready (any controller), pressed (the sampled choice),
accepted by the engine. One JSON line per game to PRESSES (A and B, so B's line is a baseline from the same games).
Humans press at 0.96% of ready decisions (IL labels); live v240 0.49%.

    python experiments/press_rate.py PRESSES.jsonl A.pt B.pt --engine ... --pool ... --games 26 --out GAMES.jsonl \\
        --a-temperature 0.05 --b-temperature 0.05 --device cuda
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arena  # noqa: E402
from cr import engine, policy  # noqa: E402

out = open(sys.argv.pop(1), 'a')
evaluate, Game = arena.evaluate, arena.Game


class Counted(Game):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine.overlap = [True, True]  # the run's rule for every seat
        self.counts = {o: dict(ready=0, pressed=0, accepted=0) for o in (0, 1)}

    def step(self, choices):
        ready = {o: any(a.ready for a in self.state.players[o].abilities) for o in (0, 1)}
        submitted, transition = super().step(choices)
        for o, c in choices.items():
            self.counts[o]['ready'] += ready[o]
            self.counts[o]['pressed'] += ready[o] and c[0] == 'ability'
        for play, ok in zip(submitted, transition.executed):
            self.counts[play.owner]['accepted'] += ok and play.slot == engine.ABILITY
        arena.last = self
        return submitted, transition


def counted(eng, models, pair, seed, a_owner, *args, **kwargs):
    result = evaluate(eng, models, pair, seed, a_owner, *args, **kwargs)
    c = arena.last.counts
    out.write(json.dumps(dict(seed=seed, a=c[a_owner], b=c[1 - a_owner])) + '\n')
    out.flush()
    return result


def load(path, device):
    model = policy.load(path, device)
    model[0].overlap = True
    return model


if __name__ == '__main__':
    arena.Game, arena.evaluate, arena.load = Counted, counted, load
    arena.main()
