"""Groundwork check 5: every field the reader gives is a model input, or is left out with a written reason.

Arron, 2026-11-03: "Model sees everything, no missing fields." The reader frame is the one record live and conversion share
(checks 3 and 4 prove it). This file says, for every field of it, what the model gets. A field the reader gains later and this
spec does not name fails the check, so a new field cannot be silently dropped on its way to the model.

Rule: everything a player could use is an input. Left out only: what is not game state (addresses, the reader's own timing and
diagnostics) and what no player can know (the battle's random seed). Hidden opponent cards are inputs too: the live client has
them empty until they are revealed, and the model gets exactly what the client has.

    python3 crx/model_inputs.py RECORDING.jsonl.gz [...]      (exit 0 only when every field is covered)
"""
import collections
import gzip
import json
import sys

IN = 'input'
OUT = 'out'

# field (as check 3 / check 4 name it) -> (IN, group) or (OUT, reason)
SPEC = {
    # the battle
    'game_tick': (IN, 'battle: the tick (time left follows from it and the mode)'),
    'game_mode': (OUT, 'the mode id: conversions run every replay under the match factory\'s default (Ladder 72000006), a live client reads the real one (Ranked1v1 72000464): the same battle would read differently in training and live (round 4, AJ). The rules it sets are inputs: clock (elixir rate, time limit, overtime) and the levels'),
    'arena': (OUT, 'the arena id: a conversion default (54000001) in training, the real one live (round 4, AJ); cosmetic'),
    'clock': (IN, 'battle: the game\'s clock: tick, elapsed ms, overtime, time limit, elixir phase / rate (the 2x / 3x changes)'),
    'logic_raw': (IN, 'battle: the logic\'s own changing words (result, finished flag, 6 raw until named)'),
    'location': (OUT, 'the layout id: a conversion default (15000199) in training, the real one live (round 4, AJ); the playable tiles are the same and are derived from the towers'),
    'battle_active': (OUT, 'whether the reader sees a battle: frames outside one are not decisions'),
    'rng': (OUT, 'the battle\'s random seed: no player can know it; with it a model could foresee random events'),
    'seed': (OUT, 'a second seed word, 0 in this build'),
    'server_tick': (OUT, 'when the newest server turn arrived: network timing, not game state'),
    'queue': (IN, 'commands about to land: each from 6 ticks before its landing until it lands, both players -- the same in a conversion (whose engine holds the whole battle from tick 0) and live (a client has a command 6-19 ticks before it lands); the reader applies it (round 5, AQ)'),
    'queue[].t': (IN, 'commands: issue tick'),
    'queue[].t2': (IN, 'commands: landing tick (issue + 20 where a live client still shows -1)'),
    'queue[].card': (IN, 'commands: the card played'),
    'queue[].a': (IN, 'commands: x (a play) or the unit (an ability)'),
    'queue[].b': (IN, 'commands: y'),
    'queue[].id': (OUT, 'the issuing account\'s ids: who, not what -- pro players in a conversion, the user live (round 7); queue[].side carries it'),
    'queue[].side': (IN, 'commands: which side issued it (matched through the players\' accounts by the reader)'),
    'queue[].vt': (OUT, 'a code address (the command class\'s vtable): differs by build and platform -- 0x1af1508 in the x86 engine, 0x1a28748 / 0x1a27348 on the phone for one card play (round 3, AI); queue[].kind carries it'),
    'queue[].kind': (IN, 'commands: 1 a card play, 2 an ability press, 0 another family -- the same on every build and platform'),
    'queue[].sid': (OUT, 'a command field: -1 in every card play seen (300 pro replays, live battles); on 160635009 a live ability press carries a rising number (11, 26, 36, 47 in one battle: a command sequence count, not game state; round 3 #36)'),
    'queue[].pd': (IN, 'commands: the card\'s selection value (which form / option it is played as)'),
    # each player
    'players[].side': (IN, 'player: which side'),
    'players[].elixir_raw': (IN, 'player: elixir'),
    'players[].hand_deck_indices': (IN, 'player: hand (deck slots -> card, level, form)'),
    'players[].next_deck_index': (IN, 'player: next card'),
    'players[].cycle_deck_indices': (IN, 'player: the cards after it, in order'),
    'players[].deck_card_ids': (IN, 'player: deck (the opponent\'s empty until revealed, as on the client)'),
    'players[].deck_form_flags': (IN, 'player: which slots are evolution / hero slots'),
    'players[].deck_level_index': (IN, 'player: card levels'),
    'players[].battle_deck': (IN, 'player: the battle\'s deck (card, level, form per slot)'),
    'players[].evo_progress': (IN, 'player: evolution progress per slot'),
    'players[].abilities[].controller_slot': (IN, 'ability: which button'),
    'players[].abilities[].charges': (IN, 'ability: charges left'),
    'players[].abilities[].button': (IN, 'ability: button state (ready, limited window, ...)'),
    'players[].abilities[].cooldown_ms': (IN, 'ability: cooldown left'),
    'players[].abilities[].configured_ms': (IN, 'ability: cooldown in all'),
    'players[].abilities[].character_id': (IN, 'ability: which hero / champion it belongs to'),
    'players[].abilities[].used': (IN, 'ability: used'),
    'players[].abilities[].effect_delay_ms': (IN, 'ability: time until its effect starts'),
    'players[].abilities[].units': (IN, 'ability: the units that carry it'),
    'players[].account': (OUT, 'account ids: who, not what; a side is the input'),
    'players[].deal_id': (OUT, 'the id word the deal seeds with: identity, like the account'),
    'players[].address': (OUT, 'an address'),
    'players[].cards_played': (IN, 'player: cards played so far'),
    'players[].elixir_spent': (IN, 'player: elixir spent so far (x10000)'),
    'players[].elixir_wasted': (IN, 'player: elixir wasted at full so far (x10000)'),
    'players[].first_cards': (IN, 'player: the first troop / spell / building they played (-1 before)'),
    'players[].card_counts': (IN, 'player: the cards played IN THIS BATTLE so far, in first-play order, with how many times each ([card id, count]; both sides -- a live client holds the account\'s career counts in the same record, which the reader subtracts at the battle\'s first frame)'),
    'players[].card_counts_since': (OUT, 'the reader\'s own bookkeeping: the tick it took the card_counts baseline (before tick 21 = complete; -1 = it first saw the battle later and outputs the record as it is, which on a live client includes the career counts)'),
    'players[].player_raw': (IN, 'player: +0x120 / +0x1FC, a level-like value (15 -> 16) and a flag set with it (raw until named)'),
    'players[].slot_words': (IN, 'player: one word per deck slot, 0 -> 1 / 2 as that card is played (raw until named)'),
    'players[].refill_timer': (OUT, 'ms until the next card is dealt after a play: in the engine 0 except 20 ticks after a play, but a live client does not know when the OPPONENT\'s next card is dealt and runs a free 1000 ms countdown for it (nonzero in ~97 % of frames; round 4, AL, side corrected on 8 live battles); derived instead from the commands (a card is dealt 1000 ms after its play lands)'),
    # each object on the board
    'entities[].category': (IN, 'object: its id (the key other fields refer to)'),
    'entities[].side': (IN, 'object: side'),
    'entities[].kind': (IN, 'object: kind (unit, building, King asleep / awake, effect)'),
    'entities[].card_id': (IN, 'object: card'),
    'entities[].data_id': (IN, 'object: what it is (unit / projectile / area data, with the static attributes by id)'),
    'entities[].x': (IN, 'object: position'),
    'entities[].y': (IN, 'object: position'),
    'entities[].tx': (IN, 'object: where it is heading'),
    'entities[].ty': (IN, 'object: where it is heading'),
    'entities[].hp': (IN, 'object: hit points'),
    'entities[].max_hp': (IN, 'object: maximum hit points'),
    'entities[].level': (IN, 'object: level'),
    'entities[].behavior_state_raw': (IN, 'object: state (moving, attacking, deploying, underground, ...)'),
    'entities[].deploy_remaining': (IN, 'object: deploy time left'),
    'entities[].has_attack': (IN, 'object: can attack'),
    'entities[].has_move': (IN, 'object: can move'),
    'entities[].target': (IN, 'object: what it is attacking (as an object id)'),
    'entities[].atk_stage': (IN, 'attack: stage (Inferno ramp, sequences)'),
    'entities[].atk_timeline': (IN, 'attack: time into the swing'),
    'entities[].atk_load': (IN, 'attack: load time left'),
    'entities[].charge': (IN, 'movement: charge progress (Prince, Ram Rider, ...)'),
    'entities[].address': (OUT, 'an address (objects are referred to by id)'),
    'entities[].us.hidden': (IN, 'unit state: hidden'),
    'entities[].us.immune': (IN, 'unit state: immune'),
    'entities[].us.airborne': (IN, 'unit state: in the air'),
    'entities[].us.invisible': (IN, 'unit state: invisible'),
    'entities[].us.invisible_again_ms': (IN, 'unit state: time to invisible again'),
    'entities[].us.shield': (IN, 'unit state: shield / its maximum'),
    'entities[].us.pushed': (IN, 'unit state: being pushed'),
    'entities[].us.push_ms': (IN, 'unit state: the push SPEED (movement +0x1EC; the name is historical): world units the push moves it each tick, falling by 25 a tick (225, 200 ... 0, -25), signed -- a friend\'s code + run check on 160635007 (2026-11-05), matched on our recordings (300, 275, 250 ...)'),
    'entities[].us.hit_count': (IN, 'attack: which hit of a sequence'),
    'entities[].us.dash_ms': (IN, 'attack: dash cooldown'),
    'entities[].us.hop': (IN, 'attack: Golden Knight\'s hop'),
    'entities[].us.shot': (IN, 'attack: a returning shot out (Executioner\'s axe, Fisherman\'s hook)'),
    'entities[].us.special_ms': (IN, 'attack: special timer (Fisherman)'),
    'entities[].us.spawn_ms': (IN, 'unit state: next spawn time (huts, Witch, Tombstone)'),
    'entities[].us.jump_ms': (IN, 'movement: jump time left'),
    'entities[].us.surface': (IN, 'movement: where an underground unit comes up'),
    'entities[].us.z': (IN, 'object: height'),
    'entities[].us.daggers': (IN, 'tower: Dagger Duchess\'s daggers'),
    'entities[].us.a10': (IN, 'unit: evolved Musketeer\'s snipe shots left'),
    'entities[].us.vars': (IN, 'unit: the card\'s own counters (54 variables: souls, rage, attack counts, ...)'),
    'entities[].us.acts': (IN, 'unit: the card actions running now (by action id: a net launcher, a target marker, a rage aura, ...)'),
    'entities[].us.waypoint': (IN, 'unit: the next path node it walks to'),
    'entities[].us.facing': (IN, 'unit: which way it faces (a vector, length ~256)'),
    'entities[].us.area_raw': (IN, 'spell area: its other words: the tick it was made, a first-tick flag, 7 raw until named'),
    'entities[].us.area_units': (IN, 'spell area: the unit ids it has caught / hit'),
    'entities[].us.prj_d8': (IN, 'projectile: its +0xD8 block\'s word +0x14, raw until named'),
    'entities[].us.prj_units': (IN, 'projectile: the unit ids it has hit'),
    'entities[].us.prj_sub': (IN, 'projectile: its sub-object (4 words, two unit-id lists), raw until named'),
    'entities[].us.braw': (IN, 'unit: per active effect (same order as buffs), its other words, raw until named'),
    'entities[].us.buffs2': (IN, 'unit: the effect component\'s per-tick list (empty between ticks)'),
    'entities[].us.bcomp': (IN, 'unit: effect component words +0x38 / +0x3C, raw until named'),
    'entities[].us.alists': (IN, 'card actions: their int32 lists (Rune Giant\'s friends, Laser Ball / attack-chain targets, captured units, timers): [action id, [[...] per list]]'),
    'entities[].us.ahead': (IN, 'card actions: each running action\'s header words +0x8 / +0x10 (a started flag in most kinds), raw'),
    'entities[].us.mov_lists': (IN, 'unit: two parallel movement lists (offsets / amounts), raw until named'),
    'entities[].us.atk_lists': (IN, 'unit: the attack component\'s three unit-id lists'),
    'entities[].us.links': (IN, 'unit: its rows in the game\'s unit link table: [other unit id, kind, flags, timer ms] (the evolved Goblin Cage pulling it, Boss Bandit / hero Mega Minion ability running, Rune Giant\'s friends)'),
    'entities[].us.hlt': (IN, 'unit: health component words: hp at spawn (2), a decaying building\'s decay rate (hp/100 a tick) and hp fraction, 2 raw until named'),
    'entities[].us.hit_by': (IN, 'unit: who has hit it: [object id, count] records'),
    'entities[].us.hl': (IN, 'unit: two small health-component lists, raw until named'),
    'entities[].us.unlinking': (IN, 'unit: its link rows queued for removal this tick: [other unit id, kind]'),
    'entities[].us.ctx': (IN, 'unit: its card actions\' context values by key (the hero Mega Minion\'s return point, the Ice Wizard hero\'s snowman, ...)'),
    'entities[].us.atk': (IN, 'unit: the attack component\'s other changing words, raw until named'),
    'entities[].us.mov': (IN, 'unit: the movement component\'s other changing words, raw until named'),
    'entities[].us.prj': (IN, 'projectile: its other changing words, raw until named (target points, timers, speed, ...)'),
    'entities[].us.ent': (IN, 'unit / princess tower: the troop object\'s other changing words, raw until named (37: timers, counters, age, ...)'),
    'entities[].us.king': (IN, 'King tower: its own changing words, raw until named (9: facing, attack timers, hit counters, ...)'),
    'entities[].us.astate': (IN, 'unit: each running card action\'s state (cooldowns, windows, delays as ms / ticks; counts; targets as object ids)'),
    'entities[].us.sched': (IN, 'unit: the card actions about to start, with ticks until each starts (delayed spawns, waits)'),
    'entities[].us.tags': (IN, 'unit: game tags (62: untargetable, no attack, ability pending, deflecting, ...)'),
    'entities[].us.buffs': (IN, 'effects on it: kind, time left / elapsed, side, speed and hit-speed change, who applied it'),
    'entities[].us.source': (IN, 'projectile: who shot it'),
    'entities[].us.dx': (IN, 'projectile: movement'),
    'entities[].us.dy': (IN, 'projectile: movement'),
    'entities[].us.aim': (IN, 'projectile: where it is aimed'),
    'entities[].us.aim_object': (IN, 'projectile: what it is homing on'),
    'entities[].us.start': (IN, 'projectile: where it was launched'),
    'entities[].us.area_ms': (IN, 'area: time left'),
    'entities[].us.area_life': (IN, 'area: duration in all'),
    # the reader's own bookkeeping
    'event': (OUT, 'frame type'), 'schema_version': (OUT, 'frame format'), 'pid': (OUT, 'process id'),
    'reader_pid': (OUT, 'process id'), 'sequence': (OUT, 'frame counter'), 'coherent': (OUT, 'read consistency flag'),
    'failure': (OUT, 'reader diagnostics'), 'applied_replay_tick': (OUT, 'reader diagnostics'),
    'native_object_count': (OUT, 'reader diagnostics (the objects themselves are inputs)'),
    'decoded_entity_count': (OUT, 'reader diagnostics'), 'filtered_object_count': (OUT, 'reader diagnostics'),
    'discovery_nodes': (OUT, 'reader diagnostics'), 'discovery_budget_exhausted': (OUT, 'reader diagnostics'),
    'sample_monotonic_us': (OUT, 'reader timing'), 'read_us': (OUT, 'reader timing'),
}
CHAIN = (OUT, 'addresses the reader followed')

# What the model also gets that is not in the frame: the static attributes of every id (crx/data/static_attributes-<build>.json,
# check 1), the names of tags and variables (crx/data/game_tags-, variables-<build>.json), and the time since each ability's
# last use (kept by the console from frames; check 3 holds it).


def keys(o, pre=''):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from keys(v, pre + '.' + k if pre else k)
    elif isinstance(o, list) and o and isinstance(o[0], dict):
        for v in o:
            yield from keys(v, pre + '[]')
    else:
        yield pre


# the reader's pointer chain, by name (fresh break test N: any key starting with 'chain' was out without review)
CHAIN_KEYS = {'chain.battle', 'chain.collection', 'chain.context', 'chain.data', 'chain.hp_state', 'chain.libg_base',
              'chain.player_state', 'chain.player_state_path', 'chain.registry', 'chain.root'}
# out reasons that state a value are held against the recordings (fresh break test O)
VALUE_CLAIMS = {'queue[].sid': -1, 'seed': 0}


def rule(key):
    if key in CHAIN_KEYS:
        return CHAIN
    return SPEC.get(key)


def claim_violations(frame):
    bad = []
    if 'seed' in frame and frame['seed'] != VALUE_CLAIMS['seed']:
        bad.append(('seed', frame['seed']))
    for q in frame.get('queue') or ():
        play = q.get('kind') == 1 if 'kind' in q else q.get('vt') in ('0x1a28748', '0x1af1508', '0x1a27348')
        if play and 'sid' in q and q['sid'] != VALUE_CLAIMS['queue[].sid']:   # the claim is made for card plays (round 3 #36)
            bad.append(('queue[].sid', q['sid']))
    return bad


# ---- the reference input builder: a frame -> the model's input, every IN field wired -------------------------------------
# Plain numeric rows (the model's encoding of them -- embeddings, scaling -- is the model's own); what matters here is that every
# input field reaches them. check_wiring() changes each input field of a frame and requires the built input to change.

def _num(v):
    if isinstance(v, bool):
        return [int(v)]
    if isinstance(v, (int, float)):
        return [v]
    if isinstance(v, str):
        try:
            return [int(v, 16)]
        except ValueError:
            return [hash(v) & 0xFFFF]
    if isinstance(v, (list, tuple)):
        out = [len(v)]
        for x in v:
            out += _num(x)
        return out
    if isinstance(v, dict):
        out = [len(v)]
        for k in sorted(v):                 # the key is information too (which variable / context): round 3 prep, 2026-11-04
            out += _num(k) + _num(v[k])
        return out
    return [0 if v is None else -1]


def build(frame):
    """{'battle': [...], 'players': [[...], [...]], 'objects': [[...] per object], 'commands': [[...] per command]}"""
    def pick(obj, prefix):
        row = []
        for k in sorted(obj):
            name = f'{prefix}.{k}'
            if isinstance(obj[k], dict) and k != 'us':
                row += pick(obj[k], name)
            elif k == 'us':
                row += pick(obj[k], f'{prefix}.us')
            elif isinstance(obj[k], list) and obj[k] and isinstance(obj[k][0], dict):
                for x in obj[k]:
                    row += pick(x, f'{name}[]')
            elif (SPEC.get(name) or ('',))[0] == IN:
                row += _num(obj[k])
        return row
    addr = {e['address']: e['category'] for e in frame.get('entities') or ()}
    objects = []
    for e in frame.get('entities') or ():
        e = dict(e, target=addr.get(e.get('target'), -1))      # a target as the object id it points to
        objects.append(pick(e, 'entities[]'))
    # every battle-level input (fresh break test L: clock and logic_raw never reached the built input)
    return {'battle': [v for k in sorted(frame) if k not in ('players', 'entities', 'queue') and (SPEC.get(k) or ('',))[0] == IN
                       for v in _num(frame.get(k))],
            'players': [pick(p, 'players[]') for p in frame.get('players') or ()],
            'objects': objects,
            'commands': [pick(q, 'queue[]') for q in frame.get('queue') or ()]}


def _paths(o, pre=''):
    """(spec key, path to the value) for every leaf of a frame"""
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _paths(v, (pre + '.' + k if pre else k)) if not isinstance(v, (dict, list)) or isinstance(v, dict) or \
                (isinstance(v, list) and v and isinstance(v[0], dict)) else [((pre + '.' + k) if pre else k, (o, k))]
            if isinstance(v, (int, float, str, bool)) or v is None:
                yield ((pre + '.' + k) if pre else k, (o, k))
    elif isinstance(o, list) and o and isinstance(o[0], dict):
        for v in o:
            yield from _paths(v, pre + '[]')


def check_wiring(frame):
    """every input field present in this frame changes the built input when it changes -> [fields that do not]"""
    import copy
    base = build(frame)
    dead, tried, empty = [], set(), {}
    for key, (holder, k) in list(_paths(frame)):
        spec = key.replace('entities[].us.', 'entities[].us.')
        if (SPEC.get(spec) or ('',))[0] != IN or spec in tried or spec.startswith('players[].abilities') and False:
            continue
        if spec == 'queue':                  # the list itself: its commands' fields are each changed below
            continue
        old = holder[k]
        if isinstance(old, (list, dict)) and not old:   # an empty one proves only its length: look for a filled one first
            empty.setdefault(spec, (holder, k))
            continue
        tried.add(spec)
        if spec == 'entities[].target':      # wired through the object id it points to: point it at another object
            others = [e['address'] for e in frame.get('entities') or () if e['address'] != old]
            if not others:
                continue
            holder[k] = others[0]
            if build(frame) == base:
                dead.append(spec)
            holder[k] = old
            continue
        # (round 2, Z) a list / dict is changed INSIDE (its last leaf, and a dict key renamed), not only made longer: appending
        # proved only that its length reaches the input
        trials = [_bumped(old)] if not isinstance(old, (list, dict)) else \
            [x for x in (_bump_leaf(old), _rename_key(old), _swap(old), _shift(old)) if x is not None]   # (round 4, AN) order, sum
        if isinstance(old, (list, dict)):
            value_proven.add(spec)
        for new in trials:
            holder[k] = new
            if build(frame) == base:
                dead.append(spec)
                break
        holder[k] = old
    # (round 6, BB) a field's values exchanged between two objects / players / commands must change the input: a builder that
    # sorted each field across objects kept every value but lost which troop or side owned it
    for group, pre in (('entities', 'entities[].'), ('players', 'players[].'), ('queue', 'queue[].')):
        items = frame.get(group) or []
        if len(items) < 2:
            continue
        for k in sorted({k for it in items for k in it}):
            spec = pre + k
            if (SPEC.get(spec) or ('',))[0] != IN:
                continue
            pair = next(((i, j) for i in range(len(items)) for j in range(i + 1, len(items))
                         if k in items[i] and k in items[j] and items[i][k] != items[j][k]), None)
            if pair is None:
                continue
            i, j = pair
            items[i][k], items[j][k] = items[j][k], items[i][k]
            if build(frame) == base:
                dead.append(spec + ' (exchanged between two)')
            items[i][k], items[j][k] = items[j][k], items[i][k]
    for spec, (holder, k) in empty.items():   # empty everywhere in this frame: the length test
        if spec in tried:
            continue
        tried.add(spec); length_tested.add(spec)
        old = holder[k]
        holder[k] = [1] if isinstance(old, list) else {'x': 1}
        if build(frame) == base:
            dead.append(spec)
        holder[k] = old
    return sorted(tried), dead


value_proven, length_tested = set(), set()   # list / dict inputs changed inside / only made longer (round 2, Z)


def _bumped(v):
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v + 7
    if isinstance(v, str):
        return v + '1'
    return 1 if v is None else None


ROTATE = [0]   # (round 3, AF) which element a list bump changes: rotates frame by frame, so every position is tried


def _bump_leaf(v):
    """a copy of a list / dict with one leaf changed (None if it has no leaf): at each list level the element at the rotating
    position (ROTATE), then the others -- not only the last (a builder keeping a list's length and last element passed)"""
    if isinstance(v, list):
        n = len(v)
        for j in range(n):
            i = (ROTATE[0] + j) % n
            b = _bump_leaf(v[i]) if isinstance(v[i], (list, dict)) else _bumped(v[i])
            if b is not None:
                return v[:i] + [b] + v[i + 1:]
        return None
    if isinstance(v, dict):
        for k in sorted(v, reverse=True):
            b = _bump_leaf(v[k]) if isinstance(v[k], (list, dict)) else _bumped(v[k])
            if b is not None:
                return {**v, k: b}
        return None
    return _bumped(v)


def _numbers(v, path=()):
    """(path, value) of every number leaf of a list / dict"""
    if isinstance(v, list):
        for i, x in enumerate(v):
            yield from _numbers(x, path + (i,))
    elif isinstance(v, dict):
        for k in sorted(v):
            yield from _numbers(v[k], path + (k,))
    elif isinstance(v, (int, float)) and not isinstance(v, bool):
        yield path, v


def _set(v, path, x):
    import copy
    v = copy.deepcopy(v)
    o = v
    for p in path[:-1]:
        o = o[p]
    o[path[-1]] = x
    return v


def _swap(v):
    """two different number leaves exchanged (a builder that sorts, or keeps only a sum / set, passes a bump of one); the pair
    starts at the rotating position (round 5, AT: always the first two let a builder keep them and sort the rest)"""
    leaves = list(_numbers(v))
    n = len(leaves)
    for a in range(n):
        i = (ROTATE[0] + a) % n
        for j in range(i + 1, n):
            if leaves[i][1] != leaves[j][1]:
                return _set(_set(v, leaves[i][0], leaves[j][1]), leaves[j][0], leaves[i][1])
    return None


def _shift(v):
    """+7 on one number leaf, -7 on another: the sum stays (round 4, AN: a builder reducing a list to [length, sum] passed)"""
    leaves = list(_numbers(v))
    if len(leaves) < 2:
        return None
    i = ROTATE[0] % len(leaves); j = (i + 1 + (ROTATE[0] // len(leaves)) % (len(leaves) - 1)) % len(leaves)
    if i == j:
        j = (i + 1) % len(leaves)
    return _set(_set(v, leaves[i][0], leaves[i][1] + 7), leaves[j][0], leaves[j][1] - 7)


def _rename_key(v):
    if isinstance(v, dict) and v:
        k = sorted(v)[-1]
        out = {x: y for x, y in v.items() if x != k}
        out[str(k) + '_other'] = v[k]
        return out
    return None


# fields that exist only when the battle has them: abilities (hero / champion decks) and commands in flight. (round 2, Y) The
# exemption from ABSENT holds only while no such element exists; every element that does exist must carry every input of its kind
CONDITIONAL = ('players[].abilities[].', 'queue[].')
ELEMENTS = {'players[].abilities[].': lambda fr: [a for p in fr.get('players') or () for a in p.get('abilities') or ()],
            'queue[].': lambda fr: list(fr.get('queue') or ())}


def required_facts(fr, facts):
    """(round 3, AG) what the recording itself says must have been filled: a side that played cards had commands in flight
    and card_counts; a side with a hero in its deck had an ability"""
    for p in fr.get('players') or ():
        s = p.get('side')
        if (p.get('cards_played') or 0) > 0:
            facts[('played', s)] = True
        if p.get('card_counts'):
            facts[('card_counts', s)] = True
        if any(f == 2 for f in p.get('deck_form_flags') or ()):
            facts[('hero', s)] = True
        if p.get('abilities'):
            facts[('abilities', s)] = True
    tk = fr.get('game_tick')
    if isinstance(tk, int):
        facts.setdefault('ticks', set()).add(tk)
    for q in fr.get('queue') or ():
        # (round 6, AY) the queue window: every entry lands 0..6 ticks after this frame, with a real landing tick
        d = (q.get('t2') if isinstance(q.get('t2'), int) else -1) - (tk if isinstance(tk, int) else 0)
        if q.get('t2', -1) < 0 or d < 0 or d > 6:
            facts['window_wrong'] = (facts['window_wrong'] or 0) + 1
        facts.setdefault('cmd_ticks', collections.defaultdict(set))[(q.get('kind'), q.get('t2'), tuple(q.get('id') or ()), q.get('a'), q.get('b'))].add(tk)
    if fr.get('queue'):
        facts['queue'] = True
        if any(q.get('kind') == 1 for q in fr['queue']):
            facts['play_kind'] = True
        for q in fr['queue']:                     # (round 7) every command's side is known (0 / 1)
            if 'side' in q and q['side'] not in (0, 1):
                facts['side_unknown'] = (facts['side_unknown'] or 0) + 1
        for q in fr['queue']:                     # (round 5, AT / AU) every command, not one example: a card -> a play (1), none -> an ability (2)
            want = 1 if (q.get('card') or -1) > 0 else 2
            if q.get('kind') != want:
                facts['kind_wrong'] = (facts['kind_wrong'] or 0) + 1
        acct = {tuple(p.get('account') or ()): p.get('side') for p in fr.get('players') or ()}
        for q in fr['queue']:
            if q.get('kind') == 1 and (q.get('card') or -1) > 0:
                facts.setdefault('played_cards', set()).add((acct.get(tuple(q.get('id') or ())), q['card']))
    for q in fr.get('queue') or ():               # (round 6, BB) each play once, by player and landing: card_counts is held against them
        if q.get('kind') == 1 and (q.get('card') or -1) > 0:
            facts.setdefault('plays', {})[(tuple(q.get('id') or ()), q.get('t2'))] = q['card']
    acct_side = {tuple(p.get('account') or ()): p.get('side') for p in fr.get('players') or ()}
    facts['acct_side'] = {**(facts.get('acct_side') or {}), **acct_side}
    for p in fr.get('players') or ():
        if p.get('card_counts'):
            facts.setdefault('last_counts', {})[p.get('side')] = (fr.get('game_tick'), p['card_counts'])
    for p in fr.get('players') or ():          # (round 4, AM) card_counts is this battle's: its counts add up to cards played
        cc = p.get('card_counts')
        for cid, _n in cc or ():
            facts.setdefault('counted_cards', set()).add((p.get('side'), cid))
        if cc and sum(c for _, c in cc) != (p.get('cards_played') or 0):
            facts['card_counts_sum_wrong'] = facts['card_counts_sum_wrong'] + 1 if isinstance(facts['card_counts_sum_wrong'], int) else 1
    units = [e for e in fr.get('entities') or () if 5000006 <= e.get('category', 0) < 6000000]
    if units:
        facts['units'] = True
        for side in (0, 1):                      # (round 6, BB) per side: one side's troops losing their components shows
            us_ = [e for e in units if e.get('side') == side]
            facts[('unit_frames', side)] = (facts[('unit_frames', side)] or 0) + len(us_)
            facts[('attack_frames', side)] = (facts[('attack_frames', side)] or 0) + sum(1 for e in us_ if e.get('has_attack'))
            facts[('move_frames', side)] = (facts[('move_frames', side)] or 0) + sum(1 for e in us_ if e.get('has_move'))
        facts['unit_frames'] = (facts['unit_frames'] or 0) + len(units)


def element_gaps(frame, gaps, exercised):
    for pre, get in ELEMENTS.items():
        want = [k[len(pre):] for k, v in SPEC.items() if v[0] == IN and k.startswith(pre)]
        for el in get(frame):
            exercised.add(pre)
            have = {k for k in keys(el)}
            for w in want:
                if w not in have:
                    gaps[pre + w] = gaps.get(pre + w, 0) + 1


def main():
    seen, frames, sample, claims, gaps, exercised = set(), 0, [], {}, {}, set()
    facts = collections.defaultdict(bool)
    for path in sys.argv[1:]:
        with gzip.open(path, 'rt') if path.endswith('.gz') else open(path) as f:
            for line in f:
                if line.startswith('{'):
                    fr = json.loads(line)
                    seen |= set(keys(fr))
                    element_gaps(fr, gaps, exercised)
                    required_facts(fr, facts)
                    for k, v in claim_violations(fr):
                        claims[k] = claims.get(k, 0) + 1
                    if frames % 50 == 0:
                        sample.append(fr)
                    frames += 1
    uncovered = sorted(k for k in seen if rule(k) is None)
    ins = sorted(k for k in seen if (rule(k) or ('',))[0] == IN)
    outs = sorted(k for k in seen if (rule(k) or ('',))[0] == OUT)
    # (fresh break test M) the other direction: every input of the spec must be in the recordings (a reader or a recording that
    # lost a field failed nothing before)
    absent = sorted(k for k, v in SPEC.items() if v[0] == IN and k not in seen and not k.startswith(CONDITIONAL))
    not_exercised = sorted(p for p in CONDITIONAL if p not in exercised)
    # (fresh break test L) the builder half: every input present changes the built input
    dead = set()
    for i, fr in enumerate(sample):
        ROTATE[0] = i
        dead |= set(check_wiring(fr)[1])
    print(f'CHECK 5: {frames} frames, {len(seen)} fields: {len(ins)} inputs, {len(outs)} left out with a reason, '
          f'{len(uncovered)} NOT COVERED, {len(absent)} spec inputs ABSENT from the recordings, {len(dead)} inputs that never reach '
          f'the built input (DEAD; wiring tried on {len(sample)} frames)')
    for k in uncovered:
        print('  NOT COVERED:', k)
    for k in absent:
        print('  ABSENT:', k)
    for k in sorted(dead):
        print('  DEAD:', k)
    for k, n in sorted(gaps.items()):
        print(f'  INCOMPLETE ELEMENT: {k} missing in {n} elements that exist')
    for p in not_exercised:
        print(f'  NOT EXERCISED: no {p.rstrip(".")} element in these recordings: its fields are unproven here')
    length_only = length_tested - value_proven
    if length_only:
        print(f'  length only (empty in every frame tried; values not proven wired): {", ".join(sorted(length_only))}')
    for k, n in claims.items():
        print(f'  OUT REASON WRONG: {k} is not {VALUE_CLAIMS[k]} in {n} frames / commands')
    for k in outs:
        print(f'  out: {k}: {rule(k)[1]}')
    must = []                                 # (round 3, AG) empty although the recording says it was filled
    for s_ in (0, 1):
        if facts[('played', s_)] and not facts[('card_counts', s_)]:
            must.append(f'players[].card_counts of side {s_}: cards were played, the list is empty in every frame')
        if facts[('hero', s_)] and not facts[('abilities', s_)]:
            must.append(f'players[].abilities of side {s_}: a hero in the deck, no ability in any frame')
    if (facts[('played', 0)] or facts[('played', 1)]) and not facts['queue']:
        must.append('queue: cards were played, no command in any frame')
    if facts['queue'] and (facts[('played', 0)] or facts[('played', 1)]) and not facts['play_kind']:
        must.append('queue[].kind: cards were played, no command has kind 1 (a card play)')
    if facts['card_counts_sum_wrong']:
        must.append(f"players[].card_counts: in {facts['card_counts_sum_wrong']} player-frames the counts do not add up to cards_played (a career record, or a lost play)")
    # (round 5, AT) shares, not one example: most troops attack and move (engine and live: 70-95 % of troop-frames)
    for side in (0, 1):                          # measured 70-100 % per side in engine and live battles
        n_ = facts[('unit_frames', side)] or 0
        if n_ >= 500 and (facts[('attack_frames', side)] or 0) < 0.6 * n_:
            must.append(f"entities[].has_attack: side {side}: only {facts[('attack_frames', side)] or 0} of {n_} troop-frames have an attack (components lost?)")
        if n_ >= 500 and (facts[('move_frames', side)] or 0) < 0.5 * n_:
            must.append(f"entities[].has_move: side {side}: only {facts[('move_frames', side)] or 0} of {n_} troop-frames move (components lost?)")
    if facts['window_wrong']:
        must.append(f"queue: {facts['window_wrong']} entries outside the window (landing 0..6 ticks after the frame, t2 known)")
    ticks = facts.get('ticks') or set()
    gaps = 0
    for (kind, t2, _id, _a, _b), seen in (facts.get('cmd_ticks') or {}).items():
        if isinstance(t2, int):
            gaps += sum(1 for t in range(t2 - 6, t2 + 1) if t in ticks and t not in seen)
    if gaps:
        must.append(f'queue: {gaps} recorded frames inside a command\'s window (landing - 6 .. landing) lack the command')
    if facts['side_unknown']:
        must.append(f"queue[].side: {facts['side_unknown']} commands whose side the reader could not match")
    if facts['kind_wrong']:
        must.append(f"queue[].kind: {facts['kind_wrong']} commands whose kind does not fit (a card -> 1, none -> 2)")
    plays = facts.get('plays') or {}
    if plays and facts.get('last_counts'):
        sides = facts.get('acct_side') or {}
        for side, (tick, counts) in facts['last_counts'].items():
            mine = sorted(((t2, card) for (acct, t2), card in plays.items() if sides.get(acct) == side and t2 is not None and t2 <= tick))
            order, n = [], collections.Counter()
            for _t2, card in mine:
                if card not in order:
                    order.append(card)
                n[card] += 1
            if mine and ([c for c, _ in counts] != order or {c: k for c, k in counts} != dict(n)):
                must.append(f'players[].card_counts of side {side} at tick {tick}: {counts[:4]} ... against the plays landed: '
                            f'{[(c, n[c]) for c in order[:4]]} ... (counts or first-play order)')
    stray = (facts.get('counted_cards') or set()) - (facts.get('played_cards') or set())
    if stray and facts.get('played_cards'):
        must.append(f'players[].card_counts: {len(stray)} (side, card) counted that no command of that side played, e.g. {sorted(stray)[:3]}')
    for m in must:
        print('  MUST BE FILLED:', m)
    bad = uncovered or absent or dead or claims or gaps or must or not frames
    print('VERDICT', 'FAIL' if bad else 'PASS')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
