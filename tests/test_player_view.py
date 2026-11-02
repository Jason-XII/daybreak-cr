"""PlayerView's hero / ability inputs and its own hand (31 Oct): what live play reads, built the same way from a record."""
import numpy as np

from cr import observation as C

KNIGHT, ARCHERS, GIANT, MUSKETEER, MINER, ZAP, LOG, ICE = 26000000, 26000001, 26000003, 26000014, 26000032, 28000008, 28000011, 26000023
DECK = [KNIGHT, ARCHERS, GIANT, MUSKETEER, MINER, ZAP, LOG, ICE]
FORMS = [2, 0, 1, 0, 0, 0, 0, 0]           # hero Knight, evolved Giant


def unit(oid, owner, card, x=9000, y=10000):
    return dict(nativeObjectId=oid, owner=owner, cardId=card, dataGlobalId=card, x=x, y=y, hp=100, maxHp=100,
                targetEntityKey=None, phaseRuntime={})


def record(tick, objects, own_carriers=(), their_carriers=(), button=2):
    hand = [dict(handIndex=i, deckSlot=s, cardId=DECK[s]) for i, s in enumerate((0, 1, 2, 3))]
    cycle = [dict(cycleIndex=i, deckSlot=s, cardId=DECK[s]) for i, s in enumerate((4, 5, 6, 7))]
    state = [dict(owner=o, elixirRaw=50000, hand=hand, cycle=cycle) for o in (0, 1)]
    controller = lambda units: [dict(controllerSlot=1, available=True, buttonState=button, remainingChargesRaw=1,
                                     remainingCooldownMs=0, configuredCooldownMs=0, championObjectIds=list(units))]
    lean = [dict(owner=0, abilityRuntime=controller(own_carriers), evolutionRuntime=[]),
            dict(owner=1, abilityRuntime=controller(their_carriers), evolutionRuntime=[])]
    return dict(tick=tick, lean=dict(state=dict(players=state), players=lean), objects=objects,
                units=[dict(nativeObjectId=o['nativeObjectId'], kind='unit') for o in objects])


def test_hero_ability_inputs_and_the_own_hand():
    vocab = C.Vocab(DECK)
    view = C.PlayerView(0, vocab, own_forms=FORMS)
    x = view.step(record(90, [unit(7, 0, ARCHERS, y=10000)]), [])
    assert (x['hand_card'] > 1).all() and (x['cycle_card'] > 1).all()          # the opening hand is known, as live
    assert x['hand_float'][0, 4] == 1 and x['hand_float'][2, 5] == 1           # hero Knight, evolved Giant
    assert x['hand_float'][1, 4:6].tolist() == [0, 0]
    assert x['ability_card'].tolist() == [0, 0] and x['opp_hero_card'].tolist() == [0, 0]   # no carrier on the board yet
    assert x['ability'][0, 5 + 2] == 1 and x['ability'][0, 1] == 1             # button state 2 one-hot, ready
    press = (121, {'kind': 'ability', 'owner': 0, 'controller': 0, 'tap': 100})
    objects = [unit(7, 0, ARCHERS, y=11000), unit(8, 0, KNIGHT), unit(9, 1, MINER)]
    x = view.step(record(97, objects, own_carriers=[8], their_carriers=[9]), [press])  # 7 ticks after the last decision
    assert x['ability_card'].tolist() == [vocab(KNIGHT), 0] and x['ability'][0, 4] == 0   # before the tap: not pending
    rows = {int(c): i for i, c in enumerate(x['obj_card']) if c}
    button = C.OBJ_FLOAT_NAMES.index('ability_button')
    assert x['obj_float'][rows[vocab(KNIGHT)], button] == 0.5 and x['obj_float'][rows[vocab(MINER)], button] == 0
    assert abs(x['obj_float'][rows[vocab(ARCHERS)], 3] - 1000 / 1000 * 5 / 7) < 1e-6   # movement scaled to 5 ticks
    assert x['opp_hero_card'].tolist() == [vocab(MINER), 0]                    # seen on the board: known from now on
    x = view.step(record(102, objects[:2], own_carriers=[8]), [press])
    assert x['ability'][0, 4] == 1 and x['opp_hero_card'].tolist() == [vocab(MINER), 0]   # in flight; still known
    x = view.step(record(122, objects[:2], own_carriers=[8]), [press])
    assert x['ability'][0, 4] == 0                                              # executed


def test_the_opponents_command_shows_from_its_queue_tick_or_a_fixed_live_lead():
    c = dict(owner=1, card_id=26000000, x=3500, y=25000)
    assert C.opp_seen(500, dict(c, seen=490)) == 490
    leads = {500 - C.opp_seen(500, dict(c, x=x)) for x in range(0, 40000, 500)}
    assert C.opp_seen(500, c) == C.opp_seen(500, dict(c)) and leads <= set(C.OPP_LEADS) and len(leads) > 5


def test_object_kind_and_hit_points_follow_what_the_live_reader_reports():
    o = lambda oid, hp=100, most=200: dict(nativeObjectId=oid, hp=hp, maxHp=most)  # noqa: E731
    assert [C.object_kind(o(i)) for i in (5000000, 3000001, 4000001, 5000100)] == ['tower', 'area', 'projectile', 'unit']
    dummy = o(5000101, 4089994, 4089994)   # Tombstone hero's visual dummy: the reader drops hp above 100,000
    assert C.hit_points(dummy) == (None, None) and C.object_kind(dummy) == 'timed'
    assert C.object_kind(o(5000102, None, None)) == 'timed' and C.hit_points(o(5000103, 900, 200)) == (None, None)
    assert C.hit_points(o(5000104, 390, 195)) == (390, 195)       # an evolved Bat healed to twice its maximum
