"""Actual wrapper sequencing with a clock/payment stub; native validation is a separate smoke test."""
from collections import defaultdict
from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from cr import engine, observation as C, policy
from cr.live_budget import LiveBudget, Pending


def simulator(balance=2, ability_cost=1, card_cost=2, tick=100):
    clock = SimpleNamespace(tick=tick, elixir=float(balance * 10000), used=False, card=26000002)
    native, events = [], []
    def capture():
        ability = engine.Ability(6 if clock.used else 2, 0, 0, int(not clock.used), 1, (123,),
                                 cost=ability_cost, name='test')
        hand = (engine.HandCard(0, clock.card, card_cost, 0, clock.card),)
        return SimpleNamespace(tick=clock.tick, players=(engine.Player(0, int(clock.elixir), hand, (), (ability,)),))
    def advance(n):
        for _ in range(n):
            clock.elixir = min(100000, clock.elixir + C.regen(clock.tick, clock.tick + 1))
            clock.tick += 1
            for event in list(events):
                due, kind, cost = event
                if due <= clock.tick:
                    assert clock.elixir + 1e-5 >= cost * 10000, 'admitted actions overspent elixir'
                    clock.elixir -= cost * 10000
                    if kind == 'ability': clock.used = True
                    else: clock.card += 1
                    native.append(kind)
                    events.remove(event)
    def activate(owner, controller):
        events.append((clock.tick + 1, 'ability', ability_cost))
        return True
    def queue_card(command, execute_tick):
        events.append((execute_tick, 'card', card_cost))
    sim = engine.Engine.__new__(engine.Engine)
    sim.state, sim.pending, sim.overlap, sim.expired = capture(), [], [True, False], 0
    sim.ability_stats, sim.ability_receipts = [defaultdict(int), defaultdict(int)], []
    sim.env = SimpleNamespace(tick=lambda: clock.tick, step=advance, queue_hand_action_at=queue_card)
    sim._capture, sim._activate = capture, activate
    sim._players = lambda: [({0: (0, clock.card)}, (), int(clock.elixir))]
    return sim, clock, native


PRESS = engine.Play(0, engine.ABILITY, 0, 0)
CARD = engine.Play(0, 0, 3500, 10000)


def test_ability_reservation_blocks_double_spend_but_releases_after_payment():
    sim, clock, native = simulator()
    assert sim.step([PRESS], 5, delay=22).executed == (True,)
    assert sim.pending[0]['cost'] == 1
    assert not sim.playable(0)[0]
    assert sim.step([CARD], 5, delay=22).executed == (False,)
    for _ in range(12): sim.step([], 5)
    assert sim.ability_stats[0]['confirmed'] == 1
    assert native == ['ability'] and sim.expired == 0
    assert sim.playable(0)[0]
    assert sim.step([CARD], 5, delay=22).executed == (True,)
    for _ in range(10): sim.step([], 5)
    assert native == ['ability', 'card'] and sim.expired == 0


def test_future_regeneration_can_fund_ability_and_card_without_artificial_hold():
    sim, clock, native = simulator(balance=2.3, tick=3000)
    assert sim.step([PRESS], 5, delay=22).executed == (True,)
    assert sim.playable(0)[0]
    assert sim.step([CARD], 5, delay=22).executed == (True,)
    card = next(p for p in sim.pending if p['slot'] != engine.ABILITY)
    assert not card['held']
    for _ in range(8): sim.step([], 5)
    assert native == ['ability', 'card'] and sim.expired == 0


def test_exact_same_tick_payments_are_checked_together():
    sim, clock, native = simulator(balance=2.7)
    assert sim.step([PRESS, CARD], 5, delay=22).executed == (True, True)
    for _ in range(8): sim.step([], 5)
    assert sorted(native) == ['ability', 'card'] and sim.expired == 0


def test_live_and_training_use_same_known_cost_timeline():
    for balance in np.arange(1, 4, .1):
        sim, clock, native = simulator(balance=float(balance))
        assert sim.step([PRESS], 5, delay=22).executed == (True,)
        live = LiveBudget(0, overlap=True, ability_costs=(1, 0))
        live.pending.append(Pending('ability', 1, 100, 122, 0, charges=1))
        expected = live.card_due(clock.elixir / 10000, clock.tick, 0, clock.card, 2)
        _, _, actual = sim._card_tap(0, 0, sim.state, delay=22)
        assert actual == (expected is not None)


def test_sampled_delay_never_breaks_the_conservative_playability_mask():
    rng = random.Random(271027)
    admitted = 0
    for _ in range(200):
        ability_cost, card_cost = rng.randrange(1, 6), rng.randrange(1, 7)
        balance = rng.uniform(ability_cost, 10)
        sim, clock, native = simulator(balance, ability_cost, card_cost, rng.choice([100, 2390, 3000, 4790, 5000]))
        assert sim.step([PRESS], 5, delay=rng.choice(engine.PAYMENT_DELAYS)).executed == (True,)
        allowed = sim.playable(0)[0]
        result = sim.step([CARD] if allowed else [], 5, delay=rng.choice(engine.PAYMENT_DELAYS))
        if allowed:
            admitted += 1
            assert result.executed == (True,)
        for _ in range(15): sim.step([], 5)
        assert sim.expired == 0
        assert native.count('card') == int(allowed)
    assert admitted > 50


def test_pending_or_unfunded_ability_stays_masked_even_with_exploration():
    logits = torch.tensor([[-5., -5., 10., 10.]])
    inputs = dict(own_flight_card=torch.zeros(1, dtype=torch.long),
                  live_abilities=torch.tensor([[False, False]]))
    for explore in (0, policy.ABILITY_EXPLORE):
        lp = policy.what_logp(logits, inputs, explore, overlap=True)
        assert torch.isneginf(lp[0, 2:]).all()
        assert lp[0, :2].exp().sum() == pytest.approx(1)


def test_selected_controller_must_be_ready_and_funded():
    sim, clock, native = simulator(balance=1, ability_cost=2)
    assert not sim.ability_playable(0, 0)
    assert not sim.ability_playable(0, 1)
    assert sim.step([PRESS], 5, delay=22).executed == (False,)
    assert not sim.pending


def test_runtime_uses_ability_name_cost_and_keeps_controller_slot_identity():
    rows = [dict(controllerSlot=2, actionDataName='Musketeer_hero_Ability', buttonState=2,
                 remainingCooldownMs=0, configuredCooldownMs=0, remainingChargesRaw=1,
                 maxCharges=1, championObjectIds=[123])]
    sim = engine.Engine.__new__(engine.Engine)
    sim.ability_costs = {'Musketeer_hero_Ability': 3}
    sim.env = SimpleNamespace(battle=None, B=SimpleNamespace(build='160402020'), reader=SimpleNamespace(
        player=lambda *a: None, logic=lambda *a: None, abilities=lambda *a: rows))
    first, second = sim._runtime(0)
    assert not first.ready and first.cost is None
    assert second.ready and second.cost == 3
    rows[0]['actionDataName'] = 'UnmappedAbility'
    with pytest.raises(ValueError, match='unknown ability cost'):
        sim._runtime(0)
