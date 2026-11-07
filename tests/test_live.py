"""Live play's bookkeeping: plays confirmed by the hand, screen coordinates, the reader's target pointers."""
import json
import pytest

import play


def frame(tick, commands=(), hand=(0, 1, 2, 3)):
    return dict(game_tick=tick, coherent=True, failure='none', queue_ok=True, commands=list(commands),
                players=[dict(side=1, deck_card_ids=[26, 27, 28, 29, 30], hand_deck_indices=list(hand))])


def command(sequence=1, card=26, due=121):
    return dict(owner=1, sequence=sequence, card_id=card, issue_tick=due - 21, execute_tick=due, x=3500, y=25000)


def test_live_plays_are_confirmed_by_the_hand_not_the_queue():
    plays = play.LivePlays()
    assert plays.update(frame(110, [command()])) == []
    assert plays.commands()[0][0] == 121 and plays.commands()[0][1]['first_seen_tick'] == 110
    landed = plays.update(frame(121, [], hand=(4, 1, 2, 3)))
    assert len(landed) == 1 and landed[0][0] == 121 and landed[0][1]['x'] == 3500
    assert plays.flying == [] and len(plays.update(frame(122, [], hand=(4, 1, 2, 3)))) == 1
    # an incoherent frame confirms nothing; a hand change with no queued command has no coordinates
    other = play.LivePlays()
    other.update(frame(110, [command()]))
    bad = frame(121, [], hand=(4, 1, 2, 3))
    bad['coherent'] = False
    assert other.update(bad) == []
    blind = play.LivePlays()
    blind.update(frame(110))
    assert 'x' not in blind.update(frame(121, [], hand=(4, 1, 2, 3)))[0][1]
    # two cards at once, and our own just-sent tap, are not counted twice
    two = play.LivePlays()
    two.update(frame(110, [command(), command(2, 27, 122)]))
    assert len(two.commands(dict(card=26, due=121, x=3500, y=25000), 1)) == 2
    assert [c['card_id'] for _, c in two.update(frame(123, [], hand=(4, -1, 2, 3)))] == [26, 27]


def test_the_opponents_plays_land_from_the_queue_without_their_hand():
    # we are side 0; side 1's hand is hidden: its command lands at its execute tick, seen at 110
    plays = play.LivePlays(0)
    hidden = lambda tick, commands=(): dict(frame(tick, commands), players=[dict(side=1, deck_card_ids=[],
                                                                                   hand_deck_indices=[-1] * 4)])
    plays.update(hidden(110, [command()]))
    assert plays.commands()[0][1]['seen'] == 110
    assert plays.update(hidden(120)) == []
    landed = plays.update(hidden(123))
    assert [(t, c['card_id'], c['seen']) for t, c in landed] == [(121, 26, 110)] and plays.update(hidden(124)) == landed



def test_a_card_reserved_after_our_ability_press():
    # 1 Nov live crash: the pending press (no card id) was matched against the next card reservation
    press = (130, dict(kind='ability', owner=0, controller=1, tap=120))
    card = (140, dict(owner=0, card_id=26, x=9000, y=8000, tap=121))
    out = play.LivePlays(0).commands(reservations=[press, card], tick=121)
    assert [c.get('card_id') for _, c in out] == [None, 26]

def test_world_to_screen_puts_our_side_at_the_bottom():
    assert play.world_screen(9000, 3000, 0)[1] > play.world_screen(9000, 29000, 0)[1]   # owner 0's king is at y 3000
    assert play.world_screen(9000, 29000, 1)[1] > play.world_screen(9000, 3000, 1)[1]


def test_live_targets_resolve_across_the_heap_tag(tmp_path):
    # the reader prints entity addresses with Android's heap tag and target pointers without it (tick 381 of the
    # 23 Oct audit); comparing the strings lost every target live
    (tmp_path / 'tables.json').write_text(json.dumps(dict(build='150535029', characters={'1': {}}, effects={'1': {}},
                                                          abilityBuffs=[])))
    client = play.Client(folder=tmp_path, name='test', tables='tables.json', tables_build='150535029', hero_card_class=None)

    def entity(address, category, side, target):
        return dict(address=address, category=category, side=side, target=target, has_attack=1, card_id=-1, data_id=0,
                    x=9000, y=3000, hp=100, max_hp=100, deploy_remaining=0, atk_load=0)
    frame = dict(game_tick=381, players=[], entities=[entity('0xb40000761f46ba70', 5000005, 1, '0x761f5331b0'),
                                                      entity('0xb40000761f5331b0', 5000006, 0, '0x761f46ba70'),
                                                      entity('0xb40000761f5452c0', 5000007, 0, '0x0')])
    objects = play.LiveRecords(client)(frame)['objects']
    assert [o['targetEntityKey'] for o in objects] == [[0, -2, 5000006], [1, -2, 5000005], None]


def test_ability_buttons_follow_the_deck_order_not_the_controller_number():
    # 1 Nov, Nulls: Golden Knight (deck slot 1) was controller 2 with the left button, Archer Queen (slot 2) controller 1
    deck = [26000049, 26000074, 26000072, 26000021, 26000050, 26000010, 26000031, 26000084]
    two = dict(deck_card_ids=deck, abilities=[dict(controller_slot=1, character_id=34000093),
                                              dict(controller_slot=2, character_id=34000092)])
    assert play.ability_buttons(two) == {1: play.ABILITY_BUTTON, 2: play.ABILITY_LEFT}
    one = dict(deck_card_ids=deck, abilities=[dict(controller_slot=1, character_id=34000093),
                                              dict(controller_slot=2, character_id=0)])
    assert play.ability_buttons(one) == {1: play.ABILITY_BUTTON}


def test_client_settings_seat_and_card_ids(tmp_path):
    (tmp_path / 'client.json').write_text(json.dumps(dict(name='t', seat='hand', hero_card_class=306)))
    client = play.Client.load(str(tmp_path))
    assert client.folder == tmp_path and client.card(306000017) == 203000017 and client.card(26000000) == 26000000
    hidden = dict(players=[dict(side=0, hand_deck_indices=[0, 1, 2, 3]), dict(side=1, hand_deck_indices=[-1] * 4)],
                  accounts=[5, 6])
    assert play.our_side(client, hidden) == (0, [5, 6])
    both = dict(hidden, players=[dict(side=0, hand_deck_indices=[0, 1, 2, 3]), dict(side=1, hand_deck_indices=[0, 1, 2, 3])])
    assert play.our_side(client, both)[0] is None
    # each button's press cost: its hero's (Archer Queen, character 34000093: 1 elixir) unless overridden
    me = dict(abilities=[dict(controller_slot=1, character_id=34000093), dict(controller_slot=2, character_id=0)])
    assert play.ability_costs_now(me, (None, None)) == (1.0, None) and play.ability_costs_now(me, (2.0, 3.0)) == (2.0, 3.0)


def test_the_opponents_plays_also_land_from_their_hand_when_the_queue_has_none():
    # Training Camp on Nulls: the trainer's plays never reach the queue, its hand is readable; a queued play is not counted
    # twice when the hand shows it too
    def two(tick, commands=(), theirs=(0, 1, 2, 3)):
        return dict(game_tick=tick, coherent=True, failure='none', queue_ok=True, commands=list(commands),
                    players=[dict(side=0, deck_card_ids=[1, 2, 3, 4, 5], hand_deck_indices=[0, 1, 2, 3]),
                             dict(side=1, deck_card_ids=[26, 27, 28, 29, 30], hand_deck_indices=list(theirs))])
    plays = play.LivePlays(0)
    plays.update(two(110))
    assert [(t, c['card_id']) for t, c in plays.update(two(121, theirs=(4, 1, 2, 3)))] == [(121, 26)]
    plays.update(two(130, [command(2, 27, 141)], theirs=(4, 1, 2, 3)))
    landed = plays.update(two(142, theirs=(4, 0, 2, 3)))
    assert [(t, c['card_id']) for t, c in landed] == [(121, 26), (141, 27)]


def test_netwatch_reads_the_game_connection_from_nettop(monkeypatch):
    import subprocess
    import play
    out = (',state,bytes_in,bytes_out,re-tx,rtt_avg,rtt_var,\n'
           'qemu-system-aar.1,,10,20,0,,,\n'
           'tcp4 127.0.0.1:5555<->127.0.0.1:631,Established,5,6,0,1.22 ms,0.19 ms,\n'
           'tcp4 10.0.0.2:63213<->135.125.24.77:9339,Established,368139,33763,823,351.75 ms,10.62 ms,\n')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, out, ''))
    watch = play.NetWatch.__new__(play.NetWatch)
    assert watch.sample({'135.125.24.77:9339'}) == {'server': '135.125.24.77:9339', 'rtt_ms': 351.75, 'rtt_var_ms': 10.62,
                                                     'retx_bytes': 823, 'bytes_in': 368139, 'bytes_out': 33763}
    assert watch.sample({'1.2.3.4:9339'}) is None


def test_next_battle_takes_a_moving_battle_at_any_tick_but_never_a_finished_one():
    from types import SimpleNamespace
    import play
    client = SimpleNamespace(seat='roster')  # 'roster': our side is found later, not here
    f = lambda battle, tick: {'battle_active': True, 'failure': 'none', 'chain': {'battle': battle}, 'game_tick': tick}  # noqa: E731
    results = [f('a', 2022)] * 50  # the finished battle's results screen: active, clock still
    # results screen, then the same battle re-entered and moving again past where it stopped
    assert play.next_battle(iter(results + [f('a', 2030), f('a', 2035)]), client, ('a', 2022))['game_tick'] == 2030
    # a re-entered battle under a new address, already at tick 2100 (the old rule wanted tick < 90)
    assert play.next_battle(iter(results + [f('b', 2100), f('b', 2105)]), client, ('a', 2022))['chain']['battle'] == 'b'
    # started while a battle runs: it counts once its clock moves; a still one does not
    assert play.next_battle(iter([f('c', 500), f('c', 500), f('c', 505)]), client, None)['game_tick'] == 505
    with pytest.raises(StopIteration):
        play.next_battle(iter(results), client, ('a', 2022))
