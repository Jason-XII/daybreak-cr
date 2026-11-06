from copy import deepcopy
from pathlib import Path
import sys
import unittest

from cr.inputs_v2.v2_contract import numeric_frame, pending_inputs, player_frame, sanitize_frame


class ContractTests(unittest.TestCase):
    def test_two_commands_and_landed_removal(self):
        commands = [dict(t=85, t2=105, side=0, kind=1, card=26000000, a=2000, b=3000),
                    dict(t=84, t2=104, side=0, kind=1, card=26000001, a=4000, b=5000)]
        frame = dict(game_tick=100, queue=commands)
        cards = pending_inputs(frame, 0, int)['own_flight_card']
        self.assertEqual(cards.tolist(), [26000001, 26000000])
        frame['game_tick'] = 104
        cards = pending_inputs(frame, 0, int)['own_flight_card']
        self.assertEqual(cards.tolist(), [26000000, 0])

    def test_future_and_outside_window_rejected(self):
        for issue, landing in ((101, 105), (80, 107)):
            frame = dict(game_tick=100, queue=[dict(t=issue, t2=landing, side=0, kind=1,
                                                   card=26000000, a=0, b=0)])
            with self.assertRaises(ValueError):
                pending_inputs(frame, 0, int)

    def test_hidden_opponent_and_unspawned_ability(self):
        opponent = dict(side=1, player_raw=[15, None], next_deck_index=3,
                        abilities=[dict(units=[], character_id=123)])
        for name in ('deck_card_ids', 'deck_form_flags', 'deck_level_index', 'battle_deck',
                     'evo_progress', 'hand_deck_indices', 'cycle_deck_indices', 'slot_words'):
            opponent[name] = [1, 2, 3]
        world = dict(players=[dict(side=0), opponent], entities=[])
        original = deepcopy(world)
        public = player_frame(world, 0)
        self.assertEqual(world, original)
        self.assertEqual(public['players'][1]['deck_card_ids'], [])
        self.assertEqual(public['players'][1]['player_raw'], [None, None])
        self.assertEqual(public['players'][1]['abilities'][0]['character_id'], 0)

    def test_seed_pointer_and_future_metadata_not_inputs(self):
        frame = dict(game_tick=100, rng=123, seed=0,
                     entities=[dict(category=5000000, address='0x123')], queue=[])
        public = sanitize_frame(frame)
        self.assertNotIn('rng', public)
        self.assertNotIn('seed', public)
        self.assertNotIn('address', public['entities'][0])
        with self.assertRaises(ValueError):
            sanitize_frame(dict(game_tick=100, future_commands=[1]))

    def test_numeric_coordinates_and_vectors_use_own_view(self):
        frame = dict(observer_side=1, players=[dict(side=0), dict(side=1)],
                     entities=[dict(category=7, side=1, x=2000, y=3000, tx=4000, ty=5000,
                                    us=dict(aim=[6000, 7000], start=[2000, 3000], surface=[-1, -1],
                                            dx=100, dy=-200, facing=[256, 0], waypoint=[9, 10], ent=[123]))],
                     queue=[dict(side=1, kind=1, a=2000, b=3000)])
        original = deepcopy(frame)
        result = numeric_frame(frame, 1)
        self.assertEqual(frame, original)
        self.assertEqual([p['side'] for p in result['players']], [1, 2])
        entity = result['entities'][0]
        self.assertEqual((entity['x'], entity['y']), (16000, 29000))
        self.assertEqual(entity['us']['aim'], [12000, 25000])
        self.assertEqual(entity['us']['surface'], [-1, -1])
        self.assertEqual((entity['us']['dx'], entity['us']['dy']), (-100, 200))
        self.assertEqual(entity['us']['facing'], [-256, 0])
        self.assertEqual(entity['us']['waypoint'], [9, 10])
        self.assertEqual(result['queue'][0]['b'], 29000)


if __name__ == '__main__':
    unittest.main()
