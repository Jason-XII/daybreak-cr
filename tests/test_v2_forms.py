from pathlib import Path
import sys
import unittest

import numpy as np
from cr import observation as C
from cr.inputs_v2.v2_dataset import update_form_and_carriers


class FormTests(unittest.TestCase):
    def test_forms_and_carriers_populate_without_changing_other_columns(self):
        data = dict(x_hand_card=np.array([[10, 11, 12, 13]]),
                    x_hand_float=np.full((1, 4, 7), .25, np.float32),
                    x_obj_float=np.zeros((3, C.OBJ_FLOATS), np.float32))
        player = dict(side=0, hand_deck_indices=[0, 1, 2, 3],
                      deck_card_ids=[10, 11, 12, 13], deck_form_flags=[2, 1, 0, 0],
                      abilities=[dict(controller_slot=1, units=[100]), dict(controller_slot=2, units=[101])])
        entities = [dict(category=100, side=0), dict(category=101, side=0), dict(category=102, side=1)]
        update_form_and_carriers(data, dict(players=[player]), 0, 0, entities, 0, 3, int)
        np.testing.assert_array_equal(data['x_hand_float'][0, :, 4:6], [[1, 0], [0, 1], [0, 0], [0, 0]])
        np.testing.assert_array_equal(data['x_hand_float'][0, :, :4], np.full((4, 4), .25))
        np.testing.assert_array_equal(data['x_obj_float'][:, C.OBJ_FLOAT_NAMES.index('ability_button')], [.5, 1, 0])

    def test_hand_identity_mismatch_rejected(self):
        data = dict(x_hand_card=np.array([[99, 11, 12, 13]]), x_hand_float=np.zeros((1, 4, 7)),
                    x_obj_float=np.zeros((0, C.OBJ_FLOATS)))
        player = dict(side=0, hand_deck_indices=[0, 1, 2, 3], deck_card_ids=[10, 11, 12, 13],
                      deck_form_flags=[0]*4, abilities=[])
        with self.assertRaises(ValueError):
            update_form_and_carriers(data, dict(players=[player]), 0, 0, [], 0, 0, int)

    def test_unknown_base_hand_is_not_a_different_known_card(self):
        data = dict(x_hand_card=np.ones((1, 4), np.int64), x_hand_float=np.zeros((1, 4, 7)),
                    x_obj_float=np.zeros((0, C.OBJ_FLOATS)))
        player = dict(side=0, hand_deck_indices=[0, 1, 2, 3], deck_card_ids=[10, 11, 12, 13],
                      deck_form_flags=[2, 1, 0, 0], abilities=[])
        update_form_and_carriers(data, dict(players=[player]), 0, 0, [], 0, 0, int)
        np.testing.assert_array_equal(data['x_hand_card'], np.ones((1, 4)))
        np.testing.assert_array_equal(data['x_hand_float'][0, :, 4:6], [[1, 0], [0, 1], [0, 0], [0, 0]])


if __name__ == '__main__':
    unittest.main()
