from types import SimpleNamespace

import numpy as np

from cr.diagnostics import Capture
from cr.observation import Vocab


def test_audit_normalizes_client_ids_without_counting_diagnostic_misses():
    rows = []
    capture = Capture.__new__(Capture)
    capture.emit = rows.append
    records = SimpleNamespace(aliases={71948640: 10000071}, characters={}, effects={7: {}},
                              card=lambda c: 203000023 if c == 306000023 else c)
    vocab = Vocab([203000023, 10000071, 7, 26000000])
    vocab(999)  # An actual inference miss must survive the audit unchanged.
    before = vocab.misses.copy()
    frame = dict(game_tick=100, entities=[
        dict(data_id=71948640, card_id=306000023, kind=0, hp=-1, us={'buffs': [[7], [8]]}),
        dict(data_id=123, card_id=-1, kind=1, hp=100),
    ], players=[dict(deck_card_ids=[26000000, -1, 0, 456])])
    capture.audit(frame, records, vocab, {'obj_side': np.array([1, 2])})
    assert vocab.misses == before
    assert rows[0]['unknown_vocab'] == [('data', 123), ('deck_card', 456), ('effect', 8)]
    assert rows[0]['missing_characters'] == [123]
    assert rows[0]['missing_effects'] == [8]
    assert frame['entities'][0]['data_id'] == 71948640
