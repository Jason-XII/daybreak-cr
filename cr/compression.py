import numpy as np


OBJECT_KEYS = ('obj_card', 'obj_data', 'obj_side', 'obj_kind', 'obj_float', 'eff_id', 'eff_left', 'obj_count')
CANNON_BARRAGE = 2119760969


def object_keys(x):
    return OBJECT_KEYS + (('v2_entity',) if 'v2_entity' in x else ())


def grouped_v2(values, groups):
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError('V2 entity features require two member slots')
    if np.any(values[:, 1] != 0):
        raise ValueError('Cannot recursively merge v2 entity pairs')
    output = np.zeros((len(groups), 2, values.shape[-1]), values.dtype)
    for index, members in enumerate(groups):
        for slot, member in enumerate(members):
            output[index, slot] = values[member, 0]
    return output


def pairs(x, eligible, coordinates, radius, unit=False):
    if not eligible.any():
        return []
    f = x['obj_float']
    columns = [x[k] for k in ('obj_side', 'obj_card', 'obj_data')]
    if unit:
        columns += [(f[:, j] > 0).astype(np.int64) for j in (6, 7, 12, 13, 14, 15, 16, 17, 30, 31, 33, 34)]
        columns += [f[:, 32]]
    identity = np.column_stack(columns + [x['eff_id']])
    result = []
    for key in np.unique(identity[eligible], axis=0):
        members = np.flatnonzero(eligible & (identity == key).all(1))
        i, j = np.triu_indices(len(members), 1)
        a, b = members[i], members[j]
        distance = np.linalg.norm(coordinates[a] - coordinates[b], axis=1)
        order = np.lexsort((b, a, distance))
        used = set()
        for k in order[distance[order] < radius]:
            u, v = int(a[k]), int(b[k])
            if u not in used and v not in used:
                result.append([u, v])
                used.update((u, v))
    return result


def compress_frame(inputs, cannon_index):
    n = len(inputs['obj_side'])
    x = {k: np.asarray(v) for k, v in inputs.items()}
    if n <= 40:
        return x | {'obj_count': np.ones(n, np.float32)}, dict(original=n, special_drop=0, pairs=0, cap_drop=0)
    f = x['obj_float']
    if not np.isfinite(f).all():
        raise ValueError('Nonfinite entity input')
    pos = f[:, :2].astype(np.float64) * [18, 32]
    aim = pos + f[:, 28:30].astype(np.float64) * 20
    kind, side = x['obj_kind'], x['obj_side']
    special = (kind == 4) & (x['obj_data'] == cannon_index)
    removed = set()
    # Select real corner entities independently per side; never synthesize counts.
    for owner in np.unique(side[special]):
        members = np.flatnonzero(special & (side == owner))
        if len(members) <= 4:
            continue
        lo, hi = pos[members].min(0), pos[members].max(0)
        corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
        kept = []
        for corner in corners:
            candidates = [i for i in members if int(i) not in kept]
            kept.append(int(min(candidates, key=lambda i: (float(np.linalg.norm(pos[i]-corner)), int(i)))))
        removed.update(int(i) for i in members if int(i) not in kept)
    eligible = np.array([i not in removed for i in range(n)]) & ~special
    known = (x['obj_card'] > 1) & (x['obj_data'] > 1)
    up = pairs(x, eligible & known & (kind == 1) & (f[:, -2] == 0), pos, 1.0, unit=True)
    pp = pairs(x, eligible & known & (kind == 4) & (f[:, 28:30] != 0).any(1), aim, 1.5)
    paired = up + pp
    used = {i for g in paired for i in g}
    groups = paired + [[i] for i in range(n) if i not in used and i not in removed]
    groups.sort(key=min)
    first = np.array([g[0] for g in groups])
    result = {k: np.stack([v[g].mean(0) for g in groups]).astype(v.dtype)
              if k in ('obj_float', 'eff_left') else v[first].copy() for k, v in x.items()}
    if 'v2_entity' in x:
        result['v2_entity'] = grouped_v2(x['v2_entity'], groups)
    result['obj_count'] = np.array([len(g) for g in groups], np.float32)
    mf = result['obj_float']
    mpos = mf[:, :2].astype(np.float64)*[18,32]
    maim = mpos + mf[:, 28:30].astype(np.float64)*20
    mk, ms = result['obj_kind'], result['obj_side']
    keep = list(range(len(groups)))
    protected_special = result['obj_data'] == cannon_index
    units = np.flatnonzero((kind == 1) | (kind == 2))
    while len(keep) > 60:
        priorities = []
        for i in keep:
            protected = mk[i] == 2 or mf[i,-2] > 0 or mf[i,6] > 0 or protected_special[i]
            tier = 3 if protected else 0 if mk[i] == 4 else 1 if mk[i] in (3,5) else 2
            located_aim = mk[i] == 4 and (mf[i,28:30] != 0).any()
            p = maim[i] if located_aim else mpos[i]
            same = [j for j in keep if j != i and mk[j] == mk[i] and ms[j] == ms[i]
                    and result['obj_card'][j] == result['obj_card'][i]
                    and result['obj_data'][j] == result['obj_data'][i]]
            nearest = min((float(np.linalg.norm(p-(maim[j] if located_aim else mpos[j]))) for j in same), default=1000.)
            enemies = units[side[units] != ms[i]]
            combat = float(np.linalg.norm(pos[enemies]-p,axis=1).min()) if len(enemies) else 1000.
            priorities.append(((tier, nearest, -combat, result['obj_count'][i], i),i))
        keep.remove(min(priorities)[1])
    cap_drop = len(groups)-len(keep)
    return {k:v[keep] for k,v in result.items()}, dict(original=n, special_drop=len(removed),
                                                       pairs=len(paired), cap_drop=cap_drop)


def compress_chunk(x, cannon_index):
    rows, stats = [], []
    for t in range(len(x['obj_side'])):
        n = int((x['obj_side'][t] > 0).sum())
        row, telemetry = compress_frame({k:x[k][t,:n] for k in object_keys(x) if k != 'obj_count'}, cannon_index)
        rows.append(row)
        stats.append(telemetry)
    width = max(1, max(len(r['obj_side']) for r in rows))
    result = dict(x)
    for k in object_keys(x):
        exemplar = rows[0][k]
        a = np.zeros((len(rows), width, *exemplar.shape[1:]), exemplar.dtype)
        for t,row in enumerate(rows):
            a[t,:len(row[k])] = row[k]
        result[k] = a
    return result, stats
