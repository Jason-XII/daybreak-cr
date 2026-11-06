import ast
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import textwrap

import torch
from torch import nn

from experiments.history_compare import ddp_runner as original

BASE_STEP = 30000
WORLD = 16
WORKERS = 2
MBS = 4
SEED = 30000
MAX_SECONDS = 20 * 3600


def absolute_lr(settings, index):
    step = BASE_STEP + index
    value = settings['lr'] * min(1., (step + 1) / max(1, settings['warmup'])) * (
        1 - (1 - settings['final_lr']) * step / 42564)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Absolute LR outside the supported positive schedule')
    return value


def milestone(index, target):
    return index == target or index in (256, 512, 768, 1024, 3000, 3200) or index % 1000 == 0


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_checkpoint(saved, arm, protocol):
    if saved.get('arm') != arm or saved.get('phase') != 'direct':
        raise ValueError('Only this arm own direct-training checkpoint is accepted')
    if saved.get('real_ddp') is not True or saved.get('world_size') != WORLD:
        raise ValueError('Require a new-run 16-rank DDP checkpoint, not old final or virtual-rank state')
    index = saved.get('index')
    if type(index) is not int or index <= 0 or saved.get('step') != BASE_STEP + index:
        raise ValueError('Checkpoint absolute step/index mismatch')
    if not saved.get('optimizer') or not saved.get('model'):
        raise ValueError('Both model and its own AdamW state are required')
    if protocol.get('arm') != arm or protocol.get('seed') != SEED or protocol.get('world_size') != WORLD:
        raise ValueError('Origin arm/seed/world mismatch')
    if saved['settings'] != protocol['settings'] or saved['config'] != protocol['config']:
        raise ValueError('Checkpoint and origin protocol settings/config differ')
    settings = saved['settings']
    for key, expected in dict(batch=64, micro_batch=4, workers=2, chunk=1280, seed=0,
                              ability_labels=0, compile_blocks=1).items():
        if settings.get(key) != expected:
            raise ValueError(f'Unsupported frozen setting: {key}')
    if saved.get('fresh_optimizer') is not True:
        raise ValueError('Missing original fresh-AdamW lineage')
    for tensor in saved['model'].values():
        if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all()):
            raise ValueError('Nonfinite or invalid checkpoint model tensor')
    return index


def restore_optimizer(model, saved, *, fused=True):
    state = saved['optimizer']
    named = list(model.named_parameters())
    groups = state['param_groups']
    if len(groups) != 1 or len(groups[0]['params']) != len(named):
        raise ValueError('AdamW parameter registration/group mismatch')
    if saved.get('parameter_names') is not None and saved['parameter_names'] != [n for n, _ in named]:
        raise ValueError('Saved named parameter order changed')
    identifiers = groups[0]['params']
    if len(set(identifiers)) != len(identifiers) or set(state['state']) - set(identifiers):
        raise ValueError('Invalid optimizer parameter identifiers')
    group = groups[0]
    if group.get('weight_decay') != saved['settings']['weight_decay'] or group.get('maximize', False):
        raise ValueError('AdamW objective/decay contract changed')
    if group.get('betas') != (0.9, 0.999) or group.get('eps') != 1e-8 or group.get('amsgrad', False):
        raise ValueError('Unexpected inherited AdamW hyperparameters')
    steps = []
    for identifier, (name, parameter) in zip(identifiers, named):
        row = state['state'].get(identifier)
        if row is None:
            continue
        for key in ('exp_avg', 'exp_avg_sq'):
            if key not in row or row[key].shape != parameter.shape or not bool(torch.isfinite(row[key]).all()):
                raise ValueError(f'Invalid AdamW {key} for {name}')
        step = float(row['step'])
        if not math.isfinite(step) or step != int(step) or not 1 <= step <= saved['index']:
            raise ValueError('AdamW counter outside this checkpoint history')
        steps.append(int(step))
    if not steps or max(steps) != saved['index']:
        raise ValueError('AdamW counters do not reach the checkpoint update')
    # Sparse heads may have fewer updates; never rewrite their individual step counters.
    optimizer = torch.optim.AdamW(model.parameters(), lr=saved['settings']['lr'],
                                  weight_decay=saved['settings']['weight_decay'], fused=fused)
    optimizer.load_state_dict(copy.deepcopy(state))
    if optimizer.param_groups[0].get('fused', False) != fused:
        if fused:
            raise ValueError('Production continuation requires the inherited fused AdamW')
        optimizer.param_groups[0]['fused'] = False
    return optimizer


def verify_order(document, entries, rank, origin_rank):
    if document['seed'] != SEED or document['rank'] != rank:
        raise ValueError('Order seed/rank mismatch')
    batches = document['batches']
    if not batches or any(len(micro) != MBS for micro in batches):
        raise ValueError('Recorded plan violates MBS4')
    checksum = canonical_hash(batches)
    if checksum != document['sha256'] or checksum != origin_rank['order_sha256']:
        raise ValueError('Order hash differs from origin rank receipt')
    generated = original.rank_order(entries, SEED, rank)
    if generated[:len(batches)] != batches:
        raise ValueError('Manifest-derived rank/worker plan differs from saved order')
    if any(e['decisions'] > 1280 for micro in batches for e in micro):
        raise ValueError('Random-offset chunking is not supported')
    return batches


class OffsetGames(original.Games):
    def __init__(self, data, batches, offset, end, rank, cannon=0):
        super().__init__(data, [], 1280, SEED, rank=rank, world=WORLD, cannon_index=cannon)
        if not 0 <= offset <= end <= len(batches):
            raise ValueError('Offset/end outside saved rank stream')
        self.batches, self.offset, self.end = batches, offset, end

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        if info is None or info.num_workers != WORKERS:
            raise ValueError('Exact offset loader requires two workers')
        # DataLoader restarts at physical worker zero. Rotate logical workers at odd offsets.
        logical = (self.offset + info.id) % WORKERS
        rng = random.Random(SEED * 1000 + self.rank * WORKERS + logical)
        for index in range(self.offset + info.id, self.end, WORKERS):
            yield from self._block(self.batches[index], rng)


def offset_loader(data, batches, offset, end, rank, cannon=0, *, dataset_class=OffsetGames,
                  collate_fn=original.collate, pin_memory=True):
    generator = torch.Generator().manual_seed(SEED + rank)
    return torch.utils.data.DataLoader(dataset_class(data, batches, offset, end, rank, cannon),
        batch_size=MBS, num_workers=WORKERS, collate_fn=collate_fn, pin_memory=pin_memory,
        generator=generator, in_order=True, multiprocessing_context='spawn')


def rng_audit(model):
    dropout = []
    risky_calls, sdpa = [], []
    inspected = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.modules.dropout._DropoutNd, nn.AlphaDropout, nn.FeatureAlphaDropout)):
            dropout.append(dict(name=name, type=type(module).__name__, p=float(module.p)))
        if isinstance(module, (nn.RNNBase, nn.MultiheadAttention)) and getattr(module, 'dropout', 0):
            dropout.append(dict(name=name, type=type(module).__name__, p=float(module.dropout)))
        functions = [type(module).forward]
        if module is model:
            functions.append(model.tiles)
        for function in functions:
            if function in inspected:
                continue
            inspected.add(function)
            try:
                text = textwrap.dedent(inspect.getsource(function))
            except (OSError, TypeError):
                continue
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
                if func in ('dropout', 'dropout2d', 'dropout3d', 'rand', 'randn', 'rand_like', 'randn_like',
                            'bernoulli', 'multinomial', 'randint', 'randperm', 'random', 'stochastic_depth'):
                    risky_calls.append(dict(function=function.__qualname__, call=func))
                if func == 'scaled_dot_product_attention':
                    value = next((k.value for k in node.keywords if k.arg == 'dropout_p'), ast.Constant(0.))
                    probability = value.value if isinstance(value, ast.Constant) else 'dynamic'
                    sdpa.append(dict(function=function.__qualname__, dropout_p=probability))
    safe = not any(row['p'] for row in dropout) and not risky_calls and all(row['dropout_p'] == 0 for row in sdpa)
    return dict(no_stochastic_training_path_found=safe, dropout_modules=dropout, stochastic_calls=risky_calls,
                sdpa=sdpa, inspected_forwards=len(inspected),
                scope='Instantiated modules and forward/tiles source, plus separately verified full-history deterministic loader; not a proof of CUDA bitwise determinism')


def elapsed_credit(saved, checkpoint, parent, explicit=None):
    if 'continuation' in saved:
        previous = float(saved['continuation']['cumulative_seconds'])
        evidence = 'checkpoint.continuation.cumulative_seconds'
        for name in ('finished.json', 'latest-checkpoint.json'):
            path = parent / name
            if path.exists():
                previous = max(previous, float(json.loads(path.read_text()).get('cumulative_seconds', 0)))
        if (parent / 'train.jsonl').exists():
            for line in (parent / 'train.jsonl').read_text().splitlines():
                try:
                    previous = max(previous, float(json.loads(line).get('cumulative_seconds', 0)))
                except json.JSONDecodeError:
                    continue
        evidence += '; max with parent cumulative receipts including rollback work'
    else:
        values = []
        terminal = parent / 'finished.json'
        if terminal.exists():
            finished = json.loads(terminal.read_text())
            if finished.get('index', -1) >= saved['index']:
                values.append(float(finished['seconds']))
        log = parent / 'train.jsonl'
        if log.exists():
            for line in log.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'seconds' in row:
                    values.append(float(row['seconds']))
        if not values and explicit is None:
            raise ValueError('Parent timing evidence missing; explicit charged seconds required')
        previous = max(values, default=0.)
        evidence = 'max(parent finished.seconds, all logged seconds); includes work after an older checkpoint'
        # Include the post-timer saving/evaluation interval where file timestamps permit a conservative bound.
        launches = list(parent.parent.glob(f'{saved["arm"]}-node*-process.json'))
        if launches:
            start = min(float(json.loads(p.read_text())['started']) for p in launches)
            end = max([Path(checkpoint).stat().st_mtime, *(p.stat().st_mtime for p in (terminal, log) if p.exists())])
            previous = max(previous, end - start)
            evidence += '; conservative launcher-to-last-artifact wall time'
    if explicit is not None:
        if not math.isfinite(explicit) or explicit < previous:
            raise ValueError('Explicit elapsed credit cannot undercharge available evidence')
        previous = explicit
        evidence += '; explicit conservative credit'
    if not math.isfinite(previous) or previous < 0:
        raise ValueError('Invalid cumulative elapsed time')
    return previous, evidence


def time_status(previous, segment, target_reached, total=MAX_SECONDS):
    if not 0 < total <= MAX_SECONDS:
        raise ValueError('Cumulative budget must be at most 20 hours')
    cumulative = previous + segment
    return dict(status='complete' if target_reached else 'bounded_time_limit', cumulative_seconds=cumulative,
                remaining_seconds=max(0., total - cumulative), numerical_failure=False)
