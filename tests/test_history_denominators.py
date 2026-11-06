from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cr import history_imitation as I, observation as C


class Heads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.what = torch.nn.Parameter(torch.randn(4))
        self.which = torch.nn.Parameter(torch.randn(4))
        self.tile = torch.nn.Parameter(torch.randn(C.CELLS))
        self.value = torch.nn.Parameter(torch.randn(()))
        self.when = torch.nn.Parameter(torch.randn(len(C.NEXT_EDGES) + 1))
        self.next = torch.nn.Parameter(torch.randn(4))

    def forward(self, x):
        b, t = x['scalars'].shape[:2]
        return dict(what=self.what.expand(b, t, 4), which=self.which.expand(b, t, 4),
                    core=self.value.expand(b * t, 1), value=self.value.expand(b, t),
                    next_when=self.when.expand(b, t, -1), next_which=self.next.expand(b, t, 4))

    def tiles(self, output, x, slot):
        return self.tile.expand(len(slot), -1)


def data():
    x = dict(scalars=torch.zeros(2, 2, C.SCALARS), hand_card=torch.full((2, 2, 4), 2),
             legal=torch.ones(2, 2, 4, C.H, C.W, dtype=torch.bool))
    t = dict(time=torch.ones(2, 2, dtype=torch.bool), ok=torch.ones(2, 2, dtype=torch.bool),
             what=torch.tensor([[C.PLAY, C.WAIT], [C.WAIT, C.WAIT]]),
             which=torch.zeros(2, 2, dtype=torch.long), which_ok=torch.ones(2, 2, dtype=torch.bool),
             tile=torch.zeros(2, 2, dtype=torch.long), value=torch.ones(2, 2),
             next_when=torch.zeros(2, 2, dtype=torch.long), next_which=torch.zeros(2, 2, dtype=torch.long),
             next_which_ok=torch.tensor([[True, True], [False, False]]))
    return x, t


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.manual_seed(1701)
    reference = Heads().train()
    x, t = data()
    I.loss_and_metrics(reference, x, t, .5, .5)[0].backward()
    model = Heads().train()
    model.load_state_dict(reference.state_dict())
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        objective = torch.nn.parallel.DistributedDataParallel(I.Objective(model, .5, .5),
                                                             find_unused_parameters=True)
        local_x, local_t = ({key: value[rank:rank + 1] for key, value in group.items()} for group in (x, t))
        objective(local_x, local_t)[0].backward()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-7)
    finally:
        dist.destroy_process_group()


def test_global_head_denominators_with_zero_label_rank(tmp_path):
    mp.spawn(worker, args=(str(tmp_path / 'gloo-store'),), nprocs=2, join=True)
