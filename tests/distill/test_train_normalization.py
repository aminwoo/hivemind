import torch

from hivemind.distill.train import TERMS, concat_batches, evaluate, losses


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wdl = torch.nn.Parameter(torch.zeros(2, 3))

    def forward(self, planes):
        source = planes[:, 0].long()
        n = len(source)
        value = torch.zeros(n, 1)
        policy = torch.zeros(n, 2)
        return value, (policy, policy), None, self.wdl[source], value


def batch(source, labelled):
    n = len(labelled)
    target = torch.tensor([[1.0, 0.0]]).expand(n, 2)
    legal = torch.ones(n, 2, dtype=torch.bool)
    mask = torch.ones(n, dtype=torch.bool)
    return dict(planes=torch.full((n, 1), source), source=torch.full((n,), source),
                value=torch.ones(n), moves_left=torch.ones(n),
                wdl=torch.tensor([[0.0, 0.0, float(valid)] for valid in labelled]),
                policy=[(target, legal, mask), (target, legal, mask)])


class ToyDataset:
    device = torch.device('cpu')

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['value'])

    def batch(self, index):
        return {k: [tuple(x[index] for x in board) for board in v] if k == 'policy'
                else v[index] for k, v in self.data.items()}


def test_search_supervision_preserves_raw_anchor_gradient():
    raw, search = batch(0, [True] * 4), batch(1, [True, False])
    model = ToyModel()
    ones = {k: 1.0 for k in TERMS}
    raw_weights = {**ones, 'moves_left': 0.1}
    search_a = {**ones, 'wdl': 0.0, 'moves_left': 0.0, 'value': 0.0}
    mixed = concat_batches(raw, search)
    a, _, _ = losses(model, mixed, {'raw': raw_weights, 'search': search_a})
    grad_a = torch.autograd.grad(a, model.wdl)[0]
    b, _, _ = losses(model, mixed, {'raw': raw_weights,
                                 'search': {**search_a, 'wdl': 1.0, 'moves_left': 0.1}})
    grad_b = torch.autograd.grad(b, model.wdl)[0]
    torch.testing.assert_close(grad_a[0], grad_b[0])
    torch.testing.assert_close(grad_a[0, 2], torch.tensor(-2 / 3))
    torch.testing.assert_close(b - a, torch.tensor(3.0).log() + 0.1)
    assert grad_b[1].abs().sum() > 0


def test_validation_is_independent_of_batching_and_row_order():
    model = ToyModel()
    data = concat_batches(batch(0, [True, True, False, False]), batch(1, [False, True]))
    weights = {'raw': {k: 1.0 for k in TERMS},
               'search': {k: 0.25 for k in TERMS}}
    weights['search']['value'] = 0.0
    expected, _, _ = losses(model, data, weights)
    dataset = ToyDataset(data)
    metrics = evaluate(model, dataset, weights, batch_size=2)
    torch.testing.assert_close(torch.tensor(metrics['loss']), expected)
    reordered = ToyDataset(dataset.batch(torch.tensor([0, 2, 4, 1, 3, 5])))
    shuffled = evaluate(model, reordered, weights, batch_size=3)
    for name in metrics:
        assert abs(metrics[name] - shuffled[name]) < 1e-6
    raw = evaluate(model, ToyDataset(batch(0, [True, True, False, False])), weights)
    search = evaluate(model, ToyDataset(batch(1, [False, True])), weights)
    assert abs(metrics['loss'] - raw['loss'] - search['loss']) < 1e-6


def test_flat_weights_keep_global_masked_mean():
    model = ToyModel()
    data = concat_batches(batch(0, [True, False]), batch(1, [True, True]))
    with torch.no_grad():
        model.wdl[1] = torch.tensor([0.0, 0.0, 2.0])
    weights = {k: float(k == 'wdl') for k in TERMS}
    total, terms, _ = losses(model, data, weights)
    expected = (-torch.log_softmax(model.wdl, 1)[:, 2] * torch.tensor([1.0, 2.0])).sum() / 3
    torch.testing.assert_close(total, expected)
    torch.testing.assert_close(terms['wdl'], expected)
    metrics = evaluate(model, ToyDataset(data), weights, batch_size=2)
    assert abs(metrics['loss'] - float(expected.detach())) < 1e-6
