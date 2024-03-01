from typing import List, Optional
from dataclasses import dataclass
import multiprocessing as mp
import os
import time
import json
import torch
from torch import nn
from torch.optim import SGD, lr_scheduler

import utils


input_vector_path = 'vectors_input'


class SICPOVM(nn.Module):
    def __init__(self, dim: int, hl: List[int]):
        super().__init__()
        self.input_layer = nn.Linear(2*dim, hl[0]*dim, dtype=torch.float64)
        self.hidden_layers = [
            nn.Linear(hl[i]*dim, hl[i+1]*dim, dtype=torch.float64)
            for i in range(len(hl) - 1)]
        self.output_layer = nn.Linear(hl[-1]*dim, 2*dim, dtype=torch.float64)

    def forward(self, x):
        x = torch.view_as_real(x)
        shape = x.shape
        x = x.reshape(-1, 2*shape[1])
        x = torch.sigmoid(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.sigmoid(layer(x))
        x = nn.functional.normalize(self.output_layer(x))
        x = x.reshape(shape)
        x = torch.view_as_complex(x)
        return x


class GMatrixLoss:
    def __init__(self, dim: int):
        self.dim = dim
        # self.indices = self.optimal_index_order()

    def optimal_index_order(self):
        indices = self.generate_indices()
        return self.sort_indices(indices)

    def generate_indices(self):
        d = self.dim
        num_indices = (d*(d+1))//2 - 1
        indices = torch.zeros(num_indices, 4, dtype=torch.int)
        i = 0
        for k in range(1, d):
            for l in range(k):
                indices[i] = torch.tensor([2, l, k, (l+k) % d])
                i += 1
        for k in range(1, d):
            indices[i] = torch.tensor([1, k, k, (2*k) % d])
            i += 1
        assert i == num_indices
        return indices

    def sort_indices(self, indices):
        sortd = torch.zeros_like(indices)
        sorted_list = sorted(indices, key=lambda x: (x[1], x[2], x[3]))
        for i, l in enumerate(sorted_list):
            sortd[i] = l
        return sortd

    def __call__(self, a: torch.Tensor):
        loss = 0.0
        for vector in a:
            loss += self.calculate_loss_one_vector_for_loop(vector)
        return loss

    def calculate_loss_one_vector_for_loop(self, a):
        d = self.dim
        A = torch.zeros(d, d, dtype=torch.complex128)
        for i in range(d):
            A[i] = a.roll(-i)
        A_conj = A.conj()
        loss = (a.abs()**4).sum()**2
        for k in range(1, d):
            for l in range(k):
                loss += 2*((a * A_conj[k] * A_conj[l] * A[(k+l) % d]).sum()).abs()**2
            loss += (a * A_conj[k]**2 * A[(2*k) % d]).sum().abs()**2
        bound = 2.0/(self.dim + 1)
        loss = loss - bound
        return loss

    def calculate_loss_one_vector_precomputed_indices(self, a):
        d = self.dim
        A = torch.zeros(d, d, dtype=torch.complex128)
        for i in range(d):
            A[i] = a.roll(-i)
        A_conj = A.conj()
        loss = (a.abs()**4).sum()**2
        for (factor, k, l, kl) in self.indices:
            loss += factor * ((a * A_conj[k] * A_conj[l] * A[kl]).sum()).abs()**2
        bound = 2.0/(self.dim + 1)
        loss = loss - bound
        return loss


def generate_input_vector(dim: int) -> None:
    x = torch.rand((dim, 2), dtype=torch.float64)
    x /= torch.linalg.vector_norm(x)
    x = torch.view_as_complex(x)
    fname = os.path.join(input_vector_path, f'{dim}.pt')
    torch.save(x, fname)


def load_input_vector(dim: int) -> torch.Tensor:
    fname = os.path.join(input_vector_path, f'{dim}.pt')
    return torch.load(fname).unsqueeze(0)


def generate_perturbed_batch(x: torch.Tensor) -> torch.Tensor:
    dim = x.shape[1]
    perturbation = 0.0001*x.mean()
    x_batch = x.repeat(dim, 1)
    for i in range(1, dim):
        x_batch[i, i] += perturbation
        x_batch[i] /= torch.linalg.vector_norm(x_batch[i])
    return x_batch


@dataclass
class FiducialResult:
    vector: torch.Tensor
    loss: torch.Tensor
    random_seed: int


def run_network_one_seed(
    dim: int,
    params,
    random_seed: int
) -> Optional[FiducialResult]:
    device = torch.device('cuda:0' if params['use_cuda'] else 'cpu')
    torch.manual_seed(random_seed)
    model = SICPOVM(dim, params['hidden_layers']).to(device)
    criterion = GMatrixLoss(dim)
    optimizer = SGD(model.parameters(), lr=params['lr'])
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=params['gamma'])

    x = load_input_vector(dim)
    if params['use_mini_batch']:
        x_batch = generate_perturbed_batch(x).to(device)
    x = x.to(device)

    losses = []
    current_loss = 1.0
    max_iter = dim*500
    for i in range(1, max_iter):
        optimizer.zero_grad()
        if params['use_mini_batch'] and current_loss > 1E-12:
            pred = model(x_batch)
        else:
            pred = model(x)
        loss = criterion(pred)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        current_loss = losses[-1]
        if current_loss < 1E-15:
            vec = model(x).detach().cpu()
            fid_res = FiducialResult(vec, torch.tensor(losses), random_seed)
            return fid_res
        if i % 100*dim == 0:
            scheduler.step()
    return None


def run_network(dim: int, params) -> Optional[FiducialResult]:
    max_init_vectors = 10*dim
    for i in range(1, max_init_vectors + 1):
        fid_res = run_network_one_seed(dim, params, random_seed=i)
        if fid_res is not None:
            return fid_res
    return None


def find_fiducial(dim: int, params) -> str:
    start = time.time()
    fid_res = run_network(dim, params)
    stop = time.time()
    elapsed = stop - start
    if fid_res is None:
        line = f'Dimension: {dim:3d} ' \
            + f'{elapsed:7.2f} s for max no. of starting vectors, ' \
            + 'SIC-POVM: False'
    else:
        a = fid_res.vector.squeeze(0).numpy()
        success = utils.verify_povm(
            utils.generate_WH_povm_from_fiducial(a), rel_tol=1E-6
        )
        utils.save_povm_to_file(a, params['save_path'])
        line = f'Dimension: {dim:3d} ' \
            + f'{elapsed:7.2f} s for ' \
            + f'{fid_res.random_seed:>10d} starting vectors, ' \
            + f'SIC-POVM: {success}'
    if params['print_progress']:
        print(line)

    # fig, ax = plt.subplots()
    # ax.set_yscale('log')
    # ax.plot(fid_res.loss)
    # fig.savefig(os.path.join(params['save_path'], f'Loss-dim-{dim}.pdf'))
    # plt.close(fig)
    return line


def run_subprocess(dims, params):
    lines = []
    for dim in dims:
        lines.append(find_fiducial(dim, params))
    return lines


def main(params):
    if not os.path.isdir(params['save_path']):
        os.mkdir(params['save_path'])
    dims = params['dims']
    num_procs = len(dims)
    with mp.Pool(num_procs) as pool:
        lines = pool.starmap(
            run_subprocess, [[proc_dims, params] for proc_dims in dims]
        )
    # flatten
    lines = [line for sub in lines for line in sub]
    with open(os.path.join(params['save_path'], 'timing.txt'), 'w') as f:
        f.write('\n'.join(lines))
    with open(os.path.join(params['save_path'], 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)


if __name__ == '__main__':
    parameters = {
        'use_cuda': False,
        'use_mini_batch': False,
        'print_progress': True,
        'hidden_layers': [3, 4, 3],
        'lr': 0.01,
        'gamma': 0.9,
        'save_path': 'output/2022-10-09-1800',
        'dims': [[i] for i in range(2, 21)]
    }
    # cProfile.run('train_network(dim=3, params=params)')
    main(parameters)
