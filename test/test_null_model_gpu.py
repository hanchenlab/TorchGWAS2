"""pymodules/NullModelGPU.py: the batched fit against a dense one-phenotype-at-a-time reference, and the
step 1 file rules.

    python -m pytest test/test_null_model_gpu.py

Needs numpy, pandas, pyarrow and torch, not the C++ module: the GPU module is
loaded on its own. The fit runs on the CPU too; the step 1 test needs CUDA.
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    'NullModelGPU', Path(__file__).resolve().parents[1] / 'pymodules' / 'NullModelGPU.py')
gpu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gpu)

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def _reference(y_all, covariates, kin_dense, tol=1e-5, max_iter=500):
    """GMMAT's AI-REML per phenotype with Sigma inverted densely over its observed samples."""
    n, traits = y_all.shape
    x_all = gpu.design_matrix(covariates, n)
    trace_all = np.trace(kin_dense)
    out = dict(pseudo=np.zeros((n, traits)), c1=np.zeros(traits), c2=np.zeros(traits), tau=np.zeros((traits, 2)))
    for t in range(traits):
        keep = ~np.isnan(y_all[:, t])
        y, x, kin = y_all[keep, t], x_all[keep], kin_dense[np.ix_(keep, keep)]
        m = x.shape[0]

        def state(tau):
            si = np.linalg.inv(tau[0] * np.eye(m) + tau[1] * kin)
            six = si @ x
            cov = np.linalg.inv(x.T @ six)
            p_mat = si - six @ cov @ six.T
            return cov @ six.T @ y, p_mat, p_mat @ y

        fixed = np.zeros(2, bool)
        while True:
            variance = y.var(ddof=1)
            tau = np.where(fixed, 0.0, [variance / 2, variance / 2 / kin.diagonal().mean()])
            alpha = np.linalg.solve(x.T @ x, x.T @ y)
            _, p_mat, py = state(tau)
            score = np.array([py @ py - np.trace(p_mat), py @ kin @ py - np.trace(p_mat @ kin)])
            tau = np.where(fixed, tau, np.maximum(0, tau + tau ** 2 * score / m))
            for _ in range(max_iter):
                tau0, alpha0 = tau.copy(), alpha.copy()
                alpha, p_mat, py = state(tau0)
                kpy = kin @ py
                ai = np.array([[py @ p_mat @ py, py @ p_mat @ kpy], [py @ p_mat @ kpy, kpy @ p_mat @ kpy]])
                score = np.array([py @ py - np.trace(p_mat), py @ kpy - np.trace(p_mat @ kin)])
                free = ~fixed
                step = np.zeros(2)
                step[free] = np.linalg.solve(ai[np.ix_(free, free)], score[free])

                def bounded(delta):
                    new = tau0 + delta
                    new[(new < tol) & (tau0 < tol)] = 0
                    return new
                tau = bounded(step)
                while (tau < 0).any():
                    step = step / 2
                    tau = bounded(step)
                tau[tau < tol] = 0
                py0, p0, tau0_e = py, p_mat, tau0[0]
                change = max(np.max(np.abs(alpha - alpha0) / (np.abs(alpha) + np.abs(alpha0) + tol)),
                             np.max(np.abs(tau - tau0) / (np.abs(tau) + np.abs(tau0) + tol)))
                if 2 * change < tol:
                    break
            newly = (tau < 1.01 * tol) & ~fixed
            if not newly.any():
                break
            fixed |= newly
        r = py0 * (1.0 if tau0_e == 0 and tau[0] == 0 else tau0_e / tau[0])
        c1 = trace_all / np.trace(p0 @ kin)
        out['pseudo'][keep, t] = c1 * r
        out['c1'][t], out['c2'][t] = c1, c1 * (r @ r) / (n - 1)
        out['tau'][t] = tau
    return out


def _expit(eta):
    return np.exp(eta) / (1.0 + np.exp(eta))


def _reference_binary(y_all, covariates, kin_dense, tol=1e-5, max_iter=500):
    """The C++ step 1 for binary phenotypes, one at a time with dense matrices, written as the C++ is.

    fitNullModel2's logistic regression; then glmmkin_fit_cs without a kinship,
    or glmmkin_ai (binomial: tau_e fixed at 1) and its boundary refit with one.
    """
    n, traits = y_all.shape
    x_all = gpu.design_matrix(covariates, n)
    out = dict(pseudo=np.zeros((n, traits)), c1=np.zeros(traits), c2=np.zeros(traits), tau=np.zeros((traits, 2)))
    for t in range(traits):
        keep = ~np.isnan(y_all[:, t])
        y, x = y_all[keep, t], x_all[keep]
        size = x.shape[0]
        beta = np.linalg.inv(x.T @ x) @ (x.T @ y)
        for _ in range(499):
            eta = x @ beta
            mu = _expit(eta)
            w = mu * (1 - mu)
            new = np.linalg.inv(x.T @ (w[:, None] * x)) @ (x.T @ (w * (eta + (y - mu) / w)))
            done = np.all(np.abs(new - beta) <= tol)
            beta = new
            if done:
                break
        else:
            raise AssertionError('logistic regression did not converge')
        if kin_dense is None:
            mu = _expit(x @ beta)
            w = mu * (1 - mu)
            cov = np.linalg.inv(x.T @ (w[:, None] * x))
            six = w[:, None] * x
            c1 = n / (w.sum() - np.sum(cov * (six.T @ six)))
            r, tau_g = y - mu, 0.0
        else:
            kin = kin_dense[np.ix_(keep, keep)]

            def fitglmm_ai(tau_g, w, big_y):
                si = np.linalg.inv(np.diag(1 / w) + tau_g * kin)
                six = si @ x
                cov = np.linalg.inv(x.T @ six)
                alpha = cov @ (six.T @ big_y)
                eta = big_y - (1 / w) * (si @ big_y - six @ alpha)
                p_mat = si - six @ cov @ six.T
                return si, six, cov, alpha, eta, p_mat

            def tr_pk(si, six, cov):
                return np.sum(si * kin) - np.sum(six * (kin @ six @ cov))

            def glmmkin_ai(fixed):
                alpha, eta = beta.copy(), x @ beta
                mu = _expit(eta)
                w = mu * (1 - mu)
                big_y = eta + (y - mu) / w
                tau_g = 0.0
                if not fixed:
                    tau_g = big_y.var(ddof=1) / 2 / kin.diagonal().mean()
                    si, six, cov, _, _, p_mat = fitglmm_ai(tau_g, w, big_y)
                    papy = p_mat @ (kin @ (p_mat @ big_y))
                    tau_g = max(0.0, tau_g + tau_g ** 2 * (big_y @ papy - tr_pk(si, six, cov)) / size)
                for _ in range(max_iter):
                    alpha0, tau0 = alpha, tau_g
                    si, six, cov, alpha, eta, p_mat = fitglmm_ai(tau0, w, big_y)
                    if not fixed:
                        py = p_mat @ big_y
                        papy = p_mat @ (kin @ py)
                        dtau = (big_y @ papy - tr_pk(si, six, cov)) / (py @ kin @ papy)
                        tau_g = tau0 + dtau
                        tau_g = 0.0 if tau_g < tol and tau0 < tol else tau_g
                        while tau_g < 0:
                            dtau /= 2
                            tau_g = tau0 + dtau
                            tau_g = 0.0 if tau_g < tol and tau0 < tol else tau_g
                        tau_g = 0.0 if tau_g < tol else tau_g
                    mu = _expit(eta)
                    w = mu * (1 - mu)
                    big_y = eta + (y - mu) / w
                    change = max(np.max(np.abs(alpha - alpha0) / (np.abs(alpha) + np.abs(alpha0) + tol)),
                                 abs(tau_g - tau0) / (abs(tau_g) + abs(tau0) + tol))
                    if 2 * change < tol or tau_g > tol ** -2:
                        break
                return tau_g, y - mu, np.trace(kin_dense) / tr_pk(si, six, cov)

            tau_g, r, c1 = glmmkin_ai(False)
            if tau_g < 1.01 * tol:
                tau_g, r, c1 = glmmkin_ai(True)
        out['pseudo'][keep, t] = c1 * r
        out['c1'][t], out['c2'][t] = c1, c1 * (r @ r) / (n - 1)
        out['tau'][t] = 1.0, tau_g
    return out


def _binary(y, seed=5):
    """0/1 phenotypes from _panel's: each column above its own quantile, from 15% to 50% cases."""
    rng = np.random.default_rng(seed)
    out = np.full_like(y, np.nan)
    for t in range(y.shape[1]):
        keep = ~np.isnan(y[:, t])
        cut = np.quantile(y[keep, t], 1 - rng.uniform(0.15, 0.5))
        out[keep, t] = (y[keep, t] > cut).astype(float)
    return out


def _panel(seed=7, n=240, traits=6, missing=0.15):
    """Families of 1-6 (2 x kinship: 1 on the diagonal, 0.5 between members), covariates and phenotypes."""
    rng = np.random.default_rng(seed)
    ids = [f's{i}' for i in range(n)]
    rows = []
    start = 0
    while start < n:
        size = min(n - start, int(rng.integers(1, 7)))
        for i in range(start, start + size):
            rows.append((ids[i], ids[i], 1.0))
            for j in range(i + 1, start + size):
                rows.append((ids[i], ids[j], 0.5))
        start += size
    dense = np.zeros((n, n))
    for a, b, v in rows:
        i, j = int(a[1:]), int(b[1:])
        dense[i, j] = dense[j, i] = v
    covariates = np.column_stack([rng.normal(size=n), rng.integers(0, 2, n)]).astype(np.float64)
    chol = np.linalg.cholesky(dense)
    y = np.empty((n, traits))
    for t, h in enumerate(np.linspace(0.0, 0.8, traits)):
        y[:, t] = 1 + covariates @ [0.3, -0.5] + np.sqrt(h) * chol @ rng.normal(size=n) + \
            np.sqrt(1 - h) * rng.normal(size=n)
    y[rng.random(y.shape) < missing] = np.nan
    return ids, rows, dense, covariates, y


def _kinship(ids, rows, diagonal=0.0):
    index = {s: i for i, s in enumerate(ids)}
    diag = np.full(len(ids), diagonal)
    pairs = {}
    for a, b, v in rows:
        if a == b:
            diag[index[a]] += v
        else:
            pairs[(index[a], index[b])] = v
    keys = np.asarray(list(pairs), dtype=np.int64)
    return gpu.Kinship(len(ids), diag, keys[:, 0], keys[:, 1], np.asarray(list(pairs.values())))


@pytest.mark.parametrize('device', DEVICES)
def test_the_batched_fit_matches_the_dense_reference(device):
    ids, rows, dense, covariates, y = _panel()
    fit = gpu.fit_null_model(y, covariates, _kinship(ids, rows), torch.device(device))
    want = _reference(y, covariates, dense)
    np.testing.assert_allclose(fit['tau'], want['tau'], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(fit['c1'], want['c1'], rtol=1e-8)
    np.testing.assert_allclose(fit['c2'], want['c2'], rtol=1e-8)
    np.testing.assert_allclose(fit['pseudo'], want['pseudo'], rtol=1e-7, atol=1e-10)
    assert fit['converged'].all()
    assert np.all(fit['pseudo'][np.isnan(y)] == 0)


@pytest.mark.parametrize('device', DEVICES)
def test_without_kinship_the_fit_is_ols_in_closed_form(device):
    ids, rows, dense, covariates, y = _panel(seed=3)
    fit = gpu.fit_null_model(y, covariates, None, torch.device(device))
    n = len(ids)
    x = gpu.design_matrix(covariates, n)
    for t in range(y.shape[1]):
        keep = ~np.isnan(y[:, t])
        residual = y[keep, t] - x[keep] @ np.linalg.lstsq(x[keep], y[keep, t], rcond=None)[0]
        np.testing.assert_allclose(fit['pseudo'][keep, t], n * residual / (keep.sum() - x.shape[1]), rtol=1e-9)
    np.testing.assert_allclose(fit['c2'], n / (n - 1))


@pytest.mark.parametrize('device', DEVICES)
def test_binary_phenotypes_match_the_dense_cpp_reference(device):
    ids, rows, dense, covariates, y = _panel(seed=0, n=600, traits=8)
    # Heritable phenotypes, and pure noise (30% cases), which puts tau_g at zero.
    noise = np.random.default_rng(0).random(y.shape) < 0.3
    y = np.where(np.isnan(y), np.nan, np.column_stack([_binary(y[:, :4]), noise[:, 4:]]))
    fit = gpu.fit_null_model(y, covariates, _kinship(ids, rows), torch.device(device))
    want = _reference_binary(y, covariates, dense)
    # Both sides of the boundary: some tau_g refitted at zero, some not.
    assert (want['tau'][:, 1] == 0).any() and (want['tau'][:, 1] > 0).any()
    assert fit['binary'].all() and fit['converged'].all()
    np.testing.assert_allclose(fit['tau'], want['tau'], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(fit['c1'], want['c1'], rtol=1e-8)
    np.testing.assert_allclose(fit['c2'], want['c2'], rtol=1e-8)
    np.testing.assert_allclose(fit['pseudo'], want['pseudo'], rtol=1e-7, atol=1e-10)
    assert np.all(fit['pseudo'][np.isnan(y)] == 0)


@pytest.mark.parametrize('device', DEVICES)
def test_binary_phenotypes_without_kinship_are_the_logistic_regression(device):
    ids, rows, dense, covariates, y = _panel(seed=19)
    y = _binary(y)
    fit = gpu.fit_null_model(y, covariates, None, torch.device(device))
    want = _reference_binary(y, covariates, None)
    np.testing.assert_allclose(fit['tau'], want['tau'])
    np.testing.assert_allclose(fit['c2'], want['c2'], rtol=1e-8)
    np.testing.assert_allclose(fit['pseudo'], want['pseudo'], rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize('device', DEVICES)
def test_quantitative_and_binary_phenotypes_fit_together(device):
    ids, rows, dense, covariates, y = _panel(seed=23, n=300, traits=6)
    y[:, 1::2] = _binary(y[:, 1::2])
    kin = _kinship(ids, rows)
    fit = gpu.fit_null_model(y, covariates, kin, torch.device(device))
    np.testing.assert_array_equal(fit['binary'], [False, True] * 3)
    for columns in ([0, 2, 4], [1, 3, 5]):
        alone = gpu.fit_null_model(y[:, columns], covariates, kin, torch.device(device))
        for key in ('pseudo', 'c2', 'tau'):
            np.testing.assert_allclose(fit[key][..., columns] if key == 'pseudo' else fit[key][columns],
                                       alone[key], rtol=1e-10, atol=1e-12)


CUDA_UNION_FIND = torch.cuda.is_available() and gpu.union_find_available('cuda')


def _labellers():
    out = [('hook-to-root, cpu', 'cpu', lambda r, c, n: gpu.hook_to_root(r, c, n)[0])]
    if torch.cuda.is_available():
        out.append(('hook-to-root, cuda', 'cuda', lambda r, c, n: gpu.hook_to_root(r, c, n)[0]))
    if CUDA_UNION_FIND:
        out.append(('union-find, cuda', 'cuda', gpu.union_find))
    return out


def _components_reference(rows, cols, n):
    """Union-find in Python; each node's label is its component's smallest node."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for u, v in zip(rows.tolist(), cols.tolist()):
        a, b = find(u), find(v)
        if a != b:
            parent[max(a, b)] = min(a, b)
    return np.array([find(x) for x in range(n)])


def _graphs():
    rng = np.random.default_rng(3)
    order = rng.permutation(200_000)               # a chain visiting the nodes in random order
    yield 'long chain', order[:-1], order[1:], 200_000
    yield 'star', np.full(49_999, 25_000), np.delete(np.arange(50_000), 25_000), 50_000
    yield 'random', rng.integers(0, 5_000, 4_000), rng.integers(0, 5_000, 4_000), 5_000
    pairs = [(a + i, a + j) for a in range(0, 3_000, 6) for i in range(6) for j in range(i + 1, 6)
             if rng.random() < 0.5]
    yield 'families, both orders and repeats', np.array([p[0] for p in pairs] + [p[1] for p in pairs[:50]]), \
        np.array([p[1] for p in pairs] + [p[0] for p in pairs[:50]]), 3_000
    yield 'no pairs', np.zeros(0, np.int64), np.zeros(0, np.int64), 10


@pytest.mark.parametrize('labeller', _labellers(), ids=lambda item: item[0])
@pytest.mark.parametrize('graph', list(_graphs()), ids=lambda g: g[0])
def test_component_labels_are_each_components_smallest_sample(labeller, graph):
    _, device, label = labeller
    _, rows, cols, n = graph
    labels = label(torch.as_tensor(rows, device=device), torch.as_tensor(cols, device=device), n)
    np.testing.assert_array_equal(labels.cpu().numpy(), _components_reference(np.asarray(rows), np.asarray(cols), n))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_the_union_find_runs_on_this_gpu():
    major, minor = torch.cuda.get_device_capability()
    assert CUDA_UNION_FIND, f'the prebuilt union-find (pymodules/native) does not launch on sm_{major}{minor}'


@pytest.mark.parametrize('device', DEVICES)
def test_families_are_the_same_whichever_labelling_runs(device, monkeypatch):
    ids, rows, dense, covariates, y = _panel(seed=31, n=400)
    kin = _kinship(ids, rows)
    default = kin.families(torch.device(device))
    monkeypatch.setattr(gpu, 'union_find_available', lambda device: False)
    fallback = kin.families(torch.device(device))
    assert torch.equal(default['diagonal'], fallback['diagonal'])
    for (size, index, blocks), (size2, index2, blocks2) in zip(default['groups'], fallback['groups']):
        assert size == size2 and torch.equal(index, index2) and torch.equal(blocks, blocks2)


def test_kinship_rows_follow_the_cpp_rules(tmp_path):
    path = tmp_path / 'kin.txt'
    path.write_text('ID1\tID2\tkinship\n'
                    'a\ta\t0.2\n' 'a\ta\t0.7\n'              # a repeated diagonal row counts once
                    'a\tb\t0.25\n' 'a\tb\t0.9\n'             # a repeated ordered pair counts once
                    'b\ta\t0.05\n'                           # the other order adds
                    '"c"\t"d"\t0.125\n' 'c\tzz\t0.5\n')       # quotes stripped; unknown IDs dropped
    kin = gpu.read_kinship(path, '\t', ['a', 'b', 'c', 'd'], diagonal=1.0)
    np.testing.assert_allclose(kin.diagonal, [1.2, 1.0, 1.0, 1.0])
    pairs = {(int(i), int(j)): v for i, j, v in zip(kin.rows, kin.cols, kin.values)}
    assert pairs.keys() == {(0, 1), (2, 3)}
    np.testing.assert_allclose([pairs[(0, 1)], pairs[(2, 3)]], [0.30, 0.125])


def _step1_files(tmp_path, binary=None, duplicate=False):
    ids, rows, dense, covariates, y = _panel(seed=11, n=120, traits=4)
    if binary is not None:   # phenotype y1 binary, its two values coded as `binary`
        y[:, 1] = np.where(np.isnan(y[:, 1]), np.nan, np.where(y[:, 1] > 1, binary[1], binary[0]))
    cov_ids = list(ids)
    if duplicate:
        cov_ids[5] = cov_ids[4]
    # Covariate rows in reverse; one sample with a missing covariate, one not genotyped.
    order = list(range(len(ids)))[::-1]
    cov = ['IID\tage\tsex'] + [f'{cov_ids[i]}\t{"NA" if i == 7 else repr(float(covariates[i, 0]))}\t{covariates[i, 1]:.0f}'
                               for i in order] + ['ghost\t1.0\t0']
    pheno = ['FID\tIID\t' + '\t'.join(f'y{k}' for k in range(y.shape[1]))]
    for i in order:
        pheno.append(f'{cov_ids[i]}\t{cov_ids[i]}\t' + '\t'.join('NA' if np.isnan(v) else repr(float(v)) for v in y[i]))
    pheno.append('ghost\tghost\t' + '\t'.join('1.0' for _ in range(y.shape[1])))
    (tmp_path / 'cov.txt').write_text('\n'.join(cov) + '\n')
    (tmp_path / 'pheno.txt').write_text('\n'.join(pheno) + '\n')
    (tmp_path / 'geno.sample').write_text('ID_1 ID_2 missing\n0 0 0\n' + ''.join(f'{s} {s} 0\n' for s in ids))
    (tmp_path / 'kin.txt').write_text('ID1\tID2\tvalue\n' + ''.join(f'{a}\t{b}\t{v}\n' for a, b, v in rows))
    args = argparse.Namespace(pheno_file=str(tmp_path / 'pheno.txt'), cov_file=str(tmp_path / 'cov.txt'),
                              kin_file=str(tmp_path / 'kin.txt'), kin_diag=0.0, sampleid_name='IID',
                              covar_names=['age', 'sex'], missing_value='NA', random_slope_name='',
                              bgen=str(tmp_path / 'geno.bgen'), sample=str(tmp_path / 'geno.sample'),
                              using_bed=False, geno_flag='--bgen', corr_file=str(tmp_path / 'corr.txt'),
                              null_log=str(tmp_path / 'null.log'))
    return args, ids, rows, dense, covariates, y


def test_unsupported_cases_are_left_to_the_cpp_step1(tmp_path):
    cpu = torch.device('cpu')
    for coding, reason in (((1.0, 2.0), 'not coded 0/1'), ((0.0, 1.0), None)):
        inputs = gpu.read_step1_inputs(_step1_files(tmp_path, binary=coding)[0], '\t', '\t')[0]
        binary, found = gpu.check_binary_coding(inputs['y'], inputs['traits'], cpu)
        np.testing.assert_array_equal(binary, [False, True, False, False])
        assert found is None if reason is None else reason in found
    args = _step1_files(tmp_path, duplicate=True)[0]
    assert 'repeated measures' in gpu.read_step1_inputs(args, '\t', '\t')[1]
    args = _step1_files(tmp_path)[0]
    args.random_slope_name = 'age'
    assert 'random slope' in gpu.read_step1_inputs(args, '\t', '\t')[1]


def _dense(rows, n):
    dense = np.zeros((n, n))
    for a, b, v in rows:
        dense[int(a[1:]), int(b[1:])] = dense[int(b[1:]), int(a[1:])] = v
    return dense


def test_a_kinship_that_is_not_relatives_only_is_thresholded(monkeypatch):
    ids, rows, dense, covariates, y = _panel()
    cpu = torch.device('cpu')
    monkeypatch.setattr(gpu, 'MAX_BLOCK', 6)     # the panel's families have up to 6 members
    sparse = _kinship(ids, rows)
    assert gpu.relatives_only(sparse, cpu) == (sparse, None)
    # Weak pairs (0.01) chaining consecutive samples join everything into one family.
    chained = _kinship(ids, rows + [(ids[i], ids[i + 1], 0.01) for i in range(len(ids) - 1)
                                    if dense[i, i + 1] == 0])
    assert chained.largest_family(cpu) == len(ids)
    kept, note = gpu.relatives_only(chained, cpu)        # default: 0.05 x the median diagonal (1)
    assert 'default' in note and kept.largest_family(cpu) <= 6
    assert sorted(zip(kept.rows, kept.cols, kept.values)) == sorted(zip(sparse.rows, sparse.cols, sparse.values))
    kept, note = gpu.relatives_only(sparse, cpu, threshold=0.6)   # drops every 0.5 pair
    assert '--kin-threshold' in note and len(kept.values) == 0
    monkeypatch.setattr(gpu, 'MAX_BLOCK', 3)
    with pytest.raises(gpu.FamilyTooLarge, match='kin-threshold'):
        gpu.relatives_only(chained, cpu)


@pytest.mark.parametrize('device', DEVICES)
def test_a_family_with_a_negative_eigenvalue_is_made_positive_semidefinite(device):
    ids, rows, dense, covariates, y = _panel(seed=29, n=300)
    # Three members of one family with kinships 0.9, 0.9, -0.9: their block's
    # eigenvalues include -0.8, so the whole family's block has a negative one.
    pairs = {(a, b) for a, b, v in rows if a != b}
    i = next(int(a[1:]) for a, b in sorted(pairs) if int(b[1:]) == int(a[1:]) + 2)
    new = {(ids[i], ids[i + 1]): 0.9, (ids[i], ids[i + 2]): 0.9, (ids[i + 1], ids[i + 2]): -0.9}
    rows = [(a, b, new.get((a, b), v)) for a, b, v in rows]
    lam, u = np.linalg.eigh(_dense(rows, len(ids)))
    assert lam[0] < -0.5
    clipped = (u * np.maximum(lam, 0)) @ u.T            # the same as clipping block by block
    kin = _kinship(ids, rows)
    y[:, 3:] = _binary(y[:, 3:])
    fit = gpu.fit_null_model(y, covariates, kin, torch.device(device))
    assert fit['clipped'] == 1 and fit['lowest_eigenvalue'] < -0.5
    for columns, reference in (([0, 1, 2], _reference), ([3, 4, 5], _reference_binary)):
        want = reference(y[:, columns], covariates, clipped)
        np.testing.assert_allclose(fit['tau'][columns], want['tau'], rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(fit['c2'][columns], want['c2'], rtol=1e-8)
        np.testing.assert_allclose(fit['pseudo'][:, columns], want['pseudo'], rtol=1e-7, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_step1_writes_the_cpp_correction_file(tmp_path):
    args, ids, rows, dense, covariates, y = _step1_files(tmp_path, binary=(0.0, 1.0))
    assert gpu.fit_step1_gpu(args, '\t', '\t', '\t') == (None, None)
    # What --step all hands to step 2 in memory is what step 2 reads from the file, as float32.
    (header, c2_kept, kept, kept_ids), reason = gpu.fit_step1_gpu(args, '\t', '\t', '\t', keep_on_device=True)
    lines = (tmp_path / 'corr.txt').read_text().splitlines()
    assert reason is None and header == lines[0].split('\t') and kept.is_cuda and kept.dtype == torch.float32
    assert kept_ids == [line.split('\t')[0] for line in lines[2:]]
    np.testing.assert_array_equal(c2_kept, [float(v) for v in lines[1].split('\t')[1:]])
    from_file = np.array([[float(v) for v in line.split('\t')[1:]] for line in lines[2:]]).astype(np.float32)
    np.testing.assert_array_equal(kept.cpu().numpy(), from_file)
    lines = (tmp_path / 'corr.txt').read_text().splitlines()
    assert lines[0] == 'sample_id\ty0\ty1\ty2\ty3'
    assert lines[1].startswith('#\t')
    c2 = np.array([float(v) for v in lines[1].split('\t')[1:]])
    got_ids = [line.split('\t')[0] for line in lines[2:]]
    got = np.array([[float(v) for v in line.split('\t')[1:]] for line in lines[2:]])
    # Genotype order; the sample with a missing covariate and the ungenotyped one are gone.
    keep = [i for i in range(len(ids)) if i != 7]
    assert got_ids == [ids[i] for i in keep]
    sub = np.ix_(keep, keep)
    want = _reference(y[keep], covariates[keep], dense[sub])
    binary = _reference_binary(y[keep][:, [1]], covariates[keep], dense[sub])
    want['c2'][1], want['pseudo'][:, 1] = binary['c2'][0], binary['pseudo'][:, 0]
    np.testing.assert_allclose(c2, want['c2'], rtol=1e-8)
    np.testing.assert_allclose(got, want['pseudo'], rtol=1e-7, atol=1e-10)
    assert np.all(got[np.isnan(y[keep])] == 0)
