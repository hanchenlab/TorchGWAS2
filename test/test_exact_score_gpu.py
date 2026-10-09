"""pymodules/ExactScoreGPU.py: the exact mixed-model score test against a dense one-phenotype-at-a-time
reference, and step 1's --exact-score outputs.

    python -m pytest test/test_exact_score_gpu.py

Needs numpy, pandas, pyarrow and torch, not the C++ module. The test runs on
the CPU too; the step 1 test needs CUDA.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import test_null_model_gpu as base

gpu = base.gpu
sys.modules.setdefault('NullModelGPU', gpu)
_root = Path(__file__).resolve().parents[1] / 'pymodules'
_modules = {}
for _name in ('ExactScoreGPU', 'TorchGWAS'):
    _spec = importlib.util.spec_from_file_location(_name, _root / f'{_name}.py')
    _modules[_name] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_modules[_name])
exact, step2 = _modules['ExactScoreGPU'], _modules['TorchGWAS']

DEVICES = base.DEVICES


def _dense(g, e, tau, c1, x, kin):
    """Per phenotype on its observed samples S: P from tau over S; U = g'(e / c1), V = g' P g.

    Returns beta, se, z (variants x phenotypes) and each phenotype's P (for P y).
    """
    traits = e.shape[1]
    out = {key: np.full((g.shape[0], traits), np.nan) for key in ('beta', 'se', 'z')}
    out['p'] = []
    g = g - g.mean(1, keepdims=True)
    for k in range(traits):
        s = ~np.isnan(e[:, k])
        si = np.linalg.inv(tau[k, 0] * np.eye(s.sum()) + tau[k, 1] * kin[np.ix_(s, s)])
        six = si @ x[s]
        p_mat = si - six @ np.linalg.pinv(x[s].T @ six) @ six.T
        u = g[:, s] @ (e[s, k] / c1[k])
        v = np.einsum('ci,ij,cj->c', g[:, s], p_mat, g[:, s])
        a = np.einsum('ci,ij,cj->c', g[:, s], si, g[:, s])
        ok = v > exact.RELATIVE_FLOOR * a
        out['beta'][ok, k] = u[ok] / v[ok]
        out['se'][ok, k] = 1 / np.sqrt(v[ok])
        out['z'][ok, k] = u[ok] / np.sqrt(v[ok])
        out['p'].append((s, p_mat))
    return out


def _genotypes(n, seed=3):
    """Hard calls at allele frequencies 0.02-0.5, dosages, one monomorphic variant and one with one carrier."""
    rng = np.random.default_rng(seed)
    freq = rng.uniform(0.02, 0.5, 40)
    calls = rng.binomial(2, freq[:, None], (40, n)).astype(np.float64)
    dosages = rng.uniform(0, 2, (8, n))
    single = np.zeros((1, n))
    single[0, 3] = 1
    return np.vstack([calls, dosages, np.ones((1, n)), single])


def _check(got, want, rows=slice(None)):
    beta, se, z = (t.cpu().numpy().astype(np.float64)[rows] for t in got)
    np.testing.assert_array_equal(np.isnan(z), np.isnan(want['z'][rows]))
    np.testing.assert_allclose(z, want['z'][rows], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(se, want['se'][rows], rtol=1e-4)
    # BETA to 1e-4 of itself or of its SE, whichever is larger.
    np.testing.assert_allclose(beta / want['se'][rows], want['beta'][rows] / want['se'][rows], rtol=1e-4, atol=1e-4)


def _panel():
    """_panel's families, with a phenotype complete, and one missing a whole family."""
    ids, rows, dense, covariates, y = base._panel(n=240, traits=6, missing=0.15)
    rng = np.random.default_rng(9)
    y[:, 5] = np.where(np.isnan(y[:, 5]), 1 + rng.normal(size=len(ids)), y[:, 5])
    sizes = (dense > 0).sum(1)
    family = np.flatnonzero(dense[np.argmax(sizes)] > 0)       # the largest family
    y[family, 0] = np.nan
    return ids, rows, dense, covariates, y


# Families of 2-6: stored blocks of Sigma^-1 (16), or rotated into their eigenbases with patches (1).
FAMILY_PATHS = pytest.mark.parametrize('dense_family', [16, 1], ids=['blocks', 'rotated'])


@FAMILY_PATHS
@pytest.mark.parametrize('device', DEVICES)
def test_the_exact_test_matches_the_dense_reference(device, dense_family, monkeypatch):
    monkeypatch.setattr(exact, 'DENSE_FAMILY', dense_family)
    ids, rows, dense, covariates, y = _panel()
    device = torch.device(device)
    kinship = base._kinship(ids, rows)
    fit = gpu.fit_null_model(y, covariates, kinship, device, exact=True)
    assert np.array_equal(np.isnan(fit['pseudo']), np.isnan(y))
    x = gpu.design_matrix(covariates, len(ids))
    test = exact.ExactScore(kinship.families(device), fit['tau'], fit['c1'], fit['pseudo'], x, device)
    # Some family misses a member for some phenotype: a patch when rotated.
    assert (bool(test.blocks), bool(test.patches)) == ((True, False) if dense_family > 1 else (False, True))
    g = _genotypes(len(ids))
    carrier = np.flatnonzero(np.isnan(y).any(1))[0]          # missing some phenotype
    g[-1] = 0
    g[-1, carrier] = 1
    want = _dense(g, fit['pseudo'], fit['tau'], fit['c1'], x, dense)
    # Step 1's e / c1 is P y at the final tau, the P that V uses.
    for k, (s, p_mat) in enumerate(want['p']):
        np.testing.assert_allclose(fit['pseudo'][s, k] / fit['c1'][k], p_mat @ y[s, k], rtol=1e-7, atol=1e-12)
    got = test(torch.as_tensor(g, dtype=torch.float32, device=device))
    _check(got, want)
    assert np.isnan(got[2][-2].cpu().numpy()).all()             # the monomorphic variant
    # One carrier is enough where the carrier's value is observed; elsewhere the variant is constant.
    np.testing.assert_array_equal(np.isfinite(got[2][-1].cpu().numpy()), ~np.isnan(y[carrier]))


@pytest.mark.parametrize('device', DEVICES)
def test_without_kinship_the_exact_test_is_the_weighted_ols_score_test(device):
    ids, rows, dense, covariates, y = _panel()
    device = torch.device(device)
    fit = gpu.fit_null_model(y, covariates, None, device, exact=True)
    x = gpu.design_matrix(covariates, len(ids))
    test = exact.ExactScore(None, fit['tau'], fit['c1'], fit['pseudo'], x, device)
    g = _genotypes(len(ids))
    want = _dense(g, fit['pseudo'], fit['tau'], fit['c1'], x, np.zeros_like(dense))
    _check(test(torch.as_tensor(g, dtype=torch.float32, device=device)), want)
    # Unrelated samples: z = t sqrt((n - p) / (n - p - 1 + t^2)), t the OLS t statistic of the variant.
    s = ~np.isnan(y[:, 1])
    design = np.column_stack([x[s], g[0, s]])
    coef, rss = np.linalg.lstsq(design, y[s, 1], rcond=None)[:2]
    n, p = s.sum(), x.shape[1]
    t = coef[-1] / np.sqrt(rss[0] / (n - p - 1) * np.linalg.inv(design.T @ design)[-1, -1])
    z = test(torch.as_tensor(g[:1], dtype=torch.float32, device=device))[2].cpu().numpy()[0, 1]
    np.testing.assert_allclose(z, t * np.sqrt((n - p) / (n - p - 1 + t ** 2)), rtol=1e-4)


@FAMILY_PATHS
@pytest.mark.parametrize('device', DEVICES)
def test_a_covariate_constant_on_a_phenotypes_samples_is_dropped(device, dense_family, monkeypatch):
    """X' Sigma^-1 X is singular there; the score test takes its pseudo-inverse."""
    monkeypatch.setattr(exact, 'DENSE_FAMILY', dense_family)
    ids, rows, dense, covariates, y = _panel()
    device = torch.device(device)
    y[covariates[:, 1] == 1, 2] = np.nan            # phenotype 2: only one sex observed
    x = gpu.design_matrix(covariates, len(ids))
    tau = np.array([[0.7, 0.3]] * y.shape[1])
    c1 = np.linspace(0.8, 1.3, y.shape[1])
    e = np.full_like(y, np.nan)
    for k in range(y.shape[1]):
        s = ~np.isnan(y[:, k])
        si = np.linalg.inv(tau[k, 0] * np.eye(s.sum()) + tau[k, 1] * dense[np.ix_(s, s)])
        six = si @ x[s]
        e[s, k] = c1[k] * (si - six @ np.linalg.pinv(x[s].T @ six) @ six.T) @ y[s, k]
    families = base._kinship(ids, rows).families(device)
    test = exact.ExactScore(families, tau, c1, e, x, device)
    g = _genotypes(len(ids))
    _check(test(torch.as_tensor(g, dtype=torch.float32, device=device)), _dense(g, e, tau, c1, x, dense))


def test_step2_needs_step1s_table(tmp_path):
    spec = exact.ExactScoreSpec(tmp_path / 'corr.txt.null.tsv')
    with pytest.raises(FileNotFoundError, match='--exact-score'):
        spec.build(['y0'], ['s0'], np.zeros((1, 1)), np.ones((1, 1)), torch.device('cpu'))
    (tmp_path / 'corr.txt.null.tsv').write_text('phenotype\tmodel\ttau_e\ttau_g\tc1\tc2\tn_observed\n'
                                                'y0\tlinear\t1.0\t0.0\t1.0\t1.0\t1\n')
    with pytest.raises(ValueError, match='no null model'):
        spec.build(['y0', 'y1'], ['s0'], np.zeros((1, 2)), np.ones((1, 1)), torch.device('cpu'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_step1_writes_what_step2s_exact_test_reads(tmp_path):
    args, ids, rows, dense, covariates, y = base._step1_files(tmp_path, binary=(0.0, 1.0))
    args.exact_score = True
    assert gpu.fit_step1_gpu(args, '\t', '\t', '\t') == (None, None)
    keep = [i for i in range(len(ids)) if i != 7]            # sample 7 has a missing covariate
    header, c2, residuals, sample_ids = step2.read_correction_file(args.corr_file)
    assert sample_ids == [ids[i] for i in keep]
    np.testing.assert_array_equal(np.isnan(residuals), np.isnan(y[keep]))
    table = gpu.null_table_path(args.corr_file).read_text().splitlines()
    assert table[0] == 'phenotype\tmodel\ttau_e\ttau_g\tc1\tc2\tn_observed'
    assert [line.split('\t')[:2] for line in table[1:]] == [['y0', 'linear'], ['y1', 'logistic'],
                                                           ['y2', 'linear'], ['y3', 'linear']]
    assert [int(line.split('\t')[6]) for line in table[1:]] == list((~np.isnan(y[keep])).sum(0))
    np.testing.assert_array_equal([float(line.split('\t')[5]) for line in table[1:]], c2)
    # Step 2: the quantitative phenotypes only, against the dense reference at the table's tau.
    device = torch.device('cuda')
    spec = exact.ExactScoreSpec(gpu.null_table_path(args.corr_file), args.kin_file, '\t', args.kin_diag)
    x = gpu.design_matrix(covariates[keep], len(keep))
    test, columns = spec.build(header[1:], sample_ids, torch.as_tensor(residuals, device=device), x, device,
                               log=lambda line: None)
    assert list(columns) == [0, 2, 3]
    fields = [line.split('\t') for line in table[1:]]
    tau = np.array([[float(f[2]), float(f[3])] for f in fields])[columns]
    c1 = np.array([float(f[4]) for f in fields])[columns]
    g = _genotypes(len(keep))
    want = _dense(g, residuals[:, columns], tau, c1, x, dense[np.ix_(keep, keep)])
    _check(test(torch.as_tensor(g, dtype=torch.float32, device=device)), want)
