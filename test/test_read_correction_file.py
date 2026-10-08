"""pymodules/TorchGWAS.py read_correction_file: the same values as the line-by-line reader it replaced.

    python -m pytest test/test_read_correction_file.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve().parents[1] / 'pymodules'
_modules = {}
for _name in ('TorchGWAS', 'NullModelGPU'):
    _spec = importlib.util.spec_from_file_location(_name, _root / f'{_name}.py')
    _modules[_name] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_modules[_name])
step2, gpu = _modules['TorchGWAS'], _modules['NullModelGPU']


def _previous_reader(file_path):
    """The reader as it was: a Python float() per value."""
    c2 = np.empty(0, dtype=np.float64)
    c_res = []
    sample_ids = []
    with open(file_path, "r") as f:
        header = f.readline().strip().split("\t")
        second_line = f.readline().strip().split("\t")
        if second_line[0].startswith("#"):
            c2 = np.array([float(val) for val in second_line if not val.startswith("#")], dtype=np.float64)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                sample_ids.append(parts[0])
                c_res.append([float(x) for x in parts[1:]])
    return header, c2, np.array(c_res, dtype=np.float64), sample_ids


def _values(n=3000, k=40, seed=2):
    rng = np.random.default_rng(seed)
    pseudo = rng.normal(size=(n, k)) * 10.0 ** rng.integers(-6, 4, size=k)
    pseudo[rng.random(pseudo.shape) < 0.1] = 0          # missing phenotype values are written as 0
    return [f'S{i:05d}' for i in range(n)], [f'trait_{j}' for j in range(k)], rng.random(k) + 0.5, pseudo


def _cpp_style(path, ids, traits, c2, pseudo):
    """As the C++ step 1 prints it: ostream's default six significant digits."""
    with open(path, 'w') as out:
        out.write('sample_id\t' + '\t'.join(traits) + '\n')
        out.write('#' + ''.join(f'\t{v:g}' for v in c2) + '\n')
        for sample, row in zip(ids, pseudo):
            out.write(sample + ''.join(f'\t{v:g}' for v in row) + '\n')


@pytest.mark.parametrize('writer', ['gpu', 'cpp', 'cpp with blank and one-field lines'])
def test_the_reader_gives_what_the_previous_one_did(tmp_path, writer):
    ids, traits, c2, pseudo = _values()
    path = tmp_path / 'corr.txt'
    if writer == 'gpu':
        gpu.write_correction_file(path, traits, c2, ids, pseudo, threads=4)
    else:
        _cpp_style(path, ids, traits, c2, pseudo)
        if writer != 'cpp':
            lines = path.read_text().splitlines()
            path.write_text('\n'.join(lines[:10] + ['', 'orphan'] + lines[10:]) + '\n\n')
    want = _previous_reader(path)
    got = step2.read_correction_file(str(path))
    assert got[0] == want[0] and got[3] == want[3]
    np.testing.assert_array_equal(got[1], want[1])
    np.testing.assert_array_equal(got[2], want[2])
    assert got[2].dtype == np.float64 and got[2].flags.c_contiguous
