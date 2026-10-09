"""
@brief Step 1 on a GPU (--null-device cuda): the C++ step 1's null model and correction file.

The model is the C++ step 1's for a quantitative phenotype, fitted per
phenotype on its own observed samples:

    y = X alpha + g + e,   g ~ N(0, tau_g K),   e ~ N(0, tau_e I),
    Sigma = tau_e I + tau_g K,   P = Sigma^-1 - Sigma^-1 X (X' Sigma^-1 X)^-1 X' Sigma^-1,

by GMMAT's average-information REML with the same iteration (an EM-type first
step, AI updates with step halving and zeroing below tol, the joint
relative-change test on (alpha, tau), and a refit with any component that
reached the boundary fixed at zero). The correction file is the same: c2 per
phenotype and the scaled residuals c1 P y per sample (zero where the phenotype
is missing), with

    c1 = tr(K) / tr(P K) over the analysed samples,   c2 = c1 ||r||^2 / (N - 1).

Without a kinship file the model is OLS, in closed form.

A binary phenotype (two or fewer distinct observed values, coded 0/1) gets the
C++ step 1's logistic model. Its fixed-effect start is the C++ logistic
regression: IRLS from the OLS fit until every coefficient moves at most tol.
Without a kinship that fit is the null model: Sigma^-1 = diag(W), W = mu (1 - mu),
and the scaled residual is y - mu. With one, GMMAT's PQL follows from it: tau_e
is fixed at 1 and

    Sigma = diag(1/W) + tau_g K,   Y = eta + (y - mu) / W   (the working response),

with the same AI iteration on tau_g, each step also updating eta = Y - P Y / W,
mu and W; the scaled residual is y - mu at the end, and c1 and c2 are formed as
above from the last iteration's Sigma.

Why it is fast. A relatives-only kinship falls apart into small families (the
connected components of its graph), so Sigma is block diagonal. Each family's
block of K is diagonalised once, K_b = U diag(lambda) U'; in those coordinates
Sigma is diagonal at every tau, and every AI-REML iteration is elementwise over
samples x phenotypes plus p x p reductions. A phenotype that misses part of a
family has that family diagonalised again on its observed members. All
phenotypes iterate together, each with its own variance components and
convergence. The binary model's diag(1/W) differs per sample and phenotype, so
no basis diagonalises it; there each family block is inverted per phenotype at
every iteration, a batch of small dense inverses, and singletons are
elementwise.

The kinship. A family with a negative eigenvalue (an estimated or thresholded
kinship can have one) is replaced by its nearest positive semi-definite matrix,
U max(lambda, 0) U', and logged. A kinship that is not relatives-only (a family
of more than MAX_BLOCK samples; a dense GRM is one family of all of them) is
thresholded by default, dropping the pairs below 0.05 times the median diagonal
(fastGWA's 0.05 on the GRM scale), and logged; --kin-threshold sets the cutoff.
The C++ step 1 uses the kinship as given, so it agrees with this one exactly
when the kinship is relatives-only and positive semi-definite.

Not handled here, and left to the C++ step 1: a binary phenotype not coded 0/1,
repeated measures (duplicated sample IDs in the covariate file) and random
slopes. fit_step1_gpu() returns the reason without writing anything, and the
caller runs the C++ step 1. When step 2 follows in the same process
(--step all), fit_step1_gpu() can also hand over the residuals still on the
GPU, so step 2 does not parse them back from the correction file.

Inputs are read with the C++ step 1's rules: genotype sample IDs are the first
token of each .sample line after its two header lines (the .fam IID for BED);
the analysed samples are those, in genotype order, with every covariate present
(neither the missing-value token nor empty); phenotype rows match covariate
rows one for one; a phenotype value is missing when empty or the token; a
kinship row (ID1, ID2, value) is used once per ordered ID pair, the diagonal is
--kin-diag plus a listed (i, i) value, and both orders of a pair add up.
"""
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TOL = 1e-5
MAX_ITER = 500
# A family larger than this is a dense (batch x s x s) block per phenotype;
# such a kinship is not relatives-only, and relatives_only() thresholds it.
MAX_BLOCK = 8192
# relatives_only()'s default cutoff, times the median kinship diagonal:
# fastGWA's 0.05 on the GRM scale.
DEFAULT_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Inputs, read as the C++ step 1 reads them
# ---------------------------------------------------------------------------

def read_genotype_sample_ids(geno_file, sample_file, using_bed):
    """Genotype sample IDs in file order: .sample (first token, after two header lines) or .fam (IID)."""
    if using_bed and not sample_file:
        fam = Path(str(geno_file)[:-4] + '.fam') if str(geno_file).endswith('.bed') else Path(str(geno_file) + '.fam')
        with open(fam) as handle:
            return [line.split()[1] for line in handle if line.strip()]
    with open(sample_file) as handle:
        lines = handle.read().splitlines()[2:]
    return [line.split()[0] for line in lines if line.split()]


def _header(path, delim):
    with open(path) as handle:
        return handle.readline().rstrip('\n').split(delim)


def read_step1_inputs(args, pheno_delim, cov_delim):
    """The analysed samples, phenotypes and covariates, or (None, reason) for a case left to the C++ step 1."""
    if args.random_slope_name:
        return None, 'a random slope (--random-slope-name)'
    if args.geno_flag == '--bgen' and not args.sample:
        return None, 'a BGEN file without a .sample file'
    if args.using_bed and args.sample:
        return None, 'a BED file with an overriding --sample file'
    id_col = args.sampleid_name
    covariates = list(args.covar_names or [])
    missing = args.missing_value

    cov_header = _header(args.cov_file, cov_delim)
    absent = [name for name in [id_col] + covariates if name not in cov_header]
    if absent:
        raise ValueError(f'covariate file has no column(s) {absent}')
    cov = pd.read_csv(args.cov_file, sep=cov_delim, dtype=str, keep_default_na=False, na_filter=False,
                      usecols=[id_col] + covariates)
    cov_ids = cov[id_col].to_numpy()
    if len(set(cov_ids)) != len(cov_ids):
        return None, 'repeated measures (duplicated sample IDs in the covariate file)'

    pheno_header = _header(args.pheno_file, pheno_delim)
    if len(set(pheno_header)) != len(pheno_header):
        raise ValueError("there are repeated columns' names in the phenotype file")
    if len(pheno_header) < 3:
        raise ValueError('the phenotype file needs at least 3 columns (FID, IID and a phenotype)')
    if id_col not in pheno_header:
        raise ValueError(f'phenotype file has no column {id_col!r}')
    traits = pheno_header[2:]
    # pyarrow parses on several threads and rounds each value correctly, as std::stod does.
    import pyarrow as pa
    import pyarrow.csv as pacsv
    threads = getattr(args, 'threads', None) or len(os.sched_getaffinity(0))
    if pa.cpu_count() < threads:      # its pool follows OMP_NUM_THREADS, which the image sets to 1
        pa.set_cpu_count(threads)
    id_cols = set(pheno_header[:2] + [id_col])
    pheno = pacsv.read_csv(args.pheno_file, parse_options=pacsv.ParseOptions(delimiter=pheno_delim),
                           convert_options=pacsv.ConvertOptions(
                               column_types={name: pa.string() if name in id_cols else pa.float64()
                                             for name in pheno_header},
                               null_values=['', missing], strings_can_be_null=False))
    pheno_ids = pheno.column(id_col).to_numpy(zero_copy_only=False).astype(str)
    if len(pheno_ids) != len(cov_ids):
        raise ValueError('the phenotype and covariate files have different numbers of samples (rows)')
    mismatch = np.flatnonzero(pheno_ids != cov_ids)
    if mismatch.size:
        row = int(mismatch[0])
        raise ValueError(f'Sample ID mismatch at line {row + 1}. Expected: {cov_ids[row]}, Found: {pheno_ids[row]}')

    complete = np.ones(len(cov), dtype=bool)
    for name in covariates:
        values = cov[name].to_numpy()
        complete &= (values != missing) & (values != '')
    row_of = {sample: row for row, sample in enumerate(cov_ids) if complete[row]}
    geno_ids = read_genotype_sample_ids(args.bgen, args.sample, args.using_bed)
    ids, rows, seen = [], [], set()
    for sample in geno_ids:                       # genotype order
        if sample in row_of and sample not in seen:
            seen.add(sample)
            ids.append(sample)
            rows.append(row_of[sample])
    if not ids:
        raise ValueError('no genotype sample has complete covariates')
    rows = np.asarray(rows)
    y = np.empty((len(rows), len(traits)), order='F')     # each phenotype's column contiguous
    for k, name in enumerate(traits):
        y[:, k] = pheno.column(name).to_numpy(zero_copy_only=False)[rows]   # nulls become NaN
    # Only the kept rows: the others may hold the missing-value token.
    x = cov[covariates].iloc[rows].to_numpy(dtype=np.float64) if covariates else None
    return dict(ids=ids, traits=traits, y=y, x=x), None


def _budget(device):
    """Bytes a phenotype batch may use on `device`."""
    return 0.2 * torch.cuda.mem_get_info(device)[0] if device.type == 'cuda' else 4 << 30


def _columns_to_device(y_all, first, stop, device):
    """y_all[:, first:stop] on `device` as a contiguous N x k tensor; any transpose happens on the device."""
    block = y_all[:, first:stop]
    if block.flags.f_contiguous and not block.flags.c_contiguous:
        return torch.as_tensor(block.T, device=device).T.contiguous()
    return torch.as_tensor(np.ascontiguousarray(block), device=device)


def column_kinds(y_all, device):
    """Each phenotype column's observed minimum and maximum, and whether it is binary, computed on `device`.

    The C++ step 1's rule: a phenotype with two or fewer distinct observed
    values is binary (logistic), that is, every observed value is its column's
    minimum or maximum.
    """
    n, traits = y_all.shape
    binary, low, high = np.empty(traits, bool), np.empty(traits), np.empty(traits)
    batch = int(max(1, min(traits, _budget(device) // (8 * 6 * n))))
    for first in range(0, traits, batch):
        stop = min(traits, first + batch)
        y = _columns_to_device(y_all, first, stop, device)
        missing = torch.isnan(y)
        lo = torch.where(missing, math.inf, y).amin(0)
        hi = torch.where(missing, -math.inf, y).amax(0)
        binary[first:stop] = (missing | (y == lo) | (y == hi)).all(0).cpu().numpy()
        low[first:stop], high[first:stop] = lo.cpu().numpy(), hi.cpu().numpy()
    return binary, low, high


def check_binary_coding(y_all, traits, device):
    """Which phenotypes are binary, and a reason for the C++ step 1 if one of them is not coded 0/1 (else None)."""
    binary, low, high = column_kinds(y_all, device)
    for k in np.flatnonzero(binary):
        if low[k] != 0 or high[k] != 1:
            return binary, (f'a binary phenotype not coded 0/1 ({traits[k]}: values '
                            f'{sorted({float(low[k]), float(high[k])})})')
    return binary, None


def read_kinship(path, delim, ids, diagonal):
    """The kinship over `ids`, as the C++ step 1 builds it (module docstring)."""
    index = {sample: position for position, sample in enumerate(ids)}
    n = len(ids)
    diag = np.full(n, float(diagonal), dtype=np.float64)
    pairs = {}
    seen = set()
    with open(path) as handle:
        if len(handle.readline().rstrip('\n').split(delim)) != 3:
            raise ValueError(f'the kinship file {path} must have 3 columns (ID1, ID2, value)')
        for line in handle:
            fields = line.rstrip('\n').split(delim)
            if len(fields) < 3:
                continue
            a, b = fields[0].replace('"', ''), fields[1].replace('"', '')
            i, j = index.get(a), index.get(b)
            if i is None or j is None or (i, j) in seen:
                continue
            seen.add((i, j))
            value = float(fields[2])
            if i == j:
                diag[i] += value
            else:
                key = (i, j) if i < j else (j, i)
                pairs[key] = pairs.get(key, 0.0) + value
    if pairs:
        keys = np.asarray(list(pairs.keys()), dtype=np.int64)
        rows, cols, values = keys[:, 0], keys[:, 1], np.asarray(list(pairs.values()), dtype=np.float64)
    else:
        rows = cols = np.zeros(0, dtype=np.int64)
        values = np.zeros(0)
    return Kinship(n, diag, rows, cols, values)


# ---------------------------------------------------------------------------
# The kinship's families: connected components of the related pairs
# ---------------------------------------------------------------------------
#
# Each sample is labelled with its component's smallest index. Two
# implementations, the same labels:
#
# union-find (native/union_find.cu, built ahead of time by
# native/compile_union_find.sh for sm_75 to sm_120 plus compute_75 PTX): ECL-CC
# style. One thread per pair finds the roots of both samples and hooks the
# larger root under the smaller with an atomic compare-and-swap, retrying from
# the root's new parent when it loses a race; a final pass points every sample
# at its root. A root is only ever hooked under a smaller index, so the
# component's smallest sample is never hooked and is the root everyone ends at,
# whatever order the threads run in.
#
# hook-to-root (Torch, any device): Shiloach-Vishkin. Each round every pair
# whose samples have different roots hooks the larger root under the smaller
# (scatter amin on the roots), then pointer jumping (label[label]) takes every
# sample to its root.
#
# component_labels() runs the union-find on any GPU it launches on, otherwise
# hook-to-root. On the UK Biobank relatedness file (147,716 samples, 107,149
# pairs; an A100) hook-to-root takes 3.9 ms in 3 rounds.

_UNION_FIND = None
_UNION_FIND_LAUNCHES = {}      # device index -> whether the union-find runs there


def _union_find_library():
    global _UNION_FIND
    if _UNION_FIND is None:
        import ctypes as ct
        library = ct.CDLL(str(Path(__file__).with_name('native') / 'libunion_find.so'))
        library.tg_union_find.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int64, ct.c_void_p, ct.c_int64, ct.c_void_p]
        library.tg_union_find.restype = ct.c_int
        library.tg_union_find_error.restype = ct.c_char_p
        _UNION_FIND = library
    return _UNION_FIND


def union_find(rows, cols, n):
    """The native union-find's labels for pairs rows[k]-cols[k] (CUDA tensors), int64."""
    library = _union_find_library()
    device = rows.device
    parent = torch.arange(n, dtype=torch.int32, device=device)
    if rows.numel():
        rows32 = rows.to(torch.int32).contiguous()
        cols32 = cols.to(torch.int32).contiguous()
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device).cuda_stream
            if library.tg_union_find(rows32.data_ptr(), cols32.data_ptr(), rows32.numel(), parent.data_ptr(), n,
                                     stream):
                raise RuntimeError(library.tg_union_find_error().decode())
    return parent.to(torch.int64)


def union_find_available(device):
    """Whether the union-find runs on `device`, found once per GPU by labelling a small graph.

    Loading the library is not enough: on a GPU the build has no code for, the
    failure comes only at launch.
    """
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        return False
    index = torch.cuda.current_device() if device.index is None else device.index
    if index not in _UNION_FIND_LAUNCHES:
        try:
            on = torch.device('cuda', index)
            labels = union_find(torch.tensor([1, 3], device=on), torch.tensor([2, 1], device=on), 5)
            _UNION_FIND_LAUNCHES[index] = labels.tolist() == [0, 1, 1, 1, 4]
        except (OSError, RuntimeError):
            _UNION_FIND_LAUNCHES[index] = False
    return _UNION_FIND_LAUNCHES[index]


def hook_to_root(rows, cols, n):
    """Shiloach-Vishkin labels and the number of hooking rounds."""
    label = torch.arange(n, device=rows.device)
    rounds = 0
    while rows.numel():
        lu, lv = label[rows], label[cols]
        differ = lu != lv
        if not bool(differ.any()):
            break
        rounds += 1
        # Every label is a root here, so this hooks roots, each under the smallest offered.
        label = label.scatter_reduce(0, torch.maximum(lu, lv)[differ], torch.minimum(lu, lv)[differ], 'amin')
        while True:                      # pointer jumping to the roots
            jumped = label[label]
            if torch.equal(jumped, label):
                break
            label = jumped
    return label, rounds


def component_labels(rows, cols, n, device):
    """Each sample's component label, its smallest index (int64, on `device`), for pairs rows[k]-cols[k]."""
    device = torch.device(device)
    rows = torch.as_tensor(rows, dtype=torch.int64, device=device)
    cols = torch.as_tensor(cols, dtype=torch.int64, device=device)
    if union_find_available(device):
        return union_find(rows, cols, n)
    return hook_to_root(rows, cols, n)[0]


class FamilyTooLarge(ValueError):
    """A kinship family larger than MAX_BLOCK, even after thresholding (relatives_only)."""


class Kinship:
    """A symmetric sparse kinship over the analysed samples: K_ii and each unordered off-diagonal pair once."""

    def __init__(self, n, diagonal, rows, cols, values):
        self.n, self.diagonal, self.rows, self.cols, self.values = n, diagonal, rows, cols, values

    def _family_of(self, device):
        """Each sample's family number (0 .., in order of smallest member), and the family sizes, on `device`."""
        label = component_labels(self.rows, self.cols, self.n, device)
        _, family = torch.unique(label, return_inverse=True)
        return family, torch.bincount(family)

    def largest_family(self, device):
        return int(self._family_of(device)[1].max())

    def without_pairs_below(self, cutoff):
        """This kinship with the off-diagonal pairs below `cutoff` dropped (the diagonal kept)."""
        kept = self.values >= cutoff
        return Kinship(self.n, self.diagonal, self.rows[kept], self.cols[kept], self.values[kept])

    def families(self, device):
        """The connected components, each made positive semi-definite, on `device`: a dict of

        groups: (size, sample indices B x size, blocks B x size x size) per family size;
        diagonal: K_ii for every sample (N), and trace, its sum;
        clipped, lowest: how many families had a negative eigenvalue below -1e-10, and the lowest.

        A family with a negative eigenvalue (a thresholded or estimated
        kinship can have one) has its block replaced by U max(lambda, 0) U',
        the nearest positive semi-definite matrix; a negative singleton
        diagonal becomes 0.
        """
        n = self.n
        rows = torch.as_tensor(self.rows, dtype=torch.int64, device=device)
        cols = torch.as_tensor(self.cols, dtype=torch.int64, device=device)
        family, sizes = self._family_of(device)
        if int(sizes.max()) > MAX_BLOCK:
            raise FamilyTooLarge(f'a family of {int(sizes.max())} related samples, more than {MAX_BLOCK}: '
                                 'the GPU null model needs a relatives-only kinship (see relatives_only)')
        order = torch.sort(family, stable=True).indices
        starts = torch.cumsum(sizes, 0) - sizes
        position = torch.empty(n, dtype=torch.int64, device=device)
        position[order] = torch.arange(n, device=device) - starts[family[order]]
        diagonal = torch.as_tensor(self.diagonal, dtype=torch.float64, device=device)
        values = torch.as_tensor(self.values, dtype=torch.float64, device=device)
        groups = []
        single = sizes[family] == 1          # a singleton's eigenvalue is its diagonal
        lowest = float(diagonal[single].min()) if single.any() else math.inf
        clipped = int((diagonal[single] < -1e-10).sum())
        out_diagonal = diagonal.clamp(min=0)
        for size in torch.unique(sizes).tolist():
            members = (sizes == size).nonzero()[:, 0]
            slot = torch.full_like(sizes, -1)
            slot[members] = torch.arange(members.numel(), device=device)
            index = torch.empty((members.numel(), size), dtype=torch.int64, device=device)
            chosen = (sizes[family] == size).nonzero()[:, 0]
            index[slot[family[chosen]], position[chosen]] = chosen
            blocks = torch.zeros((members.numel(), size, size), dtype=torch.float64, device=device)
            blocks.diagonal(dim1=1, dim2=2).copy_(diagonal[index])
            pairs = (sizes[family[rows]] == size).nonzero()[:, 0]
            b = slot[family[rows[pairs]]]
            i, j = position[rows[pairs]], position[cols[pairs]]
            blocks[b, i, j] = values[pairs]
            blocks[b, j, i] = values[pairs]
            if size > 1:
                lam, u = torch.linalg.eigh(blocks)
                lowest = min(lowest, float(lam[:, 0].min()))
                negative = lam[:, 0] < 0
                if negative.any():
                    clipped += int((lam[:, 0] < -1e-10).sum())
                    fixed = (u[negative] * lam[negative].clamp(min=0)[:, None, :]) @ u[negative].transpose(1, 2)
                    blocks[negative] = (fixed + fixed.transpose(1, 2)) / 2
                out_diagonal[index] = blocks.diagonal(dim1=1, dim2=2)
            groups.append((int(size), index, blocks))
        return dict(groups=groups, diagonal=out_diagonal, trace=float(out_diagonal.sum()), clipped=clipped,
                    lowest=lowest)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

def relatives_only(kinship, device, threshold=None):
    """The kinship to fit, and a log line saying what was done to it (None when it is used as given).

    threshold None (the default): the kinship is used as given when no family
    has more than MAX_BLOCK samples; otherwise the pairs below DEFAULT_THRESHOLD
    times the median diagonal are dropped (0.05 on the GRM scale, 2 x kinship;
    0.025 on the kinship scale). A number: the pairs below it, in the kinship
    file's units, are dropped. A family still larger than MAX_BLOCK is an error.
    """
    largest = kinship.largest_family(device)
    if threshold is None and largest <= MAX_BLOCK:
        return kinship, None
    if threshold is None:
        cutoff = DEFAULT_THRESHOLD * float(np.median(kinship.diagonal))
        why = (f'the default, {DEFAULT_THRESHOLD} x the median diagonal, as a family had {largest} samples, '
               f'more than {MAX_BLOCK}')
    else:
        cutoff, why = float(threshold), '--kin-threshold'
    thresholded = kinship.without_pairs_below(cutoff)
    after = thresholded.largest_family(device)
    if after > MAX_BLOCK:
        raise FamilyTooLarge(f'after dropping kinship pairs below {cutoff:.4g} a family of {after} samples remains, '
                             f'more than {MAX_BLOCK}; set a higher --kin-threshold (in the kinship file\'s units)')
    return thresholded, (f'kinship thresholded at {cutoff:.4g} ({why}): kept {len(thresholded.values)} of '
                         f'{len(kinship.values)} related pairs; largest family {largest} -> {after} samples')


def design_matrix(covariates, n):
    """[1, covariates], dropping a covariate collinear with those before it.

    The C++ step 1's rule: unpivoted Householder QR, column j dropped when
    |R_jj| < max |R_jj| sqrt(eps_float64).
    """
    columns = np.ones((n, 1)) if covariates is None else np.column_stack(
        [np.ones(n), np.asarray(covariates, dtype=np.float64)])
    r = np.linalg.qr(columns, mode='r')
    diagonal = np.abs(np.diag(r))
    keep = diagonal >= diagonal.max() * math.sqrt(np.finfo(np.float64).eps)
    keep[0] = True
    return columns[:, keep]


def fit_null_model(y_all, covariates, kinship, device, tol=TOL, max_iter=MAX_ITER, names=None, binary=None,
                   keep_on_device=False, exact=False):
    """Every phenotype column's null model (NaN marks a missing value): a dict of per-phenotype arrays.

    pseudo (N x K, a view of a phenotype-major array) is c1 P y with zeros
    where missing; c1, c2, tau (K x 2: tau_e, tau_g; tau_e is 1 for a binary
    phenotype), iterations, converged and binary per phenotype. Quantitative
    and binary phenotypes may be mixed; `binary` is column_kinds()'s, computed
    when not given. With keep_on_device, pseudo_device is pseudo as float32 on
    `device` (N x K), what step 2 computes with.

    exact=True prepares step 2's exact score test (ExactScoreGPU.py): the
    pseudo-phenotype is NaN where missing, and a quantitative phenotype's
    P y is taken at the final tau (the C++ step 1's convention takes it from
    the last iteration's starting point), so that U = g' P y and V = g' P g
    share one P.
    """
    n, traits = y_all.shape
    x = design_matrix(covariates, n)
    p = x.shape[1]
    xt = torch.as_tensor(x, device=device, dtype=torch.float64)
    if binary is None:
        binary = column_kinds(y_all, device)[0]
    families = None if kinship is None else kinship.families(device)
    linear = logistic = None
    per_trait = 8 * 12 * n
    if families is not None and not binary.all():
        linear = _Families(families, x, device)
        per_trait = max(per_trait, 8 * (28 * n + sum(g['index'].shape[0] * g['size'] * (4 * g['size'] + p)
                                                     for g in linear.groups)))
    if families is not None and binary.any():
        logistic = _Blocks(families, xt)
        per_trait = max(per_trait, 8 * (24 * n + sum(g['count'] * g['size'] * (6 * g['size'] + 3 * p)
                                                     for g in logistic.groups)))
    batch = int(max(1, min(traits, _budget(device) // per_trait)))
    pseudo_t = np.empty((traits, n))
    out = dict(pseudo=pseudo_t.T, c1=np.empty(traits), c2=np.empty(traits), tau=np.empty((traits, 2)),
               iterations=np.empty(traits, np.int64), converged=np.empty(traits, bool), binary=binary,
               clipped=0 if families is None else families['clipped'],
               lowest_eigenvalue=None if families is None else families['lowest'])
    if keep_on_device:      # samples x phenotypes, contiguous: the layout step 2 builds from the file
        kept = out['pseudo_device'] = torch.empty((n, traits), dtype=torch.float32, device=device)
    for first in range(0, traits, batch):
        stop = min(traits, first + batch)
        y = _columns_to_device(y_all, first, stop, device)
        observed = ~torch.isnan(y)
        if (observed.sum(0) <= p + 1).any():
            raise ValueError('a phenotype has too few observed samples for its null model')
        m = observed.to(torch.float64)
        y = torch.where(observed, y, 0.0)
        pseudo = torch.empty((stop - first, n), dtype=torch.float64, device=device)
        for kind in (False, True):
            local = np.flatnonzero(binary[first:stop] == kind)
            if not local.size:
                continue
            index = torch.as_tensor(local, device=device)
            ys, ms = (y, m) if local.size == stop - first else (y[:, index], m[:, index])
            if not kind:
                fit = (_fit_unrelated(ys, ms, xt) if linear is None
                       else _fit_mixed(linear, ys, ms, tol, max_iter, exact))
            else:
                glm = _logistic_regression(ys, ms, xt, tol)
                failed = np.flatnonzero(~glm['converged'].cpu().numpy())
                if failed.size:
                    which = [names[c] if names is not None else f'column {c}' for c in first + local[failed]]
                    raise RuntimeError(f'logistic regression failed to converge after {MAX_ITER} iterations '
                                       f'for {", ".join(map(str, which))}')
                fit = (_fit_logistic(glm, ys, ms, xt) if logistic is None
                       else _fit_pql(logistic, glm, ys, ms, tol, max_iter))
            pseudo[index] = fit['pseudo'].T
            for key in ('c1', 'c2', 'tau', 'iterations', 'converged'):
                out[key][first + local] = fit[key].cpu().numpy()
        if exact:
            pseudo = torch.where(observed.T, pseudo, torch.nan)
        torch.from_numpy(pseudo_t[first:stop]).copy_(pseudo)
        if keep_on_device:
            kept[:, first:stop] = pseudo.T
    return out


def _fit_unrelated(y, m, xt):
    """OLS per phenotype on its observed samples: e = N (y - X b) / (n - p), c2 = N / (N - 1)."""
    n, traits = y.shape
    p = xt.shape[1]
    xtx = torch.einsum('nk,ni,nj->kij', m, xt, xt)
    alpha = torch.linalg.solve(xtx, ((m * y).T @ xt).unsqueeze(-1)).squeeze(-1)
    residual = (y - xt @ alpha.T) * m
    count = m.sum(0)
    sigma2 = (residual ** 2).sum(0) / (count - p)
    c1 = n * sigma2 / (count - p)
    return dict(pseudo=n * residual / (count - p), c1=c1, c2=torch.full_like(c1, n / (n - 1)),
                tau=torch.stack((sigma2, torch.zeros_like(sigma2)), 1), alpha=alpha,
                iterations=torch.zeros(traits, dtype=torch.int64, device=y.device),
                converged=torch.ones(traits, dtype=torch.bool, device=y.device))


class _Families:
    """The kinship's families on the device, each in the eigenbasis of its own block (module docstring).

    Rotated values sit in the family's own sample rows: row index[b, i] holds
    eigencomponent i of family b. Singletons are their own eigenbasis.
    """

    def __init__(self, families, x, device):
        self.t = dict(device=device, dtype=torch.float64)
        self.n, self.p = x.shape
        self.trace_all = families['trace']
        self.kin_diag = families['diagonal']
        xt = self.x0 = torch.as_tensor(x, **self.t)
        self.x = xt.clone()
        self.groups, self.sizes = [], []
        for size, index, kin in families['groups']:
            self.sizes.append((size, int(index.shape[0])))
            if size == 1:
                continue
            lam, u = torch.linalg.eigh(kin)
            self.x[index] = torch.einsum('bji,bjp->bip', u, xt[index])
            # Every eigenvalue of a principal submatrix is >= -bound (Gershgorin);
            # a patch puts a missing member's coordinate at floor, well below.
            bound = kin.abs().sum(2).amax(1)
            self.groups.append(dict(size=size, index=index, kin=kin, lam=lam, u=u, x=xt[index],
                                    floor=-(2 * bound + 1), cut=-(bound + 0.5)))
        self.outer = (self.x[:, :, None] * self.x[:, None, :]).reshape(self.n, self.p * self.p)

    def load(self, y, m):
        """A phenotype block (N x k, zeros where missing; m its 0/1 mask) in the families' coordinates."""
        self.y = y.clone()
        self.lam = self.kin_diag[:, None].expand_as(y).clone()
        self.obs = m.clone()
        self.shared = torch.ones_like(y)
        self.patches = []
        for g in self.groups:
            index, u = g['index'], g['u']
            mb = m[index]
            self.y[index] = torch.einsum('bji,bjk->bik', u, y[index])
            self.lam[index] = g['lam'].unsqueeze(-1).expand_as(mb)
            b, k = torch.nonzero(mb.amin(1) == 0, as_tuple=True)    # (family, phenotype) missing a member
            if b.numel() == 0:
                continue
            rows, cols, mj = index[b], k[:, None], mb[b, :, k]
            masked = (g['kin'][b] * mj[:, :, None] * mj[:, None, :]
                      + torch.diag_embed((1 - mj) * g['floor'][b, None]))
            lam, uj = torch.linalg.eigh(masked)
            obs = (lam > g['cut'][b, None]).to(lam.dtype)
            self.y[rows, cols] = torch.einsum('jsi,js->ji', uj, y[rows, cols])
            self.lam[rows, cols] = lam * obs
            self.obs[rows, cols] = obs
            self.shared[rows, cols] = 0
            self.patches.append(dict(rows=rows, col=k, u=uj,
                                     x=torch.einsum('jsi,jsp->jip', uj, g['x'][b] * mj.unsqueeze(-1))))

    def gram(self, weights):
        """sum_n w X~ X~' per phenotype, for each N x k weight: (len(weights) x k x p x p)."""
        k, p = weights[0].shape[1], self.p
        stacked = torch.cat([w * self.shared for w in weights], 1)
        out = (self.outer.T @ stacked).T.reshape(len(weights), k, p, p)
        for patch in self.patches:
            rows, cols = patch['rows'], patch['col'][:, None]
            for c, w in enumerate(weights):
                out[c].index_add_(0, patch['col'], torch.einsum('jsp,js,jsq->jpq', patch['x'], w[rows, cols],
                                                                patch['x']))
        return out

    def xt(self, v):
        """X~' v per phenotype (k x p)."""
        out = (v * self.shared).T @ self.x
        for patch in self.patches:
            out.index_add_(0, patch['col'], torch.einsum('jsp,js->jp', patch['x'],
                                                         v[patch['rows'], patch['col'][:, None]]))
        return out

    def x_times(self, a):
        """X~ a_k per phenotype (N x k)."""
        out = self.x @ a.T
        for patch in self.patches:
            out[patch['rows'], patch['col'][:, None]] = torch.einsum('jsp,jp->js', patch['x'], a[patch['col']])
        return out

    def unrotate(self, v):
        """Rotated N x k values back to samples."""
        out = v.clone()
        for g in self.groups:
            out[g['index']] = torch.einsum('bij,bjk->bik', g['u'], v[g['index']])
        for patch in self.patches:
            rows, cols = patch['rows'], patch['col'][:, None]
            out[rows, cols] = torch.einsum('jis,js->ji', patch['u'], v[rows, cols])
        return out


def _fit_mixed(families, y, m, tol, max_iter, exact=False):
    """AI-REML for a block of phenotypes, with the boundary refits."""
    columns = None
    fixed = torch.zeros(y.shape[1], 2, dtype=torch.bool, device=y.device)
    for _ in range(3):  # two components: at most two boundary refits
        if columns is None:
            out = fit = _ai_reml(families, y, m, fixed, tol, max_iter, exact)
        else:
            fit = _ai_reml(families, y[:, columns], m[:, columns], fixed, tol, max_iter, exact)
            out['pseudo'][:, columns] = fit['pseudo']
            for key in ('c1', 'c2', 'tau', 'alpha', 'iterations', 'converged'):
                out[key][columns] = fit[key]
        # A component that ended below 1.01 tol is fixed at zero and the
        # phenotype refitted from the start, until no new one does.
        newly = (fit['tau'] < 1.01 * tol) & ~fixed
        redo = newly.any(1)
        if not redo.any():
            break
        chosen = redo.nonzero()[:, 0]
        columns = chosen if columns is None else columns[chosen]
        fixed = (fixed | newly)[chosen]
    return out


def _ai_reml(families, y, m, fixed, tol, max_iter, exact=False):
    """AI-REML for k phenotypes at once: y (zero where missing) and its 0/1 mask m, N x k."""
    t = families.t
    n, k = y.shape
    xt = families.x0
    count = m.sum(0)
    families.load(y, m)
    yr, lam, obs = families.y, families.lam, families.obs

    def dot(a, b):
        return (a * b).sum(0)

    def state(tau):
        """What the scores need at tau (k x 2), in rotated coordinates: Sigma^-1 is w = 1/d, K is lambda."""
        w = torch.where(obs > 0, 1.0 / (tau[:, 0] + tau[:, 1] * lam), 0.0)
        xsx, sxsx, sxksx = families.gram([w, w * w, w * w * lam])
        cov = torch.cholesky_inverse(torch.linalg.cholesky_ex(xsx)[0])
        alpha = (cov @ families.xt(w * yr).unsqueeze(-1)).squeeze(-1)
        py = w * (yr - families.x_times(alpha))

        def proj(v):         # P v = Sigma^-1 v - Sigma^-1 X cov X' Sigma^-1 v
            sv = w * v
            return sv - w * families.x_times((cov @ families.xt(sv).unsqueeze(-1)).squeeze(-1))

        tr_p = w.sum(0) - torch.einsum('kpq,kqp->k', cov, sxsx)
        tr_pk = (w * lam).sum(0) - torch.einsum('kpq,kqp->k', cov, sxksx)
        kpy = lam * py
        return dict(alpha=alpha, py=py, kpy=kpy, tr_p=tr_p, tr_pk=tr_pk, proj=proj,
                    pyy=dot(py, py), pykpy=dot(py, kpy))

    # Starting values: tau = var(y) / 2 per free component, tau_g further
    # divided by the mean kinship diagonal over the phenotype's samples.
    mean_y = (y * m).sum(0) / count
    variance = (((y - mean_y) * m) ** 2).sum(0) / (count - 1)
    mean_kin = (families.kin_diag[:, None] * m).sum(0) / count
    tau = torch.stack((variance / 2, variance / 2 / mean_kin), 1)
    tau = torch.where(fixed, torch.zeros_like(tau), tau)
    xtx = torch.einsum('nk,ni,nj->kij', m, xt, xt)
    alpha = torch.linalg.solve(xtx, torch.einsum('nk,ni->ki', m * y, xt).unsqueeze(-1)).squeeze(-1)

    # One EM-type step at the starting tau.
    s0 = state(tau)
    scores = torch.stack((s0['pyy'] - s0['tr_p'], s0['pykpy'] - s0['tr_pk']), 1)
    tau = torch.where(fixed, tau, torch.clamp(tau + tau ** 2 * scores / count[:, None], min=0))

    active = torch.ones(k, dtype=torch.bool, device=t['device'])
    converged = torch.zeros_like(active)
    iterations = torch.zeros(k, dtype=torch.int64, device=t['device'])
    final = dict(py=torch.zeros_like(yr), tr_pk=torch.zeros(k, **t), tau0_e=torch.zeros(k, **t))
    for _ in range(max_iter):
        tau0, alpha0 = tau, alpha
        s = state(tau0)
        alpha = torch.where(active[:, None], s['alpha'], alpha0)
        ppy = s['proj'](s['py'])
        pkpy = s['proj'](s['kpy'])
        ai = torch.stack((torch.stack((dot(s['py'], ppy), dot(ppy, s['kpy'])), 1),
                          torch.stack((dot(ppy, s['kpy']), dot(s['kpy'], pkpy)), 1)), 1)
        score = torch.stack((s['pyy'] - s['tr_p'], s['pykpy'] - s['tr_pk']), 1)
        # Fixed components: identity rows, zero scores, so their step is zero.
        eye = torch.eye(2, **t).expand(k, 2, 2)
        free = (~fixed).to(t['dtype'])
        ai = ai * free[:, :, None] * free[:, None, :] + eye * (1 - free)[:, :, None]
        step = torch.linalg.solve(ai, (score * free).unsqueeze(-1)).squeeze(-1)

        def bounded(delta):
            new = tau0 + delta
            return torch.where((new < tol) & (tau0 < tol), torch.zeros_like(new), new)
        new_tau = bounded(step)
        for _ in range(100):  # halve until no component is negative
            negative = (new_tau < 0).any(1)
            if not negative.any():
                break
            step = torch.where(negative[:, None], step / 2, step)
            new_tau = bounded(step)
        new_tau = torch.where(new_tau < tol, torch.zeros_like(new_tau), new_tau)
        tau = torch.where(active[:, None], new_tau, tau0)
        iterations += active.long()
        # This iteration's starting point is what the outputs use.
        final['py'] = torch.where(active, s['py'], final['py'])
        final['tr_pk'] = torch.where(active, s['tr_pk'], final['tr_pk'])
        final['tau0_e'] = torch.where(active, tau0[:, 0], final['tau0_e'])
        change = torch.maximum(
            ((alpha - alpha0).abs() / (alpha.abs() + alpha0.abs() + tol)).amax(1),
            ((tau - tau0).abs() / (tau.abs() + tau0.abs() + tol)).amax(1))
        done = active & (2 * change < tol)
        diverged = active & (tau.abs().amax(1) > tol ** -2)
        converged |= done
        active &= ~(done | diverged)
        if not active.any():
            break

    if exact:     # the exact score test's U and V need one P: the final tau's
        s = state(tau)
        final = dict(py=s['py'], tr_pk=s['tr_pk'], tau0_e=tau[:, 0])
    # Scaled residual: from the last iteration's starting point, divided by the
    # final tau_e. With tau_e fixed at zero (all variance in K) both are zero
    # and the ratio is taken as its limit, 1, where GMMAT's form gives 0/0.
    ratio = torch.where((final['tau0_e'] == 0) & (tau[:, 0] == 0), torch.ones_like(tau[:, 0]),
                        final['tau0_e'] / tau[:, 0])
    r = families.unrotate(final['py']) * m * ratio
    c1 = families.trace_all / final['tr_pk']
    c2 = c1 * (r ** 2).sum(0) / (n - 1)
    return dict(pseudo=r * c1, c1=c1, c2=c2, tau=tau, alpha=alpha, iterations=iterations, converged=converged)


# ---------------------------------------------------------------------------
# Binary phenotypes
# ---------------------------------------------------------------------------

def _gram(outer, weights, p):
    """sum_n w_n x_n x_n' per phenotype, for an N x k weight: k x p x p. `outer` is N x p^2."""
    return (outer.T @ weights).T.reshape(-1, p, p)


def _logistic_regression(y, m, xt, tol):
    """The C++ step 1's logistic regression (FitNullModel.cpp, fitNullModel2) for k phenotypes at once.

    IRLS from the OLS fit, alpha <- alpha + (X'WX)^-1 X'(y - mu), until every
    coefficient moves at most tol, for at most MAX_ITER - 1 steps; then mu,
    W = mu (1 - mu) (zero where missing) and cov = (X'WX)^-1 at the final alpha.
    """
    n, k = y.shape
    p = xt.shape[1]
    outer = (xt[:, :, None] * xt[:, None, :]).reshape(n, p * p)
    alpha = torch.linalg.solve(_gram(outer, m, p), ((m * y).T @ xt).unsqueeze(-1)).squeeze(-1)
    active = torch.ones(k, dtype=torch.bool, device=y.device)
    for _ in range(MAX_ITER - 1):
        mu = torch.sigmoid(xt @ alpha.T)
        step = torch.linalg.solve(_gram(outer, mu * (1 - mu) * m, p),
                                  (((y - mu) * m).T @ xt).unsqueeze(-1)).squeeze(-1)
        alpha = torch.where(active[:, None], alpha + step, alpha)
        active &= ~(step.abs() <= tol).all(1)
        if not active.any():
            break
    mu = torch.sigmoid(xt @ alpha.T)
    w = mu * (1 - mu) * m
    cov = torch.cholesky_inverse(torch.linalg.cholesky(_gram(outer, w, p)))
    return dict(alpha=alpha, mu=mu, w=w, cov=cov, outer=outer, converged=~active)


def _fit_logistic(glm, y, m, xt):
    """No kinship: the logistic regression is the null model (GMMAT.cpp, glmmkin_fit_cs).

    Sigma^-1 = diag(W) and K = I over the N analysed samples, so
    c1 = N / (sum W - tr(cov X'W^2X)); the scaled residual is y - mu.
    """
    n, k = y.shape
    w = glm['w']
    tr_p = w.sum(0) - torch.einsum('kpq,kqp->k', glm['cov'], _gram(glm['outer'], w * w, xt.shape[1]))
    c1 = n / tr_p
    r = (y - glm['mu']) * m
    return dict(pseudo=c1 * r, c1=c1, c2=c1 * (r ** 2).sum(0) / (n - 1),
                tau=torch.stack((torch.ones_like(c1), torch.zeros_like(c1)), 1), alpha=glm['alpha'],
                iterations=torch.zeros(k, dtype=torch.int64, device=y.device),
                converged=torch.ones(k, dtype=torch.bool, device=y.device))


class _Blocks:
    """The kinship's families as dense blocks on the device, for the binary model (module docstring).

    Sigma = diag(1/W) + tau_g K is inverted family by family and per phenotype as

        Sigma_b^-1 = W^1/2 (I + tau_g W^1/2 K_b W^1/2)^-1 W^1/2,

    which is zero in the rows and columns of a member with W = 0: that is how a
    missing phenotype value is carried. A singleton's is W / (1 + tau_g K_ii W).
    Samples are reordered (`order`): singletons first, then the families of
    each size with their members adjacent, so a family size's rows of an
    N x k array are a B x s x k view.
    """

    def __init__(self, families, xt):
        self.n, self.p = xt.shape
        self.trace_all = families['trace']
        related = [(size, index, kin) for size, index, kin in families['groups'] if size > 1]
        lone = torch.ones(self.n, dtype=torch.bool, device=xt.device)
        for _, index, _ in related:
            lone[index.reshape(-1)] = False
        self.order = torch.cat([lone.nonzero()[:, 0]] + [index.reshape(-1) for _, index, _ in related])
        self.lone = int(lone.sum())
        self.x = xt[self.order]
        single = self.x[:self.lone]
        self.outer = (single[:, :, None] * single[:, None, :]).reshape(self.lone, self.p * self.p)
        self.kin_diag = families['diagonal'][self.order, None]
        self.kin_lone = self.kin_diag[:self.lone]
        self.groups, start = [], self.lone
        for size, index, kin in related:
            count = index.shape[0]
            self.groups.append(dict(size=size, count=count, rows=slice(start, start + count * size), kin=kin,
                                    x=self.x[start:start + count * size].view(count, size, self.p)))
            start += count * size

    def _view(self, g, v):
        """A family size's rows of v (N x k) as B x s x k."""
        return v[g['rows']].view(g['count'], g['size'], v.shape[1])

    def inverse(self, tau, w):
        """Sigma^-1 at tau_g (k) and W (N x k): the singletons' diagonal (singletons x k) and each family
        size's blocks (B x k x s x s)."""
        lone = w[:self.lone]
        lone = lone / (1 + tau * self.kin_lone * lone)
        blocks = []
        for g in self.groups:
            root = self._view(g, w).sqrt().transpose(1, 2)
            inner = root[..., :, None] * g['kin'][:, None] * root[..., None, :] * tau[:, None, None]
            inner.diagonal(dim1=2, dim2=3).add_(1)
            blocks.append(root[..., :, None] * torch.linalg.inv(inner) * root[..., None, :])
        return lone, blocks

    def apply(self, inverse, v):
        """Sigma^-1 v for v N x k. A block times a vector is summed elementwise: s is small, and batched
        matrix-vector products of that size are slow."""
        lone, blocks = inverse
        parts = [lone * v[:self.lone]]
        for g, block in zip(self.groups, blocks):
            vb = self._view(g, v).transpose(1, 2)
            parts.append((block * vb[:, :, None, :]).sum(-1).transpose(1, 2).reshape(-1, v.shape[1]))
        return torch.cat(parts)

    def kin_times(self, v):
        """K v for v N x k."""
        parts = [self.kin_lone * v[:self.lone]]
        for g in self.groups:
            parts.append(torch.bmm(g['kin'], self._view(g, v)).reshape(-1, v.shape[1]))
        return torch.cat(parts)

    def reductions(self, inverse):
        """X' Sigma^-1 X and X' Sigma^-1 K Sigma^-1 X (k x p x p), and tr(Sigma^-1 K) (k)."""
        lone, blocks = inverse
        xsx = _gram(self.outer, lone, self.p)
        sxksx = _gram(self.outer, lone * lone * self.kin_lone, self.p)
        tr_sk = (lone * self.kin_lone).sum(0)
        for g, block in zip(self.groups, blocks):
            count, size, k = g['count'], g['size'], block.shape[1]
            sx = torch.bmm(block.reshape(count, k * size, size), g['x']).view(count, k, size, self.p)
            ksx = torch.einsum('bst,bktp->bksp', g['kin'], sx)
            xsx += torch.einsum('bsp,bksq->kpq', g['x'], sx)
            sxksx += torch.einsum('bksp,bksq->kpq', sx, ksx)
            tr_sk += (block * g['kin'][:, None]).sum((0, 2, 3))
        return xsx, sxksx, tr_sk


def _fit_pql(blocks, glm, y, m, tol, max_iter):
    """GMMAT's PQL for a block of binary phenotypes, with its boundary refit.

    A phenotype whose tau_g ends below 1.01 tol is refitted from its logistic
    regression with tau_g fixed at zero: IRLS under Sigma = diag(1/W).
    """
    y, m = y[blocks.order], m[blocks.order]
    fixed = torch.zeros(y.shape[1], dtype=torch.bool, device=y.device)
    out = _pql(blocks, glm['alpha'], y, m, fixed, tol, max_iter)
    redo = (out['tau'][:, 1] < 1.01 * tol).nonzero()[:, 0]
    if redo.numel():
        fit = _pql(blocks, glm['alpha'][redo], y[:, redo], m[:, redo], ~fixed[redo], tol, max_iter)
        out['pseudo'][:, redo] = fit['pseudo']
        for key in ('c1', 'c2', 'tau', 'alpha', 'iterations', 'converged'):
            out[key][redo] = fit[key]
    pseudo = torch.empty_like(out['pseudo'])
    pseudo[blocks.order] = out['pseudo']
    out['pseudo'] = pseudo
    return out


def _pql(blocks, alpha, y, m, fixed, tol, max_iter):
    """GMMAT.cpp's glmmkin_ai for binomial phenotypes, k at once, from their logistic regressions' alpha.

    y (zero where missing) and its 0/1 mask m are N x k in the blocks' sample order.
    """
    x = blocks.x
    n, k = y.shape
    count = m.sum(0)

    def dot(a, b):
        return (a * b).sum(0)

    def working(eta):
        """mu, W and the working response Y at eta (zero where missing)."""
        mu = torch.sigmoid(eta)
        w = mu * (1 - mu) * m
        return mu, w, torch.where(m > 0, eta + (y - mu) / torch.where(m > 0, w, 1.0), 0.0)

    def p_times(inverse, cov, v):         # P v = Sigma^-1 (v - X cov X' Sigma^-1 v)
        sv = blocks.apply(inverse, v)
        return blocks.apply(inverse, v - x @ (cov @ (sv.T @ x).unsqueeze(-1)).squeeze(-1).T)

    def state(tau, w, big_y):
        inverse = blocks.inverse(tau, w)
        xsx, sxksx, tr_sk = blocks.reductions(inverse)
        cov = torch.cholesky_inverse(torch.linalg.cholesky(xsx))
        alpha = (cov @ (blocks.apply(inverse, big_y).T @ x).unsqueeze(-1)).squeeze(-1)
        py = blocks.apply(inverse, big_y - x @ alpha.T)
        kpy = blocks.kin_times(py)
        tr_pk = tr_sk - torch.einsum('kpq,kqp->k', cov, sxksx)
        return dict(inverse=inverse, cov=cov, alpha=alpha, py=py, kpy=kpy, tr_pk=tr_pk, score=dot(py, kpy) - tr_pk)

    mu, w, big_y = working(x @ alpha.T)
    # Starting value: var(Y) / 2, divided by the mean kinship diagonal over the
    # phenotype's samples; then one EM-type step. tau_e stays 1.
    mean_y = (big_y * m).sum(0) / count
    variance = (((big_y - mean_y) * m) ** 2).sum(0) / (count - 1)
    mean_kin = (blocks.kin_diag * m).sum(0) / count
    tau = torch.where(fixed, 0.0, variance / 2 / mean_kin)
    if not fixed.all():
        s0 = state(tau, w, big_y)
        tau = torch.where(fixed, tau, torch.clamp(tau + tau ** 2 * s0['score'] / count, min=0))

    active = torch.ones(k, dtype=torch.bool, device=y.device)
    converged = torch.zeros_like(active)
    iterations = torch.zeros(k, dtype=torch.int64, device=y.device)
    tr_pk = torch.zeros_like(tau)
    for _ in range(max_iter):
        tau0, alpha0 = tau, alpha
        s = state(tau0, w, big_y)
        alpha = torch.where(active[:, None], s['alpha'], alpha0)
        ai = dot(s['kpy'], p_times(s['inverse'], s['cov'], s['kpy']))
        step = torch.where(fixed, 0.0, s['score'] / ai)

        def bounded(delta):
            new = tau0 + delta
            return torch.where((new < tol) & (tau0 < tol), 0.0, new)
        new_tau = bounded(step)
        for _ in range(100):  # halve until not negative
            negative = new_tau < 0
            if not negative.any():
                break
            step = torch.where(negative, step / 2, step)
            new_tau = bounded(step)
        new_tau = torch.where(new_tau < tol, 0.0, new_tau)
        tau = torch.where(active, new_tau, tau0)
        # eta = Y - diag(1/W) P Y, then mu, W and Y from it.
        eta = torch.where(m > 0, big_y - s['py'] / torch.where(m > 0, w, 1.0), 0.0)
        mu_new, w_new, y_new = working(eta)
        mu = torch.where(active, mu_new, mu)
        w = torch.where(active, w_new, w)
        big_y = torch.where(active, y_new, big_y)
        iterations += active.long()
        # This iteration's starting point is what c1 uses.
        tr_pk = torch.where(active, s['tr_pk'], tr_pk)
        change = torch.maximum(((alpha - alpha0).abs() / (alpha.abs() + alpha0.abs() + tol)).amax(1),
                               (tau - tau0).abs() / (tau.abs() + tau0.abs() + tol))
        done = active & (2 * change < tol)
        diverged = active & (tau > tol ** -2)
        converged |= done
        active &= ~(done | diverged)
        if not active.any():
            break

    r = (y - mu) * m
    c1 = blocks.trace_all / tr_pk
    c2 = c1 * (r ** 2).sum(0) / (n - 1)
    return dict(pseudo=c1 * r, c1=c1, c2=c2, tau=torch.stack((torch.ones_like(tau), tau), 1), alpha=alpha,
                iterations=iterations, converged=converged)


# ---------------------------------------------------------------------------
# Step 1
# ---------------------------------------------------------------------------

def write_correction_file(path, traits, c2, ids, pseudo, threads=8):
    """The C++ step 1's correction file: header, '#' row of c2, then one row per analysed sample.

    c2 is written at full double precision. A scaled residual is written as
    the shortest text that reads back as the same float32: step 2 converts the
    residuals to float32 on reading, so it gets the values it would from full
    double precision, from about half the text. (The C++ step 1 prints six
    significant digits; step 2 reads either.) Row chunks are formatted on
    `threads` threads and written in order.
    """
    import pyarrow as pa
    import pyarrow.csv as pacsv
    out = Path(path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as handle:
        handle.write('sample_id\t' + '\t'.join(traits) + '\n')
        handle.write('#' + ''.join('\t' + repr(float(v)) for v in c2) + '\n')
    ids = pa.array(ids, type=pa.string())
    values = np.asfortranarray(pseudo, dtype=np.float32)   # each phenotype's column contiguous
    names = ['sample_id'] + [f'c{k}' for k in range(values.shape[1])]
    options = pacsv.WriteOptions(include_header=False, delimiter='\t', quoting_style='none')

    def text(first):
        rows = slice(first, first + step)
        table = pa.Table.from_arrays([ids[rows]] + [pa.array(values[rows, k]) for k in range(values.shape[1])],
                                     names=names)
        sink = pa.BufferOutputStream()
        pacsv.write_csv(table, sink, write_options=options)
        return sink.getvalue()
    step = max(1, -(-len(ids) // (4 * threads)))
    with open(out, 'ab') as handle, ThreadPoolExecutor(threads) as pool:
        for chunk in pool.map(text, range(0, len(ids), step)):
            handle.write(chunk)


def null_table_path(corr_file):
    """Where step 1 (--exact-score) writes each phenotype's null model for step 2: beside the correction file."""
    return Path(str(corr_file) + '.null.tsv')


def write_null_table(path, traits, fit, observed):
    """One row per phenotype, in the correction file's order: model, tau_e, tau_g, c1, c2, observed samples.

    Step 2's exact score test (ExactScoreGPU.py) reads it: P y = e / c1, and
    Sigma = tau_e I + tau_g K over the phenotype's observed samples.
    """
    with open(path, 'w') as handle:
        handle.write('phenotype\tmodel\ttau_e\ttau_g\tc1\tc2\tn_observed\n')
        for k, name in enumerate(traits):
            model = 'logistic' if fit['binary'][k] else 'linear'
            handle.write(f'{name}\t{model}\t{float(fit["tau"][k, 0])!r}\t{float(fit["tau"][k, 1])!r}\t'
                         f'{float(fit["c1"][k])!r}\t{float(fit["c2"][k])!r}\t{int(observed[k])}\n')


def fit_step1_gpu(args, pheno_delim, cov_delim, kin_delim, log=print, keep_on_device=False):
    """Run step 1 on the GPU and write args.corr_file: (in_memory, None), or (None, why the C++ step 1 should run).

    in_memory is None unless keep_on_device; then it holds what step 2 reads
    from the correction file, as read_correction_file returns it (header, c2,
    residuals, sample IDs), with the residuals the float32 values on the GPU.
    """
    if not torch.cuda.is_available():
        return None, 'no CUDA device'
    started = time.time()
    inputs, reason = read_step1_inputs(args, pheno_delim, cov_delim)
    if reason is not None:
        return None, reason
    device = torch.device('cuda', torch.cuda.current_device())
    binary, reason = check_binary_coding(inputs['y'], inputs['traits'], device)
    if reason is not None:
        return None, reason
    kinship = (read_kinship(args.kin_file, kin_delim, inputs['ids'], args.kin_diag) if args.kin_file else None)
    if kinship is not None:
        kinship, note = relatives_only(kinship, device, getattr(args, 'kin_threshold', None))
        if note:
            log(f'GPU null model: {note}')
    read_seconds = time.time() - started
    fitted = time.time()
    exact = bool(getattr(args, 'exact_score', False))
    fit = fit_null_model(inputs['y'], inputs['x'], kinship, device, names=inputs['traits'], binary=binary,
                         keep_on_device=keep_on_device, exact=exact)
    if fit['clipped']:
        log(f'GPU null model: {fit["clipped"]} kinship families had a negative eigenvalue (lowest '
            f'{fit["lowest_eigenvalue"]:.4g}); each was replaced by its nearest positive semi-definite matrix')
    torch.cuda.synchronize(device)
    fit_seconds = time.time() - fitted
    written = time.time()
    threads = min(8, getattr(args, 'threads', None) or len(os.sched_getaffinity(0)))
    write_correction_file(args.corr_file, inputs['traits'], fit['c2'], inputs['ids'], fit['pseudo'], threads)
    if exact:
        write_null_table(null_table_path(args.corr_file), inputs['traits'], fit,
                         (~np.isnan(inputs['y'])).sum(0))
    write_seconds = time.time() - written
    summary = (f'GPU null model: {len(inputs["ids"])} samples, {len(inputs["traits"])} phenotypes '
               f'({int(fit["binary"].sum())} binary), {0 if kinship is None else len(kinship.values)} related pairs; '
               f'{int(fit["converged"].sum())} converged, at most {int(fit["iterations"].max(initial=0))} '
               f'AI iterations; reading {read_seconds:.2f} s, fitting {fit_seconds:.2f} s, '
               f'writing {write_seconds:.2f} s')
    log(summary)
    if args.null_log:
        with open(args.null_log, 'a') as handle:
            handle.write(summary + '\n')
            handle.write('phenotype\tmodel\ttau_e\ttau_g\tc1\tc2\titerations\tconverged\n')
            for k, name in enumerate(inputs['traits']):
                model = 'logistic' if fit['binary'][k] else 'linear'
                handle.write(f'{name}\t{model}\t{fit["tau"][k, 0]:.10g}\t{fit["tau"][k, 1]:.10g}\t'
                             f'{fit["c1"][k]:.10g}\t{fit["c2"][k]:.10g}\t{fit["iterations"][k]}\t'
                             f'{int(fit["converged"][k])}\n')
    if not keep_on_device:
        return None, None
    return (['sample_id'] + list(inputs['traits']), fit['c2'], fit['pseudo_device'], list(inputs['ids'])), None
