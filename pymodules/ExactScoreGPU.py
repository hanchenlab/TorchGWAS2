"""
@brief Step 2's exact mixed-model score test (--exact-score), for quantitative phenotypes.

The C++ step 2's statistic is TorchGWAS2's calibrated one: with e = c1 P y,
z = sqrt(c2) r sqrt(N - 2) / sqrt(1 - r^2), r the correlation of e with the
covariate-adjusted genotype. For small r that is U / sqrt(g~'g~ / c1): it
takes the score's variance g'Pg to be an OLS sum of squares divided by one
constant per phenotype, right on average over variants, not for each.

This computes the score test itself (GMMAT's), each phenotype k on its own
observed samples S_k with its own null model, Sigma_k = tau_e I + tau_g K over
S_k and P_k = Sigma^-1 - Sigma^-1 X (X' Sigma^-1 X)^+ X' Sigma^-1:

    U = g' P y,   V = g' P g = g' Sigma^-1 g - b' (X' Sigma^-1 X)^+ b,   b = X' Sigma^-1 g,
    z = U / sqrt(V),   BETA = U / V,   SE = 1 / sqrt(V).

Sigma is block diagonal over the kinship's families. A sample without a
relative has its own weight per phenotype, 1 / (tau_e + tau_g K_ii), and 0
where the phenotype is missing. Each family of at most DENSE_FAMILY samples
keeps each phenotype's block of Sigma^-1, restricted to its observed members
(a missing member's row and column zero). Per chunk of variants:

    U        = G (e / c1)                                one product
    g'S g    = (G o G) w  +  family pair products         w the samples' weights (N x K)
    b        = (G o x_j) w  +  family calls x (S_b X_b)   for each column x_j of the design

so about p + 4 products of the chunk's size, p the covariates with the
intercept, and the same result every run (no atomic additions). The stored
blocks take (sum of family sizes squared) x K floats. A larger family is
rotated into its kinship block's eigenbasis, as step 1 does
(NullModelGPU._Families), where Sigma^-1 is diagonal and its members join the
weighted products; a phenotype missing one of its members gets its own
eigenbasis for that family (a patch, (family size)^2 floats).

Per-sample weights are what a binary phenotype (W_ik = mu (1 - mu)) needs
too, and a missing value is a weight of 0 either way.

The inputs are step 1's (NullModelGPU, --exact-score): e with NaN where a
phenotype is missing, and each phenotype's tau and c1 (null_table_path). The
kinship is read as step 1 read it (read_kinship, relatives_only, families).
"""
from pathlib import Path

import numpy as np
import torch

# Families up to this size keep a block of Sigma^-1 per phenotype; larger ones are rotated.
DENSE_FAMILY = 16
# Cells (variant rows x pair or patch columns) per block of family products.
PRODUCT_CELLS = 1 << 26
# V below this fraction of g' Sigma^-1 g: the variant is (nearly) a combination of the covariates
# on the phenotype's samples, where float32 cannot resolve V; its statistics are NaN.
RELATIVE_FLOOR = 1e-5


def _null_model_module():
    """NullModelGPU, as RunTorchGWAS imports it (pymodules.NullModelGPU) or beside this file."""
    if __package__:
        from . import NullModelGPU
    else:
        import NullModelGPU
    return NullModelGPU


class ExactScore:
    """The exact score test for a set of quantitative phenotypes on the device; call it on a chunk of dosages."""

    def __init__(self, families, tau, c1, residuals, x, device):
        """families: NullModelGPU.Kinship.families() (None without kinship); tau (K x 2); c1 (K);
        residuals: e (N x K, NaN where missing); x: the design [1, covariates] (N x p)."""
        null_model = _null_model_module()
        device = torch.device(device)
        t = dict(dtype=torch.float64, device=device)
        e = torch.as_tensor(residuals).to(**t)
        observed = ~torch.isnan(e)
        n, traits = e.shape
        x = torch.as_tensor(x).to(**t)
        p = x.shape[1]
        tau = torch.as_tensor(np.asarray(tau, dtype=np.float64), **t)
        self.py = (torch.where(observed, e, 0.0) / torch.as_tensor(np.asarray(c1, dtype=np.float64), **t)).float()
        m = observed.to(torch.float64)
        del e, observed
        groups = [] if families is None else [g for g in families['groups'] if g[0] > 1]
        dense = [g for g in groups if g[0] <= DENSE_FAMILY]
        large = [g for g in groups if g[0] > DENSE_FAMILY]
        diagonal = torch.ones(n, **t) if families is None else families['diagonal'].to(**t)
        in_dense = torch.zeros(n, dtype=torch.bool, device=device)
        for _, index, _ in dense:
            in_dense[index.reshape(-1)] = True
        budget = null_model._budget(device)
        self.w = torch.empty((n, traits), dtype=torch.float32, device=device)
        information = torch.empty((traits, p, p), **t)

        # Samples without relatives and, rotated, the large families: one weight per coordinate.
        fam = null_model._Families(dict(families, groups=large), x, device) if large else None
        self.x = x.float() if fam is None else fam.x.float()
        self.rotations = [] if fam is None else [(g['index'], g['u'].float()) for g in fam.groups]
        outer = (x[:, :, None] * x[:, None, :]).reshape(n, p * p) if fam is None else None
        patches = {}
        batch = int(max(1, min(traits, budget // (48 * n))))
        for first in range(0, traits, batch):
            cols = slice(first, min(traits, first + batch))
            if fam is None:
                w = m[:, cols] / (tau[None, cols, 0] + tau[None, cols, 1] * diagonal[:, None])
                w[in_dense] = 0
                information[cols] = null_model._gram(outer, w, p)
                self.w[:, cols] = w
            else:
                fam.load(m[:, cols], m[:, cols])
                w = fam.obs / (tau[None, cols, 0] + tau[None, cols, 1] * fam.lam)
                w[in_dense] = 0
                information[cols] = fam.gram([w])[0]
                self.w[:, cols] = w * fam.shared
                for patch in fam.patches:
                    patches.setdefault(patch['rows'].shape[1], []).append(dict(
                        rows=patch['rows'], col=patch['col'] + first, u=patch['u'].float(), x=patch['x'].float(),
                        w=w[patch['rows'], patch['col'][:, None]].float()))
            del w
        del fam, outer
        # Patches of one family size, together: rows (J x s), col (J), u (J x s x s), x (J x s x p), w (J x s).
        self.patches = [{key: torch.cat([part[key] for part in parts]) for key in parts[0]}
                        for parts in patches.values()]

        # Small families: each phenotype's block of Sigma^-1 over the observed members, and S_b X_b.
        width = sum(index.numel() for _, index, _ in dense)
        self.related = torch.cat([index.reshape(-1) for _, index, _ in dense]) if dense else None
        self.cross = torch.empty((width, traits, p), dtype=torch.float32, device=device) if dense else None
        self.blocks = []
        start = 0
        for size, index, kin in dense:
            count = index.shape[0]
            rows = slice(start, start + count * size)
            blocks = torch.empty((count * size * size, traits), dtype=torch.float32, device=device)
            xb = x[index]                                                  # B x s x p
            step = int(max(1, min(traits, budget // (8 * count * size * (4 * size + 2 * p)))))
            for first in range(0, traits, step):
                cols = slice(first, min(traits, first + step))
                mb = m[index][:, :, cols].permute(0, 2, 1)                 # B x k x s
                sigma = (tau[None, cols, 0, None, None] * torch.eye(size, **t)
                         + tau[None, cols, 1, None, None] * kin[:, None])
                # Restricted to the observed members: a missing one has an identity row and column, then zero.
                sigma = sigma * mb[..., :, None] * mb[..., None, :] + torch.diag_embed(1.0 - mb)
                inverse = torch.linalg.inv(sigma) * mb[..., :, None] * mb[..., None, :]
                sx = inverse @ xb[:, None]                                 # B x k x s x p
                information[cols] += torch.einsum('bsp,bksq->kpq', xb, sx)
                blocks[:, cols] = inverse.permute(0, 2, 3, 1).reshape(count * size * size, -1)
                self.cross[rows, cols] = sx.permute(0, 2, 1, 3).reshape(count * size, -1, p)
            self.blocks.append(dict(size=size, count=count, rows=rows, blocks=blocks))
            start += count * size
        if self.cross is not None:
            self.cross = self.cross.view(width, traits * p)

        # (X' Sigma^-1 X)^+: a covariate constant on a phenotype's samples leaves it singular.
        values, vectors = torch.linalg.eigh(information)
        keep = values > 1e-9 * values.amax(-1, keepdim=True).clamp_min(1e-300)
        self.cov = (vectors * torch.where(keep, 1.0 / values, 0.0)[:, None, :]) @ vectors.transpose(1, 2)
        self.p, self.traits = p, traits

    def __call__(self, g):
        """BETA, SE and z (chunk x K, float32) for dosages g (chunk x N, on the device)."""
        # Centred: the intercept absorbs the shift (X' P = 0), and float32 keeps the digits that V needs.
        g = g.float()
        g = g - g.mean(1, keepdim=True)
        chunk, traits, p = g.shape[0], self.traits, self.p
        score = (g @ self.py).double()                                             # g' P y
        rotated = g.clone() if self.rotations else g
        for index, u in self.rotations:
            rotated[:, index] = torch.einsum('bji,cbj->cbi', u, g[:, index])
        a = ((rotated * rotated) @ self.w).double()                                # g' Sigma^-1 g
        b = torch.empty((chunk, traits, p), dtype=torch.float64, device=g.device)  # X' Sigma^-1 g
        for j in range(p):
            b[:, :, j] = ((rotated * self.x[:, j]) @ self.w).double()
        del rotated
        if self.related is not None:
            calls = g.index_select(1, self.related)
            b += (calls @ self.cross).double().view(chunk, traits, p)
            for group in self.blocks:
                step = max(1, PRODUCT_CELLS // (group['count'] * group['size'] ** 2))
                for first in range(0, chunk, step):
                    block = calls[first:first + step, group['rows']].view(-1, group['count'], group['size'])
                    pairs = (block[..., :, None] * block[..., None, :]).reshape(block.shape[0], -1)
                    a[first:first + step] += (pairs @ group['blocks']).double()
        for patch in self.patches:
            step = max(1, PRODUCT_CELLS // patch['rows'].numel())
            for first in range(0, chunk, step):
                part = g[first:first + step][:, patch['rows']]                     # c x J x s
                turned = torch.einsum('jsi,cjs->cji', patch['u'], part)
                weighted = turned * patch['w']
                a[first:first + step].index_add_(1, patch['col'], (weighted * turned).sum(-1).double())
                b[first:first + step].index_add_(1, patch['col'],
                                                 torch.einsum('cji,jip->cjp', weighted, patch['x']).double())
        v = a - torch.einsum('ckp,kpq,ckq->ck', b, self.cov, b)                  # g' P g
        valid = torch.isfinite(v) & (v > RELATIVE_FLOOR * a) & (v > 0)
        safe = torch.where(valid, v, torch.ones_like(v))
        nan = torch.full_like(v, float('nan'))
        beta = torch.where(valid, score / safe, nan)
        se = torch.where(valid, torch.rsqrt(safe), nan)
        z = torch.where(valid, score * torch.rsqrt(safe), nan)
        return beta.float(), se.float(), z.float()


class ExactScoreSpec:
    """What step 2 needs to build the exact score test: step 1's null-model table and the kinship settings."""

    def __init__(self, table, kin_file=None, kin_delim='\t', kin_diag=1.0, kin_threshold=None):
        self.table, self.kin_file, self.kin_delim = Path(table), kin_file, kin_delim
        self.kin_diag, self.kin_threshold = kin_diag, kin_threshold

    @classmethod
    def from_args(cls, args, corr_file, kin_delim):
        return cls(_null_model_module().null_table_path(corr_file), args.kin_file or None, kin_delim, args.kin_diag,
                   getattr(args, 'kin_threshold', None))

    def build(self, traits, sample_ids, residuals, x, device, log=print):
        """(ExactScore for the linear phenotypes, their columns) for step 2's residuals (N x K, NaN where missing)."""
        null_model = _null_model_module()
        if not self.table.exists():
            raise FileNotFoundError(f'--exact-score needs {self.table}, which step 1 writes on the GPU with '
                                    '--exact-score (--null-device cuda); run step 1 that way first')
        rows = {}
        with open(self.table) as handle:
            header = handle.readline().rstrip('\n').split('\t')
            for line in handle:
                values = dict(zip(header, line.rstrip('\n').split('\t')))
                rows[values['phenotype']] = values
        missing = [name for name in traits if name not in rows]
        if missing:
            raise ValueError(f'{self.table} has no null model for {missing[:5]}: it is not this correction file\'s')
        columns = np.array([k for k, name in enumerate(traits) if rows[name]['model'] == 'linear'], dtype=np.int64)
        if columns.size < len(traits):
            log(f'exact score test: {columns.size} quantitative phenotypes; {len(traits) - columns.size} binary '
                'ones keep the calibrated statistic')
        if not columns.size:
            return None, columns
        tau = np.array([[float(rows[traits[k]]['tau_e']), float(rows[traits[k]]['tau_g'])] for k in columns])
        c1 = np.array([float(rows[traits[k]]['c1']) for k in columns])
        families = None
        if self.kin_file:
            kinship = null_model.read_kinship(self.kin_file, self.kin_delim, list(sample_ids), self.kin_diag)
            kinship, _ = null_model.relatives_only(kinship, device, self.kin_threshold)
            families = kinship.families(device)
        residuals = torch.as_tensor(residuals)
        e = residuals[:, torch.as_tensor(columns, device=residuals.device)]
        return ExactScore(families, tau, c1, e, x, device), columns
