#!/usr/bin/env python3

import sys
import os
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add path for GEM module
sys.path.append("build-py")
sys.path.append("pymodules")


def calc_t(pheno_normalized, geno, beta, gamma, sqrt_c2, ph_std):
    """
    Compute per-SNP stats using pre-normalized phenotypes and on-device sqrt(c2).

    pheno_normalized: (N, P)
    geno: (M, N)
    beta, gamma: (M, P) work buffers on same device as geno
    sqrt_c2_device: (P,) tensor on same device as inputs
    ph_std: (1, P) phenotype stds computed BEFORE standardizing corrected_res
    """
    N = pheno_normalized.shape[0]
    with torch.no_grad():
        # Compute stds (keep dims for broadcasting)
        # geno_std: (M, 1) across samples; ph_std: (1, P) provided from pre-standardized corrected_res
        geno_std = geno.std(1, keepdim=True, unbiased=False).clamp_min(1e-8)
        ph_std = ph_std.clamp_min(1e-8)

        # Row-wise center and scale genotypes using saved std (so we can reuse geno_std later)
        geno.sub_(geno.mean(1, keepdim=True)).div_(geno_std)

        # Score U = X^T Y written into beta (M,P); then convert to r = U/N
        torch.matmul(geno, pheno_normalized, out=beta)  # (M,N) @ (N,P) -> (M,P)
        beta.div_(N)  # now beta holds r (correlation) when inputs are standardized

        # Keep an unscaled copy of r for SE(r)
        r = beta.clone()  # (M,P)

        # As requested: additionally scale beta by phenotype and SNP stds
        # beta: (M,P) / (1,P) / (M,1) -> (M,P)
        beta.div_(ph_std)
        beta.div_(geno_std)

        # Compute SE from r, then scale SE by the same stds
        gamma.copy_(r)                # start from r
        gamma.pow_(2).sub_(1).div_(2 - N)  # (1 - r^2)/(N - 2)
        torch.sqrt(gamma, out=gamma)  # SE(r)
        gamma.div_(ph_std)            # adjust SE for phenotype scaling
        gamma.div_(geno_std)          # adjust SE for SNP scaling

        # Null-model calibration
        gamma.div_(sqrt_c2.unsqueeze(0))

        beta_coeffs = beta.cpu()
        se = gamma.cpu()
        t_stats = beta.div_(gamma).abs_().neg_().cpu()
        return t_stats, beta_coeffs, se


def run_gwas(runner, snps_per_chunk=1000, device='cuda'):
    """
    Run GWAS using a pre-configured GEMRunner instance.
    """
    # Get C2 values from the runner (try fitting null model first)
    c2_values = None
    try:
        runner.run_fit_nullmodel()
        c2_values = runner.get_c2_values()
        print(f"Fitted null model and obtained C2 values for SE adjustment: {c2_values}")
    except Exception as e:
        print(f"Could not fit null model or get C2 values: {e}")
        print("Proceeding without C2 adjustment")
    if c2_values is None:
        return
    
    # Get corrected residuals from runner SHOULD GET CORRECTED SCALED RESIDUALS
    corrected_res = torch.from_numpy(runner.get_phenotypes()).float()
    #intercept = torch.from_numpy(runner.get_covariates()).float()
    
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    corrected_res = corrected_res.to(device)
    #covariates = covariates.to(device)
    
    n_samples, n_corrected_res = corrected_res.shape
    
    # Center phenotypes with NaN-safe mean and replace NaNs
    #ph_mean = torch.nanmean(phenotypes, dim=0, keepdim=True)
    #phenotypes = phenotypes - ph_mean
    corrected_res = torch.nan_to_num(corrected_res, nan=0.0)
    
    #c = covariates.cpu().numpy()
    #c_mean = np.nanmean(c, axis=0, keepdims=True)
    #c_std = np.nanstd(c, axis=0, keepdims=True)
    #c_std[c_std < 1e-12] = 1.0
    #c = (c - c_mean) / c_std
    #c = np.nan_to_num(c, nan=0.0)
    #covarQ, _ = np.linalg.qr(c)
    #covarQ = torch.from_numpy(covarQ).float().to(device)
    
    #pheno_normalized = phenotypes - torch.matmul(covarQ, torch.matmul(covarQ.T, phenotypes))
    # Save phenotype std BEFORE normalization for scaling beta/gamma
    ph_std_pre = torch.std(corrected_res, dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
    corrected_res = corrected_res / ph_std_pre
    corrected_res = torch.nan_to_num(corrected_res, nan=0.0)
    
    # Precompute sqrt(c2) once on the target device (clamped for stability)
    sqrt_c2 = torch.from_numpy(np.asarray(c2_values)).to(device=device, dtype=torch.float32)
    sqrt_c2.sqrt_()

    # Start dosage streaming from runner
    queue = runner.start_dosage_stream(queue_capacity=10, snps_per_chunk=snps_per_chunk)
    
    all_t_stats = []
    all_beta = []
    all_se = []
    
    # Preallocate device buffers and reuse/slice for smaller final chunks
    geno_tensor = torch.empty(snps_per_chunk, n_samples, device=device)
    beta_tensor = torch.empty(snps_per_chunk, n_corrected_res, device=device)
    gamma_tensor = torch.empty(snps_per_chunk, n_corrected_res, device=device)
    
    for chunk_data in tqdm(queue, desc="Processing SNPs"):
            
        actual_snps = chunk_data.shape[0]
        
        if actual_snps == snps_per_chunk:
            geno = geno_tensor
            beta = beta_tensor
            gamma = gamma_tensor
        else:
            geno = geno_tensor[:actual_snps, :]
            beta = beta_tensor[:actual_snps, :]
            gamma = gamma_tensor[:actual_snps, :]
        
        geno.copy_(chunk_data)
        t_stats, beta_coeffs, se = calc_t(corrected_res, geno, beta, gamma, sqrt_c2, ph_std_pre)
        
        all_t_stats.append(t_stats)
        all_beta.append(beta_coeffs)
        all_se.append(se)
    
    t_statistics = torch.cat(all_t_stats, dim=0)
    beta_coefficients = torch.cat(all_beta, dim=0)
    standard_errors = torch.cat(all_se, dim=0)
    p_values = 2 * torch.special.ndtr(t_statistics)
    
    return {
        't_stats': t_statistics,
        'beta': beta_coefficients,
        'se': standard_errors,
        'p_values': p_values,
    }
