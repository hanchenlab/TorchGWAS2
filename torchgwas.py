#!/usr/bin/env python3

import sys
import os
import numpy as np
import torch
from tqdm import tqdm
import math
import time
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add path for GEM module
sys.path.append("build")
sys.path.append("pymodules")

## function to read corrected residuals and C2
def read_correction_file(file_path: str):
    """
    @brief Read phenotype file with special format.

    @param file_path Path to the correction file.
    @return (header, c2, data_array)
        - header: list of phenotypes name (column names)
        - c2: numpy array of double (values from 2nd line after '#')
        - c_res: numpy 2D array of double (corrected residuals values)
    """
    c2 = np.empty(0, dtype=np.float64)
    c_res = []
    
    with open(file_path, "r") as f:
        header = f.readline().strip().split("\t")
        second_line = f.readline().strip().split("\t")
        if second_line[0].startswith("#"):
            c2 = np.array([float(val) for val in second_line if not val.startswith("#")],
              dtype=np.float64)
                          

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                c_res.append([float(x) for x in parts[2:]])

    c_res = np.array(c_res, dtype=np.float64)

    return header, c2, c_res


def calc_t(corrected_res, geno, beta, gamma, sqrt_c2, ph_std):
    """
    Compute per-SNP stats using pre-normalized phenotypes and on-device sqrt(c2).

    corrected_res: (N, P)
    geno: (M, N)
    beta, gamma: (M, P) work buffers on same device as geno
    sqrt_c2_device: (P,) tensor on same device as inputs
    ph_std: (1, P) phenotype stds computed BEFORE standardizing corrected_res
    """
    N = corrected_res.shape[0]
    with torch.no_grad():
        # Compute stds (keep dims for broadcasting)
        # geno_std: (M, 1) across samples; ph_std: (1, P) provided from pre-standardized corrected_res
        geno_std = geno.std(1, keepdim=True, unbiased=False).clamp_min(1e-8)
        ph_std = ph_std.clamp_min(1e-8)

        # Row-wise center and scale genotypes using saved std (so we can reuse geno_std later)
        # geno.sub_(geno.mean(1, keepdim=True)).div_(geno_std)
        geno_mean = geno.mean(1, keepdim=True)
        geno.sub_(geno_mean).div_(geno_std)
        # Score U = X^T Y written into beta (M,P); then convert to r = U/N
        torch.matmul(geno, corrected_res, out=beta)  # (M,N) @ (N,P) -> (M,P)
        beta.div_(N)  # now beta holds r (correlation) when inputs are standardized

        # Keep an unscaled copy of r for SE(r)
        r = beta.clone()  # (M,P)

        # As requested: additionally scale beta by phenotype and SNP stds
        # beta: (M,P) / (1,P) / (M,1) -> (M,P)
        beta.mul_(ph_std)
        beta.div_(geno_std)

        # Compute SE from r, then scale SE by the same stds
        gamma.copy_(r)                # start from r
        gamma.pow_(2).sub_(1).div_(2 - N)  # (1 - r^2)/(N - 2)
        torch.sqrt(gamma, out=gamma)  # SE(r)
        gamma.mul_(ph_std)            # adjust SE for phenotype scaling
        gamma.div_(geno_std)          # adjust SE for SNP scaling

        # Null-model calibration
        gamma.div_(sqrt_c2.unsqueeze(0))

        # Extract final beta and SE BEFORE computing t-stats
        beta_coeffs = beta.clone().cpu()
        se = gamma.clone().cpu()
        t_stats = beta.div_(gamma).abs_().neg_().cpu()
        return geno_mean, geno_std, t_stats, beta_coeffs, se


def run_gwas(runner, snps_per_chunk=1000, device='cuda',  compress=False):
    """
    Run GWAS using a pre-configured GEMRunner instance.
    """
    # Get C2 values from the runner (try fitting null model first)
    # c2_values = None
    # try:
    #     runner.run_fit_nullmodel()
    #     c2_values = runner.get_c2_values()
    #     print(f"Fitted null model and obtained C2 values for SE adjustment: {c2_values}")
    # except Exception as e:
    #     print(f"Could not fit null model or get C2 values: {e}")
    #     print("Proceeding without C2 adjustment")
    # if c2_values is None:
    #     return
    
    # Get corrected residuals from runner SHOULD GET CORRECTED SCALED RESIDUALS
    #ph_headers, c2_values, corrected_res = read_correction_file("outAddlie.txt") # read corrected_res, c2 and ph_headers from file
    ph_headers, c2_values, corrected_res = read_correction_file("test_output.txt") # read corrected_res, c2 and ph_headers from file
    # corrected_res = torch.from_numpy(runner.get_phenotypes()).float()

    #intercept = torch.from_numpy(runner.get_covariates()).float()
    
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # corrected_res = corrected_res.to(device)
    corrected_res = torch.from_numpy(corrected_res).float().to(device)
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
    queue = runner.start_dosage_stream(queue_capacity=20, snps_per_chunk=snps_per_chunk)
    
    ph_headers = ph_headers[2:]    
    # Preallocate device buffers and reuse/slice for smaller final chunks
    geno_tensor = torch.empty(snps_per_chunk, n_samples, device=device)
    beta_tensor = torch.empty(snps_per_chunk, n_corrected_res, device=device)
    gamma_tensor = torch.empty(snps_per_chunk, n_corrected_res, device=device)
    
    all_beta = []
    all_se = []
    all_t = []
    all_p = []
    all_mean = []
    all_std  = []
    buffer_size = 500_000
    buffer = []
    beta_se_headers = []
    for ph in ph_headers:
        beta_se_headers.append(f"{ph}_BETA")
        beta_se_headers.append(f"{ph}_SE")
    # for chunk_data in tqdm(queue, desc="Processing SNPs"):
            
    #     actual_snps = chunk_data.shape[0]
        
    #     if actual_snps == snps_per_chunk:
    #         geno = geno_tensor
    #         beta = beta_tensor
    #         gamma = gamma_tensor
    #     else:
    #         geno = geno_tensor[:actual_snps, :]
    #         beta = beta_tensor[:actual_snps, :]
    #         gamma = gamma_tensor[:actual_snps, :]

    #     # geno.copy_(chunk_data)
    #     # geno.copy_(torch.from_numpy(chunk_data).float().to(device))
    #     geno.copy_(torch.from_numpy(chunk_data).float())
    #     mean, std, t_stats, beta_coeffs, se = calc_t(corrected_res, geno, beta, gamma, sqrt_c2, ph_std_pre)
    #     pvals = 2 * torch.special.ndtr(t_stats) #added by Sam

    #     # # Convert to numpy
    #     # Only keep beta and SE
    #     b_np = beta_coeffs.cpu().numpy() 
    #     se_np = se.cpu().numpy() 
    #     all_stats = np.stack([b_np, se_np], axis=2)       # (S, P, 2)
    #     all_stats_2d = all_stats.reshape(b_np.shape[0], -1)  # (S, 2*P)

    #     # final_mat = all_stats_2d

    #     buffer.append(all_stats_2d)

    #     if sum(len(x) for x in buffer) >= buffer_size:
    #         df = pd.DataFrame(np.vstack(buffer), columns=beta_se_headers)
    #         df.to_csv("results_buffered.csv", mode="a",
    #                 header=not os.path.exists("results_buffered.csv"),
    #                 index=False)
    #         buffer = []


    # if buffer:
    #     df = pd.DataFrame(np.vstack(buffer), columns=beta_se_headers)
    #     df.to_csv("results_buffered.csv", mode="a", header=not os.path.exists("results_buffered.csv"), index=False)
  #####using parquet
    start = time.time()
    writer = None
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

        geno.copy_(torch.from_numpy(chunk_data).float())
        mean, std, t_stats, beta_coeffs, se = calc_t(
            corrected_res, geno, beta, gamma, sqrt_c2, ph_std_pre
        )

        # Convert to numpy
        b_np  = beta_coeffs.cpu().numpy()  # (num_snps, num_pheno)
        se_np = se.cpu().numpy()           # (num_snps, num_pheno)

        # Stack [beta, se] → shape (num_snps, num_pheno*2)
        all_stats = np.stack([b_np, se_np], axis=2)
        all_stats_2d = all_stats.reshape(b_np.shape[0], -1)

        buffer.append(all_stats_2d)

        # Flush when buffer full
        if sum(len(x) for x in buffer) >= buffer_size:
            df = pd.DataFrame(np.vstack(buffer), columns=beta_se_headers)
            table = pa.Table.from_pandas(df)

            if writer is None:
                writer = pq.ParquetWriter(
                    "results_buffered_ex.parquet", table.schema, compression="snappy"
                )
            writer.write_table(table)
            buffer = []


    # Flush remainder
    if buffer:
        df = pd.DataFrame(np.vstack(buffer), columns=beta_se_headers)
        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(
                "results_buffered_ex.parquet", table.schema, compression="snappy"
            )
        writer.write_table(table)

    # Close writer
    if writer:
        writer.close()
    
    end = time.time()
    print(f"time for chunck = {end - start}")

    #Read the parquet file head
    df_check = pd.read_parquet("results_buffered_ex.parquet")

    # Show first 5 rows
    print(df_check.head())

    # Show the column names
    print(df_check.columns.tolist())
