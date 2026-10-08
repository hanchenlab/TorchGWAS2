import sys
import os
import numpy as np
import torch
from tqdm import tqdm
import math
import time
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
import gc

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
    sample_ids = []
    
    with open(file_path, "r") as f:
        header = f.readline().strip().split("\t")
        second_line = f.readline().strip().split("\t")
        if second_line[0].startswith("#"):
            c2 = np.array([float(val) for val in second_line if not val.startswith("#")],
              dtype=np.float64)
                          

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                sample_ids.append(parts[0])
                c_res.append([float(x) for x in parts[1:]])

    c_res = np.array(c_res, dtype=np.float64)

    return header, c2, c_res, sample_ids


def remove_collinear_columns_np(X: np.ndarray, col_names=None):
    """
    Remove (near-)collinear columns using QR diag threshold.

    Parameters
    ----------
    X : (n, p) np.ndarray
        Design matrix.
    col_names : list[str] | None
        Optional column names length p.
    keep_first : bool
        If True, never drop column 0 (useful for intercept).

    Returns
    -------
    X_new : np.ndarray
    """
    print("Checking collinearity for the regressed covariates...\n")
    X = np.asarray(X)
    n, p = X.shape

    # QR decomposition (like HouseholderQR)
    # R is shape (min(n,p), p) in 'reduced' mode.
    _, R = np.linalg.qr(X, mode="reduced")

    # Make diagR length = p (pad zeros if p > n)
    diagR = np.zeros(p, dtype=float)
    d = min(n, p)
    if d > 0:
        diagR[:d] = np.abs(np.diag(R[:d, :d]))

    sqrtEps = np.sqrt(np.finfo(X.dtype).eps)
    maxdiag = diagR.max() if p > 0 else 0.0
    cutoff = maxdiag * sqrtEps

    dropped_idx = [j for j in range(p) if diagR[j] < cutoff]

    if dropped_idx:
        dropped_names = []
        for j in dropped_idx:
            if j == 0:
                # intercept got flagged 
                dropped_names.append("intercept")
            else:
                jj = j - 1
                if 0 <= jj < len(col_names):
                    dropped_names.append(col_names[jj])
        print(f"Dropped columns: {dropped_names}")
    else:
        print("Dropped columns: []")

    dropped_set = set(dropped_idx)
    keep_idx = [j for j in range(p) if j not in dropped_set]

    X_new = X[:, keep_idx]

    if col_names is not None:
        new_col_names = []
        for j in keep_idx:
            if j == 0:
                # skip intercept:
                continue
            else:
                jj = j - 1
                if 0 <= jj < len(col_names):
                    new_col_names.append(col_names[jj])

    return X_new, new_col_names


def has_duplicates(resid_sample_ids):
    return len(resid_sample_ids) != len(set(resid_sample_ids))


def fill_J(resid_sample_ids, cov_sample_ids, device="cpu", dtype=torch.float32):
    
    """
    resid_sample_ids: observation IDs in covariate order (after filtering)
    sample_id_for_G:  unique IDs in the SAME order as G columns
    """
    id2col = {sid: j for j, sid in enumerate(resid_sample_ids)}  # required for alignment to G
    n_obs = len(cov_sample_ids)
    n_unique = len(resid_sample_ids)

    col_idx = torch.tensor([id2col[sid] for sid in cov_sample_ids],
                           device=device, dtype=torch.long)
    row_idx = torch.arange(n_obs, device=device, dtype=torch.long)

    indices = torch.stack([row_idx, col_idx], dim=0)
    values  = torch.ones(n_obs, device=device, dtype=dtype)

    J = torch.sparse_coo_tensor(indices, values, (n_obs, n_unique)).coalesce()
    return J



def calc_cov_proj(runner_opt, resid_sample_ids, device):
    """
    geno_ids: unique IDs in the SAME order as G columns  (n_unique)
    obs_ids: observation IDs in covariate order (after filtering)
    Returns:
      proj_A: (p x n_unique) if no-dup case, else None
      cov_X:  (n_unique x p) if no-dup case, else (n_obs x p)
      has_dup: bool (dup IDs in cov file after filtering to genotype IDs)
      J: (n_obs x n_unique) if dup case, else None
      obs_ids: list[str]  (IDs in cov_X row order)
    """
    # Treat input resid_sample_ids as genotype sample IDs in G column order (unique)
    geno_id_set = set(resid_sample_ids)

    cov_file = runner_opt.cov_add
    cov_names = list(runner_opt.covariates) if hasattr(runner_opt, "covariates") else []
    sep = getattr(runner_opt, "cov_delim", "\t") or "\t"

    cov_df = pd.read_csv(cov_file, sep=sep)

    if not (hasattr(runner_opt, "sampleid_header_name") and runner_opt.sampleid_header_name):
        raise RuntimeError("You must specify --sampleid-name for covariate file.")
    sample_id_col = runner_opt.sampleid_header_name
    if sample_id_col not in cov_df.columns:
        raise RuntimeError(f"Sample ID column '{sample_id_col}' not found in covariate file.")

    sample_ids_from_cov = cov_df[sample_id_col].astype(str).to_numpy()

    if cov_names:
        cov_sel = cov_df[cov_names].copy()
    else:
        cov_sel = cov_df.select_dtypes(include=[np.number]).copy()

    # ---- filter cov rows to genotype IDs, KEEPING COV FILE ORDER ----
    keep_mask = pd.Series(sample_ids_from_cov).isin(geno_id_set).to_numpy()
    cov_sel = cov_sel.iloc[keep_mask].reset_index(drop=True)
    obs_ids = sample_ids_from_cov[keep_mask].tolist()

    has_dup = has_duplicates(obs_ids)

    # Build design matrix (in current obs order)
    cov_np = cov_sel.astype(float).to_numpy()
    intercept = np.ones((cov_np.shape[0], 1), dtype=cov_np.dtype)
    cov_X_n = np.hstack([intercept, cov_np])  # (n_rows, p)
    cov_X_n, new_cov_nam = remove_collinear_columns_np(cov_X_n, cov_names)
    # move to torch
    cov_X = torch.as_tensor(cov_X_n, device=device, dtype=torch.float32)

    if not has_dup:
        # reorder cov_X rows to match genotype order (resid_sample_ids)
        row_map = {sid: i for i, sid in enumerate(obs_ids)}  # unique now
        row_idx = [row_map[sid] for sid in resid_sample_ids]  # all should exist after filtering
        cov_X = cov_X[row_idx]  # (n_unique, p)

        # projection A = (X^T X)^-1 X^T  (more stable than inv: use solve)
        XtX = cov_X.T @ cov_X
        # # Invert XᵀX
        # inv_XtX = torch.linalg.inv(XtX)
        # # Compute projection A
        # proj_A = inv_XtX @ cov_X.T
        proj_A = torch.linalg.solve(XtX, cov_X.T)  # (p x n_unique)

        J = None
        return proj_A, cov_X, J, has_dup

    else:
        # duplicated obs IDs: keep cov order and build J to map obs->genotype columns
        J = fill_J(resid_sample_ids, obs_ids, device=device, dtype=torch.float32)
        proj_A = None
        return proj_A, cov_X, J, has_dup



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


def run_gwas(runner, corr_file, TGWAS_file, snps_per_chunk=1000, device='cuda', null_model=None):
    """
    Run GWAS using a pre-configured GEMRunner instance.

    null_model: step 1's results already in memory, as read_correction_file
    returns them (the residuals may be a float32 tensor on the GPU); used
    instead of reading them from corr_file.
    """
    # dir_name = os.path.dirname(out_file)
    # base_name = os.path.basename(out_file)
    # corr_file = os.path.join(dir_name, "intermediate_" + base_name)
    overall_start = time.time()

    if null_model is None:
        ph_headers, c2_values, corrected_res, resid_sample_ids = read_correction_file(corr_file) # read corrected_res, c2, ph_headers and sample ids from intermediate file
    else:
        ph_headers, c2_values, corrected_res, resid_sample_ids = null_model

    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # corrected_res = corrected_res.to(device)
    corrected_res = torch.as_tensor(corrected_res).to(device=device, dtype=torch.float32)
    #covariates = covariates.to(device)
    
    n_samples, n_corrected_res = corrected_res.shape
    # Validate sample ids length matches residual rows
    if len(resid_sample_ids) != n_samples:
        print(f"Warning: intermediate file sample ID count ({len(resid_sample_ids)}) != residual rows ({n_samples}).")

    # Center phenotypes with NaN-safe mean and replace NaNs
    corrected_res = torch.nan_to_num(corrected_res, nan=0.0)

    # Save phenotype std BEFORE normalization for scaling beta/gamma
    ph_std_pre = torch.std(corrected_res, dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
    corrected_res = corrected_res / ph_std_pre
    corrected_res = torch.nan_to_num(corrected_res, nan=0.0)
    
    # Precompute sqrt(c2) once on the target device (clamped for stability)
    sqrt_c2 = torch.from_numpy(np.asarray(c2_values)).to(device=device, dtype=torch.float32)
    sqrt_c2.sqrt_()

    # Start dosage streaming from runner
    queue = runner.start_dosage_stream(queue_capacity=40, snps_per_chunk=snps_per_chunk)
    
    ph_headers = ph_headers[1:]    
    # Preallocate device buffers and reuse/slice for smaller final chunks
    geno_tensor = torch.empty(snps_per_chunk, n_samples, device=device)
    beta_tensor = torch.empty(snps_per_chunk, n_corrected_res, device=device)
    gamma_tensor = torch.empty(snps_per_chunk, n_corrected_res, device=device)
    
    buffer_size = 100_000
    cov_X, proj_A, J, has_dup = None, None, None, None
    start_readcov = time.time()
    
    proj_A, cov_X, J, has_dup = calc_cov_proj(
    runner.opt,
    resid_sample_ids,
    device)

    # Precompute duplicate-ID regression constants once per run
    Jt = JT_X = XTX_i_S = counts = None
    if has_dup:
        J = J.coalesce()
        Jt = J.transpose(0, 1).coalesce()              # (n_uniq, n_obs)
        JT_X = torch.sparse.mm(Jt, cov_X)              # (n_uniq, p)
        S = JT_X.T                                     # (p, n_uniq)
        XTX = cov_X.T @ cov_X                          # (p, p)
        XTX_i_S = torch.linalg.solve(XTX, S)           # (p, n_uniq)
        counts = torch.sparse.sum(J, dim=0).to_dense() # (n_uniq,)

    end_readcov = time.time()
    print(f"Time for reading covariate file and preparing projection = {end_readcov - start_readcov:.2f}s")
 
    headers = [
    "SNPID",
    "RSID",
    "CHR",
    "POS",
    "Non_Effect_Allele",
    "Effect_Allele",
    "N_Samples",
    "AF",
    "GV"
    ]

    for ph in ph_headers:
        headers.append(f"{ph}_BETA")
        headers.append(f"{ph}_SE")
        headers.append(f"{ph}_pvalue")

    rows_in_buffer = 0
    buffer = []
    writer = None

    if os.path.exists(TGWAS_file):
        os.remove(TGWAS_file)
    
    # ====================================================================
    # PROFILING: Initialize timing accumulators
    # ====================================================================
    timing_stats = {
        'first_chunk_wait': [],     # Time waiting for first chunk from C++ queue
        'queue_wait': [],           # Time waiting for chunks from C++ queue
        'data_transfer': [],        # Time to transfer data to GPU/CPU
        'genotype_regression': [],  # Time for covariate regression on genotypes
        'gwas_computation': [],     # Time for calc_t (core GWAS math)
        'numpy_conversion': [],     # Time to convert to numpy arrays
        'arrow_formatting': [],     # Time to create Arrow/Parquet structures
        'io_write': [],             # Time for disk I/O (Parquet writing)
        'total_per_chunk': [],       # Total time per chunk
        'queue_size': [],            # Number of chunks in queue
    }
    
    chunk_count = 0
    total_snps_processed = 0
  
    start_before_loop = time.time()
    record_time_before_loop = start_before_loop - overall_start
    for chunk_data, meta in  tqdm(queue, desc="Processing SNPs"):
        # print(f"queue size :{queue.size()}")
        iteration_start = time.time()
        # Queue wait = time from end of last iteration to start of this iteration
        # Separate first chunk (includes C++ startup) from subsequent chunks
        if chunk_count == 0:
            timing_stats['first_chunk_wait'].append(iteration_start - start_before_loop)
        else:
            timing_stats['queue_wait'].append(iteration_start - iteration_end)
        
        # Track queue size (number of chunks currently in queue)
        current_queue_size = queue.size()
        timing_stats['queue_size'].append(current_queue_size)
        
        actual_snps = chunk_data.shape[0]
        rows_in_buffer += actual_snps
        total_snps_processed += actual_snps
        chunk_count += 1

        geno = geno_tensor[:actual_snps, :]
        beta = beta_tensor[:actual_snps, :]
        gamma = gamma_tensor[:actual_snps, :]

        if chunk_data is None:
            raise ValueError("chunk_data is None")

        # ---- Data Transfer ----
        transfer_start = time.time()
        G = torch.as_tensor(chunk_data, dtype=torch.float32, device=device)  # (M, n)
        if device.type == 'cuda':
            torch.cuda.synchronize()  # Ensure transfer completes
        transfer_end = time.time()
        timing_stats['data_transfer'].append(transfer_end - transfer_start)

        # ---- Genotype Regression ----
        regression_start = time.time()
        
        if has_dup:
            GD = G * counts.unsqueeze(0)                   # (M, n_uniq)
            tmp = G @ JT_X                                 # (M, p)
            corr = tmp @ XTX_i_S                           # (M, n_uniq)

            geno.copy_((GD - corr) / counts.unsqueeze(0))                      # (M, n_uniq)

        else:
            coeffs = proj_A @ G.T
            fitted = cov_X @ coeffs
            geno.copy_(G - fitted.T)
            
        if device.type == 'cuda':
            torch.cuda.synchronize()
        regression_end = time.time()
        timing_stats['genotype_regression'].append(regression_end - regression_start)
        
        # ---- GWAS Computation ----
        gwas_start = time.time()
        mean, std, t_stats, beta_coeffs, se = calc_t(
            corrected_res, geno, beta, gamma, sqrt_c2, ph_std_pre
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        gwas_end = time.time()
        timing_stats['gwas_computation'].append(gwas_end - gwas_start)

        # ---- NumPy Conversion ----
        numpy_start = time.time()
        b_np = beta_coeffs.cpu().numpy().astype(np.float32)
        se_np = se.cpu().numpy().astype(np.float32)
        neg_log10_pval = (
            -(torch.log(torch.tensor(2.0)) + torch.special.log_ndtr(t_stats))
            / torch.log(torch.tensor(10.0))
        ).cpu().numpy().astype(np.float32)

        # keep order: BETA, SE, PVAL repeating per phenotype
        all_stats = np.stack([b_np, se_np, neg_log10_pval], axis=2)
        all_stats_2d = all_stats.reshape(b_np.shape[0], -1)
        numpy_end = time.time()
        timing_stats['numpy_conversion'].append(numpy_end - numpy_start)

        # ---- Arrow Formatting ----
        arrow_start = time.time()
        # Convert metadata to Arrow arrays
        meta_arrays = []
        for k, v in meta.items():
            if isinstance(v[0], str):
                # v = [s.rstrip('\x00') for s in v]
                meta_arrays.append(pa.array(v, type=pa.string()))
            elif isinstance(v[0], (int, np.integer)):
                meta_arrays.append(pa.array(v, type=pa.int32()))
            else:
                meta_arrays.append(pa.array(v, type=pa.float32()))

        # Convert numeric results to Arrow arrays
        stat_arrays= [pa.array(col, type=pa.float32()) for col in all_stats_2d.T]
                        
        table= pa.table(meta_arrays + stat_arrays, names=headers)
        buffer.append(table)
        arrow_end = time.time()
        timing_stats['arrow_formatting'].append(arrow_end - arrow_start)

        # ---- Flush buffer ----
        if rows_in_buffer >= buffer_size:
            io_start = time.time()
            combined = pa.concat_tables(buffer)
            # write Feather (Arrow IPC v2)
            # feather.write_feather(combined, out_path + ".feather", compression="zstd")
            if writer is None:
                writer = pq.ParquetWriter(
                TGWAS_file, combined.schema, compression="snappy"
            )
            writer.write_table(combined)
            io_end = time.time()
            timing_stats['io_write'].append(io_end - io_start)
            
            rows_in_buffer = 0
            buffer.clear()
            del combined, b_np, se_np, neg_log10_pval, all_stats, all_stats_2d, table, stat_arrays, meta_arrays
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Update iteration_end at the very end of the iteration (after all processing including I/O)
        iteration_end = time.time()
        timing_stats['total_per_chunk'].append(iteration_end - iteration_start)

    # --- Flush remaining ---
    if buffer:
        io_start = time.time()
        combined = pa.concat_tables(buffer)
        # feather.write_feather(combined, out_path + ".feather", compression="zstd")
        if writer is None:
                writer = pq.ParquetWriter(
                TGWAS_file, combined.schema, compression="snappy"
            )
        writer.write_table(combined)
        io_end = time.time()
        timing_stats['io_write'].append(io_end - io_start)
        
        buffer.clear()
        del combined, b_np, se_np, neg_log10_pval, all_stats, all_stats_2d, table, stat_arrays, meta_arrays
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if writer is not None:
        writer.close()
    
    overall_end = time.time()
    total_time = overall_end - overall_start
    
    # ====================================================================
    # PROFILING: Print detailed timing statistics
    # ====================================================================
    print("\n" + "="*80)
    print("                    PERFORMANCE PROFILING REPORT")
    print("="*80)
    print(f"\nDevice: {device.type.upper()}")
    print(f"Total SNPs processed: {total_snps_processed:,}")
    print(f"Number of chunks: {chunk_count}")
    print(f"Average SNPs per chunk: {total_snps_processed / chunk_count:.1f}")
    print(f"Total time before start reading queue in Run_gwas: {record_time_before_loop:.2f}s")
    print(f"Total wall-clock time for run_gwas: {total_time:.2f}s")
    print(f"SNPs per second: {total_snps_processed / total_time:.1f}")
    
    print("\n" + "-"*80)
    print("TIMING BREAKDOWN (per chunk averages):")
    print("-"*80)
    
    def print_timing(label, times):
        if len(times) == 0:
            return
        total = sum(times)
        mean = total / len(times)
        pct = (total / total_time) * 100
        print(f"{label:.<35} {mean*1000:8.2f} ms  (total: {total:6.2f}s, {pct:5.1f}%)")
    # First chunk wait includes C++ thread startup time
    if timing_stats['first_chunk_wait']:
        first_wait = timing_stats['first_chunk_wait'][0]
        first_pct = (first_wait / total_time) * 100
        print(f"First chunk wait (C++ startup)..... {first_wait*1000:8.2f} ms  (total: {first_wait:6.2f}s, {first_pct:5.1f}%)")
    
    print_timing("Queue waiting time (chunks 2+)", timing_stats['queue_wait'])
    print_timing("Data transfer to device and create tensor", timing_stats['data_transfer'])
    print_timing("Genotype regression", timing_stats['genotype_regression'])
    print_timing("GWAS computation (calc_t)", timing_stats['gwas_computation'])
    print_timing("NumPy conversion", timing_stats['numpy_conversion'])
    print_timing("Arrow/Parquet formatting", timing_stats['arrow_formatting'])
    if timing_stats['io_write']:
        print_timing("Disk I/O (writes)", timing_stats['io_write'])
    print_timing("Total timing per chunk", timing_stats['total_per_chunk'])
    
    # Compute derived metrics
    print("\n" + "-"*80)
    print("DERIVED METRICS:")
    print("-"*80)
    
    # Python overhead = everything except queue wait and GWAS computation
    python_overhead = (sum(timing_stats['data_transfer']) + 
                      sum(timing_stats['numpy_conversion']) + 
                      sum(timing_stats['arrow_formatting']))
    python_overhead_pct = (python_overhead / total_time) * 100
    print(f"Python overhead (data handling): {python_overhead:.2f}s ({python_overhead_pct:.1f}%)")
    
    # GPU/CPU compute time
    compute_time = sum(timing_stats['gwas_computation'])
    compute_time += sum(timing_stats['genotype_regression'])
    compute_pct = (compute_time / total_time) * 100
    print(f"{device.type.upper()} compute time: {compute_time:.2f}s ({compute_pct:.1f}%)")
    
    # Queue efficiency (including first chunk startup)
    queue_wait = sum(timing_stats['queue_wait']) + sum(timing_stats['first_chunk_wait'])
    queue_pct = (queue_wait / total_time) * 100
    queue_wait_ongoing = sum(timing_stats['queue_wait'])  # Excluding first chunk
    print(f"Queue waiting time (total): {queue_wait:.2f}s ({queue_pct:.1f}%)")
    if timing_stats['first_chunk_wait']:
        print(f"  - First chunk (startup): {sum(timing_stats['first_chunk_wait']):.2f}s")
        print(f"  - Ongoing (chunks 2+): {queue_wait_ongoing:.2f}s")
    
    # Queue size statistics
    if timing_stats['queue_size']:
        queue_sizes = timing_stats['queue_size']
        avg_queue_size = sum(queue_sizes) / len(queue_sizes)
        max_queue_size = max(queue_sizes)
        min_queue_size = min(queue_sizes)
        print(f"Queue size statistics:")
        print(f"  - Average chunks in queue: {avg_queue_size:.2f}")
        print(f"  - Min chunks in queue: {min_queue_size}")
        print(f"  - Max chunks in queue: {max_queue_size}")
        print(f"  - Queue capacity: {queue.capacity()}")
    
    # I/O time
    io_time = sum(timing_stats['io_write'])
    io_pct = (io_time / total_time) * 100
    print(f"Disk I/O time: {io_time:.2f}s ({io_pct:.1f}%)")
    print("\n" + "-"*80)
