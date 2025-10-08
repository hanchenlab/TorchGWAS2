import os
import ctypes

# More aggressive MKL fixes
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'

# Try to force load a working MKL
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

import sys
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import torch

# Set path for custom modules
sys.path.append("build")  
sys.path.append("pymodules")  

# Import custom modules
import Mygen
from pymodules.ConfigueOpt import ConfOpt
from torchgwas import read_correction_file, run_gwas
from Mygen import GEMRunner

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

def report_array_info(name, arr):
    """Report information about numpy arrays"""
    print(f"{name}:")
    print(f"  shape = {arr.shape}")
    print(f"  elements = {arr.size:,}")
    print(f"  memory ≈ {arr.nbytes / 1e6:.2f} MB\n")

def save_pheno(i, pheno_name, betas, ses, tstats, pvals):
    """Save results for a single phenotype"""
    ph_df = pd.DataFrame({
        "beta": betas[:, i],
        "se": ses[:, i],
        "t_stat": tstats[:, i],
        "p_value": pvals[:, i],
    })
    ph_df.to_csv(f"results_{pheno_name}.csv", sep="\t", index=False)
    print(f"Saved results for phenotype: {pheno_name}")

def save_block(start, end, headers, betas, ses, tstats, pvals, block_id):
    cols = {}
    for i in range(start, end):
        ph = headers[i]
        cols[f"beta_{ph}"] = betas[:, i]
        cols[f"se_{ph}"] = ses[:, i]
        cols[f"t_stat_{ph}"] = tstats[:, i]
        cols[f"pval_{ph}"] = pvals[:, i]
    
    ph_df = pd.DataFrame(cols)
    ph_df.to_csv(f"results_block_{block_id}.csv", sep="\t", index=False)
    print(f"Saved results for block {block_id}: phenotypes {start}-{end-1}")

# Only proceed if all imports worked
try:
    opt_minimal = ConfOpt(
        pheno_file = "example/pheno.txt",
        cov_file = "example/cov.txt",
        delim_pheno = "\t",
        delim_cov = "\t",
        geno_file = "example/SA.bgen",
        sample_file = "",
        do_filters = False,
        use_sample_file = False,
        includeVariantFile = "",
        stream_snps = 100,
        sampleid_header_name = "id",
        random_slope_header_name = "",
        covariates = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"], 
        exposures = [],
        interactions = [],
        missing_key = "NA",
        kin_path = "example/kinfile.txt",
        delim_k = "\t",
        kin_diag = 1,
        threads = 10,
        num_chunks = 10,
        outfile = "test_output.txt"
    )
    print("✓ Minimal configuration created successfully!")
    
except Exception as e:
    print(f"✗ Configuration failed: {e}")

# Initialize the GEM runner with configuration
print("Initializing GEM Runner...")
runner = GEMRunner(opt_minimal.get())
print("GEM Runner initialized successfully!")

# Fit the null model (without genetic effects) with error handling
print("Starting null model fitting...")
start_time = time.time()


runner.run_fit_nullmodel()
null_model_time = time.time() - start_time
print(f"Null model fitting completed in {null_model_time:.2f} seconds")

# Continue with GWAS analysis
print("Starting GWAS analysis...")
gwas_start_time = time.time()

# Choose device (GPU if available, otherwise CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Run GWAS with specified parameters
results = run_gwas(runner, snps_per_chunk=1000, device=device)

gwas_end_time = time.time()
gwas_duration = gwas_end_time - gwas_start_time
print(f"GWAS analysis completed in {gwas_duration:.2f} seconds")

import pandas as pd
df_results = pd.read_parquet("results_buffered.parquet")

print(f"GWAS analysis completed! Results shape: {df_results.shape}")

# Extract phenotype names
beta_cols = [col for col in df_results.columns if col.endswith('_BETA')]
ph_headers = [col.replace('_BETA', '') for col in beta_cols]
for ph in ph_headers:
    beta_col = f"{ph}_BETA"
    se_col = f"{ph}_SE"
    pval_col = f"{ph}_PVAL"
    
    # Calculate t-statistics
    t_stats = df_results[beta_col] / df_results[se_col]
    
    # Calculate p-values (two-tailed test)
    from scipy.stats import norm
    df_results[pval_col] = 2 * (1 - norm.cdf(np.abs(t_stats)))

print("P-values calculated and added to results!")
print(df_results.head())

df_results.to_csv("df_results_tryfinal.txt", sep="\t", index=False)