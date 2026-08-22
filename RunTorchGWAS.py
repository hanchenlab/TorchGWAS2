#!/usr/bin/env python3
import pandas as pd
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "pymodules"))
from pymodules import ConfOpt
from pymodules import GEMRunner
from pymodules import run_gwas
from pymodules import parquet_to_text_duckdb
from pymodules import FDTee
import numpy as np
import time
import argparse
import logging
import io
import tempfile
from pathlib import Path
import re
import traceback
import faulthandler
import torch

# Memory recording
def reset_peak_memory(device_str="cuda:0"):
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(torch.device(device_str))
    except Exception:
        return

def get_peak_memory_gb(device_str="cuda:0") -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(torch.device(device_str)) / 1024**3
    except Exception:
        return 0.0
def setup_step1_log(log_file, mode="a"):
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.propagate = False

    fh = logging.FileHandler(log_file, mode=mode, delay=False)
    fh.setFormatter(logging.Formatter("%(asctime)s [PY] %(levelname)s: %(message)s"))
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("[PY] %(levelname)s: %(message)s"))
    root.addHandler(sh)

_TEE = None

def setup_pipeline_log(log_path: str, mode: str = "a"):
    global _TEE

    if _TEE is None:
        # _TEE = FDTee(log_path, truncate=(mode == "w"))
        _TEE = FDTee(log_path, truncate=False, tee_to_terminal=True)

        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.handlers.clear()

        h = logging.StreamHandler(sys.stderr)  # goes through FDTee -> terminal + log
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s",
                                "%Y-%m-%d %H:%M:%S")
        h.setFormatter(fmt)
        root.addHandler(h)

        def _excepthook(exc_type, exc, tb):
            logging.error("Uncaught Python exception:")
            logging.error("".join(traceback.format_exception(exc_type, exc, tb)).rstrip())
        sys.excepthook = _excepthook
    return log_path



def safe_stem(p: str) -> str:
    s = Path(p).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def normalize_delim(s):
    # Convert to single-character delimiter that C++ expects
    if s in ["\\t", "t", "tab", "TAB", r"\t"]:
        return "\t"
    if s in ["\\0", "0", "space", " "]:
        return " "
    if s == ",":
        return ","
    return s


def parse_args():
    parser = argparse.ArgumentParser(description="Run TorchGWAS using GEM2 and Torch backend.")
    parser.add_argument("--pheno-file", type=str, help="Phenotype file path (required for step1)")
    parser.add_argument("--cov-file", type=str, help="Covariate file path (required for step1 and step2)")
    parser.add_argument("--bgen", nargs="+", default=[], help="Step2: BGEN file(s). Mutually exclusive with --bed/--pgen.")
    parser.add_argument("--bed", nargs="+", default=[], help="Step2: PLINK BED file prefix(es), PLINK --bfile style "
                        "(no extension) -- e.g. --bed example looks for example.bed, example.bim, and example.fam. "
                        "--sample is not required for BED input (pass it only to override the auto-detected .fam). "
                        "Mutually exclusive with --bgen/--pgen.")
    parser.add_argument("--pgen", nargs="+", default=[], help="Step2: PLINK 2 PGEN file prefix(es), PLINK --pfile "
                        "style (no extension) -- e.g. --pgen example looks for example.pgen, example.pvar, and "
                        "example.psam. NOT YET SUPPORTED by the backend (Plink::process_plink_header_block only "
                        "accepts .bed) -- reading a .pgen file currently fails with a clear error at runtime. "
                        "Mutually exclusive with --bgen/--bed.")
    parser.add_argument("--sample", nargs="+", default=[], help="Step2: SAMPLE file(s). Required, one per file, "
                        "when using --bgen. Optional (and ignored unless supplied) when using --bed/--pgen.")
    parser.add_argument("--kin-file", type=str, default="", help="Kinship file path (optional, required for step1 and step2 if using kinship)")
    parser.add_argument("--kin-diag", type=float, default=1.0, help="Diagonal value of " \
                        "kinship matrix that not accounting for inbreeding (Default: 1.0)")
    parser.add_argument("--corr-file", type=str, default="correction.txt", help="correction file path (required for step2)")
    parser.add_argument("--pheno-delim", type=str, default=",", help="Phenotype file delimiter (default: comma)")
    parser.add_argument("--cov-delim", type=str, default=",", help="Covariate file delimiter (default: comma)")
    parser.add_argument("--kin-delim", type=str, default=",", help="Kinship file delimiter (default: comma)")
    parser.add_argument("--sampleid-name", type=str, help="sample ID herader name")
    parser.add_argument("--include-snp-file", type=str, default= "", help="Path to file containing a subset of variants in \
                        the specified genotype file to be used for analysis. The first line in this file is the header that specifies\
                        which variant identifier in the genotype file is used for ID matching. This must be 'snpid' (PLINK or BGEN)\
                        or 'rsid' (BGEN only). There should be one variantidentifier per line after the header.")
    parser.add_argument("--maf", type=float, default=0.001, help="Minimum minor allele frequency threshold; \
                        variants with MAF below this value are excluded (default: 0.001)")
    parser.add_argument("--covar-names", nargs="+", help="Covariate names list")
    parser.add_argument("--random-slope-name", type=str, default = "", help="Column name in the covariate file that contains random slope (default: "").")
    parser.add_argument("--missing-value", type=str, default="NA", help="Indicates how missing values in the phenotype and covariate files are stored.")
    parser.add_argument("--threads", type=int, help="Number of threads")
    parser.add_argument("--stream-snps", type=int, default=1000, help="Number of SNPs per chunk")
    parser.add_argument("--out", type=str, default="out.txt", help="Output file name")
    parser.add_argument("--log", type=str, default="log.log", help="log file path")
    parser.add_argument("--null-log", type=str, default="null_log.log", help="Crash log file")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Computation device (default: cuda)")
    parser.add_argument("--verbose", action="store_true", help="Print null model(default: False)")
    parser.add_argument("--convert", action="store_true", help="Convert binary to text file (default: True)")
    parser.add_argument("--step", choices=["all", "step1", "step2", "step3"], default="all",
                help=(
                    "Pipeline step to run:\n"
                    "all = perform all steps togethet\n"
                    "step1 = Fit the null model generate correction factors (write to correction file)\n"
                    "step2 = TGWAS (write TGWAS_*.parquet)\n"
                    "step3 = Convert TGWAS_*.parquet to .txt\n"
                ))

    parser.add_argument("--parquet", nargs="+", default=[], help="Input TGWAS parquet file(s) for step3.")
    return parser.parse_args() #built-in python method

def open_crash_log(log_path, mode="a"):
    try:
        log_dir = os.path.dirname(log_path)

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        return open(log_path, mode, buffering=1)

    except Exception as e:
        print(f"Warning: cannot open log file {log_path}: {e}", file=sys.stderr)
        return sys.stderr

def build_logger(log_path):
    """Only handles logger + dir/base names based on --log-file."""
    
    dir_name = os.path.dirname(log_path) or "."
    base_name = os.path.splitext(os.path.basename(log_path))[0]

    # create directory if it does not exist
    os.makedirs(dir_name, exist_ok=True)

    log_file = log_path

    return log_file


def build_output_paths(out_path):
    """Only handles logger + dir/base names based on --log-file."""
    
    dir_name = os.path.dirname(out_path) or "."
    base_name = os.path.splitext(os.path.basename(out_path))[0]

    # create directory if it does not exist
    os.makedirs(dir_name, exist_ok=True)
    return dir_name, base_name

def validate_args(args):
    """
    Ensure kinship-related options are only used when --kin-file is provided.
    """
    if not args.kin_file:
        if args.kin_delim != "," or args.kin_diag != 1.0:
            logging.error("--kin-delim or --kin-diag cannot be used without --kin-file.")
            raise SystemExit(2)

def resolve_geno_input(args):
    """
    Reconcile --bgen/--bed/--pgen/--sample into the single genotype-file list
    the rest of the pipeline consumes via args.bgen.

    --bgen, --bed, and --pgen are mutually exclusive. --bed/--pgen take a
    PLINK --bfile/--pfile-style prefix (no extension): "example" resolves to
    example.bed or example.pgen, with the companion .bim/.fam or .pvar/.psam
    auto-detected alongside it, so --sample is optional for --bed/--pgen
    (only needed to override the auto-detected sample file); --sample stays
    required for --bgen (validated downstream, unchanged).

    NOTE: --pgen is CLI plumbing only -- the backend (Plink::process_
    plink_header_block) only accepts .bed today, so a --pgen run fails at
    runtime with a clear "Only .bed is supported" error until PGEN reading
    is actually implemented.
    """
    given = [flag for flag, vals in (("--bgen", args.bgen), ("--bed", args.bed), ("--pgen", args.pgen)) if vals]
    if len(given) > 1:
        logging.error(f"{', '.join(given)} are mutually exclusive; specify only one.")
        raise SystemExit(2)

    args.using_bed = bool(args.bed)
    args.using_pgen = bool(args.pgen)
    args.geno_flag = "--bed" if args.using_bed else "--pgen" if args.using_pgen else "--bgen"

    if args.using_bed:
        args.bgen = [prefix + ".bed" for prefix in args.bed]
        if not args.sample:
            args.sample = [""] * len(args.bgen)
    elif args.using_pgen:
        args.bgen = [prefix + ".pgen" for prefix in args.pgen]
        if not args.sample:
            args.sample = [""] * len(args.bgen)

# def open_intermediate_file(int_path, mode="w"):
#     try:
#         dir_name = os.path.dirname(int_path)

#         # create directory if needed
#         if dir_name:
#             os.makedirs(dir_name, exist_ok=True)

#         return open(int_path, mode)

#     except Exception as e:
#         print(f"Warning: cannot open intermediate file {int_path}: {e}", file=sys.stderr)
#         return None

def build_conf_allsteps(args):
    """Config for all"""
    confopt = ConfOpt(
        pheno_add=args.pheno_file,
        pheno_delim=normalize_delim(args.pheno_delim),
        cov_add=args.cov_file,
        cov_delim=normalize_delim(args.cov_delim),
        geno_add=args.bgen,
        sample_add=args.sample,
        use_sample_file=bool(args.sample),
        do_filters=bool(args.include_snp_file),
        includeVariantFile=args.include_snp_file,
        maf=args.maf,
        stream_snps=args.stream_snps,
        sampleid_header_name=args.sampleid_name,
        covariates=args.covar_names,
        random_slope_header_name=args.random_slope_name,
        missing_key=args.missing_value,
        kin_add=args.kin_file,
        kin_delim=normalize_delim(args.kin_delim),
        kin_diag=args.kin_diag,
        threads=args.threads,
        out_file=args.out,
        log_file=args.log,
        verbose=args.verbose,
    )
    return confopt

def build_conf_step1(args):
    """Config for step1: needs phenotype + covariates, genotype files, kin."""
    confopt = ConfOpt(
        pheno_add=args.pheno_file,
        pheno_delim=normalize_delim(args.pheno_delim),
        cov_add=args.cov_file,
        cov_delim=normalize_delim(args.cov_delim),
        geno_add=args.bgen,
        sample_add=args.sample,
        use_sample_file=bool(args.sample),
        do_filters=bool(args.include_snp_file),
        includeVariantFile=args.include_snp_file,
        maf=args.maf,
        stream_snps=args.stream_snps,
        sampleid_header_name=args.sampleid_name,
        covariates=args.covar_names,
        random_slope_header_name=args.random_slope_name,
        missing_key=args.missing_value,
        kin_add=args.kin_file,
        kin_delim=normalize_delim(args.kin_delim),
        kin_diag=args.kin_diag,
        threads=args.threads,
        corr_file=args.corr_file,
        out_file=args.out,
        log_file=args.log,
        null_log_file=args.null_log,
        verbose=args.verbose,
    )
    return confopt


def build_conf_step2(args):
    """Config for step2: NO phenotype, but covariates, genotype files, kin."""
    confopt = ConfOpt(
        cov_add=args.cov_file,
        cov_delim=normalize_delim(args.cov_delim),
        geno_add=args.bgen,
        sample_add=args.sample,
        use_sample_file=bool(args.sample),
        do_filters=bool(args.include_snp_file),
        includeVariantFile=args.include_snp_file,
        maf=args.maf,
        stream_snps=args.stream_snps,
        sampleid_header_name=args.sampleid_name,
        covariates=args.covar_names,
        random_slope_header_name=args.random_slope_name,
        missing_key=args.missing_value,
        kin_add=args.kin_file,
        kin_delim=normalize_delim(args.kin_delim),
        kin_diag=args.kin_diag,
        threads=args.threads,
        corr_file=args.corr_file,
        out_file=args.out,
        log_file=args.log,
        verbose=args.verbose,
    )
    return confopt

def run_all(dir_name, base_name, args, log_file):
    # ------------------
    # STEP 1 (once)
    # ------------------
    logging.info("%s", "*" * 80)
    logging.info("STEP 1: Fitting null model")
    logging.info("%s", "*" * 80)
    sub_step1 = argparse.Namespace(**vars(args))
    sub_step1.bgen = args.bgen[0]
    sub_step1.sample = args.sample[0]
    conf_step1 = build_conf_step1(sub_step1)
    runner = GEMRunner(conf_step1.get())     
    corr_file = args.corr_file
    #Null model
    runner.run_fit_nullmodel()
    logging.info("correction file (correction) path: %s", corr_file)
    setup_pipeline_log(log_file, mode="a")
    # ------------------
    # STEP 2 (loop)
    # ------------------
    time_step2_satrt = time.time()
    print("*" * 80)
    print("STEP 2: Running Torch GWAS")
    print("*" * 80)

    # Reset peak memory once before the whole STEP2 batch so we can
    # measure the peak memory across the entire loop.
    if args.device == "cuda" and torch.cuda.is_available():
        device_str = f"cuda:{torch.cuda.current_device()}"
        reset_peak_memory(device_str)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    for bgen_i, sample_i in zip(args.bgen, args.sample):
        sub_step2 = argparse.Namespace(**vars(args))
        sub_step2.bgen = bgen_i
        sub_step2.sample = sample_i

        base_i = safe_stem(bgen_i) + "_" + base_name

        conf_step2 = build_conf_step2(sub_step2)

        TGWAS_file = os.path.join(dir_name, base_i + ".parquet")
        runner = GEMRunner(conf_step2.get(), True) 
        run_gwas(
            runner,
            corr_file,               # correction file
            TGWAS_file,
            snps_per_chunk=args.stream_snps,
            device=args.device,
        )
        print(f"TGWAS parquet output: {TGWAS_file}")

    # Log the peak GPU memory for the entire batch
    if args.device == "cuda" and torch.cuda.is_available():
        try:
            peak = get_peak_memory_gb(device_str)
            logging.info("Peak GPU memory (GB) for STEP2 batch: %.3f", peak)
        except Exception:
            pass
    time_step2_end = time.time()
    print("*" * 80)
    print(f"total time for step2 {time_step2_end - time_step2_satrt}")
    # ------------------
    # STEP 3
    # ------------------
    if args.convert:
        print("*" * 80)
        print("STEP 3: Converting binary to text")
        print("*" * 80)
        for bgen_i in args.bgen:
            base_i = safe_stem(bgen_i) + "_" + base_name
            parquet_file = os.path.join(dir_name, base_i + ".parquet")
            output_file = os.path.join(dir_name, base_i + ".txt")

            if not os.path.exists(parquet_file):
                logging.error(f"Parquet file not found, skipping: {parquet_file}")
                raise SystemExit(2)

            # print(f"Converting: {parquet_file} -> {output_file}")
            print(f"STEP 3: Converting {parquet_file} -> {output_file}")
            parquet_to_text_duckdb(parquet_file, output_file)


# Broken steps:
def run_step1(confopt, dir_name, base_name, args):
    """
    STEP 1:
      - GEMRunner init
      - run_fit_nullmodel
      - correction file
      -run_gwas(runner, correction, TGWAS_file, ...)
    """
    corr_file = args.corr_file
    # 1) C++ init
    runner = GEMRunner(confopt.get())

    # 2) Null model
    runner.run_fit_nullmodel()

    logging.info("correction file (correction) path: %s", corr_file)

def run_step2(confopt, dir_name, base_i, base_name, args):
    """
    STEP 2:
      - GEMRunner init
      - run_gwas(runner, correction, parquet file, ...)
    """
    corr_file = args.corr_file
    TGWAS_file = os.path.join(dir_name, base_i + ".parquet")

    # print("STEP 2: Re-initializing GEMRunner and running GWAS/TGWAS...")
    print(f"TGWAS parquet output: {TGWAS_file}")

    runner = GEMRunner(confopt.get(), True) # True to match IDs for each genotype with correction file

    print("Starting GWAS/TGWAS with run_gwas...")

    run_gwas(
        runner,
        corr_file,               # correction file
        TGWAS_file,
        snps_per_chunk=args.stream_snps,
        device=args.device,
    )

def run_step3(args):
    """
    STEP 3:
        - Convert TGWAS_<base_name>.parquet -> <base_name>.txt
        - TGWAS_file (input parquet)
        - output_file (text)
        - logger
    """
    print("STEP 3: Converting binary to text")
    if not args.parquet:
        print("STEP 3 requires --parquet  (output of step2).")
        raise SystemExit(2)

    parquet_file = args.parquet
    # output_file = os.path.join(dir_name, base_name + ".txt")
    output_file = args.out 

    print(f"STEP 3: Converting {parquet_file} -> {output_file}")
    parquet_to_text_duckdb(parquet_file, output_file)


def main():
    global _TEE
    start_time = time.time()
    args = parse_args()
    resolve_geno_input(args)

    crash_fp = open_crash_log(args.log)
    faulthandler.enable(file=crash_fp, all_threads=True)
    crash_fp.write("\n==== log start ====\n")
    crash_fp.flush()
    
    try:
        log_file = build_logger(args.log)
        dir_name, base_name = build_output_paths(args.out)
        if args.step == "all":
            setup_step1_log(log_file, mode="a")
            validate_args(args)
            run_all(dir_name, base_name, args, log_file)

        # Step-specific requirements
        if args.step == "step1":
            setup_step1_log(log_file, mode="a")
            validate_args(args)
            logging.info("%s", "*" * 80)
            logging.info("STEP 1: fitting null model...")

            if getattr(args, "convert", False):
                logging.warning("--convert is only used with --step all. Ignoring it for --step step1.")
            if not args.pheno_file:
                logging.error("STEP 1 requires --pheno-file.")
                raise SystemExit(2)
            if len(args.bgen) != 1:
                logging.error(f"STEP 1 requires exactly ONE {args.geno_flag} file.")
                raise SystemExit(2)
            if len(args.sample) != 1:
                logging.error("STEP 1 requires exactly ONE --sample file.")
                raise SystemExit(2)

            args.bgen = args.bgen[0]
            args.sample = args.sample[0]
            confopt = build_conf_step1(args)
            run_step1(confopt, dir_name, base_name, args)

        elif args.step == "step2":
            setup_pipeline_log(log_file, mode="a")
            if getattr(args, "pheno_file", None):
                print("WARNING: --pheno-file is not used in step2; ignoring it for --step step2.", file=sys.stderr)
            if getattr(args, "convert", False):
                print("WARNING: --convert is only used with --step all. Ignoring it for --step step2.", file=sys.stderr)
            if args.null_log != "null_log.log":
                print("WARNING: --null-log is only used with --step step1. Ignoring it for --step step2.", file=sys.stderr)
            if len(args.bgen) != len(args.sample):
                print(f"{args.geno_flag} count ({len(args.bgen)}) must match --sample count ({len(args.sample)}).")
                raise SystemExit(2)
            
            # Reset peak memory once before the whole step2 batch so we can
            # measure the peak memory across the entire loop.
            if args.device == "cuda" and torch.cuda.is_available():
                device_str = f"cuda:{torch.cuda.current_device()}"
                reset_peak_memory(device_str)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            for bgen_i, sample_i in zip(args.bgen, args.sample):
                sub = argparse.Namespace(**vars(args))
                sub.bgen = bgen_i   
                sub.sample = sample_i  

                base_i = safe_stem(bgen_i) + "_" + base_name   # output: TGWAS_<base_i>.parquet
                print("*" * 80)
                print(f"STEP 2 batch item: bgen={bgen_i} -> sample={sample_i}")
                confopt = build_conf_step2(sub)
                run_step2(confopt, dir_name, base_i, base_name, sub)

            # Log the peak GPU memory for the entire batch
            if args.device == "cuda" and torch.cuda.is_available():
                try:
                    peak = get_peak_memory_gb(device_str)
                    logging.info("Peak GPU memory (GB) for step2 batch: %.3f", peak)
                except Exception:
                    pass
        elif args.step == "step3":
            setup_pipeline_log(log_file, mode="a")
            for pq in args.parquet:
                sub = argparse.Namespace(**vars(args))
                sub.parquet = pq   # use string per run 
                sub.out = str(Path(pq).with_suffix(".txt"))
                print("*" * 80)
                run_step3(sub)

        end_time = time.time()
        print("\nTorchGWAS pipeline step completed successfully.")
        print(f"Wall time: {(end_time - start_time):.2f} seconds")
    except KeyboardInterrupt:
        logging.warning("Pipeline interrupted by user (Ctrl+C).")
        raise SystemExit(130)

    except SystemExit:
        raise

    except Exception:
        logging.exception("Uncaught Python exception:")
        raise SystemExit(1)
    
    finally:
        try:
            logging.shutdown() 
        finally:
            if _TEE is not None:
                _TEE.close()
                _TEE = None

if __name__ == "__main__":
    main()
