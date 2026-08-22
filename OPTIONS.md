# TorchGWAS2 options

Complete command-line reference for `RunTorchGWAS.py`. See [INSTALL.md](INSTALL.md) to build/install first.

## Contents

- [Overview](#overview)
- [Pipeline steps (`--step`)](#pipeline-steps---step)
- [Basic options](#basic-options)
- [Genotype input](#genotype-input)
- [Sample & covariate options](#sample--covariate-options)
- [Filtering options](#filtering-options)
- [Kinship / relatedness options](#kinship--relatedness-options)
- [File delimiter options](#file-delimiter-options)
- [Computational options](#computational-options)
- [Output options](#output-options)
- [Input file formats](#input-file-formats)
- [Output files](#output-files)
- [Examples](#examples)

## Overview

```
python RunTorchGWAS.py [options]
```

or, via Docker, replace `python RunTorchGWAS.py` with `docker run --rm [--gpus all] -v ...:/data torchgwas2:latest`; every flag below goes after the image name.

TorchGWAS2 runs two computational stages:

1. **Null model fitting** (CPU) — fits a linear mixed model on the phenotype/covariate data, optionally accounting for relatedness via a kinship matrix, and writes fitted residuals/correction factors to `--corr-file`.
2. **Association testing** (GPU or CPU) — streams genotype dosages in chunks and tests each variant against the fitted null model for every phenotype.

## Pipeline steps (`--step`)

| Value | Behavior |
|---|---|
| `all` (default) | Runs null model fitting once, then association testing + text conversion for every genotype file given. Requires `--pheno-file`, `--cov-file`, and exactly one `--sample` per genotype file (BGEN) or none (BED). `--covar-names` isn't CLI-enforced but should be given — without it, the null model is fit with no covariates. |
| `step1` | Fits the null model only. Requires `--pheno-file`, exactly **one** genotype file, exactly one `--sample` (BGEN) or none (BED). Writes `--corr-file` and `--null-log`. |
| `step2` | Association testing only, using an existing `--corr-file` from a prior `step1` run. `--pheno-file` is ignored (warns if given). Accepts multiple genotype files; writes one `.parquet` per file. |
| `step3` | Converts one or more `.parquet` files (`--parquet`) from `step2` into tab-separated `.txt`. |

Running `step1` and `step2` separately (instead of `all`) is useful when you want to fit the null model once and then test many genotype files against it without refitting, or to parallelize `step2` across genotype files independently.

## Basic options

| Option | Argument | Description | Default |
|---|---|---|---|
| `--pheno-file` | path | Phenotype file. Required for `step1`/`all`. | *(none)* |
| `--cov-file` | path | Covariate file. Required for `step1`/`step2`/`all`. | *(none)* |
| `--sampleid-name` | string | Header name of the individual-ID column (2nd column) in the phenotype/covariate files, e.g. `IID`. Required. | *(none)* |
| `--covar-names` | string(s) | Space-separated covariate column names to include, e.g. `age sex PC1 PC2`. Must match the covariate file header exactly (case-sensitive). | *(none)* |
| `--step` | `all`\|`step1`\|`step2`\|`step3` | Which pipeline stage to run. See [above](#pipeline-steps---step). | `all` |

## Genotype input

TorchGWAS2 currently supports **BGEN** and **PLINK 1.x BED**. `--bgen` and `--bed` are mutually exclusive — pass exactly one.

| Option | Argument | Description |
|---|---|---|
| `--bgen` | file(s) | BGEN genotype file(s) (`.bgen`). Requires a matching `--sample` file (one per `--bgen` file, same order). |
| `--sample` | file(s) | BGEN `.sample` file(s). **Required** when using `--bgen`. Optional when using `--bed`. |
| `--bed` | prefix(es) | PLINK 1.x file prefix(es), PLINK `--bfile` style — **no extension**. `--bed example` looks for `example.bed`, `example.bim`, `example.fam` (all three must exist alongside each other). |

For `step2`/`all` with multiple genotype files, pass several files/prefixes to `--bgen`/`--bed` (and, for BGEN, a same-length list to `--sample`) — one output is produced per genotype file.

Sample IDs for BED input are read from the `.fam` file's second column (IID) and must match the IDs in your covariate/phenotype files.

## Sample & covariate options

| Option | Argument | Description | Default |
|---|---|---|---|
| `--sampleid-name` | string | See [Basic options](#basic-options). | *(none)* |
| `--covar-names` | string(s) | See [Basic options](#basic-options). | *(none)* |
| `--missing-value` | string | Token used to represent missing values in the phenotype/covariate files. | `NA` |

## Filtering options

| Option | Argument | Description | Default |
|---|---|---|---|
| `--maf` | float | Minimum minor allele frequency. Variants with `MAF < value` or `MAF > 1 - value` are excluded during genotype streaming. | `0.001` |
| `--include-snp-file` | path | Restrict analysis to a subset of variants. Single-column file; the header must be exactly `snpid` (BGEN or BED) or `rsid` (BGEN only), one identifier per line after the header. All listed variants must exist in the genotype file. | *(none — all variants)* |

## Kinship / relatedness options

| Option | Argument | Description | Default |
|---|---|---|---|
| `--kin-file` | path | Pairwise kinship matrix file (three columns: ID1, ID2, value). If omitted, samples are treated as unrelated. | *(none)* |
| `--kin-diag` | float | Diagonal value for the kinship matrix (e.g. `0.5` if not accounting for inbreeding). | `1.0` |
| `--kin-delim` | delimiter | Kinship file delimiter. See [File delimiter options](#file-delimiter-options). | `,` |

## File delimiter options

`--pheno-delim`, `--cov-delim`, and `--kin-delim` each accept:

| You can pass | Resolves to |
|---|---|
| `,` (default) | comma |
| `t`, `tab`, `TAB`, `\t` | tab |
| `0`, `space`, `\0`, or a literal space | space |

| Option | Description | Default |
|---|---|---|
| `--pheno-delim` | Phenotype file delimiter. | `,` |
| `--cov-delim` | Covariate file delimiter. | `,` |
| `--kin-delim` | Kinship file delimiter (only valid with `--kin-file`). | `,` |

## Computational options

| Option | Argument | Description | Default |
|---|---|---|---|
| `--threads` | int | CPU threads for null-model fitting and genotype reading. | number of CPU cores detected |
| `--stream-snps` | int | Number of variants streamed per chunk during association testing. Higher values use more memory (and, on GPU, more VRAM) but reduce per-chunk overhead. | `1000` |
| `--device` | `cpu`\|`cuda` | Compute device for association testing. `cuda` requires a working PyTorch CUDA build (see [INSTALL.md](INSTALL.md#-known-pitfall-mkl--pytorch-version-conflicts)) and, for the Docker image, `--gpus all` with the NVIDIA Container Toolkit configured on the host. | `cuda` |

## Output options

| Option | Argument | Description | Default |
|---|---|---|---|
| `--out` | path | Output file/prefix. Genotype-file-specific outputs are named `<geno_file_stem>_<out_basename>.parquet`/`.txt` in the same directory. | `out.txt` |
| `--corr-file` | path | Correction-factors file written by `step1`, read by `step2`. Only needs to be set explicitly when running `step1`/`step2` separately (must match between the two invocations). | `correction.txt` |
| `--log` | path | Main log file. | `log.log` |
| `--null-log` | path | Null-model-fitting log/crash file (`step1` only). | `null_log.log` |
| `--verbose` | flag | Print null-model fitting details to the console. | off |
| `--convert` | flag | Also write a tab-separated `.txt` alongside each `.parquet` output. Only meaningful with `--step all` (ignored with a warning for `step1`/`step2`). | off (parquet only) |
| `--parquet` | file | Input `.parquet` file to convert. `step3` only — pass exactly **one** file per invocation (the converter does not combine multiple files; run `step3` once per `.parquet`). | *(none)* |

## Input file formats

### Phenotype file
Delimited (see [delimiters](#file-delimiter-options)), with a header row. At least 3 columns: family ID, individual ID, and one or more phenotype columns.
```
fid    IID       pheno1    pheno2
1      sample1   0.5       1.2
1      sample2   0.8       1.5
2      sample3   NA        1.3
```
- Column 2's header must match `--sampleid-name`.
- Missing values use the token given by `--missing-value` (default `NA`).
- Multiple phenotype columns are tested simultaneously.

### Covariate file
Same shape as the phenotype file: family ID, individual ID, then covariate columns.
```
FID    IID       PC1     PC2     age    sex
1      sample1   0.1     -0.2    45     1
1      sample2   0.3     0.1     52     0
2      sample3   -0.1    0.05    38     1
```
- Names passed to `--covar-names` must exactly match column headers (case-sensitive).

### Kinship file (optional)
Three columns — ID1, ID2, kinship value — with a header (any header text is accepted).
```
id1        id2        kinship
sample1    sample1    1.0
sample1    sample2    0.2
sample2    sample2    1.0
```
Either the full pairwise matrix or just the upper/lower triangle plus diagonal may be given. Diagonal entries are `0`, `0.5`, or `1.0`.

#### Example 1 of Kinship Matrix (with an inbreeding coefficient of 0.05 for the fifth individual):
```

⎡ 0.5    0    0.25  0.25   0  ⎤
⎢  0    0.5   0.25  0.25   0  ⎥
⎢ 0.25  0.25  0.5   0.25   0  ⎥
⎢ 0.25  0.25  0.25  0.5    0  ⎥
⎣  0     0     0     0   0.55 ⎦

```

#### The Kinship File Corresponding to Example 1 (along with --kin-diag 0.5):
```

| ID1 | ID2 | Kinship|
|-----|-----|--------|
|  1  |  3  |  0.25  |
|  1  |  4  |  0.25  |
|  2  |  3  |  0.25  |
|  2  |  4  |  0.25  |
|  3  |  4  |  0.25  |
|  5  |  5  |  0.05  |

```

#### Example 2 of Kinship Matrix (two times the kinship matrix in Example 1):
```
⎡  1    0    0.5   0.5   0  ⎤
⎢  0    1    0.5   0.5   0  ⎥
⎢ 0.5  0.5    1    0.5   0  ⎥
⎢ 0.5  0.5   0.5    1    0  ⎥
⎣  0    0     0     0   1.1 ⎦

```

#### The Kinship File Corresponding to Example 2 (along with --kin-diag 1):
```
| ID1 | ID2 | Kinship|
|-----|-----|--------|
|  1  |  3  |   0.5  |
|  1  |  4  |   0.5  |
|  2  |  3  |   0.5  |
|  2  |  4  |   0.5  |
|  3  |  4  |   0.5  |
|  5  |  5  |   0.1  |

```
### Include-SNP file (optional)
Single column, header exactly `snpid` or `rsid` (case-insensitive), one identifier per line, no duplicates:
```
snpid
chr1:100000:A:G
chr1:200000:C:T
```

### Genotype files
See [Genotype input](#genotype-input) above.

## Output files

Each genotype file produces:

- **`<geno_file_stem>_<out_basename>.parquet`** — primary output, written during `step2`/`all`. Columns: `SNPID`, `RSID`, `CHR`, `POS`, `Non_Effect_Allele`, `Effect_Allele`, `N_Samples`, `AF`, `GV`, then for each phenotype: `<pheno>_BETA`, `<pheno>_SE`, `<pheno>_pvalue`.
- **`<geno_file_stem>_<out_basename>.txt`** — the same data as tab-separated text. Written when `--convert` is passed (`all`) or explicitly via `--step step3`.

## Examples for Steps

**BGEN input, full pipeline, GPU:**
```bash
python RunTorchGWAS.py --step all \
  --pheno-file pheno.txt --cov-file cov.txt \
  --bgen genotypes.bgen --sample genotypes.sample \
  --sampleid-name IID --covar-names age sex PC1 PC2 \
  --kin-file kinship.txt --kin-diag 0 \
  --maf 0.01 --threads 16 --stream-snps 2000 \
  --out results/gwas --device cuda --convert
```

**BED input (prefix, no `--sample` needed), CPU:**
```bash
python RunTorchGWAS.py --step all \
  --pheno-file pheno.txt --cov-file cov.txt \
  --bed genotypes \
  --sampleid-name IID --covar-names age sex PC1 PC2 \
  --threads 8 --out results/gwas --device cpu --convert
```

**Fit the null model once, then test multiple BGEN files separately:**
```bash
# step1
python RunTorchGWAS.py --step step1 \
  --pheno-file pheno.txt --cov-file cov.txt \
  --bgen chr1.bgen --sample chr1.sample \
  --sampleid-name IID --covar-names age sex \
  --corr-file correction.txt --out results/gwas

# step2 (repeat per chromosome/file)
python RunTorchGWAS.py --step step2 \
  --cov-file cov.txt \
  --bgen chr1.bgen chr2.bgen --sample chr1.sample chr2.sample \
  --sampleid-name IID --covar-names age sex \
  --corr-file correction.txt --out results/gwas --device cuda

# step3 (one file per invocation)
python RunTorchGWAS.py --step step3 \
  --parquet results/chr1_gwas.parquet \
  --out results/chr1_gwas.txt
```

## Run example data

### Run with docker
```bash
# GPU run
nohup /usr/bin/time -v docker run --rm \
    --gpus all \
    -v example:/data \
    -v results:/results \
    -w /app \
    mzhang9/torchgwas2:latest \
    --step all \
    --pheno-file /data/example_sim_multipheno_phe100.txt \
    --cov-file /data/example_sim_multipheno_cov.txt \
    --kin-file /data/example.kinship \
    --kin-delim t --cov-delim t --pheno-delim t \
    --kin-diag 0 \
    --bgen /data/example.bgen \
    --sample /data/example.sample \
    --corr-file /results/intermediate_docker_phe100_2ksnps.txt \
    --out /results/docker_phe100_2ksnps \
    --log /results/docker_phe100_2ksnps_stepall.log \
    --null-log /results/docker_phe100_2ksnps_null.log \
    --sampleid-name IID \
    --covar-names age sex cursmk \
    --stream-snps 1000 \
    --threads 20 \
    --device cuda \
    --verbose --convert \
    > results/docker_phe100_stdout_streamsnp1000_thread20_gpu.log 2>&1 &

# CPU run
nohup /usr/bin/time -v docker run --rm \
    -v example:/data \
    -v results:/results \
    -w /app \
    mzhang9/torchgwas2:latest \
    --step all \
    --pheno-file /data/example_sim_multipheno_phe100.txt \
    --cov-file /data/example_sim_multipheno_cov.txt \
    --kin-file /data/example.kinship \
    --kin-delim t --cov-delim t --pheno-delim t \
    --kin-diag 0 \
    --bgen /data/example.bgen \
    --sample /data/example.sample \
    --corr-file /results/intermediate_docker_phe100_2ksnps.txt \
    --out /results/docker_phe100_2ksnps \
    --log /results/docker_phe100_2ksnps_stepall.log \
    --null-log /results/docker_phe100_2ksnps_null.log \
    --sampleid-name IID \
    --covar-names age sex cursmk \
    --stream-snps 1000 \
    --threads 20 \
    --device cpu \
    --verbose --convert \
    > results/docker_phe100_stdout_streamsnp1000_thread20_cpu.log 2>&1 &
```
### Run with conda environment
```bash


# conda env create -f environment.yml
# conda activate torchgwas2
# mkdir build && cd build && cmake .. && make -j$(nproc)
# conda activate torchgwas2

# GPU run
export CUDA_VISIBLE_DEVICES=0
nohup /usr/bin/time -v python RunTorchGWAS.py \
    --step all \
    --pheno-file example/example_sim_multipheno_phe100.txt \
    --cov-file example/example_sim_multipheno_cov.txt \
    --kin-file example/example.kinship \
    --kin-delim t --cov-delim t --pheno-delim t \
    --kin-diag 0 \
    --bgen example/example.bgen \
    --sample example/example.sample \
    --corr-file results/intermediate_python_phe100_10ksnps.txt \
    --out results/python_phe100_10ksnps \
    --log results/python_phe100_10ksnps_stepall.log \
    --null-log results/python_phe100_10ksnps_null.log \
    --sampleid-name IID \
    --covar-names age sex cursmk \
    --maf 0 \
    --stream-snps 1000 \
    --threads 20 \
    --device cuda \
    --verbose --convert \
    > results/python_phe100_stdout_streamsnp1000_thread20_gpu.log 2>&1 &

# CPU run
nohup /usr/bin/time -v python RunTorchGWAS.py \
    --step all \
    --pheno-file example/example_sim_multipheno_phe100.txt \
    --cov-file example/example_sim_multipheno_cov.txt \
    --kin-file example/example.kinship \
    --kin-delim t --cov-delim t --pheno-delim t \
    --kin-diag 0 \
    --bgen example/example.bgen \
    --sample example/example.sample \
    --corr-file results/intermediate_python_phe100_10ksnps.txt \
    --out results/python_phe100_10ksnps \
    --log results/python_phe100_10ksnps_stepall.log \
    --null-log results/python_phe100_10ksnps_null.log \
    --sampleid-name IID \
    --covar-names age sex cursmk \
    --maf 0 \
    --stream-snps 1000 \
    --threads 20 \
    --device cpu \
    --verbose --convert \
    > results/python_phe100_stdout_streamsnp1000_thread20_cpu.log 2>&1 &

## Bed file run
nohup /usr/bin/time -v python RunTorchGWAS.py \
    --step all \
    --pheno-file example/example_sim_multipheno_phe100.txt \
    --cov-file example/example_sim_multipheno_cov.txt \
    --kin-file example/example.kinship \
    --kin-delim t --cov-delim t --pheno-delim t \
    --kin-diag 0 \
    --bed example/example \
    --corr-file results/intermediate_python_phe100_10ksnps.txt \
    --out results/python_phe100_10ksnps \
    --log results/python_phe100_10ksnps_stepall.log \
    --null-log results/python_phe100_10ksnps_null.log \
    --sampleid-name IID \
    --covar-names age sex cursmk \
    --maf 0 \
    --stream-snps 1000 \
    --threads 20 \
    --device cpu \
    --verbose --convert \
    > results/python_phe100_stdout_streamsnp1000_thread20_bed_cpu.log 2>&1 &
```

## Contact

For comments, suggestions, bug reports and questions, please contact Han Chen (han.chen@nyu.edu) and Mengyu Zhang (mengyu1307@gmail.com). For bug reports, please include an example to reproduce the problem without having to access your confidential data.
