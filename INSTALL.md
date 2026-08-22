# Installing TorchGWAS2

There are three ways to build TorchGWAS2, all driven by the same `CMakeLists.txt`. Pick whichever matches your setup:

| Method | Setup effort | Root/sudo needed | Best for |
|---|---|---|---|
| [Docker](#1-docker-recommended) | None | No (Docker itself, yes) | Reproducible runs, no local toolchain |
| [Conda](#2-conda-environment) | Low | No | Native GPU access without Docker |
| [Manual build](#3-manual-build--custom-install-paths) | High | Usually yes | You already have Boost/MKL/SuiteSparse/etc. installed system-wide |

## Requirements (all methods)

- Linux, x86_64
- GCC/G++ 13 (required for the C++20 features used in this project)
- CMake ≥ 3.14
- An NVIDIA GPU + driver, if you want `--device cuda` (CPU-only works everywhere)

---

## 1. Docker (recommended)

Zero local setup — every dependency (GCC 13, Intel MKL, Boost, SuiteSparse, Eigen, Armadillo, PyTorch) is built into the image.

### Option A: pull the pre-built image (fastest)

```bash
docker pull mzhang9/torchgwas2:latest
# or a specific version:
docker pull mzhang9/torchgwas2:1.0.0

docker run --rm mzhang9/torchgwas2:latest --help
```

### Option B: build the image yourself

```bash
git clone https://github.com/hanchenlab/TorchGWAS2.git
cd TorchGWAS2
git checkout support-to-bed-format

docker build -t torchgwas2:latest .
```

This takes roughly 10–20 minutes and produces an image around 11 GB (base OS + MKL + SuiteSparse build toolchain + a CUDA 12.1 PyTorch wheel). `BASE_IMAGE` defaults to `nvidia/cuda:12.4.0-runtime-ubuntu22.04`; override it if you need a different CUDA runtime base.

Verify it built:

```bash
docker images torchgwas2:latest
docker run --rm torchgwas2:latest --help
```

(Substitute `torchgwas2:latest` for `mzhang9/torchgwas2:latest` in every example below if you built your own instead of pulling.)

### GPU support in Docker

`docker run --gpus all ...` requires the **NVIDIA Container Toolkit** to be installed and registered with Docker on the host — this is separate from just having an NVIDIA driver installed.

### Running

```bash
docker run --rm --gpus all \
  -v /path/to/data:/data \
  -v /path/to/results:/results \
  mzhang9/torchgwas2:latest \
  --step all \
  --pheno-file /data/pheno.txt \
  --cov-file /data/cov.txt \
  --bgen /data/genotypes.bgen \
  --sample /data/genotypes.sample \
  --sampleid-name IID \
  --covar-names age sex PC1 PC2 \
  --threads 16 \
  --out /results/gwas_results \
  --device cuda \
  --convert
```

The container's entrypoint is `python3 /app/RunTorchGWAS.py`, so every flag after the image name is a TorchGWAS2 CLI argument (see [OPTIONS.md](OPTIONS.md)). CPU-only: drop `--gpus all` and pass `--device cpu`.

---

## 2. Conda environment

No root required. Builds and runs natively on the host (direct GPU access — no Docker GPU passthrough to configure).

```bash
conda env create -f environment.yml
conda activate torchgwas2

mkdir build && cd build
cmake ..
make -j$(nproc)
```

`environment.yml` installs the full dependency set: `gcc`/`gxx=13`, `boost`, `eigen`, `armadillo`, `suitesparse`, `gmp`, `mpfr`, `mkl`/`mkl-devel`, and the Python stack (`numpy`, `pandas`, `scipy`, `duckdb`, `pyarrow`, `tqdm`). `CMakeLists.txt` auto-detects the active `$CONDA_PREFIX` and searches it first, so no extra `-D` flags are needed.

**PyTorch is installed via `pip`**. The `pip:` block at the bottom of `environment.yml` pulls the CUDA 12.1 wheel by default:
```yaml
  - pip:
    - --index-url https://download.pytorch.org/whl/cu121
    - torch
    - torchvision
    - torchaudio
```
For a CPU-only machine, change that `--index-url` to `https://download.pytorch.org/whl/cpu`.

The built module lands at `pymodules/Mygen.so`. Run directly with the env's Python:

```bash
python RunTorchGWAS.py --step all --bgen ... --sample ... [other options]
```
---

## 3. Manual build / custom install paths

Use this only if you already have Boost ≥ 1.74 (with dev headers), GMP + MPFR (dev headers), Intel MKL, and SuiteSparse installed somewhere on the system — CMake can be pointed at them, but it cannot install them for you.

```bash
cmake -B build \
      -DMKLROOT=/path/to/mkl \
      -DBOOST_ROOT=/path/to/boost \
      -DGMP_ROOT=/path/to/gmp \
      -DMPFR_ROOT=/path/to/mpfr \
      -DSUITESPARSE_ROOT=/path/to/suitesparse
cmake --build build -j$(nproc)
```

Every dependency location is a CMake cache variable with an auto-detect default (`find_path`/`find_library` against standard system locations, plus `$CONDA_PREFIX` if a conda env happens to be active), so pass only the `-D` flags that auto-detection actually fails to resolve — check the `cmake` output for which ones it found on its own.

Eigen and Armadillo are header-only: if not found locally, CMake fetches them from source automatically (no flag needed). Everything else (Boost, MKL, GMP/MPFR, SuiteSparse) must already exist somewhere — CMake will fail with a clear `FATAL_ERROR` naming exactly which one it couldn't find and what to install or point it at.

Then install PyTorch and the rest of the Python stack yourself (`numpy`, `pandas`, `scipy`, `duckdb`, `pyarrow`, `tqdm`), e.g. via `pip install torch --index-url https://download.pytorch.org/whl/cu121 torchvision torchaudio numpy pandas scipy duckdb pyarrow tqdm`.

---

## Getting the example dataset running

Once built (any method), a minimal smoke test using the bundled `example/` data:

```bash
python RunTorchGWAS.py \
  --step all \
  --pheno-file example/example_sim_multipheno_phe10.txt \
  --cov-file example/example_sim_multipheno_cov.txt \
  --pheno-delim t --cov-delim t \
  --bgen example/example.bgen \
  --sample example/example.sample \
  --sampleid-name IID \
  --covar-names age sex cursmk \
  --threads 4 \
  --out results/test_run \
  --device cpu \
  --convert
```

See [OPTIONS.md](OPTIONS.md) for the complete option reference, BED-format input, MAF filtering, and running the pipeline in separate steps.

## Contact

For comments, suggestions, bug reports and questions, please contact Han Chen (han.chen@nyu.edu) and Mengyu Zhang (mengyu1307@gmail.com). For bug reports, please include an example to reproduce the problem without having to access your confidential data.
