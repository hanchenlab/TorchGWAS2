# TorchGWAS2

A GPU-accelerated genome-wide association study (GWAS) tool ([Preprint](https://www.medrxiv.org/content/10.64898/2026.09.10.26362744v1)). TorchGWAS2 fits a mixed null model (linear for quantitative phenotypes, logistic for binary ones) on CPU (C++ / Intel MKL / SuiteSparse) or GPU (PyTorch), then streams genotype dosages and tests each variant for association on GPU or CPU (PyTorch).

## Features

- **Null model fitting** — CPU, or GPU with `--null-device cuda`, which fits every phenotype together and writes the same correction file; linear mixed model for quantitative phenotypes and logistic (PQL) for binary 0/1 ones, optional kinship/relatedness correction. With 100,000 samples (30% related) and 1000 binary phenotypes, step 1 took 8 s on one H200 against 292 s on 8 CPU threads.
- **Association testing** — GPU (CUDA) or CPU, multiple phenotypes in one pass
- **Genotype formats** — BGEN, and PLINK 1.x BED/BIM/FAM
- **Streaming I/O** — genotypes are read and tested in chunks, not loaded whole into memory
- **Three ways to build** — Docker (zero setup), Conda, or a manual build against libraries you already have

## Documentation

- **[INSTALL.md](INSTALL.md)** — how to build/install TorchGWAS2 (Docker, Conda, or manual)
- **[OPTIONS.md](OPTIONS.md)** — full command-line reference, input file formats, pipeline steps, output format, and **examples**

## Quick start

Pull the pre-built image from Docker Hub (see [INSTALL.md](INSTALL.md) to build your own instead):

```bash
docker pull mzhang9/torchgwas2:latest
mkdir results
```

```bash
docker run --rm \
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
    --verbose --convert
```

Add `--null-device cuda` to fit the null model on the GPU as well. Drop `--gpus all` and pass `--device cpu` to run without a GPU. See [OPTIONS.md](OPTIONS.md) for the full option list, BED input, MAF filtering, kinship, and pipeline steps (`--step step1`/`step2`/`step3`).

## Previous work
Please see our previous work [TorchGWAS](https://github.com/ZhiGroup/TorchGWAS) and the accompanying article: [TorchGWAS : GPU-accelerated GWAS for thousands of quantitative phenotypes](https://arxiv.org/abs/2604.21095)

## License

TorchGWAS2 is licensed under the GNU General Public License v3.0 or later (`GPL-3.0-or-later`).

You may redistribute and/or modify this project under the terms of the GNU GPL version 3, or any later version published by the Free Software Foundation. When distributing this software or derivative works, include the corresponding source code and a copy of the GPLv3 license text.

Third-party components included in `thirdparty/` remain under their respective licenses. See the license files in each third-party directory for details.

## Contact

For comments, suggestions, bug reports and questions, please contact Han Chen (han.chen@nyu.edu) and Mengyu Zhang (mengyu1307@gmail.com). For bug reports, please include an example to reproduce the problem without having to access your confidential data.
