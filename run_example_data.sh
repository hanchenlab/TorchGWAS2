cd TorchGWAS2

## Run with docker

# GPU run
nohup /usr/bin/time -v docker run --rm \
    --gpus all \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v /home/mzhang12/TorchGWAS2/example:/data \
    -v /home/mzhang12/TorchGWAS2/results:/results \
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
## CPU run
nohup /usr/bin/time -v docker run --rm \
    -v /home/mzhang12/TorchGWAS2/example:/data \
    -v /home/mzhang12/TorchGWAS2/results:/results \
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

## Run with conda environment

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