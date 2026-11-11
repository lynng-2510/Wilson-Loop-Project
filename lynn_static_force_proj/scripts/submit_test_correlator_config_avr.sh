#!/bin/bash
#SBATCH --job-name=correlator_with_E_insertion
#SBATCH --partition=compute2011
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=/home/m2427382/Wilson-Loop-Project/lynn_static_force_proj/logs/job_%A_task_%a.out

set -euo pipefail

cd /home/m2427382/Wilson-Loop-Project/lynn_static_force_proj

module purge
module load compiler/gcc/10.2.1
module load mpi/openmpi/4.1.0-no_ucx
module load mkl/latest

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_THREADING_LAYER=sequential

export NCFG="${NCFG:-${SLURM_ARRAY_TASK_ID:-1}}"
export RMAX="${RMAX:-4}" 
export TMAX="${TMAX:-8}"
export T0_MODE="${T0_MODE:-fixed}"
export T0_FIXED="${T0_FIXED:-0}"
export T0_START="${T0_START:-0}"
export T0_STRIDE="${T0_STRIDE:-1}"

echo "HOST=$(hostname)"
echo "DATE=$(date)"
echo "JOB_ID=${SLURM_JOB_ID:-none}"
echo "ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-none}"
echo "NCFG=${NCFG}"
echo "RMAX=${RMAX}"
echo "TMAX=${TMAX}"
echo "T0_MODE=${T0_MODE}"

mkdir -p results

mpirun \
  -x NCFG \
  -x RMAX \
  -x TMAX \
  -x T0_MODE \
  -x T0_FIXED \
  -x T0_START \
  -x T0_STRIDE \
  -np 1 \
  ./build/test_correlator_config_avr > results/correlator_cfg_${NCFG}.csv # append output to a text file

echo "cfg,T,R,Nsrc,Re,Im" > results/spatial_avg_RT_R1-4_T1-8_n1-100.csv
cat results/correlator_cfg_*.csv >> results/spatial_avg_RT_R1-4_T1-8_n1-100.csv