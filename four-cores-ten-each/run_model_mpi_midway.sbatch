#!/bin/bash

#SBATCH --job-name=run_model_mpi_midway
#SBATCH --ntasks=4
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr
#SBATCH --account=macs30123

# Load Python and MPI modules
module load python/anaconda-2019.03
module load intelmpi/2018.2.199+intel-18.0

# Run models with mpirun
mpirun -n 4 python3 ./run_model_mpi_midway.py > ./run_model_mpi_midway.out 