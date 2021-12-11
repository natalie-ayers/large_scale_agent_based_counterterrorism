#!/bin/bash

#SBATCH --job-name=promising_models_mpi
#SBATCH --ntasks=18
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr
#SBATCH --account=macs30123

# Load Python and MPI modules
module load python/anaconda-2019.03
module load intelmpi/2018.2.199+intel-18.0

# Run models with mpirun
mpirun -n 18 python3 ./promising_models_mpi.py > ./promising_models_mpi.out 