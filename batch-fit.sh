#!/bin/bash
#SBATCH --partition             fhs-fast
#SBATCH --ntasks                1
#SBATCH --nodes                 1
#SBATCH --tasks-per-node        5
#SBATCH --cpus-per-task         9
#SBATCH --mem-per-cpu           4G
#SBATCH --time                  120:00:00
#SBATCH --job-name              fit-nav
#SBATCH --output                log/fit-hh-vc1-cell1-NaIVCP80.%j.out
#SBATCH --error                 log/fit-hh-vc1-cell1-NaIVCP80.%j.err
##SBATCH --mail-type             ALL
##SBATCH --mail-user             chonloklei@um.edu.mo

source /etc/profile
source /etc/profile.d/modules.sh
source /home/chonloklei/m  # Load miniconda

ulimit -s unlimited

# Load module
module purge
# module load intel impi
# NOTE: Running multiple MPI jobs per node is currently only possible with IntelMPI and MVAPICH2 (i.e. OpenMPI does not work).
#       https://scitas-data.epfl.ch/confluence/display/DOC/Running+multiple+tasks+on+one+node

# Path and Python version checks
pwd
python --version
conda activate /home/chonloklei/nav-artefact-model/env  # Load miniconda venv
python --version
which python

# Set up
model="hh"
protocol="NaIVCP80"
level="1"

# We are using multiprocessing, so switch multi-threading off
# https://stackoverflow.com/a/43897781
# export OMP_NUM_THREADS=1

# Run

for data in cell1 cell2
do
    for i in 1:5
    do
        srun --exclusive --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --mem=36G python -u fit.py -m $model -p $protocol -d $data -l $level &
        sleep 5
    done
    wait
done

echo "Done."
