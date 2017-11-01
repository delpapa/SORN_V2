#!/bin/bash

#SBATCH --partition=sleuths
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=7000
#SBATCH --job-name="sorn__RandomSequenceTask_perfXtplast_test"
#SBATCH --output=sorn__RandomSequenceTask_perfXtplast_test.out
#SBATCH --mail-user=delpapa@fias.uni-frankfurt.de
#SBATCH --mail-type=END
#SBATCH --time=7-00:00:00

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES

echo "working directory = "$SLURM_SUBMIT_DIR

srun python common/run_multiple.py RandomSequenceTask
