#!/bin/bash -l
#PBS  -m bea 
#PBS -M ericalindseybusch@gmail.com
#PBS -N run_cha
#PBS -l nodes=1:ppn=16
#PBS -l feature='cellm' 
#PBS -l walltime=120:00:00
#PBS -o /dartfs-hpc/rc/home/4/f002d44/h2a/connhyper/log/${PBS_JOBID}.o
#PBS -e /dartfs-hpc/rc/home/4/f002d44/h2a/connhyper/log/${PBS_JOBID}.e

unset PYTHONPATH
module load python/anaconda2
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate comp_meth_env

python /dartfs/rc/lab/D/DBIC/DBIC/f002d44/code/hyperalignment_example.py