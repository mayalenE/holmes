#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --input=none
#SBATCH -o run_calc_statistic_space_representation.slurm.out
#SBATCH -e run_calc_statistic_space_representation.slurm.err


STATUSFILE=run_calc_statistic_space_representation.slurm.status

# Uncomment if needed on a cluster
echo "Activate holmes conda environment ..."
export PYTHON_EXEC="$WORK/miniconda3/envs/holmes/bin/python"

echo "Run ..."
STATE='Running'

date "+%Y/%m/%d %H:%M:%S" >> $STATUSFILE
echo $STATE >>  $STATUSFILE

$PYTHON_EXEC calc_statistic_space_representation.py
RETURN_CODE=$?

echo "Write status file ..."
if [ $RETURN_CODE == 0 ] 
then
	STATE='Finished'
else
	STATE='Error'
fi

date "+%Y/%m/%d %H:%M:%S" >> $STATUSFILE
echo $STATE >> $STATUSFILE

echo "Finished."

