#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time=80:00:00
#SBATCH --input=none
#SBATCH -o run_experiment.slurm.out
#SBATCH -e run_experiment.slurm.err


STATUSFILE=run_experiment.slurm.status

# Uncomment if needed on a cluster
echo "Activate holmes conda environment ..."
export PYTHON_EXEC="$WORK/miniconda3/envs/holmes/bin/python"


echo "Run the experiment ..."
STATE='Running'

date "+%Y/%m/%d %H:%M:%S" >> $STATUSFILE
echo $STATE >>  $STATUSFILE

$PYTHON_EXEC run_experiment.py
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




