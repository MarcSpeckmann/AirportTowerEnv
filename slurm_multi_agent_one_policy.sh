#!/bin/bash -l
#SBATCH --job-name=AirportTowerEnvMulti
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=55
#SBATCH --time=05:00:00
#SBATCH --mem=55G
# node the job ran on
echo "Job ran on:" $(hostname)
# load the relevant modules
module load Miniconda3
conda activate AirportTowerEnv
# run the simulation
python3 multi_agent_one_policy.py