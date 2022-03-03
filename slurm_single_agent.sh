#!/bin/bash -l
#SBATCH --job-name=AirportTowerEnv3
#SBATCH --mail-user=marc.speckmann@stud.uni-hannover.de
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --time=05:00:00
#SBATCH --mem=100G
# node the job ran on
echo "Job ran on:" $(hostname)
# load the relevant modules
module load Miniconda3
conda activate AirportTowerEnv
# run the simulation
python3 single_agent.py