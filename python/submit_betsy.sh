#!/usr/bin/env bash

# launch script for running jobs on Betzy
# run with "sbatch submit_betsy.sh"

# inactive settings saved for later are prepended with 'inactive'


### define job settings
#SBATCH --job-name=SICPOVM
#SBATCH --account=nn9284k


### set job type
#SBATCH --partition=preproc     # set job type to 'preproc'
                                # this gives 1 node by default
# inactive #SBATCH --qos=devel             # development job, limited resources



### set number of nodes, tasks, cpus and available memory
# inactive #SBATCH --nodes=4
# inactive #SBATCH --ntasks-per-node=128
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64     # should be 128/ntasks
#SBATCH --mem-per-cpu=2G
# inactive #SBATCH --mem-per-cpu=6G        # specific to 'preproc' job, details from Are


### set wall time limit
#SBATCH --time=1-00:00:00       # DD-HH:MM:SS


### set up job environment
set -o errexit  # exit the script on any error
set -o nounset  # treat any unset variables as an error

module --quiet purge    # reset the modules to the system default

module load Anaconda3/2022.05
source ${EBROOTANACONDA3}/bin/activate

conda activate /cluster/projects/nn9284k/conda/povm
POVM_DEBUG=4 python main_betzy.py
