#!/bin/bash           # the shell language when run outside of the job scheduler
#                     # lines starting with #$ is an instruction to the job scheduler
#$ -S /bin/bash       # the shell language when run via the job scheduler [IMPORTANT]
#$ -cwd               # job should run in the current working directory
#$ -j y               # STDERR and STDOUT should be joined
#$ -l mem_free=16    # job requires up to 1 GiB of RAM per slot
#$ -l scratch=70G      # job requires up to 60 GiB of local /scratch space
#$ -l h_rt=16:00:00   # job requires up to 2 hours of runtime
##$ -t 1-10           # array job with 10 tasks (remove first '#' to enable)
##$ -r y               # if job crashes, it should be restarted

## If you array jobs (option -t), this script will run T times, once per task.
## For each run, $SGE_TASK_ID is set to the corresponding task index (here 1-10).
## To configure different parameters for each task index, one can use a Bash 
## array to map from the task index to a parameter string.

## All possible parameters
# params=(1bac 2xyz 3ijk 4abc 5def 6ghi 7jkl 8mno 9pqr 10stu)

## Select the parameter for the current task index
## Arrays are indexed from 0, so we subtract one from the task index
# param="${params[$((SGE_TASK_ID - 1))]}"

echo "sigcifs2biommtf_job.sh START $1 STOP $2"
date
hostname

module load Sali
conda activate wyndevpy39
which python3
python3 --version

for file in $(ls significant_cifs/*.cif | sed -n "$1,$2p" ); do
    pdb_id=${file:17:-4}
    if [ -f significant_cifs/${pdb_id}.bio.mmtf ]; then
        echo "BASH SKIP ${pdb_id}"
    else
        echo "BASH RUN ${pdb_id}"
        python3 pdb2mmtf.py -i $file -o significant_cifs -from "cif" -to "biommtf" 
    fi
done

echo "End python job"
## End-of-job summary, if running as a job
[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"  # This is useful for debugging and usage purposes,
                                          # e.g. "did my job exceed its memory request?"

