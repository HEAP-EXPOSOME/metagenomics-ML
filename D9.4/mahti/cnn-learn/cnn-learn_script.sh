#!/bin/bash
#SBATCH --job-name=CNN-learn
#SBATCH --account=project_2002659
#SBATCH --time=0-01:00:00 
#SBATCH --mem=480000
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --output="job-%j.out"
#SBATCH --error="job-%j.err"

timer=`date +%s`

function log {
  local output=$(echo "";echo "`date +'%y-%m-%d %H:%M:%S'`" ; while IFS= read -r line; do echo -e "\t$line"; done < <(echo -e "$@");echo "")
  echo "$output"
  echo "$output" 1>&2
}

function esc {
    echo "$@" | sed 's#\\#\\\\\\\\#g'
}

log "Current date: $(esc "`date`")"
log "Master host: $(esc "`/bin/hostname`")"
log "Working directory: $(esc "`pwd`")"
log "Current job: $SLURM_JOB_ID\n$(esc "`scontrol show job $SLURM_JOB_ID`")"
log "Current script: $0\n$(esc "`cat -n $0`")"
log "Python script: cnn-learn.py\n$(esc "`cat -n cnn-learn.py`")"

# --- LOADING MODULES ---
log "Loading modules"
module load tensorflow

# --- PREPARING WORKING DIR ---
JOB_ID=$SLURM_JOB_ID
mkdir job-$JOB_ID && cd job-$JOB_ID
cp ../cnn-learn.py ./

# --- PREPARING NODES LIST ---
log "Preparing nodes list"
srun hostname -s | sort > all_nodes.txt

log "All nodes: `uniq -c all_nodes.txt | xargs -I {} echo ' {}' | paste -sd ','`"

# --- RUNNING TESTS ---
log "Running tests"

python3 cnn-learn.py \
            | tee >(cat >&2)

# --- COMPLETED ---
timer=$(( `date +%s` - $timer ))
h=$(( $timer / (60 * 60) ))
m=$(( ($timer / 60) % 60 ))
s=$(( $timer % 60 ))
log "Script completed after ${h}h ${m}m ${s}s."

# EOF
