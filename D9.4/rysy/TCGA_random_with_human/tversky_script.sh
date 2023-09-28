#!/bin/bash
#SBATCH --job-name=tversky_with
#SBATCH --account=g93-1594
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=360000M
#SBATCH --ntasks-per-node=1
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

PYTHON_SCRIPT="tversky.py"

log "Current date: $(esc "`date`")"
log "Master host: $(esc "`/bin/hostname`")"
log "Working directory: $(esc "`pwd`")"
log "Current job: $SLURM_JOB_ID\n$(esc "`scontrol show job $SLURM_JOB_ID`")"
log "Current script: $0\n$(esc "`cat -n $0`")"
log "Python script: ${PYTHON_SCRIPT}\n$(esc "`cat -n ${PYTHON_SCRIPT}`")"

# --- RUNNING TESTS ---
log "Running tests"

python3 ${PYTHON_SCRIPT} \
                | tee >(cat >&2)

# --- COMPLETED ---
timer=$(( `date +%s` - $timer ))
h=$(( $timer / (60 * 60) ))
m=$(( ($timer / 60) % 60 ))
s=$(( $timer % 60 ))
log "Script completed after ${h}h ${m}m ${s}s."

# EOF
