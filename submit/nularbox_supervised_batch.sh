#!/bin/bash

REPO=${REPO:-$HOME/larch}
source "$REPO/submit/common.sh"

## High level slurm control
QOS=regular
NODES=2
WALLTIME=480
IMAGE=docker:wilkinsonnu/ml_tools:ME

## These are fixed for now
NEPOCH=50
BATCH_SIZE=1024
CONFIG=default_nularbox_supervised.yaml
NEVENTS=2000000
NEVTSTRING="2M"
AUG_TYPE="v1"

EXPT=DUNEND
GENIE_TUNE=GENIE10c
PRESEL="CCCONT256"

## Make a directory for collating logs in
COLLATE_LOG_DIR=${PWD}/logs
mkdir -p "$COLLATE_LOG_DIR"

for WEIGHT_DECAY in 1E-7; do
    for LR in 0.03; do

	## Setup the basic names
	DATA_DIR=${PSCRATCH}/NULARBOX/${GENIE_TUNE}_${EXPT}_${PRESEL}    
	ROOT_NAME=${GENIE_TUNE}${EXPT}_supervised_WGT${WEIGHT_DECAY}_BATCH${BATCH_SIZE}_${NEPOCH}_AUG${AUG_TYPE}_${NEVTSTRING}_N${NODES}
	
	## The file to save the output into
	STATE_FILE=state_${ROOT_NAME}.pth
	    
	## Define the running directory, make it at submission time, and snapshot the repo into it...
	RUN_DIR=$PSCRATCH/larch_runs/$(date +%Y%m%d_%H%M)_$ROOT_NAME
	export RUN_DIR
	snapshot_repo

	## Make the log_dir, and symlink to a convenience location for tensorboard
	LOG_FILE=log_${ROOT_NAME}
	ln -sfn "$RUN_DIR/log_${ROOT_NAME}" "$COLLATE_LOG_DIR/log_${ROOT_NAME}"
	
	## Write the script in the run directory
	JOBSCRIPT=$RUN_DIR/jobscript.sh
	cat > "$JOBSCRIPT" <<EOF
#!/bin/bash
#SBATCH --image=${IMAGE}
#SBATCH --account=dune
#SBATCH --qos=${QOS:-regular}
#SBATCH --constraint=gpu
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --cpus-per-task=32
#SBATCH --time=${WALLTIME}
#SBATCH --licenses=scratch,cfs
#SBATCH --job-name=job_${ROOT_NAME}
#SBATCH --output=${RUN_DIR}/slurm-%j.out

RUN_DIR=${RUN_DIR}
source "\${RUN_DIR}/common.sh"
export PYTHONPATH="\$RUN_DIR/src"

srun --cpu-bind=cores shifter --image=${IMAGE} \\
     python3 -m larch.experiments.nularbox_supervised \\
     --config=${RUN_DIR}/configs/${CONFIG} \\
     --run_dir=${RUN_DIR} \\
     --data_dir=${DATA_DIR} \\
     --log=${LOG_FILE} \\
     --state_file=${STATE_FILE} \\
     --nepoch=${NEPOCH} \\
     --nevents=${NEVENTS} \\
     --lr=${LR} \\
     --weight_decay=${WEIGHT_DECAY} \\
     --batch_size=${BATCH_SIZE} \\
     --seed=${RANDOM}

STATUS=\$?
echo "\$STATUS" > "${RUN_DIR}/exit_code"
if [ "\$STATUS" -eq 0 ]; then touch "${RUN_DIR}/DONE"; else touch "${RUN_DIR}/FAILED"; fi
EOF

	## Do the business
	JOBID=$(sbatch --parsable "$JOBSCRIPT")
	echo "Submitted $JOBID $ROOT_NAME"

	## Also symlink the slurm.out so I can keep track...
        ln -sfn "$RUN_DIR/slurm-${JOBID}.out" "$COLLATE_LOG_DIR/${ROOT_NAME}.out"
    done
done


