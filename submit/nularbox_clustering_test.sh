#!/bin/bash 

## Get an interactive job to run this with:
# salloc --nodes 1 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 --qos interactive --time 04:00:00 --constraint gpu --account dune
export IMAGE=docker:wilkinsonnu/ml_tools:ME
export REPO=$HOME/larch
source "$REPO/submit/common.sh"

## Setup the run directory
JOBNAME=clustering_testing
export RUN_DIR=$PSCRATCH/larch_test_runs/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}_${JOBNAME}
snapshot_repo && activate_snapshot

## Basic inputs and outputs
DATA_DIR=$PSCRATCH/NULARBOX/GENIE10c_NuMIME_CCCONT256
LOGFILE=log_${JOBNAME}
STATE_FILE=state_${JOBNAME}.pth

## Trained encoder from here
PRETRAINED=/global/cfs/cdirs/dune/users/cwilk/larch_models/state_GENIE10cNuMIME_MAXPOOL_VICReg_WGT1E-6_BATCH1024_50_AUGv1m_2M_N2.pth

## Get the basic config here:
CONFIG=$RUN_DIR/configs/default_nularbox_clustering.yaml

## Run specific arguments
NEVENTS=200000
NEPOCH=50

## Do the business
srun --cpu-bind=cores shifter --image=${IMAGE} \
     python3 -m larch.experiments.nularbox_clustering \
     --config=${CONFIG} \
     --run_dir=${RUN_DIR} \
     --data_dir=${DATA_DIR} \
     --pretrained=${PRETRAINED} \
     --log=${LOGFILE} \
     --state_file=${STATE_FILE} \
     --nevents=${NEVENTS} \
     --nepoch=${NEPOCH}


