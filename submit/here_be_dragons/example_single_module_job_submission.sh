#!/bin/bash
#SBATCH --image=docker:wilkinsonnu/ml_tools:ME
#SBATCH --account=dune
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=1440
#SBATCH --ntasks-per-node=4
export OMP_NUM_THREADS=16
export MASTER_ADDR=$(hostname)
export MASTER_PORT='12355'
export WORLD_SIZE=4

## These are variable parameters which you may want to change
indir=/global/cfs/cdirs/dune/users/cwilk/single_module_unsupervised/h5_inputs_v9/
nevents=5000000
nstep=80
lr=5e-6
latent=128
nchan=32
hidden_act=silu
latent_act=tanh
loss=NTXentMerged
temperature=0.5
batch_size=1536
aug_type=bigmodblock10x10
dropout=0
scheduler=onecycle

## Change log and state_file names based on the above (be careful to make these distinct)
log=log/log_lat${latent}_nchan${nchan}_${lr}_${batch_size}_${loss}${temperature}_${scheduler}_${aug_type}_5M_ME_v9
state_file=${SCRATCH}/state_lat${latent}_nchan${nchan}_${lr}_${batch_size}_${loss}${temperature}_${scheduler}_${aug_type}_5M_ME_v9.pth

echo "Trained model will be saved in ${state_file}"
echo "Writing log to ${log}"
shifter python3 single_module_contrastive_dist_ME.py \
	--indir=${indir} --log=${log} --nevents=${nevents} --nstep=${nstep} --lr=${lr} --latent=${latent} \
	--nchan=${nchan} --hidden_act=${hidden_act} --latent_act=${latent_act} --latent_loss=${loss} \
	--ntx_temp=${temperature} --batch_size=${batch_size} --aug_type=${aug_type} --dropout=${dropout} \
	--state_file=${state_file} --scheduler=${scheduler} --world_size=${WORLD_SIZE}
