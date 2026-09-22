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
data_dir=/pscratch/sd/c/cwilk/FSD/DATA
sim_dir=/pscratch/sd/c/cwilk/FSD/SIMULATION
nevents=5000000
frac_data=1.0

nstep=60
lr=5e-6
latent=128
nchan=32
hidden_act=silu
latent_act=tanh
loss=NTXentMerged
temperature=0.5
batch_size=1536
aug_type=unitcharge
dropout=0
scheduler=onecycle
arch=shallow

## Change log and state_file names based on the above (be careful to make these distinct)
log=log/log_lat${latent}_nchan${nchan}_${lr}_${batch_size}_${loss}${temperature}_${scheduler}_${aug_type}_${arch}_5M${frac_data}_FSD
state_file=${SCRATCH}/state_lat${latent}_nchan${nchan}_${lr}_${batch_size}_${loss}${temperature}_${scheduler}_${aug_type}_${arch}_5M${frac_data}_FSD.pth

echo "Trained model will be saved in ${state_file}"
echo "Writing log to ${log}"
shifter python3 FSD_contrastive_dist_ME.py \
	--data_dir=${data_dir} --sim_dir=${sim_dir} --frac_data=${frac_data} \
	--log=${log} --nevents=${nevents} --nstep=${nstep} --lr=${lr} --latent=${latent} \
	--nchan=${nchan} --hidden_act=${hidden_act} --latent_act=${latent_act} --latent_loss=${loss} \
	--ntx_temp=${temperature} --batch_size=${batch_size} --aug_type=${aug_type} --dropout=${dropout} \
	--state_file=${state_file} --scheduler=${scheduler} --world_size=${WORLD_SIZE} --arch=${arch}
