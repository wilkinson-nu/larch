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
shifter python3 FSD_contrastive_dist_ME.py 	--data_dir=/pscratch/sd/c/cwilk/FSD/DATA --sim_dir=/pscratch/sd/c/cwilk/FSD/SIMULATION --frac_data=1.0 	--log=log_FSDCC_180725/log_lat128_clust20_nchan128_5E-6_1024_PROJ0.5CLUST0.5_onecycle50_unitcharge_2M_FSDCC --nevents=2000000 --nstep=50 --lr=5E-6 --latent=128 --nclusters=20 	--nchan=128 --enc_act=silu --proj_loss=NTXentMerged --proj_temp=0.5 	--clust_temp=0.5 --batch_size=1024 --aug_type=unitcharge --dropout=0 	--state_file=/pscratch/sd/c/cwilk/state_lat128_clust20_nchan128_5E-6_1024_PROJ0.5CLUST0.5_onecycle50_unitcharge_2M_FSDCC.pth --scheduler=onecycle --world_size=4
