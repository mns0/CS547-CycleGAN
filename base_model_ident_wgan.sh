#!/bin/bash
# setup the environment for python within container
# $MY_PATH and $MY_LD_LIBRARY_PATH are passed in from the 
# job script

export PATH=$MY_PATH
export LD_LIBRARY_PATH=$MY_LD_LIBRARY_PATH
# want this first in PYTHONPATH so that we get the mpi4py that was built
# here on blue waters inside the container
export PYTHONPATH=\
/u/staff/arnoldg/pytorch/lib/python:\
$PYTHONPATH
# Change to your python file
#python test_torch.py

#mpitest
#python main.py --datafolder "./datasets" --dataset "monet2photo" --name "tmp" --mode "train" --res 256 --crop_size 256 --num_epochs 100 --batch_size 8 --identity_loss 0.5 --lambda_identity_x 10 --lambda_identity_y 10 --save_dir "./ckpts/02" --wgan_lambda 10
#python main.py --datafolder "./datasets" --dataset "monet2photo" --name "tmp" --mode "train" --res 256 --crop_size 256 --num_epochs 100 --batch_size 8 --identity_loss 0.5 --lambda_identity_x 10 --lambda_identity_y 10 --save_dir "./ckpts/model_ident" --wgan_lambda 0

python main.py --datafolder "./datasets" --dataset "monet2photo" --name "tmp" --mode "train" --res 256 --crop_size 256 --num_epochs 600 --batch_size 4 --identity_loss 0.5 --lambda_identity_x 10 --lambda_identity_y 10 --save_dir "./ckpts/model_base_model_ident_wgan" --wgan_lambda 10
