all:
	python main.py --datafolder "./datasets" --dataset "monet2photo" --name "tmp" --mode "train" --res 256 --crop_size 256 --num_epochs 100 --batch_size 8 --identity_loss 0.5 --lambda_identity_x 10 --lambda_identity_y 10 --save_dir "./ckpts/02" --wgan_lambda 10

par:
	python main_parallel.py --datafolder "./datasets" --dataset "monet2photo" --name "tmp" --mode "train" --res 128 --crop_size 32 --num_epochs 100 --batch_size 32

clean:
	rm -f *err *out
