all:
	python main.py --datafolder "./datasets" --dataset "monet2photo" --name "tmp" --mode "train" --res 256 --crop_size 256 --num_epochs 600 --batch_size 8 --identity_loss 0.0 --lambda_identity_x 0 --lambda_identity_y 0 --save_dir "./ckpts/model_base_model" --wgan_lambda 0


ident_loss:
	python main.py --datafolder "./datasets" --dataset "monet2photo" --name "tmp" --mode "train" --res 256 --crop_size 256 --num_epochs 600 --batch_size 4 --identity_loss 0.5 --lambda_identity_x 10 --lambda_identity_y 10 --save_dir "./ckpts/model_base_model_ident" --wgan_lambda 0

ident_loss_wgan:
	python main.py --datafolder "./datasets" --dataset "monet2photo" --name "tmp" --mode "train" --res 256 --crop_size 256 --num_epochs 600 --batch_size 4 --identity_loss 0.5 --lambda_identity_x 10 --lambda_identity_y 10 --save_dir "./ckpts/model_base_model_ident_wgan" --wgan_lambda 10





clean:
	rm -f *err *out
