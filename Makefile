all:
	python main.py --datafolder "./datasets" --dataset "monet2photo" --name "tmp" --mode "train" --res 128 --crop_size 32 --num_epochs 100 --batch_size 16

par:
	python main_parallel.py --datafolder "./datasets" --dataset "monet2photo" --name "tmp" --mode "train" --res 128 --crop_size 32 --num_epochs 100 --batch_size 32

clean:
	rm -f *err *out
