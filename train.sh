# train all-in-one
python train.py --device 0 --mode all_in_one --epochs 347 --val-epochs 5 --epoch-chg 135 238 314 347 --train-dir /home/sht/wxf/all_in_one_dataset
# train denoising
# python train.py --device 0 --mode noise --epochs 1005 --val-epochs 10 --epoch-chg 390 689 908 1005 --train-dir /home/sht/wxf/all_in_one_dataset
# train deraining
# python train.py --device 0 --mode rain --epochs 1054 --val-epochs 10 --epoch-chg 409 722 952 1054 --train-dir /home/sht/wxf/all_in_one_dataset
# train dehazing
# python train.py --device 0 --mode haze --epochs 1054 --val-epochs 10 --epoch-chg 409 722 952 1054 --train-dir /home/sht/wxf/all_in_one_dataset