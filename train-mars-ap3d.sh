CUDA_VISIBLE_DEVICES=0 python train.py --root /mnt/File/shamgholi/datasets/ -d mars --arch ap3dres50 --gpu 0 --save_dir ./logs/log-mars-ap3d-HistByProf --train_batch 14 --test_batch 14 --lr 0.0003 --pretrain "./logs/log-mars-ap3d/best_model.pth.tar"  --eval_step 10 
#python test-all.py --root /home/guxinqian/data/ -d mars --arch ap3dres50 --gpu 0 --resume log-mars-ap3d
