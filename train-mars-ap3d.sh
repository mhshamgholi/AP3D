CUDA_VISIBLE_DEVICES=0 python train.py --root /mnt/File/shamgholi/datasets/ -d mars --arch ap3dres50 --gpu 0 --save_dir /mnt/File/shamgholi/projects/person_reid/AP3D/logs/HistByProf2_row30 --train_batch 10 --test_batch 10 --lr 0.0003 --eval_step 10 
#--pretrain "./logs/log-mars-ap3d/best_model.pth.tar"
#python test-all.py --root /home/guxinqian/data/ -d mars --arch ap3dres50 --gpu 0 --resume log-mars-ap3d
