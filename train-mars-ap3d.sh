# run in cccloud
# CUDA_VISIBLE_DEVICES=0 python train.py --root /mnt/File/shamgholi/datasets/ -d mars --arch ap3dres50 --gpu 0 --save_dir /mnt/storage/shamgholi/ap3d_logs/logs/row45 --train_batch 32 --test_batch 32 --lr 0.0003 --eval_step 10 #--weight_decay 0.001

# run in 3080 windows
# set CUDA_VISIBLE_DEVICES=0 & C:\\Users\\STOCKLAND\\.conda\\envs\\shmghli1\\python.exe train.py --root C:\\Users\\STOCKLAND\\Documents\\shamgholi\\datasets -d mars --arch ap3dres50 --gpu 0 --save_dir D:\\shamgholi\\AP3D\\logs\\ap3d_res18_3080 --train_batch 32 --test_batch 32 --lr 0.0003 --eval_step 10 #--weight_decay 0.001

#run in 213
# CUDA_VISIBLE_DEVICES=0 python train.py --root /mnt/File/shamgholi/datasets/ -d mars --arch ap3dres50 --gpu 0 --save_dir /mnt/File/shamgholi/projects/person_reid/AP3D/logs/row44 --train_batch 32 --test_batch 32 --lr 0.0003 --eval_step 10 #--weight_decay 0.001
#--pretrain "./logs/log-mars-ap3d/best_model.pth.tar"
#python test-all.py --root /home/guxinqian/data/ -d mars --arch ap3dres50 --gpu 0 --resume log-mars-ap3d

#run in wsl arman my desktop cpu
# CUDA_VISIBLE_DEVICES=-1 python train.py --root ~/iust/datasets/ -d mars --arch ap3dres50 --gpu 0 --save_dir ./logs/removeme --train_batch 32 --test_batch 32 --lr 0.0003 --eval_step 10

#run in arman 11.5 
CUDA_VISIBLE_DEVICES=0 python train.py --root ~/iust/datasets/ -d mars --arch ap3dres50 --gpu 0 --save_dir ./logs/row51 --train_batch 32 --test_batch 32 --lr 0.0003 --eval_step 10 --distance euclidean
