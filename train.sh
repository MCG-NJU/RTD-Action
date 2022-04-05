# First stage
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1
--master_port=10001 --use_env main.py --window_size 100 --batch_size 128 --num_queries 100 --point_prob_normalize --absolute_position --lr 5e-5

# Second stage for relaxation mechanism
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=10002 --use_env main.py --window_size 100 --batch_size 128 --lr 5e-6 --stage 2 --epochs 20 --num_queries 100 --point_prob_normalize --absolute_position --load outputs/checkpoint_best_auc.pth

# Third stage for completeness head
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=10003 --use_env main.py --window_size 100 --batch_size 128 --lr 5e-6 --stage 3 --epochs 10 --num_queries 100 --point_prob_normalize --absolute_position --load outputs/checkpoint_best_auc.pth
