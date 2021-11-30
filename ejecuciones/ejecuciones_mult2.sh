/002/usuarios/ivonne.monter/anaconda3/envs/p38C/bin/python train.py --batch_sz 1 --gradient_accumulation_steps 32 \
 --savedir checkpoints/ --name mult_1_vqgan --type train \
 --data_path data_sets/proc/ \
 --task moviescope --task_type multilabel \
 --model mult2  --num_heads 2 --orig_d_v 1024\
 --patience 5 --dropout 0.1 --lr 0.5 --warmup 0.1 --max_epochs 30 --seed 2
 
 
 
 