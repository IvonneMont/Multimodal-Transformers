/002/usuarios/ivonne.monter/anaconda3/envs/p38C/bin/python train.py --batch_sz 32 --gradient_accumulation_steps 32 \
 --savedir checkpoints2/ --name mftp_mo_5 --type train \
 --data_path data_sets/proc/ \
 --task moviescope --task_type multilabel \
 --model MultFusion  \
 --patience 20 --dropout 0.5 --lr 0.5 --warmup 0.1 --max_epochs 100 --seed 5
 
 
 
 
