/002/usuarios/ivonne.monter/anaconda3/envs/p38C/bin/python  train.py --batch_sz 8 --gradient_accumulation_steps 32 \
     --savedir checkpoints/ --name emb_dalle_2_uda \
 --data_path data_sets/proc/ \
 --task moviescope --task_type multilabel \
 --model clf_dalle  --type train --uda 1\
 --patience 10 --dropout 0.1 --lr 5e-04 --warmup 0.1 --max_epochs 100 --seed 2