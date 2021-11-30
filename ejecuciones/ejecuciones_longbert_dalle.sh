/002/usuarios/ivonne.monter/anaconda3/envs/p38C/bin/python  train.py --batch_sz 8 --gradient_accumulation_steps 32 \
     --savedir checkpoints/ --name longbert_dalle_1\
 --data_path data_sets/proc/ \
 --task moviescope --task_type multilabel \
 --model longBert  --type train\
 --patience 10 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1