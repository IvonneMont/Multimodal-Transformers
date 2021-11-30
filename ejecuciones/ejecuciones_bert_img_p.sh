/002/usuarios/ivonne.monter/anaconda3/envs/p38C/bin/python  train.py --batch_sz 16 --gradient_accumulation_steps 32 \
     --savedir checkpoints/ --name bert_imgt_256_1 \
 --data_path data_sets/proc/ \
 --task moviescope --task_type multilabel \
 --model bert_img  --type train --hidden_sz 1536\
 --patience 10 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 500 --seed 1