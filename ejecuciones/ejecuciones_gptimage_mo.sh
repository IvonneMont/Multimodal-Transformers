/002/usuarios/ivonne.monter/anaconda3/envs/taming/bin/python  train.py --batch_sz 4 --gradient_accumulation_steps 32 \
     --savedir checkpoints2/ --name gpt2img_mo_1_dp_0105 \
 --data_path data_sets/proc/ \
 --task moviescope --task_type multilabel \
 --model gpt2image  --type train --aug 0\
 --patience 5 --dropout 0.1 --lr 1e-03 --warmup 0.1 --max_epochs 100 --seed 1