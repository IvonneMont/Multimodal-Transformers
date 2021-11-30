/002/usuarios/ivonne.monter/anaconda3/envs/p38C/bin/python  train.py --batch_sz 4 --gradient_accumulation_steps 32 \
 --savedir checkpoints/ --name mmbt_mo_1\
 --data_path data_sets/proc/ \
 --task moviescope --task_type multilabel \
 --model mmbt  --type train --num_image_embeds 3 --aug 0 --uda 0 --tuning_metric mif1 --poster raw\
 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1