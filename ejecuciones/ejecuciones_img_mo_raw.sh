/002/usuarios/ivonne.monter/anaconda3/envs/p38C/bin/python train.py --batch_sz 4 --gradient_accumulation_steps 32 \
 --savedir checkpoints2/ --name  img_mo_5  --type train --modality ti\
 --data_path data_sets/proc/ \
 --task moviescope --task_type multilabel \
 --model img  --num_image_embeds 1 --img_hidden_sz 2048  --aug 0 --uda 0\
 --patience 10 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 5