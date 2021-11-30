/002/usuarios/ivonne.monter/anaconda3/envs/p38C/bin/python train.py --batch_sz 8 --gradient_accumulation_steps 32 \
 --savedir checkpoints/ --name mmbt_mo_raw_5 --type train --modality ti\
 --data_path data_sets/proc/ \
 --task moviescope --task_type multilabel \
 --model mmbt --num_image_embeds 3 --img_hidden_sz 2048 --encoder Resnet152 --poster raw --aug 0\
 --patience 10 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 5