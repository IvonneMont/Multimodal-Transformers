/002/usuarios/ivonne.monter/anaconda3/envs/p38C/bin/python train.py --batch_sz 4 --gradient_accumulation_steps 32 \
 --savedir checkpoints2/ --name mmbt_mo_ct_ll_1_10_256 --type train --modality ti\
 --data_path data_sets/proc/ \
 --task moviescope --task_type multilabel \
 --model mmbtct --num_image_embeds 256 --aug 0 --uda 0 --max_seq_len 512 --poster raw --img_hidden_sz 1536\
 --freeze_txt 20 \
 --patience 10 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1