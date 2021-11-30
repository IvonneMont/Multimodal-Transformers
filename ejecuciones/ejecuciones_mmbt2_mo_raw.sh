CUDA_LAUNCH_BLOCKING=1 /002/usuarios/ivonne.monter/anaconda3/envs/p38C/bin/python train.py --batch_sz 10 --gradient_accumulation_steps 32 \
 --savedir checkpoints/ --name mmbt2_mo_raw_256_12 --type train --modality ti\
 --data_path data_sets/proc/ \
 --task moviescope --task_type multilabel \
 --model mmbt2 --num_image_embeds 256  --encoder Resnet152 --poster raw --aug 1\
 --patience 20 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 200 --seed 1