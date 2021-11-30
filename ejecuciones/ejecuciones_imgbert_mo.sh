CUDA_LAUNCH_BLOCKING=1 /002/usuarios/ivonne.monter/anaconda3/envs/p38C/bin/python train.py --batch_sz 32 --gradient_accumulation_steps 32 \
 --savedir checkpoints/ --name imgbert_mo_puda_4 --type train --modality ti\
 --data_path data_sets/proc/ \
 --task moviescope --task_type multilabel \
 --model imgbert --aug 0 --dir_bertML bert_img_uda\
 --patience 5 --dropout 0.1 --lr 5e-04 --warmup 0.1 --max_epochs 100 --seed 4